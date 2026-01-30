import argparse
import csv
import json
import math
import os
import random
import shutil
import time
import sys

from arm_sim.behavior.aggregate_cache import AggregateCacheBehavior
from arm_sim.behavior.base import ZoneBehavior
from arm_sim.core.events import EventType
from arm_sim.core.sim import Simulator
from arm_sim.metrics.recorder import Recorder
from arm_sim.metrics.window import (
    compute_overload_ratio,
    compute_window_kpis,
    write_window_kpis_csv,
)
from arm_sim.model.central import CentralSupervisor
from arm_sim.model.arrival import ZoneArrivalProcess
from arm_sim.model.robot import Robot
from arm_sim.model.topology import Topology
from arm_sim.model.zone import ZoneController
from arm_sim.model.spatial import SpatialModel
from arm_sim.policy.hotspot_zone import HotspotZonePolicy, RobotTelemetry
from arm_sim.policy.noop import NoopPolicy
from arm_sim.routing.s0_centralized import CentralizedRouter
from arm_sim.routing.s1_static_edge import StaticEdgeRouter
from arm_sim.routing.s2_adaptive import AdaptiveRouter
from arm_sim.workload.hotspot import apply_hotspot
from arm_sim.experiments.repro_stamp import write_repro_stamp
from arm_sim.experiments.output_utils import normalize_output_root

HOTSPOT_START_S = 30.0
HOTSPOT_END_S = 80.0
HOTSPOT_JITTER_S = 5.0
HOTSPOT_RATIO_BASE = 0.5
HOTSPOT_RATIO_MULTIPLIER_RANGE = (0.90, 1.10)


class ForwardBehavior(ZoneBehavior):
    def on_arrival(self, zone, msg, now):
        zone.recorder.record_zone_done(msg.msg_id, now, zone.zone_id)
        zone.forward_to_central(msg)

    def on_done(self, zone, token, now):
        return None


def _read_migration_penalty_lambda(value=None):
    if value is not None:
        return float(value)
    text = os.getenv("MIGRATION_PENALTY_LAMBDA", "").strip()
    if not text:
        return 1.0
    try:
        return float(text)
    except ValueError as exc:
        raise RuntimeError(f"Invalid MIGRATION_PENALTY_LAMBDA value: {text}") from exc


def _parse_float(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_int(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _parse_seeds(text):
    if text is None:
        return []
    parts = []
    for chunk in str(text).replace(";", ",").split(","):
        value = chunk.strip()
        if not value:
            continue
        parts.append(int(value))
    return parts


def _derive_seed(base_seed, stream_id, salt=0):
    value = (int(base_seed) & 0xFFFFFFFFFFFFFFFF) + 0x9E3779B97F4A7C15 + int(stream_id) + int(salt)
    value &= 0xFFFFFFFFFFFFFFFF
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
    value = (value ^ (value >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
    value = value ^ (value >> 31)
    return value & 0xFFFFFFFFFFFFFFFF


def _mean(values):
    if not values:
        return float("nan")
    return sum(values) / len(values)


def _is_within_interval(value, low, high):
    return value is not None and not math.isnan(value) and low <= value <= high


def _distance_to_interval(value, low, high):
    if value is None or math.isnan(value):
        return float("inf")
    if value < low:
        return low - value
    if value > high:
        return value - high
    return 0.0


def _deterministic_hotspot_zones(n_zones, k_hot, center_zone=0):
    if n_zones <= 0:
        return []
    center = center_zone % n_zones
    distances = []
    for zone_id in range(n_zones):
        diff = abs(zone_id - center)
        ring = min(diff, n_zones - diff)
        distances.append((ring, zone_id))
    distances.sort(key=lambda pair: (pair[0], pair[1]))
    k_hot = max(1, min(int(k_hot), n_zones))
    return [zone_id for _, zone_id in distances[:k_hot]]


def _solve_hot_cold(lambda_mean, phi, skew):
    if lambda_mean is None or math.isnan(lambda_mean):
        return float("nan"), float("nan")
    if phi <= 0.0 or phi >= 1.0 or skew is None or skew <= 0.0 or math.isnan(skew):
        return lambda_mean, lambda_mean
    lambda_cold = lambda_mean / (phi * skew + (1.0 - phi))
    lambda_hot = skew * lambda_cold
    return lambda_hot, lambda_cold


def _central_rate_override(state_rate_hz, n_robots, n_zones, zone_rate, slack=0.10):
    if n_zones <= 0 or zone_rate <= 0:
        return n_zones * zone_rate, False
    total_arrival = state_rate_hz * n_robots
    base_central = n_zones * zone_rate
    if total_arrival >= base_central:
        return (1.0 + slack) * total_arrival, True
    return base_central, False


def _bootstrap_ci(values, alpha=0.05, iters=1000, rng=None):
    # Bootstrap CI over seed-level means using percentile bounds.
    values = [value for value in values if value is not None and not math.isnan(value)]
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], values[0]
    rng = rng or random.Random(0)
    means = []
    n = len(values)
    for _ in range(iters):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    low_idx = int((alpha / 2.0) * (len(means) - 1))
    high_idx = int((1.0 - alpha / 2.0) * (len(means) - 1))
    return means[low_idx], means[high_idx]


def _format_float(value, decimals):
    if value is None or math.isnan(value):
        return ""
    return f"{value:.{decimals}f}"


def _hotspot_mean_p95(rows, t_hot_start, t_hot_end):
    hot_rows = [
        row for row in rows if row["t_start"] >= t_hot_start and row["t_end"] <= t_hot_end
    ]
    if not hot_rows:
        return float("nan")
    return sum(row["p95_ms"] for row in hot_rows) / len(hot_rows)


def _overload_duration(overload_flags, t_hot_start, t_hot_end, window_s, duration_s):
    if not overload_flags:
        return 0.0
    count = 0
    for index, flag in enumerate(overload_flags):
        t_start = index * window_s
        t_end = min((index + 1) * window_s, duration_s)
        if t_start >= t_hot_start and t_end <= t_hot_end and flag:
            count += 1
    return count * window_s


def _queue_overload_ratio(central, zones):
    central_over = central.busy or len(central.queue) > 0
    zone_over = any(zone.queue_len() > 0 for zone in zones)
    return 1.0 if (central_over or zone_over) else 0.0


def _queue_overload_threshold(central, zones, threshold):
    central_q = len(central.queue) + (1 if central.busy else 0)
    zone_q = max((zone.queue_len() for zone in zones), default=0)
    return 1.0 if (central_q > threshold or zone_q > threshold) else 0.0


def _zone_overload_threshold(zones, zone_id, threshold):
    if zone_id < 0 or zone_id >= len(zones):
        return 0.0
    zone_q = zones[zone_id].queue_len()
    return 1.0 if zone_q > threshold else 0.0


def _zones_overload_threshold(zones, zone_ids, threshold):
    if not zone_ids:
        return 0.0
    max_q = 0
    for zone_id in zone_ids:
        if zone_id < 0 or zone_id >= len(zones):
            continue
        max_q = max(max_q, zones[zone_id].queue_len())
    return 1.0 if max_q > threshold else 0.0


def _dynamic_hotspot_overload_ratio(
    queue_all_by_window,
    recorder,
    window_s,
    k_hot,
    threshold,
    warmup_s=0.0,
    fallback_zone_ids=None,
):
    num_windows = len(queue_all_by_window)
    emit_counts = [dict() for _ in range(num_windows)]
    for record in recorder.records.values():
        emit_time = record.get("emit_time")
        emit_zone = record.get("emit_zone_id")
        if emit_time is None or emit_zone is None:
            continue
        index = int(emit_time // window_s)
        if index >= num_windows:
            index = num_windows - 1
        zone_id = int(emit_zone)
        emit_counts[index][zone_id] = emit_counts[index].get(zone_id, 0) + 1
    ratios = []
    k_hot = max(1, int(k_hot))
    for index, queues in enumerate(queue_all_by_window):
        t_end = (index + 1) * window_s
        if warmup_s and t_end <= warmup_s:
            ratios.append(None)
            continue
        counts = emit_counts[index] if index < len(emit_counts) else {}
        if counts:
            sorted_zones = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
            zone_ids = [zone for zone, _ in sorted_zones[:k_hot]]
        else:
            zone_ids = list(fallback_zone_ids or [])
        if not zone_ids or not queues:
            ratios.append(0.0)
            continue
        count = sum(
            1 for zone_id in zone_ids if 0 <= zone_id < len(queues) and queues[zone_id] > threshold
        )
        ratios.append(count / len(zone_ids))
    return ratios


def _serving_hotspot_overload_ratio(
    queue_all_by_window,
    recorder,
    window_s,
    k_hot,
    threshold,
    warmup_s=0.0,
    hotspot_zone_ids=None,
):
    num_windows = len(queue_all_by_window)
    counts = [dict() for _ in range(num_windows)]
    hotspot_set = {int(z) for z in (hotspot_zone_ids or [])}
    for record in recorder.records.values():
        home_zone = record.get("home_zone_id")
        if hotspot_set and (home_zone is None or int(home_zone) not in hotspot_set):
            continue
        admit_zone = record.get("admit_zone_id")
        if admit_zone is None:
            continue
        t_ref = record.get("arrive_zone")
        if t_ref is None:
            t_ref = record.get("emit_time")
        if t_ref is None:
            continue
        index = int(t_ref // window_s)
        if index >= num_windows:
            index = num_windows - 1
        zone_id = int(admit_zone)
        counts[index][zone_id] = counts[index].get(zone_id, 0) + 1
    ratios = []
    serving_sets = []
    topk_queues = []
    k_hot = max(1, int(k_hot))
    for index, queues in enumerate(queue_all_by_window):
        t_end = (index + 1) * window_s
        if warmup_s and t_end <= warmup_s:
            ratios.append(None)
            serving_sets.append([])
            topk_queues.append([])
            continue
        counts_i = counts[index] if index < len(counts) else {}
        if counts_i:
            sorted_zones = sorted(counts_i.items(), key=lambda kv: (-kv[1], kv[0]))
            zone_ids = [zone for zone, _ in sorted_zones[:k_hot]]
        else:
            zone_ids = list(hotspot_set)[:k_hot] if hotspot_set else []
        serving_sets.append(zone_ids)
        if not zone_ids or not queues:
            ratios.append(0.0)
            topk_queues.append([])
            continue
        count = 0
        qvals = []
        for zone_id in zone_ids:
            if 0 <= zone_id < len(queues):
                qval = queues[zone_id]
                qvals.append(qval)
                if qval > threshold:
                    count += 1
        ratios.append(count / len(zone_ids))
        topk_queues.append(qvals)
    return ratios, serving_sets, topk_queues


def _global_topk_overload_ratio(
    queue_all_by_window,
    k_hot,
    threshold,
    window_s,
    warmup_s=0.0,
):
    ratios = []
    topk_queues = []
    k_hot = max(1, int(k_hot))
    for index, queues in enumerate(queue_all_by_window):
        t_end = (index + 1) * window_s
        if warmup_s and t_end <= warmup_s:
            ratios.append(None)
            topk_queues.append([])
            continue
        if not queues:
            ratios.append(0.0)
            topk_queues.append([])
            continue
        sorted_qs = sorted(enumerate(queues), key=lambda kv: (-kv[1], kv[0]))
        topk = sorted_qs[:k_hot]
        qvals = [value for _, value in topk]
        count = sum(1 for value in qvals if value > threshold)
        ratios.append(count / len(qvals))
        topk_queues.append(qvals)
    return ratios, topk_queues

WINDOW_REQUIRED_COLUMNS = {
    "t_start",
    "t_end",
    "p95_ms",
    "mean_ms",
    "overload_ratio_q500",
    "overload_ratio_q1000",
    "overload_ratio_q1500",
    "fixed_hotspot_overload_ratio_q500",
    "fixed_hotspot_overload_ratio_q1000",
    "fixed_hotspot_overload_ratio_q1500",
    "serving_hotspot_overload_ratio_q500",
    "serving_hotspot_overload_ratio_q1000",
    "serving_hotspot_overload_ratio_q1500",
    "global_topk_overload_ratio_q500",
    "global_topk_overload_ratio_q1000",
    "global_topk_overload_ratio_q1500",
    "generated",
    "completed",
    "total_completed",
    "global_generated",
    "global_completed",
    "global_completed_total",
    "global_completion_ratio",
    "global_total_completed",
    "hotspot_generated_fixed",
    "hotspot_completed_fixed",
    "fixed_hotspot_completed_total",
    "fixed_hotspot_completion_ratio",
    "fixed_hotspot_completed",
    "hotspot_overload_ratio_q1000",
    "queue_mean_hot",
    "queue_p95_hot",
    "migrated_weight_window",
    "reconfig_actions_window",
}
WINDOW_REQUIRED_QUEUE = WINDOW_REQUIRED_COLUMNS | {"queue_max"}


def _window_has_columns(path, required=None):
    if not os.path.isfile(path):
        return False
    with open(path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = set(reader.fieldnames or [])
        required = set(required or WINDOW_REQUIRED_COLUMNS)
        return required.issubset(headers)


def _load_window_rows(path, required=None):
    if not os.path.isfile(path):
        raise RuntimeError(f"Missing window_kpis CSV: {path}")
    with open(path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        required = set(required or WINDOW_REQUIRED_COLUMNS)
        missing = [col for col in required if col not in headers]
        if missing:
            raise RuntimeError(
                f"Missing required columns in {path}: {', '.join(sorted(missing))}"
            )
        return list(reader)


def _hotspot_mean_metric(rows, t_hot_start, t_hot_end, key):
    values = []
    for row in rows:
        t_start = _parse_float(row.get("t_start"))
        t_end = _parse_float(row.get("t_end"))
        if t_start is None or t_end is None:
            continue
        if t_start >= t_hot_start and t_end <= t_hot_end:
            value = _parse_float(row.get(key))
            if value is not None and not math.isnan(value):
                values.append(value)
    if not values:
        return float("nan"), 0
    return _mean(values), len(values)


def _hotspot_queue_stats(rows, t_hot_start, t_hot_end):
    values = []
    for row in rows:
        t_start = _parse_float(row.get("t_start"))
        t_end = _parse_float(row.get("t_end"))
        if t_start is None or t_end is None:
            continue
        if t_start >= t_hot_start and t_end <= t_hot_end:
            value = _parse_float(row.get("queue_max"))
            if value is not None and not math.isnan(value):
                values.append(value)
    if not values:
        return float("nan"), float("nan")
    values.sort()
    max_queue = values[-1]
    p99_index = int(math.ceil(0.99 * len(values)) - 1)
    p99_queue = values[p99_index]
    return max_queue, p99_queue


def _queue_percentile(values, pct):
    if not values:
        return float("nan")
    values = sorted(values)
    index = int(math.ceil(pct * len(values)) - 1)
    index = max(0, min(index, len(values) - 1))
    return values[index]


def _queue_percentiles(rows, t_hot_start, t_hot_end):
    hotspot_values = []
    nonhot_values = []
    for row in rows:
        t_start = _parse_float(row.get("t_start"))
        t_end = _parse_float(row.get("t_end"))
        if t_start is None or t_end is None:
            continue
        if t_start >= t_hot_start and t_end <= t_hot_end:
            hot_val = _parse_float(row.get("queue_max"))
            if hot_val is not None and not math.isnan(hot_val):
                hotspot_values.append(hot_val)
            nonhot_val = _parse_float(row.get("queue_min_nonhot"))
            if nonhot_val is not None and not math.isnan(nonhot_val):
                nonhot_values.append(nonhot_val)
    return {
        "hotspot_queue_p90": _queue_percentile(hotspot_values, 0.90),
        "nonhot_queue_p10": _queue_percentile(nonhot_values, 0.10),
    }


def _mean_overall(rows, key, label):
    values = []
    for row in rows:
        value = _parse_float(row.get(key))
        if value is None or math.isnan(value):
            raise RuntimeError(f"Invalid {label} value.")
        values.append(value)
    if not values:
        return float("nan")
    return _mean(values)


def _summarize_window_rows(rows, t_hot_start, t_hot_end, scheme_label, seed):
    hotspot_p95, p95_count = _hotspot_mean_metric(
        rows, t_hot_start, t_hot_end, "p95_ms"
    )
    hotspot_mean, mean_count = _hotspot_mean_metric(
        rows, t_hot_start, t_hot_end, "mean_ms"
    )
    hotspot_latency_n = sum(
        int(_parse_float(row.get("completed")) or 0)
        for row in rows
        if _parse_float(row.get("t_start")) is not None
        and _parse_float(row.get("t_end")) is not None
        and _parse_float(row.get("t_start")) >= t_hot_start
        and _parse_float(row.get("t_end")) <= t_hot_end
    )
    overload_ratio_q500, _ = _hotspot_mean_metric(
        rows, t_hot_start, t_hot_end, "overload_ratio_q500"
    )
    overload_ratio_q1000, _ = _hotspot_mean_metric(
        rows, t_hot_start, t_hot_end, "overload_ratio_q1000"
    )
    overload_ratio_q1500, _ = _hotspot_mean_metric(
        rows, t_hot_start, t_hot_end, "overload_ratio_q1500"
    )
    fixed_hotspot_overload_ratio_q500, _ = _hotspot_mean_metric(
        rows, t_hot_start, t_hot_end, "fixed_hotspot_overload_ratio_q500"
    )
    fixed_hotspot_overload_ratio_q1000, _ = _hotspot_mean_metric(
        rows, t_hot_start, t_hot_end, "fixed_hotspot_overload_ratio_q1000"
    )
    fixed_hotspot_overload_ratio_q1500, _ = _hotspot_mean_metric(
        rows, t_hot_start, t_hot_end, "fixed_hotspot_overload_ratio_q1500"
    )
    hotspot_overload_ratio_q1000, _ = _hotspot_mean_metric(
        rows, t_hot_start, t_hot_end, "hotspot_overload_ratio_q1000"
    )
    serving_overload_q500, _ = _hotspot_mean_metric(
        rows, t_hot_start, t_hot_end, "serving_hotspot_overload_ratio_q500"
    )
    serving_overload_q1000, _ = _hotspot_mean_metric(
        rows, t_hot_start, t_hot_end, "serving_hotspot_overload_ratio_q1000"
    )
    serving_overload_q1500, _ = _hotspot_mean_metric(
        rows, t_hot_start, t_hot_end, "serving_hotspot_overload_ratio_q1500"
    )
    global_topk_overload_q500, _ = _hotspot_mean_metric(
        rows, t_hot_start, t_hot_end, "global_topk_overload_ratio_q500"
    )
    global_topk_overload_q1000, _ = _hotspot_mean_metric(
        rows, t_hot_start, t_hot_end, "global_topk_overload_ratio_q1000"
    )
    global_topk_overload_q1500, _ = _hotspot_mean_metric(
        rows, t_hot_start, t_hot_end, "global_topk_overload_ratio_q1500"
    )
    generated_requests = sum(int(_parse_float(row.get("generated")) or 0) for row in rows)
    completed_requests = sum(int(_parse_float(row.get("completed")) or 0) for row in rows)
    total_completed = sum(int(_parse_float(row.get("total_completed")) or 0) for row in rows)
    global_generated_requests = sum(
        int(_parse_float(row.get("global_generated")) or 0) for row in rows
    )
    global_completed_requests = sum(
        int(_parse_float(row.get("global_completed")) or 0) for row in rows
    )
    global_total_completed = sum(
        int(_parse_float(row.get("global_total_completed")) or 0) for row in rows
    )
    admitted_requests = generated_requests
    dropped_requests = 0
    global_admitted_requests = global_generated_requests
    global_dropped_requests = 0
    completion_ratio = (
        completed_requests / generated_requests if generated_requests > 0 else float("nan")
    )
    global_completion_ratio = (
        global_completed_requests / global_generated_requests
        if global_generated_requests > 0
        else float("nan")
    )
    fixed_hotspot_completion_ratio = (
        completed_requests / generated_requests if generated_requests > 0 else float("nan")
    )
    if not math.isnan(completion_ratio) and completion_ratio > 1.0 + 1.0e-9:
        raise RuntimeError(
            "Invalid completion_ratio > 1.0: "
            f"completed={completed_requests} generated={generated_requests} "
            f"edge_completed={sum(int(_parse_float(row.get('edge_completed')) or 0) for row in rows)} "
            f"central_completed={sum(int(_parse_float(row.get('central_completed')) or 0) for row in rows)} "
            f"total_completed={total_completed}"
        )
    if not math.isnan(global_completion_ratio) and global_completion_ratio > 1.0 + 1.0e-9:
        raise RuntimeError(
            "Invalid global_completion_ratio > 1.0: "
            f"completed={global_completed_requests} generated={global_generated_requests} "
            f"edge_completed={sum(int(_parse_float(row.get('edge_completed')) or 0) for row in rows)} "
            f"central_completed={sum(int(_parse_float(row.get('central_completed')) or 0) for row in rows)} "
            f"total_completed={global_total_completed}"
        )
    if (
        not math.isnan(fixed_hotspot_completion_ratio)
        and fixed_hotspot_completion_ratio > 1.0 + 1.0e-9
    ):
        raise RuntimeError(
            "Invalid fixed_hotspot_completion_ratio > 1.0: "
            f"completed={completed_requests} generated={generated_requests} "
            f"edge_completed={sum(int(_parse_float(row.get('edge_completed')) or 0) for row in rows)} "
            f"central_completed={sum(int(_parse_float(row.get('central_completed')) or 0) for row in rows)} "
            f"total_completed={total_completed}"
        )
    first = rows[0] if rows else {}
    migrated_weight_total = _parse_float(first.get("migrated_weight_total")) or 0.0
    accepted_migrations = int(_parse_float(first.get("policy_reassign_ops")) or 0)
    reconfig_action_count = int(_parse_float(first.get("reconfig_action_count")) or 0)
    if accepted_migrations and not reconfig_action_count:
        reconfig_action_count = accepted_migrations
    rejected_no_feasible_target = int(
        _parse_float(first.get("rejected_no_feasible_target")) or 0
    )
    rejected_budget = int(_parse_float(first.get("rejected_budget")) or 0)
    rejected_safety = int(_parse_float(first.get("rejected_safety")) or 0)
    fallback_attempts = int(_parse_float(first.get("fallback_attempts")) or 0)
    fallback_success = int(_parse_float(first.get("fallback_success")) or 0)
    dmax_rejects = int(_parse_float(first.get("dmax_rejects")) or 0)
    max_queue, p99_queue = _hotspot_queue_stats(rows, t_hot_start, t_hot_end)
    if overload_ratio_q1000 > 0.2 and hotspot_latency_n == 0:
        print(
            f"WARNING: overload_ratio_q1000={overload_ratio_q1000:.3f} but "
            f"hotspot_latency_n=0 for scheme={scheme_label} seed={seed}"
        )
        hotspot_p95 = float("nan")
        hotspot_mean = float("nan")
    horizon_s = 0.0
    for row in rows:
        t_end = _parse_float(row.get("t_end"))
        if t_end is not None and t_end > horizon_s:
            horizon_s = t_end
    cost_per_request = (
        migrated_weight_total / global_generated_requests
        if global_generated_requests > 0
        else float("nan")
    )
    reconfig_actions_per_s = (
        reconfig_action_count / horizon_s if horizon_s > 0 else float("nan")
    )
    return {
        "hotspot_p95_mean_ms": hotspot_p95,
        "hotspot_mean_ms": hotspot_mean,
        "hotspot_latency_n": hotspot_latency_n,
        "overload_ratio_q500": overload_ratio_q500,
        "overload_ratio_q1000": overload_ratio_q1000,
        "overload_ratio_q1500": overload_ratio_q1500,
        "fixed_hotspot_overload_ratio_q1000": fixed_hotspot_overload_ratio_q1000,
        "fixed_hotspot_overload_ratio_q500": fixed_hotspot_overload_ratio_q500,
        "fixed_hotspot_overload_ratio_q1500": fixed_hotspot_overload_ratio_q1500,
        "hotspot_overload_ratio_q1000": hotspot_overload_ratio_q1000,
        "serving_hotspot_overload_ratio_q500": serving_overload_q500,
        "serving_hotspot_overload_ratio_q1000": serving_overload_q1000,
        "serving_hotspot_overload_ratio_q1500": serving_overload_q1500,
        "global_topk_overload_ratio_q500": global_topk_overload_q500,
        "global_topk_overload_ratio_q1000": global_topk_overload_q1000,
        "global_topk_overload_ratio_q1500": global_topk_overload_q1500,
        "max_queue": max_queue,
        "p99_queue": p99_queue,
        "migrated_weight_total": migrated_weight_total,
        "accepted_migrations": accepted_migrations,
        "reconfig_action_count": reconfig_action_count,
        "rejected_no_feasible_target": rejected_no_feasible_target,
        "rejected_budget": rejected_budget,
        "rejected_safety": rejected_safety,
        "fallback_attempts": fallback_attempts,
        "fallback_success": fallback_success,
        "dmax_rejects": dmax_rejects,
        "generated_requests": generated_requests,
        "completed_requests": completed_requests,
        "total_completed": total_completed,
        "global_generated_requests": global_generated_requests,
        "global_completed_requests": global_completed_requests,
        "global_total_completed": global_total_completed,
        "global_completed_total": global_completed_requests,
        "admitted_requests": admitted_requests,
        "dropped_requests": dropped_requests,
        "global_admitted_requests": global_admitted_requests,
        "global_dropped_requests": global_dropped_requests,
        "hotspot_generated_requests_fixed": generated_requests,
        "hotspot_completed_requests_fixed": completed_requests,
        "fixed_hotspot_generated_requests": generated_requests,
        "fixed_hotspot_completed_requests": completed_requests,
        "fixed_hotspot_completed": completed_requests,
        "fixed_hotspot_completed_total": completed_requests,
        "hotspot_admitted_requests_fixed": generated_requests,
        "hotspot_dropped_requests_fixed": 0,
        "completion_ratio": completion_ratio,
        "global_completion_ratio": global_completion_ratio,
        "fixed_hotspot_completion_ratio": fixed_hotspot_completion_ratio,
        "cost_per_request": cost_per_request,
        "reconfig_actions_per_s": reconfig_actions_per_s,
        "hotspot_window_start_end": f"{t_hot_start:.3f}-{t_hot_end:.3f}",
        "overload_window_kind": "hotspot_window",
    }


def _collect_debug_from_rows(rows, t_hot_start, t_hot_end):
    first = rows[0] if rows else {}
    feasible_mean = _parse_float(first.get("feasible_ratio_mean"))
    feasible_count = int(_parse_float(first.get("feasible_ratio_count")) or 0)
    feasible_sum = (
        feasible_mean * feasible_count
        if feasible_mean is not None and not math.isnan(feasible_mean)
        else 0.0
    )
    queue_stats = _queue_percentiles(rows, t_hot_start, t_hot_end)
    return {
        "accepted_migrations": int(_parse_float(first.get("policy_reassign_ops")) or 0),
        "rejected_no_feasible_target": int(
            _parse_float(first.get("rejected_no_feasible_target")) or 0
        ),
        "rejected_budget": int(_parse_float(first.get("rejected_budget")) or 0),
        "rejected_safety": int(_parse_float(first.get("rejected_safety")) or 0),
        "fallback_attempts": int(_parse_float(first.get("fallback_attempts")) or 0),
        "fallback_success": int(_parse_float(first.get("fallback_success")) or 0),
        "dmax_rejects": int(_parse_float(first.get("dmax_rejects")) or 0),
        "feasible_ratio_sum": feasible_sum,
        "feasible_ratio_count": feasible_count,
        "hotspot_queue_p90": queue_stats.get("hotspot_queue_p90"),
        "nonhot_queue_p10": queue_stats.get("nonhot_queue_p10"),
    }


def _group_rows(rows, keys):
    grouped = {}
    for row in rows:
        key = tuple(row.get(k) for k in keys)
        grouped.setdefault(key, []).append(row)
    return grouped


def _mean_or_nan(values):
    values = [value for value in values if value is not None and not math.isnan(value)]
    if not values:
        return float("nan")
    return _mean(values)


def _aggregate_rows(rows, group_keys, rng=None):
    grouped = _group_rows(rows, group_keys)
    aggregated = []
    rng = rng or random.Random(0)
    for key, items in grouped.items():
        sample = items[0]
        p95_vals_ms = [row.get("hotspot_p95_mean_ms") for row in items]
        mean_vals_ms = [row.get("hotspot_mean_ms") for row in items]
        p95_vals_s = [
            value / 1000.0
            for value in p95_vals_ms
            if value is not None and not math.isnan(value)
        ]
        overload_q1000_vals = [
            row.get("overload_ratio_q1000")
            for row in items
            if row.get("overload_ratio_q1000") is not None
            and not math.isnan(row.get("overload_ratio_q1000"))
        ]
        p95_ci_low, p95_ci_high = _bootstrap_ci(p95_vals_s, rng=rng)
        overload_ci_low, overload_ci_high = _bootstrap_ci(overload_q1000_vals, rng=rng)
        aggregated.append(
            {
                "N": sample.get("N"),
                "tag": sample.get("tag"),
                "scheme": sample.get("scheme"),
                "profile": sample.get("profile"),
                "load_multiplier": sample.get("load_multiplier"),
                "hotspot_p95_mean_ms": _mean_or_nan(p95_vals_ms),
                "hotspot_mean_ms": _mean_or_nan(mean_vals_ms),
                "hotspot_p95_mean_s": _mean_or_nan(
                    [value / 1000.0 for value in p95_vals_ms if value is not None]
                ),
                "hotspot_mean_s": _mean_or_nan(
                    [value / 1000.0 for value in mean_vals_ms if value is not None]
                ),
                "hotspot_p95_s_ci_low": p95_ci_low,
                "hotspot_p95_s_ci_high": p95_ci_high,
                "overload_ratio_q500": _mean_or_nan(
                    [row.get("overload_ratio_q500") for row in items]
                ),
                "overload_ratio_q1000": _mean_or_nan(
                    [row.get("overload_ratio_q1000") for row in items]
                ),
                "overload_ratio_q1500": _mean_or_nan(
                    [row.get("overload_ratio_q1500") for row in items]
                ),
                "fixed_hotspot_overload_ratio_q500": _mean_or_nan(
                    [row.get("fixed_hotspot_overload_ratio_q500") for row in items]
                ),
                "fixed_hotspot_overload_ratio_q1000": _mean_or_nan(
                    [row.get("fixed_hotspot_overload_ratio_q1000") for row in items]
                ),
                "fixed_hotspot_overload_ratio_q1500": _mean_or_nan(
                    [row.get("fixed_hotspot_overload_ratio_q1500") for row in items]
                ),
                "overload_q1000_ci_low": overload_ci_low,
                "overload_q1000_ci_high": overload_ci_high,
                "hotspot_overload_ratio_q1000": _mean_or_nan(
                    [row.get("hotspot_overload_ratio_q1000") for row in items]
                ),
                "serving_hotspot_overload_ratio_q500": _mean_or_nan(
                    [row.get("serving_hotspot_overload_ratio_q500") for row in items]
                ),
                "serving_hotspot_overload_ratio_q1000": _mean_or_nan(
                    [row.get("serving_hotspot_overload_ratio_q1000") for row in items]
                ),
                "serving_hotspot_overload_ratio_q1500": _mean_or_nan(
                    [row.get("serving_hotspot_overload_ratio_q1500") for row in items]
                ),
                "global_topk_overload_ratio_q500": _mean_or_nan(
                    [row.get("global_topk_overload_ratio_q500") for row in items]
                ),
                "global_topk_overload_ratio_q1000": _mean_or_nan(
                    [row.get("global_topk_overload_ratio_q1000") for row in items]
                ),
                "global_topk_overload_ratio_q1500": _mean_or_nan(
                    [row.get("global_topk_overload_ratio_q1500") for row in items]
                ),
                "max_queue": _mean_or_nan([row.get("max_queue") for row in items]),
                "p99_queue": _mean_or_nan([row.get("p99_queue") for row in items]),
                "migrated_weight_total": _mean_or_nan(
                    [row.get("migrated_weight_total") for row in items]
                ),
                "accepted_migrations": _mean_or_nan(
                    [row.get("accepted_migrations") for row in items]
                ),
                "reconfig_action_count": _mean_or_nan(
                    [row.get("reconfig_action_count") for row in items]
                ),
                "rejected_no_feasible_target": _mean_or_nan(
                    [row.get("rejected_no_feasible_target") for row in items]
                ),
                "rejected_budget": _mean_or_nan(
                    [row.get("rejected_budget") for row in items]
                ),
                "rejected_safety": _mean_or_nan(
                    [row.get("rejected_safety") for row in items]
                ),
                "fallback_attempts": _mean_or_nan(
                    [row.get("fallback_attempts") for row in items]
                ),
                "fallback_success": _mean_or_nan(
                    [row.get("fallback_success") for row in items]
                ),
                "dmax_rejects": _mean_or_nan(
                    [row.get("dmax_rejects") for row in items]
                ),
                "generated_requests": _mean_or_nan(
                    [row.get("generated_requests") for row in items]
                ),
                "completed_requests": _mean_or_nan(
                    [row.get("completed_requests") for row in items]
                ),
                "total_completed": _mean_or_nan(
                    [row.get("total_completed") for row in items]
                ),
                "global_generated_requests": _mean_or_nan(
                    [row.get("global_generated_requests") for row in items]
                ),
                "global_completed_requests": _mean_or_nan(
                    [row.get("global_completed_requests") for row in items]
                ),
                "global_total_completed": _mean_or_nan(
                    [row.get("global_total_completed") for row in items]
                ),
                "global_completed_total": _mean_or_nan(
                    [row.get("global_completed_total") for row in items]
                ),
                "admitted_requests": _mean_or_nan(
                    [row.get("admitted_requests") for row in items]
                ),
                "dropped_requests": _mean_or_nan(
                    [row.get("dropped_requests") for row in items]
                ),
                "global_admitted_requests": _mean_or_nan(
                    [row.get("global_admitted_requests") for row in items]
                ),
                "global_dropped_requests": _mean_or_nan(
                    [row.get("global_dropped_requests") for row in items]
                ),
                "hotspot_generated_requests_fixed": _mean_or_nan(
                    [row.get("hotspot_generated_requests_fixed") for row in items]
                ),
                "hotspot_completed_requests_fixed": _mean_or_nan(
                    [row.get("hotspot_completed_requests_fixed") for row in items]
                ),
                "fixed_hotspot_completed": _mean_or_nan(
                    [row.get("fixed_hotspot_completed") for row in items]
                ),
                "fixed_hotspot_completed_total": _mean_or_nan(
                    [row.get("fixed_hotspot_completed_total") for row in items]
                ),
                "hotspot_admitted_requests_fixed": _mean_or_nan(
                    [row.get("hotspot_admitted_requests_fixed") for row in items]
                ),
                "hotspot_dropped_requests_fixed": _mean_or_nan(
                    [row.get("hotspot_dropped_requests_fixed") for row in items]
                ),
                "completion_ratio": _mean_or_nan(
                    [row.get("completion_ratio") for row in items]
                ),
                "global_completion_ratio": _mean_or_nan(
                    [row.get("global_completion_ratio") for row in items]
                ),
                "fixed_hotspot_completion_ratio": _mean_or_nan(
                    [row.get("fixed_hotspot_completion_ratio") for row in items]
                ),
                "cost_per_request": _mean_or_nan(
                    [row.get("cost_per_request") for row in items]
                ),
                "reconfig_actions_per_s": _mean_or_nan(
                    [row.get("reconfig_actions_per_s") for row in items]
                ),
                "hotspot_latency_n": _mean_or_nan(
                    [row.get("hotspot_latency_n") for row in items]
                ),
                "budget_scale_factor": _mean_or_nan(
                    [row.get("budget_scale_factor") for row in items]
                ),
                "hotspot_window_start_end": sample.get("hotspot_window_start_end"),
                "overload_window_kind": sample.get("overload_window_kind"),
                "seeds": ",".join(str(row.get("seed")) for row in items),
            }
        )
    return aggregated


def _seed_variation(rows, tol_p95_s=1.0e-3, tol_over=1.0e-4):
    p95_vals = [
        (row.get("hotspot_p95_mean_ms") or 0.0) / 1000.0
        for row in rows
        if row.get("hotspot_p95_mean_ms") is not None
        and not math.isnan(row.get("hotspot_p95_mean_ms"))
    ]
    over_vals = [
        row.get("fixed_hotspot_overload_ratio_q1000")
        if row.get("fixed_hotspot_overload_ratio_q1000") is not None
        else row.get("overload_ratio_q1000")
        for row in rows
        if (
            row.get("fixed_hotspot_overload_ratio_q1000") is not None
            and not math.isnan(row.get("fixed_hotspot_overload_ratio_q1000"))
        )
        or (
            row.get("fixed_hotspot_overload_ratio_q1000") is None
            and row.get("overload_ratio_q1000") is not None
            and not math.isnan(row.get("overload_ratio_q1000"))
        )
    ]
    if len(p95_vals) < 2 and len(over_vals) < 2:
        return False
    p95_diff = (max(p95_vals) - min(p95_vals)) if len(p95_vals) >= 2 else 0.0
    over_diff = (max(over_vals) - min(over_vals)) if len(over_vals) >= 2 else 0.0
    return p95_diff >= tol_p95_s or over_diff >= tol_over


def _print_seed_metrics(rows, label):
    print(f"{label} per-seed metrics:")
    for row in rows:
        p95_s = (
            row.get("hotspot_p95_mean_ms") / 1000.0
            if row.get("hotspot_p95_mean_ms") is not None
            else float("nan")
        )
        ratio = row.get("fixed_hotspot_overload_ratio_q1000")
        if ratio is None:
            ratio = row.get("overload_ratio_q1000")
        print(
            f"seed={row.get('seed')} "
            f"hotspot_p95_s={p95_s:.6f} "
            f"overload_q1000={ratio}"
        )


def _seed_variation_by_group(rows):
    grouped = {}
    for row in rows:
        key = (row.get("scheme"), row.get("profile"), row.get("tag"))
        grouped.setdefault(key, []).append(row)
    for group_rows in grouped.values():
        if _seed_variation(group_rows):
            return True
    return False


def _assert_generated_consistency(rows, label):
    grouped = {}
    for row in rows:
        key = (row.get("N"), row.get("seed"))
        grouped.setdefault(key, []).append(row)
    for key, items in grouped.items():
        if not items:
            continue
        base_global = _parse_float(items[0].get("global_generated_requests"))
        base_hot = _parse_float(
            items[0].get("fixed_hotspot_generated_requests")
            or items[0].get("hotspot_generated_requests_fixed")
        )
        for row in items:
            global_val = _parse_float(row.get("global_generated_requests"))
            hot_val = _parse_float(
                row.get("fixed_hotspot_generated_requests")
                or row.get("hotspot_generated_requests_fixed")
            )
            dropped_val = _parse_float(row.get("dropped_requests"))
            if base_global is None or global_val is None or abs(global_val - base_global) > 1.0e-6:
                raise RuntimeError(
                    f"{label} generated mismatch for N={key[0]} seed={key[1]}: "
                    f"{base_global} vs {global_val}"
                )
            if base_hot is None or hot_val is None or abs(hot_val - base_hot) > 1.0e-6:
                raise RuntimeError(
                    f"{label} hotspot generated mismatch for N={key[0]} seed={key[1]}: "
                    f"{base_hot} vs {hot_val}"
                )
            if dropped_val is not None and abs(dropped_val) > 1.0e-9:
                raise RuntimeError(
                    f"{label} dropped_requests must be 0 for N={key[0]} seed={key[1]}: "
                    f"{dropped_val}"
                )


def _write_summary_long(path, rows):
    columns = [
        "N",
        "tag",
        "scheme",
        "profile",
        "seed",
        "load_multiplier",
        "hotspot_p95_mean_ms",
        "hotspot_mean_ms",
        "hotspot_p95_mean_s",
        "hotspot_mean_s",
        "overload_ratio_q500",
        "overload_ratio_q1000",
        "overload_ratio_q1500",
        "fixed_hotspot_overload_ratio_q500",
        "fixed_hotspot_overload_ratio_q1000",
        "fixed_hotspot_overload_ratio_q1500",
        "hotspot_overload_ratio_q1000",
        "serving_hotspot_overload_ratio_q500",
        "serving_hotspot_overload_ratio_q1000",
        "serving_hotspot_overload_ratio_q1500",
        "global_topk_overload_ratio_q500",
        "global_topk_overload_ratio_q1000",
        "global_topk_overload_ratio_q1500",
        "max_queue",
        "p99_queue",
        "migrated_weight_total",
        "accepted_migrations",
        "reconfig_action_count",
        "rejected_no_feasible_target",
        "rejected_budget",
        "rejected_safety",
        "fallback_attempts",
        "fallback_success",
        "dmax_rejects",
        "generated_requests",
        "completed_requests",
        "total_completed",
        "admitted_requests",
        "dropped_requests",
        "global_generated_requests",
        "global_completed_requests",
        "global_total_completed",
        "global_completed_total",
        "global_completion_ratio",
        "global_admitted_requests",
        "global_dropped_requests",
        "hotspot_generated_requests_fixed",
        "hotspot_completed_requests_fixed",
        "fixed_hotspot_generated_requests",
        "fixed_hotspot_completed_requests",
        "fixed_hotspot_completed",
        "fixed_hotspot_completed_total",
        "fixed_hotspot_completion_ratio",
        "hotspot_admitted_requests_fixed",
        "hotspot_dropped_requests_fixed",
        "completion_ratio",
        "cost_per_request",
        "reconfig_actions_per_s",
        "hotspot_latency_n",
        "budget_scale_factor",
        "hotspot_window_start_end",
        "overload_window_kind",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow(
                [
                    row.get("N", ""),
                    row.get("tag", ""),
                    row.get("scheme", ""),
                    row.get("profile", ""),
                    row.get("seed", ""),
                    _format_float(row.get("load_multiplier"), 4),
                    _format_float(row.get("hotspot_p95_mean_ms"), 6),
                    _format_float(row.get("hotspot_mean_ms"), 6),
                    _format_float(row.get("hotspot_p95_mean_ms") / 1000.0, 6)
                    if row.get("hotspot_p95_mean_ms") is not None
                    else "",
                    _format_float(row.get("hotspot_mean_ms") / 1000.0, 6)
                    if row.get("hotspot_mean_ms") is not None
                    else "",
                    _format_float(row.get("overload_ratio_q500"), 6),
                    _format_float(row.get("overload_ratio_q1000"), 6),
                    _format_float(row.get("overload_ratio_q1500"), 6),
                    _format_float(row.get("fixed_hotspot_overload_ratio_q500"), 6),
                    _format_float(row.get("fixed_hotspot_overload_ratio_q1000"), 6),
                    _format_float(row.get("fixed_hotspot_overload_ratio_q1500"), 6),
                    _format_float(row.get("hotspot_overload_ratio_q1000"), 6),
                    _format_float(row.get("serving_hotspot_overload_ratio_q500"), 6),
                    _format_float(row.get("serving_hotspot_overload_ratio_q1000"), 6),
                    _format_float(row.get("serving_hotspot_overload_ratio_q1500"), 6),
                    _format_float(row.get("global_topk_overload_ratio_q500"), 6),
                    _format_float(row.get("global_topk_overload_ratio_q1000"), 6),
                    _format_float(row.get("global_topk_overload_ratio_q1500"), 6),
                    _format_float(row.get("max_queue"), 6),
                    _format_float(row.get("p99_queue"), 6),
                    _format_float(row.get("migrated_weight_total"), 6),
                    _format_float(row.get("accepted_migrations"), 3),
                    _format_float(row.get("reconfig_action_count"), 3),
                    _format_float(row.get("rejected_no_feasible_target"), 3),
                    _format_float(row.get("rejected_budget"), 3),
                    _format_float(row.get("rejected_safety"), 3),
                    _format_float(row.get("fallback_attempts"), 3),
                    _format_float(row.get("fallback_success"), 3),
                    _format_float(row.get("dmax_rejects"), 3),
                    _format_float(row.get("generated_requests"), 3),
                    _format_float(row.get("completed_requests"), 3),
                    _format_float(row.get("total_completed"), 3),
                    _format_float(row.get("admitted_requests"), 3),
                    _format_float(row.get("dropped_requests"), 3),
                    _format_float(row.get("global_generated_requests"), 3),
                    _format_float(row.get("global_completed_requests"), 3),
                    _format_float(row.get("global_total_completed"), 3),
                    _format_float(row.get("global_completed_total"), 3),
                    _format_float(row.get("global_completion_ratio"), 6),
                    _format_float(row.get("global_admitted_requests"), 3),
                    _format_float(row.get("global_dropped_requests"), 3),
                    _format_float(row.get("hotspot_generated_requests_fixed"), 3),
                    _format_float(row.get("hotspot_completed_requests_fixed"), 3),
                    _format_float(row.get("fixed_hotspot_generated_requests"), 3),
                    _format_float(row.get("fixed_hotspot_completed_requests"), 3),
                    _format_float(row.get("fixed_hotspot_completed"), 3),
                    _format_float(row.get("fixed_hotspot_completed_total"), 3),
                    _format_float(row.get("fixed_hotspot_completion_ratio"), 6),
                    _format_float(row.get("hotspot_admitted_requests_fixed"), 3),
                    _format_float(row.get("hotspot_dropped_requests_fixed"), 3),
                    _format_float(row.get("completion_ratio"), 6),
                    _format_float(row.get("cost_per_request"), 6),
                    _format_float(row.get("reconfig_actions_per_s"), 6),
                    _format_float(row.get("hotspot_latency_n"), 3),
                    _format_float(row.get("budget_scale_factor"), 4),
                    row.get("hotspot_window_start_end", ""),
                    row.get("overload_window_kind", ""),
                ]
            )


def _write_summary_agg(path, rows):
    columns = [
        "N",
        "tag",
        "scheme",
        "profile",
        "load_multiplier",
        "hotspot_p95_mean_ms",
        "hotspot_mean_ms",
        "hotspot_p95_mean_s",
        "hotspot_p95_s",
        "hotspot_mean_s",
        "hotspot_p95_s_ci_low",
        "hotspot_p95_s_ci_high",
        "overload_ratio_q500",
        "overload_ratio_q1000",
        "overload_ratio_q1500",
        "fixed_hotspot_overload_ratio_q500",
        "fixed_hotspot_overload_ratio_q1000",
        "fixed_hotspot_overload_ratio_q1500",
        "overload_q1000_ci_low",
        "overload_q1000_ci_high",
        "hotspot_overload_ratio_q1000",
        "serving_hotspot_overload_ratio_q500",
        "serving_hotspot_overload_ratio_q1000",
        "serving_hotspot_overload_ratio_q1500",
        "global_topk_overload_ratio_q500",
        "global_topk_overload_ratio_q1000",
        "global_topk_overload_ratio_q1500",
        "max_queue",
        "p99_queue",
        "migrated_weight_total",
        "accepted_migrations",
        "reconfig_action_count",
        "rejected_no_feasible_target",
        "rejected_budget",
        "rejected_safety",
        "fallback_attempts",
        "fallback_success",
        "dmax_rejects",
        "generated_requests",
        "completed_requests",
        "total_completed",
        "admitted_requests",
        "dropped_requests",
        "global_generated_requests",
        "global_completed_requests",
        "global_total_completed",
        "global_completed_total",
        "global_completion_ratio",
        "global_admitted_requests",
        "global_dropped_requests",
        "hotspot_generated_requests_fixed",
        "hotspot_completed_requests_fixed",
        "fixed_hotspot_generated_requests",
        "fixed_hotspot_completed_requests",
        "fixed_hotspot_completed",
        "fixed_hotspot_completed_total",
        "fixed_hotspot_completion_ratio",
        "hotspot_admitted_requests_fixed",
        "hotspot_dropped_requests_fixed",
        "completion_ratio",
        "cost_per_request",
        "reconfig_actions_per_s",
        "hotspot_latency_n",
        "budget_scale_factor",
        "hotspot_window_start_end",
        "overload_window_kind",
        "seeds",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow(
                [
                    row.get("N", ""),
                    row.get("tag", ""),
                    row.get("scheme", ""),
                    row.get("profile", ""),
                    _format_float(row.get("load_multiplier"), 4),
                    _format_float(row.get("hotspot_p95_mean_ms"), 6),
                    _format_float(row.get("hotspot_mean_ms"), 6),
                    _format_float(row.get("hotspot_p95_mean_s"), 6),
                    _format_float(row.get("hotspot_p95_mean_s"), 6),
                    _format_float(row.get("hotspot_mean_s"), 6),
                    _format_float(row.get("hotspot_p95_s_ci_low"), 6),
                    _format_float(row.get("hotspot_p95_s_ci_high"), 6),
                    _format_float(row.get("overload_ratio_q500"), 6),
                    _format_float(row.get("overload_ratio_q1000"), 6),
                    _format_float(row.get("overload_ratio_q1500"), 6),
                    _format_float(row.get("fixed_hotspot_overload_ratio_q500"), 6),
                    _format_float(row.get("fixed_hotspot_overload_ratio_q1000"), 6),
                    _format_float(row.get("fixed_hotspot_overload_ratio_q1500"), 6),
                    _format_float(row.get("overload_q1000_ci_low"), 6),
                    _format_float(row.get("overload_q1000_ci_high"), 6),
                    _format_float(row.get("hotspot_overload_ratio_q1000"), 6),
                    _format_float(row.get("serving_hotspot_overload_ratio_q500"), 6),
                    _format_float(row.get("serving_hotspot_overload_ratio_q1000"), 6),
                    _format_float(row.get("serving_hotspot_overload_ratio_q1500"), 6),
                    _format_float(row.get("global_topk_overload_ratio_q500"), 6),
                    _format_float(row.get("global_topk_overload_ratio_q1000"), 6),
                    _format_float(row.get("global_topk_overload_ratio_q1500"), 6),
                    _format_float(row.get("max_queue"), 6),
                    _format_float(row.get("p99_queue"), 6),
                    _format_float(row.get("migrated_weight_total"), 6),
                    _format_float(row.get("accepted_migrations"), 3),
                    _format_float(row.get("reconfig_action_count"), 3),
                    _format_float(row.get("rejected_no_feasible_target"), 3),
                    _format_float(row.get("rejected_budget"), 3),
                    _format_float(row.get("rejected_safety"), 3),
                    _format_float(row.get("fallback_attempts"), 3),
                    _format_float(row.get("fallback_success"), 3),
                    _format_float(row.get("dmax_rejects"), 3),
                    _format_float(row.get("generated_requests"), 3),
                    _format_float(row.get("completed_requests"), 3),
                    _format_float(row.get("total_completed"), 3),
                    _format_float(row.get("admitted_requests"), 3),
                    _format_float(row.get("dropped_requests"), 3),
                    _format_float(row.get("global_generated_requests"), 3),
                    _format_float(row.get("global_completed_requests"), 3),
                    _format_float(row.get("global_total_completed"), 3),
                    _format_float(row.get("global_completed_total"), 3),
                    _format_float(row.get("global_completion_ratio"), 6),
                    _format_float(row.get("global_admitted_requests"), 3),
                    _format_float(row.get("global_dropped_requests"), 3),
                    _format_float(row.get("hotspot_generated_requests_fixed"), 3),
                    _format_float(row.get("hotspot_completed_requests_fixed"), 3),
                    _format_float(row.get("fixed_hotspot_generated_requests"), 3),
                    _format_float(row.get("fixed_hotspot_completed_requests"), 3),
                    _format_float(row.get("fixed_hotspot_completed"), 3),
                    _format_float(row.get("fixed_hotspot_completed_total"), 3),
                    _format_float(row.get("fixed_hotspot_completion_ratio"), 6),
                    _format_float(row.get("hotspot_admitted_requests_fixed"), 3),
                    _format_float(row.get("hotspot_dropped_requests_fixed"), 3),
                    _format_float(row.get("completion_ratio"), 6),
                    _format_float(row.get("cost_per_request"), 6),
                    _format_float(row.get("reconfig_actions_per_s"), 6),
                    _format_float(row.get("hotspot_latency_n"), 3),
                    _format_float(row.get("budget_scale_factor"), 4),
                    row.get("hotspot_window_start_end", ""),
                    row.get("overload_window_kind", ""),
                    row.get("seeds", ""),
                ]
            )


def run_scheme(
    scheme,
    seed,
    output_dir,
    state_rate_hz,
    zone_service_rate_msgs_s,
    write_csv=True,
    n_zones_override=None,
    n_robots_override=None,
    central_service_rate_msgs_s_override=None,
    budget_gamma_override=None,
    candidate_sample_m_override=None,
    p2c_k_override=None,
    beta_capacity_override=None,
    q_high_override=None,
    q_low_override=None,
    move_k_override=None,
    cooldown_s_override=None,
    weight_scale_override=None,
    min_gain_override=None,
    policy_period_s_override=None,
    migrate_penalty_lambda=None,
    dmax_ms=1.0e9,
    disable_fallback=False,
    disable_budget=False,
    fixed_k=False,
    move_k_fixed=5,
    hotspot_target_zones=None,
    extra_robots_per_zone=None,
    hotspot_ratio_override=None,
    arrival_mode="robot_emit",
    tag=None,
):
    duration_s = 120.0
    n_robots = 100
    n_zones = 8
    if n_zones_override is not None:
        n_zones = int(n_zones_override)
    if n_robots_override is not None:
        n_robots = int(n_robots_override)
    arrival_mode = (arrival_mode or "robot_emit").strip().lower()
    if arrival_mode not in ("robot_emit", "zone_poisson"):
        raise RuntimeError(f"Invalid arrival_mode: {arrival_mode}")
    zone_to_central_base_ms = 5
    zone_to_central_jitter_ms = 2
    if central_service_rate_msgs_s_override is not None:
        central_service_rate_msgs_s = float(central_service_rate_msgs_s_override)
    elif scheme == "S0":
        central_service_rate_msgs_s = 400
    else:
        central_service_rate_msgs_s = 10 * zone_service_rate_msgs_s

    control_interval_s = 1.0
    if policy_period_s_override is not None:
        control_interval_s = float(policy_period_s_override)
    policy_period_s = control_interval_s
    q_high = 1000
    q_low = 900
    move_k = 200
    cooldown_s = 1.0
    beta_capacity = 0.80
    weight_w500 = 1.0
    weight_w1000 = 2.0
    weight_w1500 = 4.0
    weight_scale = 1.0
    min_gain = 5.0
    if min_gain_override is not None:
        min_gain = float(min_gain_override)
    alpha_ema = 0.20
    cost_factor_mode = "two_level"
    cost_factor_light = 1.0
    cost_factor_heavy = 3.0
    heavy_fraction = 0.20
    migrate_penalty_ms = 200.0
    migrate_penalty_ms *= _read_migration_penalty_lambda(migrate_penalty_lambda)
    migrate_penalty_ttl_s = 2.0
    candidate_sample_m = 10
    p2c_k = 2
    budget_gamma = 0.060
    if budget_gamma_override is not None:
        budget_gamma = float(budget_gamma_override)
    if candidate_sample_m_override is not None:
        candidate_sample_m = int(candidate_sample_m_override)
    if p2c_k_override is not None:
        p2c_k = int(p2c_k_override)
    if beta_capacity_override is not None:
        beta_capacity = float(beta_capacity_override)
    if q_high_override is not None:
        q_high = int(q_high_override)
    if q_low_override is not None:
        q_low = int(q_low_override)
    if move_k_override is not None:
        move_k = int(move_k_override)
    if cooldown_s_override is not None:
        cooldown_s = float(cooldown_s_override)
    if weight_scale_override is not None:
        weight_scale = float(weight_scale_override)

    rng = random.Random(seed)
    policy_rng = random.Random(seed + 17)
    telemetry_rng = random.Random(seed + 23)
    spatial_rng = random.Random(seed + 31)
    service_rng = random.Random(seed + 41)
    hotspot_rng = random.Random(seed + 47)
    arrival_zone_rngs = {
        zone_id: random.Random(_derive_seed(seed, zone_id, 101))
        for zone_id in range(n_zones)
    }
    arrival_robot_rngs = {
        robot_id: random.Random(_derive_seed(seed, robot_id, 201))
        for robot_id in range(n_robots)
    }
    delay_robot_rngs = {
        robot_id: random.Random(_derive_seed(seed, robot_id, 301))
        for robot_id in range(n_robots)
    }
    zone_rngs = {
        zone_id: random.Random(_derive_seed(seed, zone_id, 401))
        for zone_id in range(n_zones)
    }
    zone_delay_rngs = {
        zone_id: random.Random(_derive_seed(seed, zone_id, 501))
        for zone_id in range(n_zones)
    }

    t_hot_start = HOTSPOT_START_S
    t_hot_end = HOTSPOT_END_S
    hotspot_jitter_s = HOTSPOT_JITTER_S
    hotspot_duration = t_hot_end - t_hot_start
    hot_start = t_hot_start + rng.uniform(-hotspot_jitter_s, hotspot_jitter_s)
    hot_start = max(0.0, min(hot_start, duration_s - hotspot_duration))
    hot_end = hot_start + hotspot_duration
    t_hot_start = hot_start
    t_hot_end = hot_end
    hotspot_ratio_base = HOTSPOT_RATIO_BASE
    if arrival_mode == "zone_poisson" and extra_robots_per_zone is not None:
        extra_robots_per_zone = None
    if (
        arrival_mode == "robot_emit"
        and hotspot_target_zones is not None
        and hotspot_ratio_override is not None
        and extra_robots_per_zone is None
    ):
        k_hot = len(hotspot_target_zones)
        base_per_zone = n_robots / float(n_zones) if n_zones else 0.0
        desired_total = hotspot_ratio_override * n_robots
        baseline_total = k_hot * base_per_zone
        extra_total = max(0.0, desired_total - baseline_total)
        extra_robots_per_zone = (
            int(round(extra_total / k_hot)) if k_hot > 0 else 0
        )
    if hotspot_target_zones is not None and extra_robots_per_zone is not None:
        k_hot = len(hotspot_target_zones)
        base_per_zone = n_robots / float(n_zones) if n_zones else 0.0
        hotspot_ratio = (
            (k_hot * (base_per_zone + extra_robots_per_zone)) / n_robots
            if n_robots > 0
            else 0.0
        )
        hotspot_multiplier = 1.0
        target_zone = hotspot_target_zones[0] if hotspot_target_zones else 0
    elif hotspot_ratio_override is not None:
        hotspot_ratio = float(hotspot_ratio_override)
        hotspot_multiplier = 1.0
        target_zone = hotspot_target_zones[0] if hotspot_target_zones else 0
    else:
        hotspot_multiplier = hotspot_rng.uniform(
            HOTSPOT_RATIO_MULTIPLIER_RANGE[0],
            HOTSPOT_RATIO_MULTIPLIER_RANGE[1],
        )
        hotspot_ratio = hotspot_ratio_base * hotspot_multiplier
        target_zone = hotspot_rng.randrange(n_zones)

    violation_ms = 50.0
    window_s = 1.0
    sim = Simulator()
    recorder = Recorder()

    mapping = {robot_id: robot_id % n_zones for robot_id in range(n_robots)}
    home_robots_by_zone = {zone_id: [] for zone_id in range(n_zones)}
    for robot_id, zone_id in mapping.items():
        home_robots_by_zone[zone_id].append(robot_id)
    neighbors = {
        zone_id: [(zone_id - 1) % n_zones, (zone_id + 1) % n_zones]
        for zone_id in range(n_zones)
    }
    topology = Topology(mapping, n_zones, neighbors=neighbors)
    spatial_model = SpatialModel(
        robot_to_zone=mapping,
        zones_count=n_zones,
        rng=spatial_rng,
        zone_scale=10.0,
        robot_sigma=0.5,
        base_ms=3.0,
        k_ms_per_unit=1.0,
        jitter_ms=1.0,
    )

    total_arrival_rate = state_rate_hz * n_robots
    hotspot_zone_ids_fixed = (
        list(hotspot_target_zones)
        if hotspot_target_zones is not None
        else [target_zone]
    )
    k_hot_fixed = max(1, len(hotspot_zone_ids_fixed))
    lambda_hot = None
    lambda_cold = None
    if arrival_mode == "zone_poisson":
        if hotspot_target_zones is not None and extra_robots_per_zone is not None:
            robots_per_zone = n_robots / float(n_zones) if n_zones else 0.0
            per_robot_rate = state_rate_hz
            lambda_cold = per_robot_rate * robots_per_zone
            lambda_hot = per_robot_rate * (robots_per_zone + extra_robots_per_zone)
        else:
            lambda_hot = (
                total_arrival_rate * hotspot_ratio / k_hot_fixed if k_hot_fixed else 0.0
            )
            if n_zones > k_hot_fixed:
                lambda_cold = (total_arrival_rate - lambda_hot * k_hot_fixed) / (
                    n_zones - k_hot_fixed
                )
            else:
                lambda_cold = 0.0
    if scheme == "S0":
        router = CentralizedRouter(topology)
        behavior = ForwardBehavior()
        policy = NoopPolicy()
        telemetry = None
    elif scheme == "S1":
        router = StaticEdgeRouter(topology)
        behavior = AggregateCacheBehavior()
        policy = NoopPolicy()
        telemetry = None
    else:
        router = AdaptiveRouter(topology)
        behavior = AggregateCacheBehavior()
        policy = HotspotZonePolicy(
            policy_period_s,
            q_high,
            q_low,
            move_k,
            cooldown_s,
            beta_capacity,
            budget_gamma,
            candidate_sample_m,
            p2c_k,
            dmax_ms,
            disable_fallback=disable_fallback,
            disable_budget=disable_budget,
            fixed_k=fixed_k,
            move_k_fixed=move_k_fixed,
            total_arrival_rate=total_arrival_rate,
            weight_w500=weight_w500,
            weight_w1000=weight_w1000,
            weight_w1500=weight_w1500,
            weight_scale=weight_scale,
            min_gain=min_gain,
        )
        telemetry = RobotTelemetry(
            robot_ids=range(n_robots),
            alpha_ema=alpha_ema,
            heavy_fraction=heavy_fraction,
            cost_factor_light=cost_factor_light,
            cost_factor_heavy=cost_factor_heavy,
            cost_factor_mode=cost_factor_mode,
            migrate_penalty_ms=migrate_penalty_ms,
            migrate_penalty_ttl_s=migrate_penalty_ttl_s,
            spatial_model=spatial_model,
            rng=telemetry_rng,
        )

    service_time_jitter = 0.50
    emit_jitter_s = 0.02
    central = CentralSupervisor(
        sim,
        recorder,
        central_service_rate_msgs_s,
        rng=service_rng,
        service_time_jitter=service_time_jitter,
    )
    zones = [
        ZoneController(
            zone_id=i,
            sim=sim,
            recorder=recorder,
            base_ms=zone_to_central_base_ms,
            jitter_ms=zone_to_central_jitter_ms,
            rng=zone_rngs.get(i, rng),
            central=central,
            zone_service_rate_msgs_s=zone_service_rate_msgs_s,
            behavior=behavior,
            service_time_jitter=service_time_jitter,
        )
        for i in range(n_zones)
    ]
    robots = []
    zone_arrivals = []
    zone_arrival_map = {}
    if arrival_mode == "robot_emit":
        robots = [
            Robot(
                robot_id=i,
                state_rate_hz=state_rate_hz,
                sim=sim,
                router=router,
                recorder=recorder,
                base_ms=0.0,
                jitter_ms=0.0,
                rng=rng,
                arrival_rng=arrival_robot_rngs.get(i, rng),
                delay_rng=delay_robot_rngs.get(i, rng),
                telemetry=telemetry,
                delay_model=spatial_model,
                emit_jitter_s=emit_jitter_s,
                poisson_arrivals=True,
            )
            for i in range(n_robots)
        ]
    else:
        for zone_id in range(n_zones):
            rate = 0.0
            if lambda_hot is not None and lambda_cold is not None:
                rate = lambda_hot if zone_id in hotspot_zone_ids_fixed else lambda_cold
            zone_arrivals.append(
                ZoneArrivalProcess(
                    zone_id=zone_id,
                    rate_hz=rate,
                    sim=sim,
                    router=router,
                    recorder=recorder,
                    rng=arrival_zone_rngs.get(zone_id, rng),
                    arrival_rng=arrival_zone_rngs.get(zone_id, rng),
                    delay_rng=zone_delay_rngs.get(zone_id, rng),
                    home_robot_ids=home_robots_by_zone.get(zone_id, []),
                    telemetry=telemetry,
                    delay_model=spatial_model,
                )
            )
        zone_arrival_map = {process.zone_id: process for process in zone_arrivals}

    num_windows = int(math.ceil(duration_s / window_s))
    overload_ratios = [0.0] * num_windows
    overload_ratios_q100 = [0.0] * num_windows
    overload_ratios_q500 = [0.0] * num_windows
    overload_ratios_q1000 = [0.0] * num_windows
    overload_ratios_q1500 = [0.0] * num_windows
    hotspot_overload_ratios_q1000 = [0.0] * num_windows
    queue_max_per_window = [0.0] * num_windows
    nonhot_queue_min_per_window = [0.0] * num_windows
    hotspot_queue_by_window = [[] for _ in range(num_windows)]
    queue_all_by_window = [[] for _ in range(num_windows)]
    queue_mean_hot_per_window = [0.0] * num_windows
    queue_p95_hot_per_window = [0.0] * num_windows
    migrated_weight_window = [0.0] * num_windows
    reconfig_actions_window = [0.0] * num_windows
    last_migrated_weight = 0.0
    last_reconfig_actions = 0
    hotspot_params = {
        "t_hot_start": t_hot_start,
        "t_hot_end": t_hot_end,
        "hotspot_ratio": hotspot_ratio,
        "target_zone": target_zone,
    }
    if hotspot_target_zones is not None and extra_robots_per_zone is not None:
        hotspot_params["target_zones"] = list(hotspot_target_zones)
        hotspot_params["extra_robots_per_zone"] = int(extra_robots_per_zone)
    hotspot_zone_ids = list(hotspot_zone_ids_fixed)
    nonhot_zone_ids = [
        zone_id for zone_id in range(n_zones) if zone_id not in set(hotspot_zone_ids)
    ]
    hotspot_max_central_q = 0
    hotspot_max_zone_q = 0

    def handle_robot_emit(robot_id):
        if arrival_mode != "robot_emit":
            return
        robots[robot_id].on_emit()

    def handle_zone_emit(payload):
        zone_id = payload["zone_id"]
        process = zone_arrival_map.get(zone_id)
        if process is not None:
            process.on_emit(sim.now)

    def handle_arrive_zone(payload):
        msg = payload["msg"]
        zone_id = payload["zone_id"]
        zones[zone_id].on_arrive(msg)

    def handle_zone_done(token):
        zone_id = token["zone_id"]
        zones[zone_id].on_done(token)

    def handle_arrive_central(msg):
        central.on_arrive(msg)

    def handle_central_done(msg):
        central.on_done(msg)

    def handle_policy_tick(payload):
        nonlocal hotspot_max_central_q, hotspot_max_zone_q
        nonlocal last_migrated_weight, last_reconfig_actions
        kind = payload.get("kind", "policy_tick")
        if kind in ("hotspot_start", "hotspot_end"):
            if arrival_mode == "zone_poisson":
                return
            hotspot_params["action"] = "start" if kind == "hotspot_start" else "end"
            changed = apply_hotspot(topology, sim.now, hotspot_params, spatial_model)
            hotspot_params.pop("action", None)
            if changed:
                router.on_policy_update(topology)
            return
        if kind == "metrics_tick":
            tick_index = payload["tick_index"]
            if t_hot_start <= sim.now <= t_hot_end:
                zone_q = max(
                    (zones[z].queue_len() for z in hotspot_zone_ids if 0 <= z < len(zones)),
                    default=0,
                )
                if zone_q > hotspot_max_zone_q:
                    hotspot_max_zone_q = zone_q
            if tick_index < num_windows:
                all_qs = [zones[z].queue_len() for z in range(n_zones)]
                queue_all_by_window[tick_index] = all_qs
                zone_qs = [
                    zones[z].queue_len()
                    for z in hotspot_zone_ids
                    if 0 <= z < len(zones)
                ]
                hotspot_queue_by_window[tick_index] = zone_qs
                queue_len = max(zone_qs, default=0)
                queue_max_per_window[tick_index] = queue_len
                if zone_qs:
                    queue_mean_hot_per_window[tick_index] = sum(zone_qs) / len(zone_qs)
                    sorted_qs = sorted(zone_qs)
                    p95_index = int(math.ceil(0.95 * len(sorted_qs)) - 1)
                    queue_p95_hot_per_window[tick_index] = sorted_qs[p95_index]
                else:
                    queue_mean_hot_per_window[tick_index] = 0.0
                    queue_p95_hot_per_window[tick_index] = 0.0
                nonhot_min = min(
                    (zones[z].queue_len() for z in nonhot_zone_ids if 0 <= z < len(zones)),
                    default=0,
                )
                nonhot_queue_min_per_window[tick_index] = nonhot_min
                current_migrated = getattr(policy, "migrated_weight_total", 0.0) or 0.0
                delta_migrated = max(0.0, current_migrated - last_migrated_weight)
                migrated_weight_window[tick_index] = delta_migrated
                last_migrated_weight = current_migrated
                current_actions = getattr(
                    policy, "reconfig_action_count", getattr(policy, "reassign_ops", 0)
                )
                delta_actions = max(0, current_actions - last_reconfig_actions)
                reconfig_actions_window[tick_index] = delta_actions
                last_reconfig_actions = current_actions
            next_time = sim.now + window_s
            next_index = tick_index + 1
            if next_time <= duration_s:
                sim.schedule(
                    next_time,
                    EventType.POLICY_TICK,
                    {"kind": "metrics_tick", "tick_index": next_index},
                )
            return

        tick_index = payload["tick_index"]
        policy.on_interval_start(sim.now, zones, topology, router, telemetry, policy_rng)
        policy.on_tick(sim.now, zones, topology, router, telemetry, policy_rng)
        next_time = sim.now + control_interval_s
        next_index = tick_index + 1
        if next_time <= duration_s:
            sim.schedule(
                next_time,
                EventType.POLICY_TICK,
                {"kind": "policy_tick", "tick_index": next_index},
            )

    sim.handlers[EventType.ROBOT_EMIT] = handle_robot_emit
    if arrival_mode == "zone_poisson":
        sim.handlers[EventType.ZONE_EMIT] = handle_zone_emit
    sim.handlers[EventType.ARRIVE_ZONE] = handle_arrive_zone
    sim.handlers[EventType.ZONE_DONE] = handle_zone_done
    sim.handlers[EventType.ARRIVE_CENTRAL] = handle_arrive_central
    sim.handlers[EventType.CENTRAL_DONE] = handle_central_done
    sim.handlers[EventType.POLICY_TICK] = handle_policy_tick

    if arrival_mode == "robot_emit":
        for robot in robots:
            if robot.state_rate_hz > 0:
                interval = 1.0 / robot.state_rate_hz
                start_offset = robot.arrival_rng.uniform(0.0, interval)
            else:
                start_offset = 0.0
            robot.schedule_first(start_offset)
    else:
        for process in zone_arrivals:
            if process.rate_hz > 0:
                interval = 1.0 / process.rate_hz
                start_offset = process.arrival_rng.uniform(0.0, interval)
            else:
                start_offset = 0.0
            process.schedule_first(start_offset)

    sim.schedule(t_hot_start, EventType.POLICY_TICK, {"kind": "hotspot_start"})
    if t_hot_end is not None:
        sim.schedule(t_hot_end, EventType.POLICY_TICK, {"kind": "hotspot_end"})
    sim.schedule(
        control_interval_s,
        EventType.POLICY_TICK,
        {"kind": "policy_tick", "tick_index": 0},
    )
    sim.schedule(
        window_s,
        EventType.POLICY_TICK,
        {"kind": "metrics_tick", "tick_index": 0},
    )
    sim.run(duration_s)

    overload_ratios_q100 = compute_overload_ratio(
        hotspot_queue_by_window,
        100,
        sample_dt=window_s,
        warmup_s=0.0,
        scope="hotspot_only",
    )
    overload_ratios_q500 = compute_overload_ratio(
        hotspot_queue_by_window,
        500,
        sample_dt=window_s,
        warmup_s=0.0,
        scope="hotspot_only",
    )
    overload_ratios_q1000 = compute_overload_ratio(
        hotspot_queue_by_window,
        1000,
        sample_dt=window_s,
        warmup_s=0.0,
        scope="hotspot_only",
    )
    overload_ratios_q1500 = compute_overload_ratio(
        hotspot_queue_by_window,
        1500,
        sample_dt=window_s,
        warmup_s=0.0,
        scope="hotspot_only",
    )
    dynamic_hotspot_overload_ratios_q1000 = _dynamic_hotspot_overload_ratio(
        queue_all_by_window,
        recorder,
        window_s,
        k_hot_fixed,
        1000,
        warmup_s=0.0,
        fallback_zone_ids=hotspot_zone_ids,
    )
    serving_hotspot_overload_ratios_q500, _, _ = _serving_hotspot_overload_ratio(
        queue_all_by_window,
        recorder,
        window_s,
        k_hot_fixed,
        500,
        warmup_s=0.0,
        hotspot_zone_ids=hotspot_zone_ids,
    )
    (
        serving_hotspot_overload_ratios_q1000,
        serving_hotspot_sets,
        serving_hotspot_queues,
    ) = _serving_hotspot_overload_ratio(
        queue_all_by_window,
        recorder,
        window_s,
        k_hot_fixed,
        1000,
        warmup_s=0.0,
        hotspot_zone_ids=hotspot_zone_ids,
    )
    serving_hotspot_overload_ratios_q1500, _, _ = _serving_hotspot_overload_ratio(
        queue_all_by_window,
        recorder,
        window_s,
        k_hot_fixed,
        1500,
        warmup_s=0.0,
        hotspot_zone_ids=hotspot_zone_ids,
    )
    global_topk_overload_ratios_q500, _ = _global_topk_overload_ratio(
        queue_all_by_window, k_hot_fixed, 500, window_s, warmup_s=0.0
    )
    global_topk_overload_ratios_q1000, global_topk_queues = _global_topk_overload_ratio(
        queue_all_by_window, k_hot_fixed, 1000, window_s, warmup_s=0.0
    )
    global_topk_overload_ratios_q1500, _ = _global_topk_overload_ratio(
        queue_all_by_window, k_hot_fixed, 1500, window_s, warmup_s=0.0
    )
    overload_ratios = overload_ratios_q1000
    hotspot_overload_ratios_q1000 = overload_ratios_q1000

    rows = compute_window_kpis(
        recorder,
        duration_s,
        window_s,
        overload_ratios,
        violation_ms,
        overload_ratios_q500=overload_ratios_q500,
        overload_ratios_q1000=overload_ratios_q1000,
        overload_ratios_q1500=overload_ratios_q1500,
        hotspot_overload_ratios_q1000=hotspot_overload_ratios_q1000,
        queue_max_per_window=queue_max_per_window,
        queue_min_nonhot_per_window=nonhot_queue_min_per_window,
        hotspot_zone_ids=hotspot_zone_ids,
        warmup_s=0.0,
    )
    global_rows = compute_window_kpis(
        recorder,
        duration_s,
        window_s,
        overload_ratios,
        violation_ms,
        overload_ratios_q500=overload_ratios_q500,
        overload_ratios_q1000=overload_ratios_q1000,
        overload_ratios_q1500=overload_ratios_q1500,
        hotspot_overload_ratios_q1000=hotspot_overload_ratios_q1000,
        queue_max_per_window=queue_max_per_window,
        queue_min_nonhot_per_window=nonhot_queue_min_per_window,
        hotspot_zone_ids=None,
        warmup_s=0.0,
    )
    for index, row in enumerate(rows):
        if index < len(global_rows):
            row["global_generated"] = global_rows[index].get("generated")
            row["global_completed"] = global_rows[index].get("completed")
            row["global_total_completed"] = global_rows[index].get("total_completed")
            row["global_completed_total"] = row.get("global_completed")
        row["hotspot_generated_fixed"] = row.get("generated")
        row["hotspot_completed_fixed"] = row.get("completed")
        row["fixed_hotspot_completed"] = row.get("completed")
        row["fixed_hotspot_completed_total"] = row.get("fixed_hotspot_completed")
        global_gen = row.get("global_generated") or 0
        global_comp = row.get("global_completed_total") or 0
        if global_gen:
            row["global_completion_ratio"] = global_comp / global_gen
            if row["global_completion_ratio"] > 1.0 + 1.0e-9:
                raise RuntimeError(
                    "Invalid global_completion_ratio > 1.0 "
                    f"(global_completed={global_comp}, global_generated={global_gen})"
                )
        fixed_gen = row.get("hotspot_generated_fixed") or 0
        fixed_comp = row.get("fixed_hotspot_completed_total") or 0
        if fixed_gen:
            row["fixed_hotspot_completion_ratio"] = fixed_comp / fixed_gen
            if row["fixed_hotspot_completion_ratio"] > 1.0 + 1.0e-9:
                raise RuntimeError(
                    "Invalid fixed_hotspot_completion_ratio > 1.0 "
                    f"(completed={fixed_comp}, generated={fixed_gen})"
                )
        if index < len(overload_ratios_q1000):
            row["fixed_hotspot_overload_ratio_q1000"] = overload_ratios_q1000[index]
        if index < len(overload_ratios_q500):
            row["fixed_hotspot_overload_ratio_q500"] = overload_ratios_q500[index]
        if index < len(overload_ratios_q1500):
            row["fixed_hotspot_overload_ratio_q1500"] = overload_ratios_q1500[index]
        if index < len(dynamic_hotspot_overload_ratios_q1000):
            row["dynamic_hotspot_overload_ratio_q1000"] = (
                dynamic_hotspot_overload_ratios_q1000[index]
            )
        if index < len(serving_hotspot_overload_ratios_q500):
            row["serving_hotspot_overload_ratio_q500"] = (
                serving_hotspot_overload_ratios_q500[index]
            )
        if index < len(serving_hotspot_overload_ratios_q1000):
            row["serving_hotspot_overload_ratio_q1000"] = (
                serving_hotspot_overload_ratios_q1000[index]
            )
        if index < len(serving_hotspot_overload_ratios_q1500):
            row["serving_hotspot_overload_ratio_q1500"] = (
                serving_hotspot_overload_ratios_q1500[index]
            )
        if index < len(global_topk_overload_ratios_q500):
            row["global_topk_overload_ratio_q500"] = global_topk_overload_ratios_q500[index]
        if index < len(global_topk_overload_ratios_q1000):
            row["global_topk_overload_ratio_q1000"] = (
                global_topk_overload_ratios_q1000[index]
            )
        if index < len(global_topk_overload_ratios_q1500):
            row["global_topk_overload_ratio_q1500"] = global_topk_overload_ratios_q1500[index]
        if index < len(serving_hotspot_sets):
            row["serving_hotspot_set"] = ",".join(str(z) for z in serving_hotspot_sets[index])
        if index < len(serving_hotspot_queues):
            row["serving_hotspot_topk_queues"] = ",".join(
                _format_float(val, 2) for val in serving_hotspot_queues[index]
            )
        if index < len(global_topk_queues):
            row["global_topk_queues"] = ",".join(
                _format_float(val, 2) for val in global_topk_queues[index]
            )
        if index < len(queue_mean_hot_per_window):
            row["queue_mean_hot"] = queue_mean_hot_per_window[index]
        if index < len(queue_p95_hot_per_window):
            row["queue_p95_hot"] = queue_p95_hot_per_window[index]
        if index < len(migrated_weight_window):
            row["migrated_weight_window"] = migrated_weight_window[index]
        if index < len(reconfig_actions_window):
            row["reconfig_actions_window"] = reconfig_actions_window[index]
    arrival_trace = _arrival_trace_from_recorder(recorder, limit=None)
    arrivals_first_60s = sum(
        1
        for record in recorder.records.values()
        if record.get("emit_time") is not None and record.get("emit_time") <= 60.0
    )
    output_path = None
    if write_csv:
        os.makedirs(output_dir, exist_ok=True)
        if tag:
            output_path = os.path.join(
                output_dir, f"window_kpis_{tag}_{scheme.lower()}.csv"
            )
        else:
            output_path = os.path.join(output_dir, f"window_kpis_{scheme.lower()}.csv")
        if tag:
            trace_path = os.path.join(
                output_dir, f"arrival_trace_{tag}_{scheme.lower()}.csv"
            )
        else:
            trace_path = os.path.join(output_dir, f"arrival_trace_{scheme.lower()}.csv")
        extra_fields = {
            "policy_reassign_ops": getattr(policy, "reassign_ops", 0),
            "reconfig_action_count": getattr(
                policy, "reconfig_action_count", getattr(policy, "reassign_ops", 0)
            ),
            "migrated_robots_total": getattr(policy, "migrated_robots_total", 0),
            "migrated_weight_total": getattr(policy, "migrated_weight_total", 0.0),
            "rejected_no_feasible_target": getattr(policy, "rejected_no_feasible_target", 0),
            "rejected_budget": getattr(policy, "rejected_budget", 0),
            "rejected_safety": getattr(policy, "rejected_safety", 0),
            "fallback_attempts": getattr(policy, "fallback_attempts", 0),
            "fallback_success": getattr(policy, "fallback_success", 0),
            "dmax_rejects": getattr(policy, "dmax_rejects", 0),
            "hotspot_start_s": f"{t_hot_start:.3f}",
            "hotspot_end_s": f"{t_hot_end:.3f}",
            "hotspot_zone_id": target_zone,
            "hotspot_zone_ids": ",".join(str(z) for z in hotspot_zone_ids),
            "hotspot_ratio": f"{hotspot_ratio:.4f}",
            "hotspot_multiplier": f"{hotspot_multiplier:.4f}",
            "arrivals_first_60s": arrivals_first_60s,
        }
        feasible_ratio_sum = getattr(policy, "feasible_ratio_sum", 0.0)
        feasible_ratio_count = getattr(policy, "feasible_ratio_count", 0)
        feasible_ratio_mean = (
            feasible_ratio_sum / feasible_ratio_count
            if feasible_ratio_count
            else 0.0
        )
        extra_fields["feasible_ratio_mean"] = feasible_ratio_mean
        extra_fields["feasible_ratio_count"] = feasible_ratio_count
        write_window_kpis_csv(output_path, scheme, seed, rows, extra_fields=extra_fields)
        _write_arrival_trace(trace_path, arrival_trace)
        debug_path = os.path.join(
            output_dir, f"debug_windows_{tag or scheme.lower()}_{scheme.lower()}.csv"
        )
        with open(debug_path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "t_start",
                    "t_end",
                    "queue_mean_hot",
                    "queue_p95_hot",
                    "overload_ratio_q1000",
                    "fixed_hotspot_overload_ratio_q1000",
                    "dynamic_hotspot_overload_ratio_q1000",
                    "serving_hotspot_overload_ratio_q1000",
                    "global_topk_overload_ratio_q1000",
                    "serving_hotspot_set",
                    "global_topk_queues",
                    "migrated_weight_window",
                    "reconfig_actions_window",
                ]
            )
            for row in rows:
                writer.writerow(
                    [
                        row.get("t_start"),
                        row.get("t_end"),
                        row.get("queue_mean_hot"),
                        row.get("queue_p95_hot"),
                        row.get("overload_ratio_q1000"),
                        row.get("fixed_hotspot_overload_ratio_q1000"),
                        row.get("dynamic_hotspot_overload_ratio_q1000"),
                        row.get("serving_hotspot_overload_ratio_q1000"),
                        row.get("global_topk_overload_ratio_q1000"),
                        row.get("serving_hotspot_set"),
                        row.get("global_topk_queues"),
                        row.get("migrated_weight_window"),
                        row.get("reconfig_actions_window"),
                    ]
                )

    mean_p95 = _hotspot_mean_p95(rows, t_hot_start, t_hot_end)
    overload_duration = _overload_duration(
        overload_ratios, t_hot_start, t_hot_end, window_s, duration_s
    )
    overload_duration_q100 = _overload_duration(
        overload_ratios_q100, t_hot_start, t_hot_end, window_s, duration_s
    )
    overload_duration_q1000 = _overload_duration(
        overload_ratios_q1000, t_hot_start, t_hot_end, window_s, duration_s
    )
    policy_reassign_ops = getattr(policy, "reassign_ops", 0)
    migrated_robots_total = getattr(policy, "migrated_robots_total", 0)
    migrated_weight_total = getattr(policy, "migrated_weight_total", 0.0)
    rejected_no_feasible_target = getattr(policy, "rejected_no_feasible_target", 0)
    rejected_budget = getattr(policy, "rejected_budget", 0)
    rejected_safety = getattr(policy, "rejected_safety", 0)
    fallback_attempts = getattr(policy, "fallback_attempts", 0)
    fallback_success = getattr(policy, "fallback_success", 0)
    dmax_rejects = getattr(policy, "dmax_rejects", 0)
    return (
        output_path,
        mean_p95,
        overload_duration,
        overload_duration_q100,
        overload_duration_q1000,
        hotspot_max_central_q,
        hotspot_max_zone_q,
        policy_reassign_ops,
        migrated_robots_total,
        migrated_weight_total,
        rejected_no_feasible_target,
        rejected_budget,
        rejected_safety,
        fallback_attempts,
        fallback_success,
        dmax_rejects,
        migrate_penalty_ms,
        q_high,
        q_low,
        move_k,
        cooldown_s,
        beta_capacity,
        budget_gamma,
        candidate_sample_m,
        p2c_k,
    )


def _format_multiplier_tag(value):
    return f"{value:.3f}".replace(".", "p")


def _calibration_tag(n_zones, multiplier, standard_load, seed, prefix):
    load_text = f"{standard_load:.3f}".replace(".", "p")
    mult_text = _format_multiplier_tag(multiplier)
    return f"{prefix}N{n_zones}_std{load_text}_m{mult_text}_seed{seed}"


def _seeded_tag(base_tag, seed):
    return f"{base_tag}_seed{seed}"


def _enforce_monotone_nonincreasing(values):
    # Pool-adjacent-violators for non-increasing sequence.
    blocks = []
    for value in values:
        blocks.append([value, 1])
        while len(blocks) >= 2:
            prev_sum, prev_count = blocks[-2]
            curr_sum, curr_count = blocks[-1]
            prev_mean = prev_sum / prev_count
            curr_mean = curr_sum / curr_count
            if prev_mean >= curr_mean:
                break
            merged_sum = prev_sum + curr_sum
            merged_count = prev_count + curr_count
            blocks = blocks[:-2]
            blocks.append([merged_sum, merged_count])
    result = []
    for total, count in blocks:
        mean_val = total / count
        result.extend([mean_val] * count)
    return result


def _fig4_budget_scale(n_zones, mode="sqrt"):
    if mode == "linear":
        return n_zones / 16.0
    return math.sqrt(n_zones / 16.0)


def _apply_budget_scale(overrides, scale):
    scaled = dict(overrides)
    if "budget_gamma_override" in scaled:
        scaled["budget_gamma_override"] = scaled["budget_gamma_override"] * scale
    if scaled.get("fixed_k"):
        base_k = scaled.get("move_k_fixed")
        if base_k is not None:
            scaled["move_k_fixed"] = max(1, int(round(base_k * scale)))
    else:
        base_k = scaled.get("move_k_override")
        if base_k is not None:
            scaled["move_k_override"] = max(1, int(round(base_k * scale)))
    return scaled


def _window_has_audit_columns(path):
    required = {
        "hotspot_start_s",
        "hotspot_end_s",
        "hotspot_zone_id",
        "hotspot_zone_ids",
        "hotspot_ratio",
        "hotspot_multiplier",
        "arrivals_first_60s",
    }
    return _window_has_columns(path, required=required)


def _collect_seed_audit(path, scheme_key, profile, n_zones, load_multiplier, seed):
    rows = _load_window_rows(path, required=None)
    if not rows:
        return None
    row = rows[0]
    return {
        "seed": seed,
        "N": n_zones,
        "scheme": scheme_key,
        "profile": profile,
        "load_multiplier": load_multiplier,
        "hotspot_start_s": _parse_float(row.get("hotspot_start_s")),
        "hotspot_end_s": _parse_float(row.get("hotspot_end_s")),
        "hotspot_zone_id": _parse_int(row.get("hotspot_zone_id")),
        "hotspot_multiplier": _parse_float(row.get("hotspot_multiplier")),
        "hotspot_ratio": _parse_float(row.get("hotspot_ratio")),
        "arrivals_first_60s": _parse_int(row.get("arrivals_first_60s")),
    }


def _write_seed_audit(path, rows):
    if not rows:
        return
    columns = [
        "seed",
        "N",
        "scheme",
        "profile",
        "load_multiplier",
        "hotspot_start_s",
        "hotspot_end_s",
        "hotspot_zone_id",
        "hotspot_multiplier",
        "hotspot_ratio",
        "arrivals_first_60s",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow(
                [
                    row.get("seed", ""),
                    row.get("N", ""),
                    row.get("scheme", ""),
                    row.get("profile", ""),
                    _format_float(row.get("load_multiplier"), 6),
                    _format_float(row.get("hotspot_start_s"), 3),
                    _format_float(row.get("hotspot_end_s"), 3),
                    row.get("hotspot_zone_id", ""),
                    _format_float(row.get("hotspot_multiplier"), 4),
                    _format_float(row.get("hotspot_ratio"), 4),
                    row.get("arrivals_first_60s", ""),
                ]
            )


def _assert_seed_audit_variation(rows, label):
    if not rows:
        return True
    grouped = {}
    for row in rows:
        key = (row.get("N"), row.get("scheme"), row.get("profile"), row.get("load_multiplier"))
        grouped.setdefault(key, []).append(row)
    for key, items in grouped.items():
        seeds = {item.get("seed") for item in items}
        if len(seeds) <= 1:
            continue
        tuples = {
            (
                item.get("hotspot_start_s"),
                item.get("hotspot_zone_id"),
                item.get("hotspot_multiplier"),
                item.get("arrivals_first_60s"),
            )
            for item in items
        }
        if len(tuples) <= 1:
            raise RuntimeError(
                f"Seed audit failed for {label} group={key}: "
                "stochastic parameters identical across seeds."
            )
    return True


def _arrival_trace_from_recorder(recorder, limit=None):
    events = []
    for msg_id, record in recorder.records.items():
        emit_time = record.get("emit_time")
        if emit_time is None:
            continue
        home_zone_id = record.get("home_zone_id")
        if home_zone_id is None:
            home_zone_id = record.get("emit_zone_id")
        events.append(
            (
                float(emit_time),
                home_zone_id,
                record.get("emit_robot_id"),
                msg_id,
            )
        )
    events.sort(key=lambda item: (item[0], item[3]))
    if limit is not None:
        return events[:limit]
    return events


def _write_arrival_trace(path, events):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["emit_time", "home_zone_id", "robot_id", "msg_id"])
        for emit_time, home_zone, robot_id, msg_id in events:
            writer.writerow([_format_float(emit_time, 6), home_zone, robot_id, msg_id])


def _load_arrival_trace(path):
    if not os.path.isfile(path):
        return []
    rows = []
    with open(path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                (
                    _parse_float(row.get("emit_time")),
                    _parse_int(row.get("home_zone_id")),
                    _parse_int(row.get("robot_id")),
                    _parse_int(row.get("msg_id")),
                )
            )
    return rows


def _assert_arrival_trace_consistency(traces, label):
    if not traces:
        return
    base_label, base = traces[0]
    base_map = {item[3]: item for item in base}
    for scheme_label, events in traces[1:]:
        if events == base:
            continue
        diffs = []
        event_map = {item[3]: item for item in events}
        for msg_id, base_item in list(base_map.items())[:5]:
            other = event_map.get(msg_id)
            if other != base_item:
                diffs.append((msg_id, base_item, other))
        raise RuntimeError(
            f"Arrival trace mismatch for {label}: {scheme_label} differs from {base_label}. "
            f"First diffs: {diffs}"
        )

def _s2_profile_overrides(tag_suffix="", base_profiles=None):
    suffix = tag_suffix or ""
    base_profiles = base_profiles or {
        "conservative": {
            "budget_gamma_override": 0.025,
            "min_gain_override": 8.0,
        },
        "neutral": {
            "budget_gamma_override": 0.080,
            "min_gain_override": 5.0,
        },
        "aggressive": {
            "budget_gamma_override": 0.200,
            "min_gain_override": 2.0,
        },
    }
    return {
        f"S2_conservative{suffix}": dict(base_profiles.get("conservative", {})),
        f"S2_neutral{suffix}": dict(base_profiles.get("neutral", {})),
        f"S2_aggressive{suffix}": dict(base_profiles.get("aggressive", {})),
    }


def _write_profile_grid(path, rows):
    columns = [
        "N",
        "weight_scale",
        "budget_gamma",
        "cooldown_s",
        "fixed_hotspot_overload_ratio_q1000_mean",
        "migrated_weight_total_mean",
        "hotspot_p95_mean_ms",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow(
                [
                    row.get("N", ""),
                    _format_float(row.get("weight_scale_override"), 4),
                    _format_float(row.get("budget_gamma_override"), 6),
                    _format_float(row.get("cooldown_s_override"), 3),
                    _format_float(row.get("fixed_hotspot_overload_ratio_q1000_mean"), 6),
                    _format_float(row.get("migrated_weight_total_mean"), 6),
                    _format_float(row.get("hotspot_p95_mean_ms"), 6),
                ]
            )


def _select_profiles_from_grid(rows, baseline_overload=None):
    if not rows:
        raise RuntimeError("No grid search results available.")

    def _overload_value(row):
        val = row.get("fixed_hotspot_overload_ratio_q1000_mean")
        return val if val is not None and not math.isnan(val) else None

    def _expand_candidates(low, high, require_lt=None):
        step = 0.05
        for expand in range(12):
            low_exp = max(0.0, low - step * expand)
            high_exp = min(1.0, high + step * expand)
            candidates = []
            for row in rows:
                over = _overload_value(row)
                if over is None:
                    continue
                if over < low_exp - 1.0e-9 or over > high_exp + 1.0e-9:
                    continue
                if require_lt is not None and not (over < require_lt - 1.0e-9):
                    continue
                candidates.append(row)
            if candidates:
                return candidates, low_exp, high_exp
        return [], low, high

    baseline_cap = baseline_overload if baseline_overload is not None else 1.0
    lite_cap = baseline_cap * 0.8 if baseline_overload is not None else baseline_cap
    strong_candidates, _, _ = _expand_candidates(0.0, 0.10)
    balanced_candidates, _, _ = _expand_candidates(0.12, 0.30)
    lite_candidates, _, _ = _expand_candidates(0.35, 0.55, require_lt=lite_cap)

    if not strong_candidates or not balanced_candidates or not lite_candidates:
        raise RuntimeError("Failed to select Lite/Balanced/Strong profiles from grid.")

    best = None
    for min_gap in (0.05, 0.03, 0.0):
        for strong in sorted(strong_candidates, key=lambda r: _overload_value(r) or 0.0):
            strong_over = _overload_value(strong)
            for balanced in sorted(
                balanced_candidates,
                key=lambda r: r.get("migrated_weight_total_mean", float("inf")),
            ):
                balanced_over = _overload_value(balanced)
                if balanced_over is None or strong_over is None:
                    continue
                if balanced_over <= strong_over + min_gap:
                    continue
                for lite in sorted(
                    lite_candidates,
                    key=lambda r: r.get("migrated_weight_total_mean", float("inf")),
                ):
                    lite_over = _overload_value(lite)
                    if lite_over is None:
                        continue
                    if lite_over <= balanced_over + min_gap:
                        continue
                    cost_sum = (
                        (strong.get("migrated_weight_total_mean") or 0.0)
                        + (balanced.get("migrated_weight_total_mean") or 0.0)
                        + (lite.get("migrated_weight_total_mean") or 0.0)
                    )
                    best = (cost_sum, lite, balanced, strong)
                    break
                if best:
                    break
            if best:
                break
        if best:
            break

    if not best:
        raise RuntimeError("Failed to select Lite/Balanced/Strong profiles from grid.")

    _, lite, balanced, strong = best

    def _as_override(row):
        return {
            "weight_scale_override": row.get("weight_scale_override"),
            "budget_gamma_override": row.get("budget_gamma_override"),
            "cooldown_s_override": row.get("cooldown_s_override"),
        }

    return {
        "conservative": _as_override(lite),
        "neutral": _as_override(balanced),
        "aggressive": _as_override(strong),
    }, {"lite": lite, "balanced": balanced, "strong": strong}


def _grid_search_s2_profiles(
    output_dir,
    seeds,
    state_rate_hz,
    zone_rate,
    n_zones,
    n_robots,
    central_override,
    load_multiplier,
    arrival_mode,
    deterministic_hotspots,
    hotspot_target_zones,
    extra_robots_per_zone,
    hotspot_ratio_override,
    profile_version,
):
    weight_scales = [1.0]
    budget_gammas = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10]
    cooldown_vals = [0.5]
    rows = []
    for weight_scale in weight_scales:
        for budget_gamma in budget_gammas:
            for cooldown_s in cooldown_vals:
                tag_base = (
                    f"Grid_{profile_version}_w{weight_scale:.2f}_b{budget_gamma:.3f}_c{cooldown_s:.2f}"
                ).replace(".", "p")
                seed_rows = []
                for seed in seeds:
                    seed_tag = _seeded_tag(tag_base, seed)
                    path = os.path.join(output_dir, f"window_kpis_{seed_tag}_s2.csv")
                    if not _window_has_columns(path, required=WINDOW_REQUIRED_QUEUE) or not _window_has_audit_columns(path):
                        run_scheme(
                            "S2",
                            seed,
                            output_dir,
                            state_rate_hz=state_rate_hz,
                            zone_service_rate_msgs_s=zone_rate,
                            write_csv=True,
                            n_zones_override=n_zones,
                            n_robots_override=n_robots,
                            central_service_rate_msgs_s_override=central_override,
                            dmax_ms=30.0,
                            tag=seed_tag,
                            arrival_mode=arrival_mode,
                            hotspot_target_zones=hotspot_target_zones if deterministic_hotspots else None,
                            extra_robots_per_zone=extra_robots_per_zone
                            if (deterministic_hotspots and arrival_mode == "robot_emit")
                            else None,
                            hotspot_ratio_override=hotspot_ratio_override,
                            weight_scale_override=weight_scale,
                            budget_gamma_override=budget_gamma,
                            cooldown_s_override=cooldown_s,
                        )
                    seed_rows.append(
                        _collect_seed_summary(
                            output_dir,
                            seed_tag,
                            "S2",
                            "Grid",
                            "grid",
                            seed,
                            n_zones,
                            load_multiplier,
                            summary_tag=tag_base,
                        )
                    )
                overload_mean = _mean_or_nan(
                    [row.get("fixed_hotspot_overload_ratio_q1000") for row in seed_rows]
                )
                cost_mean = _mean_or_nan(
                    [row.get("migrated_weight_total") for row in seed_rows]
                )
                p95_mean = _mean_or_nan(
                    [row.get("hotspot_p95_mean_ms") for row in seed_rows]
                )
                rows.append(
                    {
                        "N": n_zones,
                        "weight_scale_override": weight_scale,
                        "budget_gamma_override": budget_gamma,
                        "cooldown_s_override": cooldown_s,
                        "fixed_hotspot_overload_ratio_q1000_mean": overload_mean,
                        "migrated_weight_total_mean": cost_mean,
                        "hotspot_p95_mean_ms": p95_mean,
                    }
                )
    return rows


def _write_profile_overrides(path, overrides, selection_info=None):
    payload = {
        "profiles": overrides,
        "selection": selection_info or {},
        "timestamp": int(time.time()),
    }
    with open(path, "w", newline="") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _load_profile_overrides(path):
    if not os.path.isfile(path):
        return None
    with open(path, "r") as handle:
        payload = json.load(handle)
    return payload.get("profiles")


def _load_summary_rows(path):
    if not os.path.isfile(path):
        return []
    with open(path, "r", newline="") as handle:
        return list(csv.DictReader(handle))


def _resolve_load_multiplier(row):
    for key in ("load_multiplier", "load_scale", "load"):
        if key in row:
            return _parse_float(row.get(key))
    return None


def _load_legacy_main_multiplier(legacy_dir):
    if not legacy_dir:
        return None
    summary_path = os.path.join(legacy_dir, "summary_main_and_ablations.csv")
    rows = _load_summary_rows(summary_path)
    if not rows:
        raise RuntimeError(f"Legacy summary not found or empty: {summary_path}")
    selected = None
    for row in rows:
        if _parse_int(row.get("N")) != 16:
            continue
        if row.get("scheme") == "S1" and row.get("profile") == "baseline":
            selected = row
            break
    if selected is None:
        selected = rows[0]
    multiplier = _resolve_load_multiplier(selected)
    if multiplier is None:
        raise RuntimeError(f"Missing load_multiplier in legacy summary: {summary_path}")
    overload = _parse_float(selected.get("fixed_hotspot_overload_ratio_q1000"))
    if overload is None:
        overload = _parse_float(selected.get("overload_ratio_q1000"))
    return {"multiplier": multiplier, "overload_q1000": overload}


def _load_legacy_fig4_multipliers(legacy_dir):
    if not legacy_dir:
        return None
    summary_path = os.path.join(legacy_dir, "summary_fig4_scaledload.csv")
    rows = _load_summary_rows(summary_path)
    if not rows:
        raise RuntimeError(f"Legacy fig4 summary not found or empty: {summary_path}")
    calibration = {}
    for row in rows:
        if row.get("scheme") != "S1" or row.get("profile") != "baseline":
            continue
        n_val = _parse_int(row.get("N"))
        if n_val is None:
            continue
        multiplier = _resolve_load_multiplier(row)
        if multiplier is None:
            continue
        calibration[n_val] = {
            "multiplier": multiplier,
            "baseline_overload_q1000_mean": _parse_float(
                row.get("fixed_hotspot_overload_ratio_q1000")
            )
            or _parse_float(row.get("overload_ratio_q1000")),
            "baseline_overload_q1000_ci_low": _parse_float(row.get("overload_q1000_ci_low")),
            "baseline_overload_q1000_ci_high": _parse_float(row.get("overload_q1000_ci_high")),
        }
    if not calibration:
        raise RuntimeError(f"No baseline rows found in legacy fig4 summary: {summary_path}")
    return calibration


def _resolve_legacy_repro_dir(output_dir):
    marker = os.path.join(output_dir, "repro_20260122_like_latest.txt")
    if os.path.isfile(marker):
        with open(marker, "r") as handle:
            path = handle.read().strip()
        if path and os.path.isdir(path):
            return path
    stamp = time.strftime("%Y%m%d_%H%M%S")
    repro_dir = os.path.join(output_dir, f"repro_20260122_like_{stamp}")
    os.makedirs(repro_dir, exist_ok=True)
    with open(marker, "w") as handle:
        handle.write(repro_dir)
    return repro_dir


def _write_legacy_used(repro_dir, snapshot_dir, main_multiplier, fig4_map, arrival_mode, k_hot, deterministic_hotspots):
    path = os.path.join(repro_dir, "legacy_used.txt")
    with open(path, "w", newline="") as handle:
        handle.write(f"snapshot_dir={snapshot_dir}\n")
        handle.write(f"arrival_mode={arrival_mode}\n")
        handle.write(f"deterministic_hotspots={deterministic_hotspots}\n")
        handle.write(f"k_hot={k_hot}\n")
        if main_multiplier is not None:
            handle.write(f"main_load_multiplier={main_multiplier}\n")
        if fig4_map:
            for n_val in sorted(fig4_map):
                handle.write(f"fig4_N{n_val}_multiplier={fig4_map[n_val]}\n")
    return path


def _legacy_copy_outputs(output_dir, repro_dir):
    figures_main = os.path.join(output_dir, "figure_z16_noc")
    figures_fig4 = os.path.join(output_dir, "figures")
    candidates = [
        os.path.join(figures_main, "summary_main_and_ablations.csv"),
        os.path.join(figures_main, "summary_s2_profiles.csv"),
        os.path.join(figures_main, "summary_s2_selected.csv"),
        os.path.join(figures_fig4, "summary_fig4_scaledload.csv"),
        os.path.join(figures_main, "fig1_hotspot_p95_main_constrained.png"),
        os.path.join(figures_main, "fig2_overload_ratio_main_constrained.png"),
        os.path.join(figures_main, "fig3_tradeoff_scatter_constrained.png"),
        os.path.join(figures_fig4, "fig4_scaledload_scalability.png"),
    ]
    for src in candidates:
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(repro_dir, os.path.basename(src)))


def calibrate_load_multiplier_for_N(
    output_dir,
    seeds,
    state_rate_hz_base,
    standard_load,
    zone_rate,
    n_zones,
    n_robots,
    central_rate_override,
    target_low,
    target_high,
    m_low=0.5,
    m_high=2.5,
    max_iters=10,
    prefix="Calib",
    calibration_log=None,
    hotspot_target_zones=None,
    extra_robots_per_zone=None,
    hotspot_ratio_override=None,
    arrival_mode="robot_emit",
    force_rerun=False,
):
    cache = {}

    def eval_multiplier(multiplier):
        if multiplier in cache:
            return cache[multiplier]
        ratios = []
        baseline_overloads = []
        for seed in seeds:
            tag = _calibration_tag(n_zones, multiplier, standard_load, seed, prefix)
            path = os.path.join(output_dir, f"window_kpis_{tag}_s1.csv")
            if force_rerun or not _window_has_columns(path, required=WINDOW_REQUIRED_QUEUE) or not _window_has_audit_columns(path):
                state_rate_hz = state_rate_hz_base * standard_load * multiplier
                central_override, _ = _central_rate_override(
                    state_rate_hz, n_robots, n_zones, zone_rate
                )
                run_scheme(
                    "S1",
                    seed,
                    output_dir,
                    state_rate_hz=state_rate_hz,
                    zone_service_rate_msgs_s=zone_rate,
                    write_csv=True,
                    n_zones_override=n_zones,
                    n_robots_override=n_robots,
                    central_service_rate_msgs_s_override=central_override,
                    dmax_ms=30.0,
                    tag=tag,
                    hotspot_target_zones=hotspot_target_zones,
                    extra_robots_per_zone=extra_robots_per_zone,
                    hotspot_ratio_override=hotspot_ratio_override,
                    arrival_mode=arrival_mode,
                )
            rows = _load_window_rows(path)
            if not rows:
                raise RuntimeError(
                    f"Empty window rows for N={n_zones} seed={seed} "
                    f"(hotspot_window_start_end unknown, rows=0)."
                )
            t_hot_start = 30.0
            t_hot_end = 80.0
            if rows:
                hot_start = _parse_float(rows[0].get("hotspot_start_s"))
                hot_end = _parse_float(rows[0].get("hotspot_end_s"))
                if hot_start is not None and hot_end is not None:
                    t_hot_start = hot_start
                    t_hot_end = hot_end
            ratio, _ = _hotspot_mean_metric(
                rows, t_hot_start, t_hot_end, "fixed_hotspot_overload_ratio_q1000"
            )
            if ratio is None or (isinstance(ratio, float) and math.isnan(ratio)):
                raise RuntimeError(
                    "Invalid overload ratio during calibration "
                    f"(N={n_zones}, seed={seed}, rows={len(rows)}, "
                    f"hotspot_window={t_hot_start}-{t_hot_end})."
                )
            ratios.append(ratio)
        mean_ratio = _mean(ratios)
        ci_low, ci_high = _bootstrap_ci(ratios)
        cache[multiplier] = (mean_ratio, ci_low, ci_high)
        status = "ok"
        if mean_ratio < target_low:
            status = "low"
        elif mean_ratio > target_high:
            status = "high"
        print(
            f"Calibrate N={n_zones} m={multiplier:.3f} "
            f"overload_q1000={mean_ratio:.3f} ({status})"
        )
        return cache[multiplier]

    best_multiplier = m_low
    ratio_low, _, _ = eval_multiplier(m_low)
    ratio_high, _, _ = eval_multiplier(m_high)
    best_ratio = ratio_low
    best_dist = abs(_distance_to_interval(ratio_low, target_low, target_high))
    if abs(_distance_to_interval(ratio_high, target_low, target_high)) < best_dist:
        best_dist = abs(_distance_to_interval(ratio_high, target_low, target_high))
        best_ratio = ratio_high
        best_multiplier = m_high

    expand_iters = 0
    while ratio_low > target_high and expand_iters < 6:
        m_high = m_low
        m_low = max(0.05, m_low / 2.0)
        ratio_low, _, _ = eval_multiplier(m_low)
        expand_iters += 1
        best_multiplier = m_low
        best_ratio = ratio_low
        best_dist = abs(_distance_to_interval(ratio_low, target_low, target_high))

    expand_iters = 0
    while ratio_high < target_low and expand_iters < 6:
        m_low = m_high
        m_high = m_high * 2.0
        ratio_high, _, _ = eval_multiplier(m_high)
        expand_iters += 1
        dist = abs(_distance_to_interval(ratio_high, target_low, target_high))
        if dist < best_dist:
            best_dist = dist
            best_multiplier = m_high
            best_ratio = ratio_high

    for _ in range(max_iters):
        mid = (m_low + m_high) / 2.0
        ratio_mid, _, _ = eval_multiplier(mid)
        dist = abs(_distance_to_interval(ratio_mid, target_low, target_high))
        if dist < best_dist:
            best_dist = dist
            best_multiplier = mid
            best_ratio = ratio_mid
        if _is_within_interval(ratio_mid, target_low, target_high):
            best_multiplier = mid
            best_ratio = ratio_mid
            break
        if ratio_mid < target_low:
            m_low = mid
        else:
            m_high = mid

    final_ratio, ci_low, ci_high = eval_multiplier(best_multiplier)
    if calibration_log is not None:
        calibration_log.append(
            {
                "N": n_zones,
                "multiplier": best_multiplier,
                "multiplier_raw": best_multiplier,
                "multiplier_monotone": best_multiplier,
                "baseline_overload_q1000_mean": final_ratio,
                "baseline_overload_q1000_ci_low": ci_low,
                "baseline_overload_q1000_ci_high": ci_high,
                "seeds": ",".join(str(seed) for seed in seeds),
                "timestamp": int(time.time()),
            }
        )
    if not _is_within_interval(final_ratio, target_low, target_high):
        print(
            f"Warning: could not hit target for N={n_zones}; "
            f"using m={best_multiplier:.3f} ratio={final_ratio:.3f}"
        )
    return best_multiplier, final_ratio, ci_low, ci_high


def calibrate_multiplier_by_N(
    output_dir,
    seeds,
    state_rate_hz_base,
    standard_load,
    zone_rate,
    n_zones,
    n_robots,
    central_rate_override,
    target_low,
    target_high,
    m_low=0.5,
    m_high=2.5,
    max_iters=10,
    prefix="Fig4Calib",
):
    return calibrate_load_multiplier_for_N(
        output_dir,
        seeds,
        state_rate_hz_base,
        standard_load,
        zone_rate,
        n_zones,
        n_robots,
        central_rate_override,
        target_low,
        target_high,
        m_low=m_low,
        m_high=m_high,
        max_iters=max_iters,
        prefix=prefix,
        calibration_log=None,
    )


def _collect_seed_summary(
    output_dir,
    file_tag,
    scheme_key,
    scheme_label,
    profile,
    seed,
    n_zones,
    load_multiplier,
    summary_tag=None,
    t_hot_start=30.0,
    t_hot_end=80.0,
    budget_scale_factor=1.0,
):
    summary_tag = summary_tag or file_tag
    path = os.path.join(output_dir, f"window_kpis_{file_tag}_{scheme_key.lower()}.csv")
    rows = _load_window_rows(path)
    if rows:
        hot_start = _parse_float(rows[0].get("hotspot_start_s"))
        hot_end = _parse_float(rows[0].get("hotspot_end_s"))
        if hot_start is not None and hot_end is not None:
            t_hot_start = hot_start
            t_hot_end = hot_end
    summary = _summarize_window_rows(rows, t_hot_start, t_hot_end, scheme_label, seed)
    summary.update(
        {
            "N": n_zones,
            "tag": summary_tag,
            "scheme": scheme_key,
            "profile": profile,
            "seed": seed,
            "load_multiplier": load_multiplier,
            "budget_scale_factor": budget_scale_factor,
        }
    )
    return summary


def _write_policy_profiles(path, profiles, budget_scale_factor=1.0):
    columns = [
        "scheme",
        "q_set",
        "q_low",
        "beta",
        "weight_scale",
        "cooldown_s",
        "policy_period_s",
        "max_migrate_per_action",
        "budget_scale_factor",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for scheme, overrides in profiles.items():
            move_k = (
                overrides.get("move_k_fixed")
                if overrides.get("fixed_k")
                else overrides.get("move_k_override")
            )
            writer.writerow(
                [
                    scheme,
                    overrides.get("q_high_override", ""),
                    overrides.get("q_low_override", ""),
                    overrides.get("budget_gamma_override", ""),
                    overrides.get("weight_scale_override", ""),
                    overrides.get("cooldown_s_override", ""),
                    overrides.get("policy_period_s_override", ""),
                    move_k if move_k is not None else "",
                    _format_float(budget_scale_factor, 3),
                ]
            )


def _write_policy_profiles_by_n(path, profiles, n_values, scale_fn):
    columns = [
        "N",
        "scheme",
        "q_set",
        "q_low",
        "beta",
        "weight_scale",
        "cooldown_s",
        "policy_period_s",
        "max_migrate_per_action",
        "budget_scale_factor",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for n_val in n_values:
            scale = scale_fn(n_val)
            for scheme, overrides in profiles.items():
                move_k = (
                    overrides.get("move_k_fixed")
                    if overrides.get("fixed_k")
                    else overrides.get("move_k_override")
                )
                writer.writerow(
                    [
                        n_val,
                        scheme,
                        overrides.get("q_high_override", ""),
                        overrides.get("q_low_override", ""),
                        overrides.get("budget_gamma_override", ""),
                        overrides.get("weight_scale_override", ""),
                        overrides.get("cooldown_s_override", ""),
                        overrides.get("policy_period_s_override", ""),
                        move_k if move_k is not None else "",
                        _format_float(scale, 3),
                    ]
                )


def _write_calibration_log(path, rows):
    columns = [
        "N",
        "multiplier_raw",
        "multiplier_monotone",
        "multiplier",
        "baseline_overload_q1000_mean",
        "baseline_overload_q1000_ci_low",
        "baseline_overload_q1000_ci_high",
        "seeds",
        "timestamp",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow(
                [
                    row.get("N", ""),
                    _format_float(row.get("multiplier_raw", row.get("multiplier")), 4),
                    _format_float(
                        row.get("multiplier_monotone", row.get("multiplier")), 4
                    ),
                    _format_float(row.get("multiplier"), 4),
                    _format_float(row.get("baseline_overload_q1000_mean"), 6),
                    _format_float(row.get("baseline_overload_q1000_ci_low"), 6),
                    _format_float(row.get("baseline_overload_q1000_ci_high"), 6),
                    row.get("seeds", ""),
                    row.get("timestamp", ""),
                ]
                )


def _write_fig4_calibration(path, rows):
    columns = [
        "N",
        "multiplier",
        "baseline_overload_q1000_mean",
        "baseline_overload_q1000_ci_low",
        "baseline_overload_q1000_ci_high",
        "seeds",
        "timestamp",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow(
                [
                    row.get("N", ""),
                    _format_float(row.get("multiplier"), 8),
                    _format_float(row.get("baseline_overload_q1000_mean"), 6),
                    _format_float(row.get("baseline_overload_q1000_ci_low"), 6),
                    _format_float(row.get("baseline_overload_q1000_ci_high"), 6),
                    row.get("seeds", ""),
                    row.get("timestamp", ""),
                ]
            )


def _write_selected_profiles(path, selections):
    columns = [
        "N",
        "profile_name",
        "selected_role",
        "tag",
        "scheme",
        "profile",
        "x_cost",
        "y_overload",
        "x_norm",
        "y_norm",
        "score",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for selection in selections:
            row = selection["row"]
            overload = row.get("fixed_hotspot_overload_ratio_q1000_mean")
            if overload is None or (isinstance(overload, float) and math.isnan(overload)):
                overload = row.get("overload_ratio_q1000")
            cost = row.get("migrated_weight_total_mean")
            if cost is None or (isinstance(cost, float) and math.isnan(cost)):
                cost = row.get("migrated_weight_total")
            writer.writerow(
                [
                    row.get("N", ""),
                    selection["profile_name"],
                    selection["selected_role"],
                    row.get("tag", ""),
                    row.get("scheme", ""),
                    row.get("profile", ""),
                    _format_float(cost, 6),
                    _format_float(overload, 6),
                    "",
                    "",
                    "",
                ]
            )


def _effective_policy_config(overrides, dmax_ms, total_arrival_rate=None):
    q_high = 1000
    q_low = 900
    move_k = 200
    cooldown_s = 1.0
    beta_capacity = 0.80
    candidate_sample_m = 10
    p2c_k = 2
    budget_gamma = 0.060
    policy_period_s = 1.0
    weight_w500 = 1.0
    weight_w1000 = 2.0
    weight_w1500 = 4.0
    weight_scale = 1.0
    min_gain = 5.0
    fixed_k = overrides.get("fixed_k", False)
    move_k_fixed = overrides.get("move_k_fixed")

    if overrides.get("q_high_override") is not None:
        q_high = overrides.get("q_high_override")
    if overrides.get("q_low_override") is not None:
        q_low = overrides.get("q_low_override")
    if overrides.get("move_k_override") is not None:
        move_k = overrides.get("move_k_override")
    if overrides.get("cooldown_s_override") is not None:
        cooldown_s = overrides.get("cooldown_s_override")
    if overrides.get("beta_capacity_override") is not None:
        beta_capacity = overrides.get("beta_capacity_override")
    if overrides.get("candidate_sample_m_override") is not None:
        candidate_sample_m = overrides.get("candidate_sample_m_override")
    if overrides.get("p2c_k_override") is not None:
        p2c_k = overrides.get("p2c_k_override")
    if overrides.get("budget_gamma_override") is not None:
        budget_gamma = overrides.get("budget_gamma_override")
    if overrides.get("weight_scale_override") is not None:
        weight_scale = overrides.get("weight_scale_override")
    if overrides.get("policy_period_s_override") is not None:
        policy_period_s = overrides.get("policy_period_s_override")
    if overrides.get("min_gain_override") is not None:
        min_gain = overrides.get("min_gain_override")
    if fixed_k and move_k_fixed is not None:
        move_k = move_k_fixed

    tokens_per_s = (
        budget_gamma * float(total_arrival_rate)
        if total_arrival_rate is not None
        else None
    )
    return {
        "q_high": q_high,
        "q_set": q_high,
        "q_low": q_low,
        "cooldown_s": cooldown_s,
        "policy_period_s": policy_period_s,
        "move_k": move_k,
        "candidate_sample_m": candidate_sample_m,
        "p2c_k": p2c_k,
        "beta_capacity": beta_capacity,
        "budget_gamma": budget_gamma,
        "beta": budget_gamma,
        "weight_w500": weight_w500,
        "weight_w1000": weight_w1000,
        "weight_w1500": weight_w1500,
        "weight_scale": weight_scale,
        "min_gain": min_gain,
        "tokens_per_s": tokens_per_s,
        "fixed_k": fixed_k,
        "move_k_fixed": move_k_fixed if fixed_k else None,
        "dmax_ms": dmax_ms,
    }


def _write_run_config(path, config):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)


def _load_run_config(path):
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_main_run_config(output_dir, figures_dir):
    main_config_path = os.path.join(output_dir, "figure_z16_noc", "run_config.json")
    if not os.path.isfile(main_config_path):
        main_config_path = os.path.join(figures_dir, "run_config_main.json")
    if not os.path.isfile(main_config_path):
        main_config_path = os.path.join(figures_dir, "run_config.json")
    main_config = _load_run_config(main_config_path)
    if not main_config:
        raise RuntimeError("Missing main run_config.json; cannot load baseline config.")
    runs = main_config.get("runs") or []
    if not runs:
        raise RuntimeError("Main run_config.json has no runs to derive parameters.")
    run0 = runs[0]
    arrival = run0.get("arrival", {})
    service = run0.get("service", {})
    hotspot = run0.get("hotspot", {})
    network = run0.get("network", {})
    state_rate_hz_base = _parse_float(arrival.get("state_rate_hz_base"))
    standard_load = _parse_float(main_config.get("standard_load"))
    zone_rate = _parse_float(service.get("zone_service_rate_msgs_s"))
    n_robots_main = _parse_int(run0.get("n_robots"))
    n_zones_main = _parse_int(run0.get("N"))
    if state_rate_hz_base is None or standard_load is None or zone_rate is None:
        raise RuntimeError("Missing arrival/service parameters in main run_config.json.")
    if not n_robots_main or not n_zones_main:
        raise RuntimeError("Missing N/n_robots in main run_config.json.")
    robots_per_zone = n_robots_main / float(n_zones_main)
    base_lambda_unscaled = _parse_float(main_config.get("base_lambda_per_zone_unscaled"))
    base_lambda = _parse_float(main_config.get("base_lambda_per_zone"))
    base_delta_unscaled = _parse_float(main_config.get("hotspot_delta_per_zone_unscaled"))
    base_delta = _parse_float(main_config.get("hotspot_delta_per_zone"))
    hotspot_skew = _parse_float(main_config.get("hotspot_skew"))
    hotspot_frac_zones = _parse_float(main_config.get("hotspot_frac_zones"))
    deterministic_hotspots = bool(main_config.get("deterministic_hotspots", False))
    arrival_mode_main = main_config.get("arrival_mode")
    deterministic_hotspot_zones = main_config.get("deterministic_hotspot_zones")
    if base_lambda_unscaled is None:
        base_lambda_unscaled = base_lambda
    if base_delta_unscaled is None:
        base_delta_unscaled = base_delta
    return {
        "path": main_config_path,
        "config": main_config,
        "arrival": arrival,
        "service": service,
        "hotspot": hotspot,
        "network": network,
        "state_rate_hz_base": state_rate_hz_base,
        "standard_load": standard_load,
        "calibration_multiplier": _parse_float(main_config.get("calibration_multiplier")),
        "zone_rate": zone_rate,
        "robots_per_zone": robots_per_zone,
        "n_zones_main": n_zones_main,
        "n_robots_main": n_robots_main,
        "base_lambda_per_zone": base_lambda,
        "base_lambda_per_zone_unscaled": base_lambda_unscaled,
        "base_mu_per_zone": _parse_float(main_config.get("base_mu_per_zone")),
        "hotspot_frac_zones": hotspot_frac_zones,
        "hotspot_delta_per_zone": base_delta,
        "hotspot_delta_per_zone_unscaled": base_delta_unscaled,
        "hotspot_skew": hotspot_skew,
        "deterministic_hotspots": deterministic_hotspots,
        "arrival_mode_main": arrival_mode_main,
        "deterministic_hotspot_zones": deterministic_hotspot_zones,
    }


def _diff_dicts(lhs, rhs, prefix=""):
    diffs = []
    keys = set(lhs.keys()) | set(rhs.keys())
    for key in sorted(keys):
        left_val = lhs.get(key)
        right_val = rhs.get(key)
        if isinstance(left_val, dict) and isinstance(right_val, dict):
            diffs.extend(_diff_dicts(left_val, right_val, prefix=f"{prefix}{key}."))
        else:
            if left_val != right_val:
                diffs.append((f"{prefix}{key}", left_val, right_val))
    return diffs


def _diff_run_configs(main_config, fig4_config, n_zones, scheme_label):
    if not main_config or not fig4_config:
        return []
    main_runs = main_config.get("runs", [])
    fig4_runs = fig4_config.get("runs", [])
    main_row = next(
        (
            row
            for row in main_runs
            if row.get("N") == n_zones and row.get("scheme") == scheme_label
        ),
        None,
    )
    fig4_row = next(
        (
            row
            for row in fig4_runs
            if row.get("N") == n_zones and row.get("scheme") == scheme_label
        ),
        None,
    )
    if not main_row or not fig4_row:
        return []
    return _diff_dicts(main_row, fig4_row)


def run_main_pipeline(
    output_dir,
    figures_dir,
    seeds,
    profile_version,
    arrival_mode="robot_emit",
    deterministic_hotspots=False,
    hotspot_frac_zones=None,
    auto_profile_search=False,
    legacy_reproduce_dir=None,
    profile_overrides_path=None,
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    state_rate_hz_base = 10
    standard_load = 2.0
    zone_rate = 200
    n_zones = 16
    base_n_zones = 8
    base_n_robots = 100
    robots_per_zone = base_n_robots / float(base_n_zones)
    n_robots = int(round(robots_per_zone * n_zones))
    central_rate_override = n_zones * zone_rate
    tag_suffix_base = f"_z{n_zones}_std{standard_load:.1f}_{profile_version}"
    hotspot_zone_id = 0
    hotspot_frac_zones_used = (
        float(hotspot_frac_zones) if hotspot_frac_zones is not None else 1.0 / n_zones
    )
    hotspot_zones_count = max(1, int(round(hotspot_frac_zones_used * n_zones)))
    deterministic_hotspot_zones = _deterministic_hotspot_zones(
        n_zones, hotspot_zones_count, center_zone=hotspot_zone_id
    )
    extra_robots_per_zone = None
    if deterministic_hotspots:
        base_per_zone = n_robots / float(n_zones) if n_zones else 0.0
        desired_total = HOTSPOT_RATIO_BASE * n_robots
        baseline_total = hotspot_zones_count * base_per_zone
        extra_total = max(0.0, desired_total - baseline_total)
        extra_robots_per_zone = (
            int(round(extra_total / hotspot_zones_count))
            if hotspot_zones_count > 0
            else 0
        )
    hotspot_ratio_override = HOTSPOT_RATIO_BASE if deterministic_hotspots else None
    if arrival_mode == "zone_poisson":
        extra_robots_per_zone = None

    calibration_log = []
    legacy_main = _load_legacy_main_multiplier(legacy_reproduce_dir)
    if legacy_main is not None:
        calib_multiplier = legacy_main["multiplier"]
        calib_ratio = legacy_main.get("overload_q1000")
        ci_low = None
        ci_high = None
        calibration_log.append(
            {
                "N": n_zones,
                "multiplier": calib_multiplier,
                "baseline_overload_q1000_mean": calib_ratio,
                "baseline_overload_q1000_ci_low": ci_low,
                "baseline_overload_q1000_ci_high": ci_high,
                "seeds": ",".join(str(seed) for seed in seeds),
                "timestamp": int(time.time()),
            }
        )
        auto_profile_search = False
    else:
        calib_multiplier, calib_ratio, ci_low, ci_high = calibrate_load_multiplier_for_N(
            output_dir,
            seeds,
            state_rate_hz_base,
            standard_load,
            zone_rate,
            n_zones,
            n_robots,
            central_rate_override,
            target_low=0.55,
            target_high=0.65,
            m_low=0.5,
            m_high=2.5,
            calibration_log=calibration_log,
            prefix=f"Calib_{profile_version}",
            arrival_mode=arrival_mode,
            hotspot_target_zones=deterministic_hotspot_zones if deterministic_hotspots else None,
            extra_robots_per_zone=extra_robots_per_zone if deterministic_hotspots else None,
            hotspot_ratio_override=hotspot_ratio_override,
        )
    calib_path = os.path.join(results_dir, "fig1_3_calibration.csv")
    _write_calibration_log(calib_path, calibration_log)
    tag_suffix = f"{tag_suffix_base}_m{calib_multiplier:.3f}".replace(".", "p")
    main_tag = f"Main_constrained{tag_suffix}"
    state_rate_hz = state_rate_hz_base * standard_load * calib_multiplier
    total_arrival_rate = state_rate_hz * n_robots
    central_override, central_scaled = _central_rate_override(
        state_rate_hz, n_robots, n_zones, zone_rate
    )

    profile_overrides_base = None
    if profile_overrides_path:
        profile_overrides_base = _load_profile_overrides(profile_overrides_path)
        if not profile_overrides_base:
            raise RuntimeError(f"Failed to load profile overrides: {profile_overrides_path}")
        auto_profile_search = False
        overrides_path = os.path.join(figures_dir, "s2_profile_overrides.json")
        _write_profile_overrides(overrides_path, profile_overrides_base)
    if auto_profile_search:
        grid_rows = _grid_search_s2_profiles(
            output_dir,
            seeds,
            state_rate_hz,
            zone_rate,
            n_zones,
            n_robots,
            central_override,
            calib_multiplier,
            arrival_mode,
            deterministic_hotspots,
            deterministic_hotspot_zones,
            extra_robots_per_zone,
            hotspot_ratio_override,
            profile_version,
        )
        grid_path = os.path.join(figures_dir, "summary_s2_profile_grid.csv")
        _write_profile_grid(grid_path, grid_rows)
        profile_overrides_base, selection_info = _select_profiles_from_grid(
            grid_rows, baseline_overload=calib_ratio
        )
        overrides_path = os.path.join(figures_dir, "s2_profile_overrides.json")
        _write_profile_overrides(overrides_path, profile_overrides_base, selection_info)
        print(f"S2 profile grid search saved: {grid_path}")
        print(f"S2 profile overrides saved: {overrides_path}")

    profile_overrides = _s2_profile_overrides(tag_suffix, base_profiles=profile_overrides_base)
    overrides_used_path = os.path.join(figures_dir, "s2_overrides_used.json")
    _write_profile_overrides(overrides_used_path, profile_overrides_base or {})
    profile_map = [
        ("conservative", f"S2_conservative{tag_suffix}", "Ours-Lite"),
        ("neutral", f"S2_neutral{tag_suffix}", "Ours-Balanced"),
        ("aggressive", f"S2_aggressive{tag_suffix}", "Ours-Strong"),
    ]

    summary_long = []
    force_rerun = bool(auto_profile_search)
    seed_audit_rows = []
    for seed in seeds:
        seed_tag = _seeded_tag(main_tag, seed)
        path = os.path.join(output_dir, f"window_kpis_{seed_tag}_s1.csv")
        if (
            force_rerun
            or not _window_has_columns(path)
            or not _window_has_audit_columns(path)
        ):
            run_scheme(
                "S1",
                seed,
                output_dir,
                state_rate_hz=state_rate_hz,
                zone_service_rate_msgs_s=zone_rate,
                write_csv=True,
                n_zones_override=n_zones,
                n_robots_override=n_robots,
                central_service_rate_msgs_s_override=central_override,
                dmax_ms=30.0,
                tag=seed_tag,
                arrival_mode=arrival_mode,
                hotspot_target_zones=deterministic_hotspot_zones if deterministic_hotspots else None,
                extra_robots_per_zone=extra_robots_per_zone
                if (deterministic_hotspots and arrival_mode == "robot_emit")
                else None,
                hotspot_ratio_override=hotspot_ratio_override,
            )
        summary_long.append(
            _collect_seed_summary(
                output_dir,
                seed_tag,
                "S1",
                "Static-edge",
                "baseline",
                seed,
                n_zones,
                calib_multiplier,
                summary_tag=main_tag,
                budget_scale_factor=1.0,
            )
        )
        audit = _collect_seed_audit(
            path, "S1", "baseline", n_zones, calib_multiplier, seed
        )
        if audit:
            seed_audit_rows.append(audit)

        for profile_name, profile_tag, label in profile_map:
            seed_profile_tag = _seeded_tag(profile_tag, seed)
            path = os.path.join(output_dir, f"window_kpis_{seed_profile_tag}_s2.csv")
            if (
                force_rerun
                or not _window_has_columns(path, required=WINDOW_REQUIRED_QUEUE)
                or not _window_has_audit_columns(path)
            ):
                run_scheme(
                    "S2",
                    seed,
                    output_dir,
                    state_rate_hz=state_rate_hz,
                    zone_service_rate_msgs_s=zone_rate,
                    write_csv=True,
                    n_zones_override=n_zones,
                    n_robots_override=n_robots,
                    central_service_rate_msgs_s_override=central_override,
                    dmax_ms=30.0,
                    tag=seed_profile_tag,
                    arrival_mode=arrival_mode,
                    hotspot_target_zones=deterministic_hotspot_zones if deterministic_hotspots else None,
                    extra_robots_per_zone=extra_robots_per_zone
                    if (deterministic_hotspots and arrival_mode == "robot_emit")
                    else None,
                    hotspot_ratio_override=hotspot_ratio_override,
                    **profile_overrides[profile_tag],
                )
            summary_long.append(
                _collect_seed_summary(
                    output_dir,
                    seed_profile_tag,
                    "S2",
                    label,
                    profile_name,
                    seed,
                    n_zones,
                    calib_multiplier,
                    summary_tag=profile_tag,
                    budget_scale_factor=1.0,
                )
            )
            audit = _collect_seed_audit(
                path, "S2", profile_name, n_zones, calib_multiplier, seed
            )
            if audit:
                seed_audit_rows.append(audit)

        if arrival_mode == "zone_poisson":
            trace_entries = []
            base_trace_path = os.path.join(
                output_dir, f"arrival_trace_{seed_tag}_s1.csv"
            )
            trace_entries.append(("Static-edge", _load_arrival_trace(base_trace_path)))
            for profile_name, profile_tag, label in profile_map:
                seed_profile_tag = _seeded_tag(profile_tag, seed)
                trace_path = os.path.join(
                    output_dir, f"arrival_trace_{seed_profile_tag}_s2.csv"
                )
                trace_entries.append((label, _load_arrival_trace(trace_path)))
            print(f"Arrival trace (first 50) seed={seed}:")
            for item in trace_entries[0][1]:
                print(item)
            _assert_arrival_trace_consistency(trace_entries, f"main seed={seed}")

    static_seed_rows = [
        row
        for row in summary_long
        if row.get("scheme") == "S1" and row.get("profile") == "baseline"
    ]
    duration_s = 120.0
    hotspot_ratio_vals = []
    for row in static_seed_rows:
        seed = row.get("seed")
        if seed is not None:
            seed_tag = _seeded_tag(main_tag, seed)
            path = os.path.join(output_dir, f"window_kpis_{seed_tag}_s1.csv")
            if os.path.isfile(path):
                rows = _load_window_rows(path, required=None)
                if rows:
                    ratio = _parse_float(rows[0].get("hotspot_ratio"))
                    if ratio is not None and not math.isnan(ratio):
                        hotspot_ratio_vals.append(ratio)
    hotspot_ratio_mean = _mean(hotspot_ratio_vals) if hotspot_ratio_vals else HOTSPOT_RATIO_BASE
    hotspot_frac_zones = 1.0 / n_zones
    state_rate_hz_calib = state_rate_hz_base * standard_load * calib_multiplier
    base_lambda_per_zone = (
        (state_rate_hz_calib * n_robots) / n_zones if n_zones else float("nan")
    )
    base_mu_per_zone = zone_rate
    k_hot = max(1, int(round(hotspot_frac_zones * n_zones)))
    phi = k_hot / float(n_zones) if n_zones else 0.0
    if hotspot_ratio_mean >= 1.0 or k_hot <= 0 or n_zones <= k_hot:
        hotspot_skew = float("nan")
        base_hotspot_delta_per_zone = 0.0
    else:
        hotspot_skew = (
            (hotspot_ratio_mean * (n_zones - k_hot))
            / ((1.0 - hotspot_ratio_mean) * k_hot)
        )
        lambda_cold = base_lambda_per_zone / (phi * hotspot_skew + (1.0 - phi))
        lambda_hot = hotspot_skew * lambda_cold
        base_hotspot_delta_per_zone = max(0.0, lambda_hot - lambda_cold)
    base_lambda_per_zone_unscaled = (
        base_lambda_per_zone / calib_multiplier if calib_multiplier else base_lambda_per_zone
    )
    base_hotspot_delta_per_zone_unscaled = (
        base_hotspot_delta_per_zone / calib_multiplier
        if calib_multiplier
        else base_hotspot_delta_per_zone
    )
    # Use per-zone hotspot fraction (one hotspot zone in main) and derived skew.
    lite_seed_rows = [
        row
        for row in summary_long
        if row.get("scheme") == "S2" and row.get("profile") == "conservative"
    ]
    balanced_seed_rows = [
        row
        for row in summary_long
        if row.get("scheme") == "S2" and row.get("profile") == "neutral"
    ]
    strong_seed_rows = [
        row
        for row in summary_long
        if row.get("scheme") == "S2" and row.get("profile") == "aggressive"
    ]
    _print_seed_metrics(static_seed_rows, "Main Static-edge")
    _print_seed_metrics(lite_seed_rows, "Main Ours-Lite")
    _print_seed_metrics(balanced_seed_rows, "Main Ours-Balanced")
    _print_seed_metrics(strong_seed_rows, "Main Ours-Strong")
    if not _seed_variation_by_group(summary_long):
        print("WARNING: Seeds appear ineffective for main scenario.")
        raise SystemExit(1)
    if arrival_mode == "zone_poisson":
        _assert_generated_consistency(summary_long, "main")

    summary_long_path = os.path.join(figures_dir, "summary_main_long.csv")
    _write_summary_long(summary_long_path, summary_long)
    seed_audit_path = os.path.join(figures_dir, "seed_audit_main.csv")
    _write_seed_audit(seed_audit_path, seed_audit_rows)
    _assert_seed_audit_variation(seed_audit_rows, "main")

    summary_agg = _aggregate_rows(
        summary_long, group_keys=["tag", "scheme", "profile", "N", "load_multiplier"]
    )
    summary_main_path = os.path.join(figures_dir, "summary_main_and_ablations.csv")
    _write_summary_agg(summary_main_path, summary_agg)

    summary_profiles = [row for row in summary_agg if row.get("scheme") == "S2"]
    summary_profiles_path = os.path.join(figures_dir, "summary_s2_profiles.csv")
    _write_summary_agg(summary_profiles_path, summary_profiles)

    selected_profiles = []
    for profile_name, profile_tag, label in profile_map:
        match = next(
            (row for row in summary_profiles if row.get("tag") == profile_tag),
            None,
        )
        if match:
            selected_profiles.append(
                {"profile_name": label, "selected_role": profile_name, "row": match}
            )
    selected_profiles_path = os.path.join(figures_dir, "summary_s2_selected.csv")
    _write_selected_profiles(selected_profiles_path, selected_profiles)

    profiles_for_table = {
        "Static-edge": {},
        "Ours-Lite": profile_overrides[profile_map[0][1]],
        "Ours-Balanced": profile_overrides[profile_map[1][1]],
        "Ours-Strong": profile_overrides[profile_map[2][1]],
    }
    profiles_path = os.path.join(figures_dir, "policy_profiles_used.csv")
    _write_policy_profiles(profiles_path, profiles_for_table, budget_scale_factor=1.0)

    config_path = os.path.join(figures_dir, "main_config.txt")
    with open(config_path, "w") as handle:
        handle.write(f"N={n_zones}\n")
        handle.write(f"standard_load={standard_load:.3f}\n")
        handle.write(f"load_multiplier={calib_multiplier:.3f}\n")
        handle.write(f"seeds={','.join(str(seed) for seed in seeds)}\n")

    arrival_config = {
        "model": "poisson",
        "state_rate_hz_base": state_rate_hz_base,
        "emit_jitter_s": 0.02,
        "poisson_arrivals": True,
    }
    service_config = {
        "zone_service_rate_msgs_s": zone_rate,
        "central_service_rate_msgs_s": central_override,
        "service_time_jitter": 0.50,
    }
    network_config = {
        "zone_to_central_base_ms": 5,
        "zone_to_central_jitter_ms": 2,
        "robot_to_zone_base_ms": 3.0,
        "robot_to_zone_k_ms_per_unit": 1.0,
        "robot_to_zone_jitter_ms": 1.0,
        "zone_scale": 10.0,
        "robot_sigma": 0.5,
    }
    hotspot_config = {
        "base_start_s": HOTSPOT_START_S,
        "base_end_s": HOTSPOT_END_S,
        "jitter_s": HOTSPOT_JITTER_S,
        "ratio_base": HOTSPOT_RATIO_BASE,
        "ratio_multiplier_range": [
            HOTSPOT_RATIO_MULTIPLIER_RANGE[0],
            HOTSPOT_RATIO_MULTIPLIER_RANGE[1],
        ],
        "target_zone_random": not deterministic_hotspots,
        "hotspot_frac_zones": hotspot_frac_zones_used,
        "deterministic_hotspot_zones": deterministic_hotspot_zones
        if deterministic_hotspots
        else [],
    }
    tag_map = {name: tag for name, tag, _ in profile_map}
    runs = []
    for label, profile_name in (
        ("Static-edge", "baseline"),
        ("Ours-Lite", "conservative"),
        ("Ours-Balanced", "neutral"),
        ("Ours-Strong", "aggressive"),
    ):
        if profile_name == "baseline":
            scheme_key = "S1"
            overrides = {}
        else:
            scheme_key = "S2"
            profile_tag = tag_map.get(profile_name)
            overrides = profile_overrides.get(profile_tag, {}) if profile_tag else {}
        policy_config = (
            _effective_policy_config(overrides, 30.0, total_arrival_rate)
            if scheme_key == "S2"
            else {}
        )
        for seed in seeds:
            runs.append(
                {
                    "N": n_zones,
                    "n_robots": n_robots,
                    "seed": seed,
                    "scheme": label,
                    "profile": profile_name,
                    "load_multiplier": calib_multiplier,
                    "horizon_s": 120.0,
                    "warmup_s": 0.0,
                    "arrival": arrival_config,
                    "service": service_config,
                    "network": network_config,
                    "hotspot": hotspot_config,
                    "policy": policy_config,
                    "budget_scale_factor": 1.0,
                    "derived": {
                        "k_hot": max(1, int(round(hotspot_frac_zones * n_zones))),
                        "phi_used": hotspot_frac_zones,
                        "phi_actual": (
                            max(1, int(round(hotspot_frac_zones * n_zones))) / float(n_zones)
                            if n_zones
                            else 0.0
                        ),
                        "lambda_mean": base_lambda_per_zone,
                        "lambda_cold": base_lambda_per_zone
                        / (hotspot_frac_zones * hotspot_skew + (1.0 - hotspot_frac_zones))
                        if hotspot_skew and hotspot_frac_zones is not None
                        else None,
                        "lambda_hot": (
                            hotspot_skew
                            * base_lambda_per_zone
                            / (hotspot_frac_zones * hotspot_skew + (1.0 - hotspot_frac_zones))
                        )
                        if hotspot_skew and hotspot_frac_zones is not None
                        else None,
                        "total_arrival_rate": total_arrival_rate,
                        "mu_zone": zone_rate,
                        "mu_c": central_override,
                        "tokens_per_s": policy_config.get("tokens_per_s") if scheme_key == "S2" else 0.0,
                        "q_set": policy_config.get("q_set") if scheme_key == "S2" else None,
                        "q_low": policy_config.get("q_low") if scheme_key == "S2" else None,
                        "control_period_s": policy_config.get("policy_period_s") if scheme_key == "S2" else None,
                    },
                }
            )
    run_config_path = os.path.join(figures_dir, "run_config.json")
    _write_run_config(
        run_config_path,
        {
            "N": n_zones,
            "standard_load": standard_load,
            "calibration_multiplier": calib_multiplier,
            "arrival_mode": arrival_mode,
            "deterministic_hotspots": deterministic_hotspots,
            "deterministic_hotspot_zones": deterministic_hotspot_zones
            if deterministic_hotspots
            else [],
            "canonical_scaling_mode": "per_zone_constant",
            "base_lambda_per_zone": base_lambda_per_zone,
            "base_lambda_per_zone_unscaled": base_lambda_per_zone_unscaled,
            "base_mu_per_zone": base_mu_per_zone,
            "hotspot_frac_zones": hotspot_frac_zones,
            "hotspot_frac_zones_used": hotspot_frac_zones_used,
            "hotspot_delta_per_zone": base_hotspot_delta_per_zone,
            "hotspot_delta_per_zone_unscaled": base_hotspot_delta_per_zone_unscaled,
            "hotspot_ratio_mean": hotspot_ratio_mean,
            "hotspot_skew": hotspot_skew,
            "seeds": seeds,
            "runs": runs,
        },
    )
    snapshot_root = os.path.join(output_dir, "config_snapshots")
    stamp_path = write_repro_stamp(
        figures_dir,
        {"mode": "main", "run_config_path": run_config_path},
        cmdline=" ".join(sys.argv),
        snapshot_root=snapshot_root,
        artifacts={
            "run_config": run_config_path,
            "s2_profile_overrides": overrides_path if os.path.isfile(overrides_path) else None,
        },
    )
    overrides_path = os.path.join(figures_dir, "s2_profile_overrides.json")
    if os.path.isfile(overrides_path):
        shutil.copy2(
            overrides_path, os.path.join(figures_dir, "s2_profile_overrides.copy.json")
        )
    print(f"Wrote repro stamp: {stamp_path}")

    print("Main calibration summary:")
    print(
        f"N={n_zones} m16={calib_multiplier:.3f} "
        f"Static-edge overload_q1000={calib_ratio:.3f} "
        f"CI=[{ci_low:.3f},{ci_high:.3f}]"
    )

    summary_by_label = {row.get("tag"): row for row in summary_agg}
    static_row = next(
        (row for row in summary_agg if row.get("scheme") == "S1"), None
    )
    lite_row = next(
        (row for row in summary_agg if row.get("tag") == profile_map[0][1]), None
    )
    balanced_row = next(
        (row for row in summary_agg if row.get("tag") == profile_map[1][1]), None
    )
    strong_row = next(
        (row for row in summary_agg if row.get("tag") == profile_map[2][1]), None
    )

    print("Verification table (N=16 calibrated):")
    for label, row in (
        ("Static-edge", static_row),
        ("Ours-Lite", lite_row),
        ("Ours-Balanced", balanced_row),
        ("Ours-Strong", strong_row),
    ):
        if row is None:
            continue
        print(
            f"{label} hotspot_p95_ms={row.get('hotspot_p95_mean_ms')} "
            f"overload_q500={row.get('overload_ratio_q500')} "
            f"overload_q1000={row.get('overload_ratio_q1000')} "
            f"overload_q1500={row.get('overload_ratio_q1500')} "
            f"reconfig_cost={row.get('migrated_weight_total')} "
            f"actions={row.get('reconfig_action_count')}"
        )

    static_ratio = static_row.get("overload_ratio_q1000") if static_row else None
    lite_ratio = lite_row.get("overload_ratio_q1000") if lite_row else None
    balanced_ratio = balanced_row.get("overload_ratio_q1000") if balanced_row else None
    strong_ratio = strong_row.get("overload_ratio_q1000") if strong_row else None
    static_p95 = static_row.get("hotspot_p95_mean_ms") if static_row else None
    lite_p95 = lite_row.get("hotspot_p95_mean_ms") if lite_row else None
    lite_cost = lite_row.get("migrated_weight_total") if lite_row else None
    balanced_cost = balanced_row.get("migrated_weight_total") if balanced_row else None
    strong_cost = strong_row.get("migrated_weight_total") if strong_row else None
    lite_q500 = lite_row.get("overload_ratio_q500") if lite_row else None
    lite_q1500 = lite_row.get("overload_ratio_q1500") if lite_row else None
    balanced_q1500 = balanced_row.get("overload_ratio_q1500") if balanced_row else None
    strong_q1500 = strong_row.get("overload_ratio_q1500") if strong_row else None

    if not _is_within_interval(static_ratio, 0.55, 0.65):
        raise RuntimeError(
            f"Static-edge overload_ratio_q1000 out of range: {static_ratio}"
        )
    if lite_ratio is None or static_ratio is None or lite_ratio >= static_ratio:
        raise RuntimeError(
            "Ours-Lite overload_ratio_q1000 must be lower than Static-edge "
            f"(lite={lite_ratio}, static={static_ratio})"
        )
    if lite_p95 is None or static_p95 is None or lite_p95 >= static_p95:
        raise RuntimeError(
            "Ours-Lite hotspot_p95_mean_ms must improve over Static-edge "
            f"(lite={lite_p95}, static={static_p95})"
        )
    for label, row in (
        ("Static-edge", static_row),
        ("Ours-Lite", lite_row),
        ("Ours-Balanced", balanced_row),
        ("Ours-Strong", strong_row),
    ):
        if row is None:
            continue
        ratio = row.get("global_completion_ratio")
        if ratio is not None and not math.isnan(ratio) and ratio < 0.99:
            print(
                f"Warning: {label} global_completion_ratio={ratio:.3f} below 0.99; "
                "completion backlog may bias latency."
            )
        fixed_ratio = row.get("fixed_hotspot_completion_ratio")
        if ratio is not None and not math.isnan(ratio) and fixed_ratio is not None and not math.isnan(fixed_ratio):
            print(
                f"{label} completion_ratio_global={ratio:.6f} "
                f"completion_ratio_fixed_hotspot={fixed_ratio:.6f}"
            )
    if lite_q500 is None or lite_q500 <= 0.0 or lite_ratio is None or lite_ratio <= 0.0:
        raise RuntimeError("Ours-Lite overload ratios must be non-zero for q500/q1000.")
    if balanced_ratio is None or lite_ratio is None or balanced_ratio > lite_ratio + 1.0e-9:
        raise RuntimeError(
            "Ours-Balanced overload_ratio_q1000 must be <= Ours-Lite "
            f"(balanced={balanced_ratio}, lite={lite_ratio})"
        )
    if strong_ratio is None or balanced_ratio is None or strong_ratio > balanced_ratio + 1.0e-9:
        raise RuntimeError(
            "Ours-Strong overload_ratio_q1000 must be <= Ours-Balanced "
            f"(strong={strong_ratio}, balanced={balanced_ratio})"
        )
    if lite_cost is None or balanced_cost is None or strong_cost is None:
        raise RuntimeError("Missing reconfiguration cost for at least one scheme.")
    if not (
        strong_cost >= balanced_cost
        and balanced_cost >= lite_cost
        and lite_cost >= 0
        and balanced_cost >= 0
    ):
        print(
            "Warning: Reconfiguration cost ordering not strictly monotone "
            f"(lite={lite_cost}, balanced={balanced_cost}, strong={strong_cost})."
        )

    return {
        "calibration_multiplier": calib_multiplier,
        "summary_main_path": summary_main_path,
        "summary_profiles_path": summary_profiles_path,
        "summary_selected_path": selected_profiles_path,
        "summary_long_path": summary_long_path,
        "calibration_path": calib_path,
        "profiles_path": profiles_path,
        "config_path": config_path,
        "hotspot_zone_id": hotspot_zone_id,
    }


def _read_fig4_calibration(path):
    if not os.path.isfile(path):
        raise RuntimeError(f"Missing fig4 calibration file: {path}")
    with open(path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    result = {}
    for row in rows:
        n_val = _parse_int(row.get("N"))
        multiplier = _parse_float(row.get("multiplier"))
        if multiplier is None:
            multiplier = _parse_float(
                row.get("multiplier_monotone") or row.get("multiplier_raw")
            )
        if n_val is None or multiplier is None:
            continue
        result[n_val] = multiplier
    return result


def run_fig4_calibrate(
    output_dir,
    figures_dir,
    seeds,
    profile_version,
    arrival_mode="robot_emit",
    exclude_small_n=False,
    force_rerun=False,
    legacy_reproduce_dir=None,
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    main_bundle = _load_main_run_config(output_dir, figures_dir)
    state_rate_hz_base = main_bundle["state_rate_hz_base"]
    standard_load = main_bundle["standard_load"]
    zone_rate = main_bundle["zone_rate"]
    robots_per_zone = main_bundle["robots_per_zone"]
    hotspot_frac_zones = main_bundle.get("hotspot_frac_zones")
    base_lambda_per_zone_unscaled = main_bundle.get("base_lambda_per_zone_unscaled")
    hotspot_skew = main_bundle.get("hotspot_skew")
    if hotspot_skew is None or (isinstance(hotspot_skew, float) and math.isnan(hotspot_skew)):
        hotspot_skew = 1.0

    rows = []
    n_values = [16, 32, 64] if exclude_small_n else [8, 16, 32, 64]
    legacy_calibration = _load_legacy_fig4_multipliers(legacy_reproduce_dir)
    if legacy_calibration is not None:
        for n_zones in n_values:
            if n_zones not in legacy_calibration:
                raise RuntimeError(
                    f"Legacy fig4 summary missing load_multiplier for N={n_zones}"
                )
            entry = legacy_calibration[n_zones]
            rows.append(
                {
                    "N": n_zones,
                    "multiplier": entry.get("multiplier"),
                    "baseline_overload_q1000_mean": entry.get(
                        "baseline_overload_q1000_mean"
                    ),
                    "baseline_overload_q1000_ci_low": entry.get(
                        "baseline_overload_q1000_ci_low"
                    ),
                    "baseline_overload_q1000_ci_high": entry.get(
                        "baseline_overload_q1000_ci_high"
                    ),
                    "seeds": ",".join(str(seed) for seed in seeds),
                    "timestamp": int(time.time()),
                }
            )
        fig4_calib_path = os.path.join(figures_dir, "fig4_calibration.csv")
        _write_fig4_calibration(fig4_calib_path, rows)
        snapshot_root = os.path.join(output_dir, "config_snapshots")
        stamp_path = write_repro_stamp(
            figures_dir,
            {"mode": "fig4_calibrate", "calibration_path": fig4_calib_path},
            cmdline=" ".join(sys.argv),
            snapshot_root=snapshot_root,
            artifacts={"fig4_calibration": fig4_calib_path},
        )
        print(f"Wrote repro stamp: {stamp_path}")
        print(fig4_calib_path)
        return fig4_calib_path
    for n_zones in n_values:
        n_robots = int(round(robots_per_zone * n_zones))
        hotspot_zones_count = max(
            1, int(round((hotspot_frac_zones or (1.0 / n_zones)) * n_zones))
        )
        phi_actual = hotspot_zones_count / float(n_zones) if n_zones else 0.0
        hotspot_ratio_override = (
            (phi_actual * hotspot_skew)
            / (phi_actual * hotspot_skew + (1.0 - phi_actual))
            if phi_actual > 0.0 and hotspot_skew and hotspot_skew > 0.0
            else None
        )
        deterministic_hotspot_zones = _deterministic_hotspot_zones(
            n_zones, hotspot_zones_count, center_zone=0
        )
        calib_multiplier, calib_ratio, ci_low, ci_high = calibrate_load_multiplier_for_N(
            output_dir,
            seeds,
            state_rate_hz_base,
            standard_load,
            zone_rate,
            n_zones,
            n_robots,
            n_zones * zone_rate,
            target_low=0.55,
            target_high=0.65,
            m_low=0.05,
            m_high=0.5,
            max_iters=12,
            calibration_log=None,
            prefix=f"Fig4Calib_{profile_version}",
            arrival_mode=arrival_mode,
            hotspot_target_zones=deterministic_hotspot_zones if arrival_mode == "zone_poisson" else None,
            extra_robots_per_zone=None,
            hotspot_ratio_override=hotspot_ratio_override,
            force_rerun=force_rerun,
        )
        rows.append(
            {
                "N": n_zones,
                "multiplier": calib_multiplier,
                "baseline_overload_q1000_mean": calib_ratio,
                "baseline_overload_q1000_ci_low": ci_low,
                "baseline_overload_q1000_ci_high": ci_high,
                "seeds": ",".join(str(seed) for seed in seeds),
                "timestamp": int(time.time()),
            }
        )
        print(
            f"Fig4 calibration N={n_zones} multiplier={calib_multiplier:.3f} "
            f"overload_q1000={calib_ratio:.3f}"
        )

    fig4_calib_path = os.path.join(figures_dir, "fig4_calibration.csv")
    _write_fig4_calibration(fig4_calib_path, rows)
    snapshot_root = os.path.join(output_dir, "config_snapshots")
    stamp_path = write_repro_stamp(
        figures_dir,
        {"mode": "fig4_calibrate", "calibration_path": fig4_calib_path},
        cmdline=" ".join(sys.argv),
        snapshot_root=snapshot_root,
        artifacts={"fig4_calibration": fig4_calib_path},
    )
    print(f"Wrote repro stamp: {stamp_path}")
    print(fig4_calib_path)
    return fig4_calib_path


def run_fig4_sweep(
    output_dir,
    figures_dir,
    seeds,
    profile_version,
    calibration_path,
    enforce_n16_match=False,
    arrival_mode="robot_emit",
    deterministic_hotspots=False,
    hotspot_frac_zones_override=None,
    exclude_small_n=False,
    force_rerun=False,
    profile_overrides_path=None,
    force_match_main=True,
    legacy_single_hotspot=False,
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    def _load_summary_csv(path):
        if not os.path.isfile(path):
            return []
        with open(path, "r", newline="") as handle:
            return list(csv.DictReader(handle))

    main_summary_path = os.path.join(output_dir, "figure_z16_noc", "summary_main_and_ablations.csv")
    main_rows = _load_summary_csv(main_summary_path)
    main_tags = {}
    for row in main_rows:
        if _parse_int(row.get("N")) != 16:
            continue
        if row.get("scheme") == "S1" and row.get("profile") == "baseline":
            main_tags["Static-edge"] = row.get("tag")
        if row.get("scheme") == "S2" and row.get("profile") == "conservative":
            main_tags["Ours-Lite"] = row.get("tag")
        if row.get("scheme") == "S2" and row.get("profile") == "neutral":
            main_tags["Ours-Balanced"] = row.get("tag")
        if row.get("scheme") == "S2" and row.get("profile") == "aggressive":
            main_tags["Ours-Strong"] = row.get("tag")

    canonical_scaling_mode = "per_zone_constant"

    main_bundle = _load_main_run_config(output_dir, figures_dir)
    main_config = main_bundle["config"]
    state_rate_hz_base = main_bundle["state_rate_hz_base"]
    standard_load = main_bundle["standard_load"]
    calibration_multiplier = main_bundle["calibration_multiplier"]
    zone_rate = main_bundle["zone_rate"]
    robots_per_zone = main_bundle["robots_per_zone"]
    base_lambda_per_zone = main_bundle["base_lambda_per_zone"]
    base_lambda_per_zone_unscaled = main_bundle.get("base_lambda_per_zone_unscaled")
    base_mu_per_zone = main_bundle["base_mu_per_zone"]
    hotspot_frac_zones_main = main_bundle["hotspot_frac_zones"]
    hotspot_frac_zones = (
        float(hotspot_frac_zones_override)
        if hotspot_frac_zones_override is not None
        else hotspot_frac_zones_main
    )
    hotspot_delta_per_zone = main_bundle["hotspot_delta_per_zone"]
    hotspot_skew = main_bundle.get("hotspot_skew")
    deterministic_hotspots_main = main_bundle.get("deterministic_hotspots", False)
    arrival_mode_main = main_bundle.get("arrival_mode_main")
    hotspot_delta_per_zone_unscaled = main_bundle.get("hotspot_delta_per_zone_unscaled")
    arrival_cfg = main_bundle["arrival"]
    service_cfg = main_bundle["service"]
    hotspot_cfg = main_bundle["hotspot"]
    network_cfg = main_bundle["network"]
    if base_lambda_per_zone is None or base_mu_per_zone is None:
        raise RuntimeError("Missing base per-zone rates in main run_config.json.")
    if hotspot_frac_zones is None or hotspot_delta_per_zone is None:
        raise RuntimeError("Missing hotspot scaling parameters in main run_config.json.")
    if base_lambda_per_zone_unscaled is None:
        base_lambda_per_zone_unscaled = base_lambda_per_zone
    if hotspot_delta_per_zone_unscaled is None:
        hotspot_delta_per_zone_unscaled = hotspot_delta_per_zone
    if hotspot_skew is None or (isinstance(hotspot_skew, float) and math.isnan(hotspot_skew)):
        if hotspot_delta_per_zone is not None and base_lambda_per_zone is not None:
            phi_tmp = hotspot_frac_zones if hotspot_frac_zones else (1.0 / n_zones_main)
            lambda_cold = base_lambda_per_zone - (phi_tmp * hotspot_delta_per_zone)
            if lambda_cold > 0:
                lambda_hot = lambda_cold + hotspot_delta_per_zone
                hotspot_skew = lambda_hot / lambda_cold
            else:
                hotspot_skew = 1.0
    if calibration_multiplier is None:
        raise RuntimeError("Missing calibration_multiplier in main run_config.json.")

    calibration_map = _read_fig4_calibration(calibration_path)

    n_values = [16, 32, 64] if exclude_small_n else [8, 16, 32, 64]
    load_multiplier_default = calibration_multiplier
    n_zones_main = main_bundle["n_zones_main"]
    if hotspot_skew is None or (isinstance(hotspot_skew, float) and math.isnan(hotspot_skew)):
        hotspot_skew = 1.0

    summary_long = []
    seed_audit_rows = []
    effective_params = []
    fig4_runs = []
    debug_acc = {}
    selected_profiles = []
    profile_overrides_base = None
    if profile_overrides_path:
        profile_overrides_base = _load_profile_overrides(profile_overrides_path)
        if not profile_overrides_base:
            raise RuntimeError(
                f"Failed to load profile overrides: {profile_overrides_path}"
            )
        overrides_used_path = os.path.join(figures_dir, "s2_overrides_used.json")
        _write_profile_overrides(overrides_used_path, profile_overrides_base)

    def _hot_window(rows):
        t_hot_start = 30.0
        t_hot_end = 80.0
        if rows:
            hot_start = _parse_float(rows[0].get("hotspot_start_s"))
            hot_end = _parse_float(rows[0].get("hotspot_end_s"))
            if hot_start is not None and hot_end is not None:
                t_hot_start = hot_start
                t_hot_end = hot_end
        return t_hot_start, t_hot_end

    def _accumulate_debug(n_val, scheme_label, debug):
        key = (n_val, scheme_label)
        entry = debug_acc.setdefault(
            key,
            {
                "accepted_migrations": 0,
                "rejected_no_feasible_target": 0,
                "rejected_budget": 0,
                "rejected_safety": 0,
                "fallback_attempts": 0,
                "fallback_success": 0,
                "dmax_rejects": 0,
                "feasible_ratio_sum": 0.0,
                "feasible_ratio_count": 0,
                "hotspot_queue_p90_sum": 0.0,
                "hotspot_queue_p90_count": 0,
                "nonhot_queue_p10_sum": 0.0,
                "nonhot_queue_p10_count": 0,
            },
        )
        entry["accepted_migrations"] += debug.get("accepted_migrations", 0)
        entry["rejected_no_feasible_target"] += debug.get("rejected_no_feasible_target", 0)
        entry["rejected_budget"] += debug.get("rejected_budget", 0)
        entry["rejected_safety"] += debug.get("rejected_safety", 0)
        entry["fallback_attempts"] += debug.get("fallback_attempts", 0)
        entry["fallback_success"] += debug.get("fallback_success", 0)
        entry["dmax_rejects"] += debug.get("dmax_rejects", 0)
        entry["feasible_ratio_sum"] += debug.get("feasible_ratio_sum", 0.0)
        entry["feasible_ratio_count"] += debug.get("feasible_ratio_count", 0)
        hotspot_p90 = debug.get("hotspot_queue_p90")
        if hotspot_p90 is not None and not math.isnan(hotspot_p90):
            entry["hotspot_queue_p90_sum"] += hotspot_p90
            entry["hotspot_queue_p90_count"] += 1
        nonhot_p10 = debug.get("nonhot_queue_p10")
        if nonhot_p10 is not None and not math.isnan(nonhot_p10):
            entry["nonhot_queue_p10_sum"] += nonhot_p10
            entry["nonhot_queue_p10_count"] += 1

    for n_zones in n_values:
        if n_zones not in calibration_map:
            raise RuntimeError(f"Missing calibration multiplier for N={n_zones}")
        load_multiplier = calibration_map[n_zones]
        if (
            force_match_main
            and n_zones == n_zones_main
            and calibration_multiplier is not None
        ):
            if abs(load_multiplier - calibration_multiplier) > 1.0e-9:
                print(
                    "Fig4: overriding N=16 load_multiplier to match main "
                    f"(fig4={load_multiplier:.6f}, main={calibration_multiplier:.6f})"
                )
            load_multiplier = calibration_multiplier
        n_robots = int(round(robots_per_zone * n_zones))
        per_robot_rate = (
            (base_lambda_per_zone_unscaled * load_multiplier) / robots_per_zone
            if robots_per_zone > 0
            else 0.0
        )
        state_rate_hz = state_rate_hz_base * standard_load * load_multiplier
        total_arrival_rate = state_rate_hz * n_robots
        zone_rate_local = zone_rate
        central_override, central_scaled = _central_rate_override(
            state_rate_hz, n_robots, n_zones, zone_rate_local
        )
        per_zone_arrivals_mean = base_lambda_per_zone_unscaled * load_multiplier
        per_zone_service_mean = zone_rate_local
        use_main_outputs = (
            n_zones == n_zones_main
            and arrival_mode == arrival_mode_main
            and (not deterministic_hotspots or deterministic_hotspots_main)
            and (
                hotspot_frac_zones is None
                or hotspot_frac_zones_main is None
                or abs(hotspot_frac_zones - hotspot_frac_zones_main) <= 1.0e-9
            )
        )
        use_redistribution = deterministic_hotspots or not use_main_outputs
        hotspot_zones_count = max(1, int(round(hotspot_frac_zones * n_zones)))
        phi_used = hotspot_frac_zones if hotspot_frac_zones is not None else 0.0
        if legacy_single_hotspot:
            hotspot_zones_count = 1
            phi_used = 1.0 / n_zones if n_zones else 0.0
        phi_actual = hotspot_zones_count / float(n_zones) if n_zones else 0.0
        if deterministic_hotspots:
            lambda_hot_scaled, lambda_cold_scaled = _solve_hot_cold(
                per_zone_arrivals_mean, phi_actual, hotspot_skew
            )
            hotspot_ratio_override = (
                (phi_actual * hotspot_skew)
                / (phi_actual * hotspot_skew + (1.0 - phi_actual))
                if phi_actual > 0.0 and hotspot_skew is not None and hotspot_skew > 0.0
                else None
            )
        else:
            lambda_mean = per_zone_arrivals_mean
            lambda_hot_scaled = lambda_mean + hotspot_delta_per_zone_unscaled * load_multiplier
            total_lambda = lambda_mean * n_zones
            lambda_cold_scaled = (
                (total_lambda - hotspot_zones_count * lambda_hot_scaled)
                / (n_zones - hotspot_zones_count)
                if n_zones > hotspot_zones_count
                else lambda_mean
            )
            hotspot_ratio_override = None
        extra_robots_per_zone = None
        if arrival_mode == "robot_emit" and use_redistribution:
            delta_lambda = max(0.0, lambda_hot_scaled - lambda_cold_scaled)
            extra = delta_lambda / state_rate_hz if state_rate_hz > 0.0 else 0.0
            extra_robots_per_zone = max(0, int(round(extra)))
        rho = (
            per_zone_arrivals_mean / per_zone_service_mean
            if per_zone_service_mean
            else float("nan")
        )
        print(
            f"Fig4 N={n_zones} k_hot={hotspot_zones_count} "
            f"lambda_mean={per_zone_arrivals_mean:.3f} "
            f"lambda_hot={lambda_hot_scaled:.3f} lambda_cold={lambda_cold_scaled:.3f} "
            f"mu_zone={per_zone_service_mean:.3f} rho={rho:.3f}"
        )

        arrival_config = dict(arrival_cfg)
        arrival_config["state_rate_hz_base"] = state_rate_hz_base
        arrival_config["canonical_scaling_mode"] = canonical_scaling_mode
        arrival_config["base_lambda_per_zone"] = base_lambda_per_zone
        arrival_config["per_robot_rate"] = per_robot_rate
        arrival_config["load_multiplier"] = load_multiplier
        arrival_config["hotspot_frac_zones"] = phi_used
        arrival_config["hotspot_ratio_override"] = hotspot_ratio_override
        arrival_config["deterministic_hotspots"] = deterministic_hotspots
        service_config = dict(service_cfg)
        service_config["zone_service_rate_msgs_s"] = zone_rate_local
        service_config["central_service_rate_msgs_s"] = central_override
        service_config.setdefault("service_time_jitter", 0.50)
        network_config = dict(network_cfg)
        hotspot_config = dict(hotspot_cfg)
        hotspot_config["target_zone_random"] = not use_redistribution
        hotspot_config["hotspot_zones_count"] = hotspot_zones_count
        hotspot_config["hotspot_delta_per_zone"] = lambda_hot_scaled - lambda_cold_scaled
        hotspot_config["hotspot_frac_zones"] = phi_used
        hotspot_config["deterministic_hotspot_zones"] = (
            _deterministic_hotspot_zones(n_zones, hotspot_zones_count, center_zone=0)
            if use_redistribution
            else []
        )
        target_zones = None
        if use_redistribution:
            target_zones = list(hotspot_config.get("deterministic_hotspot_zones") or [])

        baseline_overloads = []
        for seed in seeds:
            target_zones = target_zones if use_redistribution else None
            base_tag = (
                main_tags.get("Static-edge")
                if use_main_outputs and main_tags.get("Static-edge")
                else f"Fig4N{n_zones}_static_{profile_version}_m{load_multiplier:.3f}".replace(".", "p")
            )
            tag = _seeded_tag(base_tag, seed)
            path = os.path.join(output_dir, f"window_kpis_{tag}_s1.csv")
            if (
                force_rerun
                or not _window_has_columns(path, required=WINDOW_REQUIRED_QUEUE)
                or not _window_has_audit_columns(path)
            ):
                run_scheme(
                    "S1",
                    seed,
                    output_dir,
                    state_rate_hz=state_rate_hz,
                    zone_service_rate_msgs_s=zone_rate_local,
                    write_csv=True,
                    n_zones_override=n_zones,
                    n_robots_override=n_robots,
                    central_service_rate_msgs_s_override=central_override,
                    dmax_ms=30.0,
                    tag=tag,
                    hotspot_target_zones=target_zones,
                    extra_robots_per_zone=extra_robots_per_zone
                    if (use_redistribution and arrival_mode == "robot_emit")
                    else None,
                    hotspot_ratio_override=hotspot_ratio_override,
                    arrival_mode=arrival_mode,
                )
            summary_long.append(
                _collect_seed_summary(
                    output_dir,
                    tag,
                    "S1",
                    "Static-edge",
                    "baseline",
                    seed,
                    n_zones,
                    load_multiplier,
                    summary_tag=base_tag,
                    budget_scale_factor=1.0,
                )
            )
            baseline_overloads.append(
                summary_long[-1].get("fixed_hotspot_overload_ratio_q1000")
            )
            rows = _load_window_rows(path, required=None)
            t_hot_start, t_hot_end = _hot_window(rows)
            debug = _collect_debug_from_rows(rows, t_hot_start, t_hot_end)
            _accumulate_debug(n_zones, "Static-edge", debug)
            audit = _collect_seed_audit(
                path, "S1", "baseline", n_zones, load_multiplier, seed
            )
            if audit:
                seed_audit_rows.append(audit)
            if use_redistribution:
                hotspot_delta_value = lambda_hot_scaled - lambda_cold_scaled
            else:
                ratio = _parse_float(rows[0].get("hotspot_ratio")) if rows else None
                total_lambda = per_zone_arrivals_mean * n_zones
                if ratio is None or n_zones <= 1:
                    hotspot_delta_value = float("nan")
                else:
                    lambda_hot_seed = total_lambda * ratio
                    lambda_cold_seed = (total_lambda - lambda_hot_seed) / (n_zones - 1)
                    hotspot_delta_value = lambda_hot_seed - lambda_cold_seed
            effective_params.append(
                {
                    "N": n_zones,
                    "scheme": "Static-edge",
                    "profile": "baseline",
                    "seed": seed,
                    "per_zone_arrivals_mean": per_zone_arrivals_mean,
                    "per_zone_service_mean": per_zone_service_mean,
                    "hotspot_zones_count": hotspot_zones_count,
                    "hotspot_delta_per_zone": hotspot_delta_value,
                    "S2_budget_effective": "",
                    "S2_max_migrate_effective": "",
                }
            )
            fig4_runs.append(
                {
                    "N": n_zones,
                    "n_robots": n_robots,
                    "seed": seed,
                    "scheme": "Static-edge",
                    "profile": "baseline",
                    "load_multiplier": load_multiplier,
                    "horizon_s": 120.0,
                    "warmup_s": 0.0,
                    "arrival": arrival_config,
                    "service": service_config,
                    "network": network_config,
                    "hotspot": hotspot_config,
                    "policy": {},
                    "budget_scale_factor": 1.0,
                    "derived": {
                        "k_hot": hotspot_zones_count,
                        "phi_used": phi_used,
                        "phi_actual": phi_actual,
                        "lambda_mean": per_zone_arrivals_mean,
                        "lambda_cold": lambda_cold_scaled,
                        "lambda_hot": lambda_hot_scaled,
                        "total_arrival_rate": total_arrival_rate,
                        "mu_zone": per_zone_service_mean,
                        "mu_c": central_override,
                        "tokens_per_s": 0.0,
                        "q_set": None,
                        "q_low": None,
                        "control_period_s": None,
                    },
                }
            )

        baseline_overload_mean = _mean_or_nan(baseline_overloads)
        if baseline_overload_mean is None or math.isnan(baseline_overload_mean):
            raise RuntimeError(f"Invalid baseline overload for N={n_zones}.")

        selection_info = {}
        if profile_overrides_base is None:
            grid_rows = _grid_search_s2_profiles(
                output_dir,
                seeds,
                state_rate_hz,
                zone_rate_local,
                n_zones,
                n_robots,
                central_override,
                load_multiplier,
                arrival_mode,
                deterministic_hotspots,
                target_zones,
                extra_robots_per_zone if (use_redistribution and arrival_mode == "robot_emit") else None,
                hotspot_ratio_override,
                profile_version,
            )
            grid_path = os.path.join(figures_dir, f"summary_s2_profile_grid_N{n_zones}.csv")
            _write_profile_grid(grid_path, grid_rows)
            profile_overrides_base, selection_info = _select_profiles_from_grid(
                grid_rows, baseline_overload=baseline_overload_mean
            )
            overrides_path = os.path.join(figures_dir, "s2_profile_overrides.json")
            _write_profile_overrides(overrides_path, profile_overrides_base, selection_info)
        else:
            overrides_path = os.path.join(figures_dir, "s2_profile_overrides.json")
            _write_profile_overrides(overrides_path, profile_overrides_base)

        profile_overrides = _s2_profile_overrides(base_profiles=profile_overrides_base)
        profile_map = [
            ("conservative", "Ours-Lite", profile_overrides["S2_conservative"]),
            ("neutral", "Ours-Balanced", profile_overrides["S2_neutral"]),
            ("aggressive", "Ours-Strong", profile_overrides["S2_aggressive"]),
        ]
        for profile_name, role in (
            ("conservative", "Lite"),
            ("neutral", "Balanced"),
            ("aggressive", "Strong"),
        ):
            row = selection_info.get(profile_name)
            if row is None:
                continue
            row = dict(row)
            row["N"] = n_zones
            row["scheme"] = "S2"
            row["profile"] = profile_name
            selected_profiles.append(
                {
                    "profile_name": f"Ours-{role}",
                    "selected_role": role,
                    "row": row,
                }
            )

        for profile_name, label, overrides in profile_map:
            for seed in seeds:
                profile_base_tag = f"Fig4N{n_zones}_{profile_name}_{profile_version}_m{load_multiplier:.3f}".replace(".", "p")
                profile_tag = _seeded_tag(profile_base_tag, seed)
                path = os.path.join(output_dir, f"window_kpis_{profile_tag}_s2.csv")
                if (
                    force_rerun
                    or not _window_has_columns(path, required=WINDOW_REQUIRED_QUEUE)
                    or not _window_has_audit_columns(path)
                ):
                    run_scheme(
                        "S2",
                        seed,
                        output_dir,
                        state_rate_hz=state_rate_hz,
                        zone_service_rate_msgs_s=zone_rate_local,
                        write_csv=True,
                        n_zones_override=n_zones,
                        n_robots_override=n_robots,
                        central_service_rate_msgs_s_override=central_override,
                        dmax_ms=30.0,
                        tag=profile_tag,
                        hotspot_target_zones=target_zones,
                        extra_robots_per_zone=extra_robots_per_zone
                        if (use_redistribution and arrival_mode == "robot_emit")
                        else None,
                        hotspot_ratio_override=hotspot_ratio_override,
                        arrival_mode=arrival_mode,
                        **overrides,
                    )
                summary_long.append(
                    _collect_seed_summary(
                        output_dir,
                        profile_tag,
                        "S2",
                        label,
                        profile_name,
                        seed,
                        n_zones,
                        load_multiplier,
                        summary_tag=profile_base_tag,
                        budget_scale_factor=1.0,
                    )
                )
                rows = _load_window_rows(path, required=None)
                t_hot_start, t_hot_end = _hot_window(rows)
                debug = _collect_debug_from_rows(rows, t_hot_start, t_hot_end)
                _accumulate_debug(n_zones, label, debug)
                audit = _collect_seed_audit(
                    path, "S2", profile_name, n_zones, load_multiplier, seed
                )
                if audit:
                    seed_audit_rows.append(audit)
                policy_cfg = _effective_policy_config(overrides, 30.0, total_arrival_rate)
                if use_redistribution:
                    hotspot_delta_value = lambda_hot_scaled - lambda_cold_scaled
                else:
                    ratio = _parse_float(rows[0].get("hotspot_ratio")) if rows else None
                    total_lambda = per_zone_arrivals_mean * n_zones
                    if ratio is None or n_zones <= 1:
                        hotspot_delta_value = float("nan")
                    else:
                        lambda_hot_seed = total_lambda * ratio
                        lambda_cold_seed = (total_lambda - lambda_hot_seed) / (n_zones - 1)
                        hotspot_delta_value = lambda_hot_seed - lambda_cold_seed
                effective_params.append(
                    {
                        "N": n_zones,
                        "scheme": label,
                        "profile": profile_name,
                        "seed": seed,
                        "per_zone_arrivals_mean": per_zone_arrivals_mean,
                        "per_zone_service_mean": per_zone_service_mean,
                        "hotspot_zones_count": hotspot_zones_count,
                        "hotspot_delta_per_zone": hotspot_delta_value,
                        "S2_budget_effective": policy_cfg.get("tokens_per_s"),
                        "S2_max_migrate_effective": policy_cfg.get("move_k"),
                    }
                )
                fig4_runs.append(
                    {
                        "N": n_zones,
                        "n_robots": n_robots,
                        "seed": seed,
                        "scheme": label,
                        "profile": profile_name,
                        "load_multiplier": load_multiplier,
                        "horizon_s": 120.0,
                        "warmup_s": 0.0,
                        "arrival": arrival_config,
                        "service": service_config,
                        "network": network_config,
                        "hotspot": hotspot_config,
                        "policy": policy_cfg,
                        "budget_scale_factor": 1.0,
                        "derived": {
                            "k_hot": hotspot_zones_count,
                            "phi_used": phi_used,
                            "phi_actual": phi_actual,
                            "lambda_mean": per_zone_arrivals_mean,
                            "lambda_cold": lambda_cold_scaled,
                            "lambda_hot": lambda_hot_scaled,
                            "total_arrival_rate": total_arrival_rate,
                            "mu_zone": per_zone_service_mean,
                            "mu_c": central_override,
                            "tokens_per_s": policy_cfg.get("tokens_per_s"),
                            "q_set": policy_cfg.get("q_set"),
                            "q_low": policy_cfg.get("q_low"),
                            "control_period_s": policy_cfg.get("policy_period_s"),
                        },
                    }
                )


    summary_agg = _aggregate_rows(
        summary_long, group_keys=["tag", "scheme", "profile", "N", "load_multiplier"]
    )

    summary_long_path = os.path.join(figures_dir, "summary_fig4_scaledload_long.csv")
    _write_summary_long(summary_long_path, summary_long)
    if selected_profiles:
        selected_profiles_path = os.path.join(figures_dir, "summary_s2_selected.csv")
        _write_selected_profiles(selected_profiles_path, selected_profiles)
    seed_audit_path = os.path.join(figures_dir, "seed_audit_fig4.csv")
    _write_seed_audit(seed_audit_path, seed_audit_rows)
    _assert_seed_audit_variation(seed_audit_rows, "fig4")
    if arrival_mode == "zone_poisson":
        _assert_generated_consistency(summary_long, "fig4")

    for row in summary_long:
        ratio = row.get("fixed_hotspot_overload_ratio_q1000")
        ratio_label = "<1e-3" if ratio is not None and ratio < 1.0e-3 else _format_float(ratio, 6)
        print(
            f"N={row.get('N')} scheme={row.get('scheme')} seed={row.get('seed')} "
            f"load_multiplier={row.get('load_multiplier')} budget_scale_factor={row.get('budget_scale_factor')} "
            f"overload_ratio_q1000={ratio_label} max_queue={row.get('max_queue')} "
            f"p99_queue={row.get('p99_queue')} hotspot_window_start_end={row.get('hotspot_window_start_end')} "
            f"overload_window_kind={row.get('overload_window_kind')}"
        )

    n64_rows = [row for row in summary_long if row.get("N") == 64]
    n64_static = [
        row
        for row in n64_rows
        if row.get("scheme") == "S1" and row.get("profile") == "baseline"
    ]
    _print_seed_metrics(n64_static, "Fig4 N=64 Static-edge")
    if not _seed_variation_by_group(n64_rows):
        print("WARNING: Seeds appear ineffective for Fig4 N=64.")
        raise SystemExit(1)

    summary_fig4_path = os.path.join(figures_dir, "summary_fig4_scaledload.csv")
    _write_summary_agg(summary_fig4_path, summary_agg)

    effective_params_path = os.path.join(figures_dir, "effective_params.csv")
    with open(effective_params_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "N",
                "scheme",
                "profile",
                "seed",
                "per_zone_arrivals_mean",
                "per_zone_service_mean",
                "hotspot_zones_count",
                "hotspot_delta_per_zone",
                "S2_budget_effective",
                "S2_max_migrate_effective",
            ]
        )
        for row in effective_params:
            writer.writerow(
                [
                    row.get("N"),
                    row.get("scheme"),
                    row.get("profile"),
                    row.get("seed"),
                    row.get("per_zone_arrivals_mean"),
                    row.get("per_zone_service_mean"),
                    row.get("hotspot_zones_count"),
                    row.get("hotspot_delta_per_zone"),
                    row.get("S2_budget_effective"),
                    row.get("S2_max_migrate_effective"),
                ]
            )

    def _assert_effective_params(params):
        expected = {}
        for row in params:
            key = (row.get("N"), row.get("seed"))
            payload = (
                row.get("per_zone_arrivals_mean"),
                row.get("per_zone_service_mean"),
                row.get("hotspot_zones_count"),
                row.get("hotspot_delta_per_zone"),
            )
            if key not in expected:
                expected[key] = payload
                continue
            for left, right, label in zip(
                expected[key],
                payload,
                (
                    "per_zone_arrivals_mean",
                    "per_zone_service_mean",
                    "hotspot_zones_count",
                    "hotspot_delta_per_zone",
                ),
            ):
                if left is None or right is None:
                    raise RuntimeError(f"Missing {label} in effective_params for N={key}")
                if isinstance(left, float) or isinstance(right, float):
                    if abs(float(left) - float(right)) > 1.0e-6:
                        raise RuntimeError(
                            f"Load scale mismatch for N={key} on {label}: {left} vs {right}"
                        )
                elif left != right:
                    raise RuntimeError(
                        f"Load scale mismatch for N={key} on {label}: {left} vs {right}"
                    )

    _assert_effective_params(effective_params)

    policy_profiles_path = os.path.join(figures_dir, "policy_profiles_used.csv")
    profiles_for_table = {
        "Ours-Lite": profile_overrides["S2_conservative"],
        "Ours-Balanced": profile_overrides["S2_neutral"],
        "Ours-Strong": profile_overrides["S2_aggressive"],
    }
    _write_policy_profiles(policy_profiles_path, profiles_for_table, budget_scale_factor=1.0)

    debug_rows = []
    for n_val in n_values:
        rows_n = [row for row in summary_agg if row.get("N") == n_val]
        baseline = next(
            (row for row in rows_n if row.get("scheme") == "S1" and row.get("profile") == "baseline"),
            None,
        )
        if not baseline:
            continue
        load_multiplier = load_multiplier_default
        per_zone_rho = (
            (base_lambda_per_zone_unscaled * load_multiplier) / zone_rate
            if zone_rate
            else float("nan")
        )
        debug_rows.append(
            {
                "N": n_val,
                "per_zone_rho": per_zone_rho,
                "overload_ratio_q1000": baseline.get("overload_ratio_q1000"),
            }
        )
        print(
            f"Fig4 debug N={n_val} per_zone_rho={per_zone_rho:.3f} "
            f"overload_q1000={baseline.get('overload_ratio_q1000')}"
        )
    debug_path = os.path.join(figures_dir, "fig4_static_edge_debug.csv")
    with open(debug_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["N", "per_zone_rho", "overload_ratio_q1000"])
        for row in debug_rows:
            writer.writerow(
                [row.get("N"), row.get("per_zone_rho"), row.get("overload_ratio_q1000")]
            )

    run_config_path = os.path.join(figures_dir, "run_config.json")
    if os.path.isfile(run_config_path):
        backup_path = os.path.join(figures_dir, "run_config_main.json")
        if not os.path.isfile(backup_path):
            shutil.copyfile(run_config_path, backup_path)
    _write_run_config(
        run_config_path,
        {
            "canonical_scaling_mode": canonical_scaling_mode,
            "standard_load": standard_load,
            "arrival_mode": arrival_mode,
            "deterministic_hotspots": deterministic_hotspots,
            "deterministic_hotspot_zones": _deterministic_hotspot_zones(
                n_zones_main,
                max(1, int(round((hotspot_frac_zones or 0.0) * n_zones_main))),
                center_zone=0,
            )
            if deterministic_hotspots
            else [],
            "base_lambda_per_zone": base_lambda_per_zone,
            "base_lambda_per_zone_unscaled": base_lambda_per_zone_unscaled,
            "base_mu_per_zone": base_mu_per_zone,
            "hotspot_frac_zones": hotspot_frac_zones,
            "hotspot_delta_per_zone": hotspot_delta_per_zone,
            "hotspot_delta_per_zone_unscaled": hotspot_delta_per_zone_unscaled,
            "seeds": seeds,
            "calibration_path": calibration_path,
            "overload_window_kind": "hotspot_window",
            "overload_threshold_q1000": 1000,
            "runs": fig4_runs,
        },
    )
    snapshot_root = os.path.join(output_dir, "config_snapshots")
    stamp_path = write_repro_stamp(
        figures_dir,
        {"mode": "fig4_run", "run_config_path": run_config_path},
        cmdline=" ".join(sys.argv),
        snapshot_root=snapshot_root,
        artifacts={"run_config": run_config_path, "fig4_calibration": calibration_path},
    )
    print(f"Wrote repro stamp: {stamp_path}")

    main_summary_path = os.path.join(
        output_dir, "figure_z16_noc", "summary_main_and_ablations.csv"
    )
    main_rows = _load_summary_csv(main_summary_path)
    if not main_rows:
        raise RuntimeError("Missing main summary for N=16 consistency check.")
    scheme_profiles = [("S1", "baseline", "Static-edge")]
    for scheme_key, profile, label in scheme_profiles:
        main_row = next(
            (
                row
                for row in main_rows
                if _parse_int(row.get("N")) == 16
                and row.get("scheme") == scheme_key
                and row.get("profile") == profile
            ),
            None,
        )
        fig4_row = next(
            (
                row
                for row in summary_agg
                if row.get("N") == 16
                and row.get("scheme") == scheme_key
                and row.get("profile") == profile
            ),
            None,
        )
        if not main_row or not fig4_row:
            raise RuntimeError(f"Missing N=16 rows for {label} in Fig4 or main summary.")
        for metric in (
            "fixed_hotspot_overload_ratio_q500",
            "fixed_hotspot_overload_ratio_q1000",
            "fixed_hotspot_overload_ratio_q1500",
        ):
            main_val = _parse_float(main_row.get(metric))
            fig4_val = _parse_float(fig4_row.get(metric))
            if (
                main_val is None
                or fig4_val is None
                or abs(main_val - fig4_val) > 1.0e-3
            ):
                diffs = _diff_run_configs(main_config, {"runs": fig4_runs}, 16, label)
                print(f"Config diff ({label}, N=16):")
                for key, left, right in diffs:
                    print(f"  {key}: main={left} fig4={right}")
                raise RuntimeError(
                    f"Fig4 N=16 {label} {metric} mismatch: "
                    f"main={main_val} fig4={fig4_val}"
                )

    def _slug(value):
        return value.lower().replace(" ", "_").replace("-", "_")

    for (n_val, scheme_label), entry in debug_acc.items():
        ratio_mean = (
            entry["feasible_ratio_sum"] / entry["feasible_ratio_count"]
            if entry["feasible_ratio_count"]
            else float("nan")
        )
        hotspot_p90 = (
            entry["hotspot_queue_p90_sum"] / entry["hotspot_queue_p90_count"]
            if entry["hotspot_queue_p90_count"]
            else float("nan")
        )
        nonhot_p10 = (
            entry["nonhot_queue_p10_sum"] / entry["nonhot_queue_p10_count"]
            if entry["nonhot_queue_p10_count"]
            else float("nan")
        )
        payload = {
            "N": n_val,
            "scheme": scheme_label,
            "accepted_migrations_total": entry["accepted_migrations"],
            "rejected_no_feasible_target_total": entry["rejected_no_feasible_target"],
            "rejected_budget_total": entry["rejected_budget"],
            "rejected_safety_total": entry["rejected_safety"],
            "fallback_attempts_total": entry["fallback_attempts"],
            "fallback_success_total": entry["fallback_success"],
            "dmax_rejects_total": entry["dmax_rejects"],
            "feasible_target_ratio_mean": ratio_mean,
            "feasible_target_ratio_count": entry["feasible_ratio_count"],
            "hotspot_queue_p90": hotspot_p90,
            "nonhot_queue_p10": nonhot_p10,
        }
        debug_path = os.path.join(
            figures_dir, f"debug_stats_N{n_val}_{_slug(scheme_label)}.json"
        )
        with open(debug_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    print(summary_fig4_path)
    for row in summary_agg:
        print(
            f"N={row.get('N')} load_multiplier={row.get('load_multiplier'):.3f} "
            f"{row.get('scheme')} hotspot_p95_s={row.get('hotspot_p95_mean_s')} "
            f"overload_q1000={row.get('overload_ratio_q1000')}"
        )

    return summary_fig4_path


def run_n16_strong_ablation(
    output_dir,
    figures_dir,
    seeds,
    profile_version,
    main_multiplier,
    fig4_multiplier,
):
    state_rate_hz_base = 10
    standard_load = 2.0
    zone_rate = 200
    base_n_zones = 8
    base_n_robots = 100
    robots_per_zone = base_n_robots / float(base_n_zones)
    n_zones = 16
    n_robots = int(round(robots_per_zone * n_zones))
    central_rate_override = n_zones * zone_rate

    profile_overrides = _s2_profile_overrides()
    profile_base = profile_overrides["S2_aggressive"]

    combos = [
        ("main", main_multiplier, False),
        ("main", main_multiplier, True),
        ("fig4", fig4_multiplier, False),
        ("fig4", fig4_multiplier, True),
    ]
    summary_rows = []
    for source, multiplier, budget_on in combos:
        if multiplier is None:
            continue
        scale = _fig4_budget_scale(n_zones) if budget_on else 1.0
        overrides = _apply_budget_scale(profile_base, scale)
        state_rate_hz = state_rate_hz_base * standard_load * multiplier
        for seed in seeds:
            base_tag = (
                f"AblN16Strong_{source}_{'on' if budget_on else 'off'}_"
                f"{profile_version}_m{multiplier:.3f}"
            ).replace(".", "p")
            tag = _seeded_tag(base_tag, seed)
            path = os.path.join(output_dir, f"window_kpis_{tag}_s2.csv")
            if not _window_has_columns(path, required=WINDOW_REQUIRED_QUEUE) or not _window_has_audit_columns(path):
                run_scheme(
                    "S2",
                    seed,
                    output_dir,
                    state_rate_hz=state_rate_hz,
                    zone_service_rate_msgs_s=zone_rate,
                    write_csv=True,
                    n_zones_override=n_zones,
                    n_robots_override=n_robots,
                    central_service_rate_msgs_s_override=central_rate_override,
                    dmax_ms=30.0,
                    tag=tag,
                    **overrides,
                )
            summary_rows.append(
                _collect_seed_summary(
                    output_dir,
                    tag,
                    "S2",
                    "Ours-Strong",
                    "aggressive",
                    seed,
                    n_zones,
                    multiplier,
                    summary_tag=base_tag,
                    budget_scale_factor=scale,
                )
            )

    aggregated = _aggregate_rows(
        summary_rows, group_keys=["tag", "scheme", "profile", "N", "load_multiplier"]
    )
    ablation_path = os.path.join(figures_dir, "ablation_n16_strong.csv")
    with open(ablation_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "load_multiplier_source",
                "budget_scaling",
                "load_multiplier",
                "hotspot_p95_mean_s",
                "overload_ratio_q1000",
            ]
        )
        for row in aggregated:
            tag = row.get("tag", "")
            parts = tag.split("_")
            source = parts[1] if len(parts) > 1 else ""
            budget = parts[2] if len(parts) > 2 else ""
            writer.writerow(
                [
                    source,
                    budget,
                    _format_float(row.get("load_multiplier"), 4),
                    _format_float(row.get("hotspot_p95_mean_s"), 6),
                    _format_float(row.get("overload_ratio_q1000"), 6),
                ]
            )
    print(ablation_path)
    return ablation_path


def run_fig4_sensitivity(
    output_dir, figures_dir, seeds, calibration_path, profile_version, arrival_mode="robot_emit"
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    state_rate_hz_base = 10
    standard_load = 2.0
    zone_rate = 200
    base_n_zones = 8
    base_n_robots = 100
    robots_per_zone_fallback = base_n_robots / float(base_n_zones)

    main_config_path = os.path.join(figures_dir, "run_config_main.json")
    if not os.path.isfile(main_config_path):
        main_config_path = os.path.join(figures_dir, "run_config.json")
    main_config = _load_run_config(main_config_path)
    if not main_config:
        raise RuntimeError("Missing run_config.json for Fig4 sensitivity.")
    base_lambda_per_zone = _parse_float(main_config.get("base_lambda_per_zone"))
    base_mu_per_zone = _parse_float(main_config.get("base_mu_per_zone"))
    hotspot_frac_zones = _parse_float(main_config.get("hotspot_frac_zones"))
    hotspot_delta_per_zone = _parse_float(main_config.get("hotspot_delta_per_zone"))
    if base_lambda_per_zone is None or base_mu_per_zone is None:
        raise RuntimeError("Missing base per-zone rates in run_config.json.")
    if hotspot_frac_zones is None or hotspot_delta_per_zone is None:
        raise RuntimeError("Missing hotspot scaling parameters in run_config.json.")

    robots_per_zone = robots_per_zone_fallback
    runs = main_config.get("runs") or []
    if runs:
        n_robots_main = _parse_float(runs[0].get("n_robots"))
        n_zones_main = _parse_float(runs[0].get("N"))
        if n_robots_main and n_zones_main:
            robots_per_zone = n_robots_main / n_zones_main

    base_multiplier = 1.0
    multipliers = [0.9 * base_multiplier, base_multiplier, 1.1 * base_multiplier]
    n_zones = 64
    n_robots = int(round(robots_per_zone * n_zones))
    zone_rate_local = base_mu_per_zone
    central_rate_override = n_zones * zone_rate_local
    per_robot_rate = base_lambda_per_zone / robots_per_zone if robots_per_zone > 0 else 0.0
    hotspot_zones_count = max(1, int(round(hotspot_frac_zones * n_zones)))
    target_zones = list(range(hotspot_zones_count))
    extra_robots_per_zone = (
        int(round(hotspot_delta_per_zone / per_robot_rate))
        if per_robot_rate > 0.0
        else 0
    )

    profile_overrides = _s2_profile_overrides()
    budget_scale = _fig4_budget_scale(n_zones)
    profile_map = [
        ("S1", "Static-edge", {}, "baseline"),
        (
            "S2",
            "Ours-Lite",
            _apply_budget_scale(profile_overrides["S2_conservative"], budget_scale),
            "conservative",
        ),
        (
            "S2",
            "Ours-Balanced",
            _apply_budget_scale(profile_overrides["S2_neutral"], budget_scale),
            "neutral",
        ),
        (
            "S2",
            "Ours-Strong",
            _apply_budget_scale(profile_overrides["S2_aggressive"], budget_scale),
            "aggressive",
        ),
    ]

    summary_rows = []
    seed_audit_rows = []
    for multiplier in multipliers:
        state_rate_hz = per_robot_rate * multiplier
        for scheme_key, label, overrides, profile in profile_map:
            for seed in seeds:
                base_tag = f"SensN64_{label.replace(' ', '')}_{profile_version}_m{multiplier:.3f}_b{budget_scale:.2f}".replace(
                    ".", "p"
                )
                tag = _seeded_tag(base_tag, seed)
                path = os.path.join(output_dir, f"window_kpis_{tag}_{scheme_key.lower()}.csv")
                if not _window_has_columns(path, required=WINDOW_REQUIRED_QUEUE) or not _window_has_audit_columns(path):
                    run_scheme(
                        scheme_key,
                        seed,
                        output_dir,
                        state_rate_hz=state_rate_hz,
                        zone_service_rate_msgs_s=zone_rate_local,
                        write_csv=True,
                        n_zones_override=n_zones,
                        n_robots_override=n_robots,
                        central_service_rate_msgs_s_override=central_rate_override,
                        dmax_ms=30.0,
                        tag=tag,
                        hotspot_target_zones=target_zones,
                        extra_robots_per_zone=extra_robots_per_zone,
                        arrival_mode=arrival_mode,
                        **overrides,
                    )
                summary_rows.append(
                    _collect_seed_summary(
                        output_dir,
                        tag,
                        scheme_key,
                        label,
                        profile,
                        seed,
                        n_zones,
                        multiplier,
                        summary_tag=base_tag,
                        budget_scale_factor=budget_scale,
                    )
                )
                audit = _collect_seed_audit(
                    path, scheme_key, profile, n_zones, multiplier, seed
                )
                if audit:
                    seed_audit_rows.append(audit)

    aggregated = _aggregate_rows(
        summary_rows, group_keys=["scheme", "profile", "N", "load_multiplier"]
    )
    sensitivity_path = os.path.join(figures_dir, "fig4_sensitivity_N64.csv")
    with open(sensitivity_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "N",
                "multiplier_scale",
                "multiplier",
                "scheme",
                "hotspot_p95_mean_s",
                "hotspot_p95_ci_low",
                "hotspot_p95_ci_high",
                "overload_q1000_mean",
                "overload_q1000_ci_low",
                "overload_q1000_ci_high",
                "reconfig_cost_mean",
                "reconfig_action_count_mean",
            ]
        )
        for row in aggregated:
            scale = row.get("load_multiplier") / base_multiplier if base_multiplier else float("nan")
            writer.writerow(
                [
                    row.get("N"),
                    _format_float(scale, 3),
                    _format_float(row.get("load_multiplier"), 4),
                    row.get("scheme"),
                    _format_float(row.get("hotspot_p95_mean_s"), 6),
                    _format_float(row.get("hotspot_p95_s_ci_low"), 6),
                    _format_float(row.get("hotspot_p95_s_ci_high"), 6),
                    _format_float(row.get("overload_ratio_q1000"), 6),
                    _format_float(row.get("overload_q1000_ci_low"), 6),
                    _format_float(row.get("overload_q1000_ci_high"), 6),
                    _format_float(row.get("migrated_weight_total"), 6),
                    _format_float(row.get("reconfig_action_count"), 3),
                ]
            )
    print(sensitivity_path)
    seed_audit_path = os.path.join(figures_dir, "seed_audit_fig4_sensitivity.csv")
    _write_seed_audit(seed_audit_path, seed_audit_rows)
    _assert_seed_audit_variation(seed_audit_rows, "fig4_sensitivity")
    return sensitivity_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["main", "fig4_calibrate", "fig4_run", "fig4_sensitivity"],
        default="main",
    )
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--enforce_n16_match", action="store_true")
    parser.add_argument("--arrival_mode", default="robot_emit")
    parser.add_argument("--deterministic_hotspots", action="store_true")
    parser.add_argument("--hotspot_frac_zones", default=None)
    parser.add_argument("--exclude_small_n", action="store_true")
    parser.add_argument("--auto_profile_search", action="store_true")
    parser.add_argument("--force_rerun", action="store_true")
    parser.add_argument("--legacy_p28_reproduce_dir", default=None)
    parser.add_argument("--legacy_snapshot_dir", default=None)
    parser.add_argument("--s2_overrides_json", default=None)
    args = parser.parse_args()

    seeds = _parse_seeds(args.seeds)
    if not seeds:
        seeds = [0, 1, 2]

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    output_dir = normalize_output_root(base_dir, args.output_root, prefix="figure")
    figures_dir_main = os.path.join(output_dir, "figure_z16_noc")
    figures_dir_fig4 = os.path.join(output_dir, "figures")

    profile_version = "p98"
    legacy_dir = args.legacy_snapshot_dir or args.legacy_p28_reproduce_dir
    legacy_mode = bool(legacy_dir)
    if legacy_mode:
        if args.arrival_mode != "robot_emit":
            print("Legacy snapshot mode forcing arrival_mode=robot_emit.")
        args.arrival_mode = "robot_emit"
        args.deterministic_hotspots = False
        args.auto_profile_search = False

    if args.mode == "main":
        run_main_pipeline(
            output_dir,
            figures_dir_main,
            seeds,
            profile_version,
            arrival_mode=args.arrival_mode,
            deterministic_hotspots=args.deterministic_hotspots,
            hotspot_frac_zones=args.hotspot_frac_zones,
            auto_profile_search=args.auto_profile_search,
            legacy_reproduce_dir=legacy_dir,
            profile_overrides_path=args.s2_overrides_json,
        )
        if legacy_mode:
            repro_dir = _resolve_legacy_repro_dir(output_dir)
            fig4_map = _load_legacy_fig4_multipliers(legacy_dir) or {}
            main_mult = _load_legacy_main_multiplier(legacy_dir).get("multiplier")
            _write_legacy_used(
                repro_dir,
                legacy_dir,
                main_mult,
                fig4_map,
                args.arrival_mode,
                1,
                args.deterministic_hotspots,
            )
            _legacy_copy_outputs(output_dir, repro_dir)
        return
    if args.mode == "fig4_calibrate":
        run_fig4_calibrate(
            output_dir,
            figures_dir_fig4,
            seeds,
            profile_version,
            arrival_mode=args.arrival_mode,
            exclude_small_n=args.exclude_small_n,
            force_rerun=args.force_rerun,
            legacy_reproduce_dir=legacy_dir,
        )
        if legacy_mode:
            repro_dir = _resolve_legacy_repro_dir(output_dir)
            fig4_map = _load_legacy_fig4_multipliers(legacy_dir) or {}
            main_mult = _load_legacy_main_multiplier(legacy_dir).get("multiplier")
            _write_legacy_used(
                repro_dir,
                legacy_dir,
                main_mult,
                fig4_map,
                args.arrival_mode,
                1,
                args.deterministic_hotspots,
            )
            _legacy_copy_outputs(output_dir, repro_dir)
        return
    if args.mode == "fig4_run":
        calibration_path = os.path.join(figures_dir_fig4, "fig4_calibration.csv")
        run_fig4_sweep(
            output_dir,
            figures_dir_fig4,
            seeds,
            profile_version,
            calibration_path,
            enforce_n16_match=args.enforce_n16_match,
            arrival_mode=args.arrival_mode,
            deterministic_hotspots=args.deterministic_hotspots,
            hotspot_frac_zones_override=args.hotspot_frac_zones,
            exclude_small_n=args.exclude_small_n,
            force_rerun=args.force_rerun,
            profile_overrides_path=args.s2_overrides_json,
            force_match_main=not legacy_mode,
            legacy_single_hotspot=legacy_mode,
        )
        if legacy_mode:
            repro_dir = _resolve_legacy_repro_dir(output_dir)
            fig4_map = _load_legacy_fig4_multipliers(legacy_dir) or {}
            main_mult = _load_legacy_main_multiplier(legacy_dir).get("multiplier")
            _write_legacy_used(
                repro_dir,
                legacy_dir,
                main_mult,
                fig4_map,
                args.arrival_mode,
                1,
                args.deterministic_hotspots,
            )
            _legacy_copy_outputs(output_dir, repro_dir)
        return
    if args.mode == "fig4_sensitivity":
        calibration_path = os.path.join(figures_dir_fig4, "fig4_calibration.csv")
        run_fig4_sensitivity(
            output_dir,
            figures_dir_fig4,
            seeds,
            calibration_path,
            profile_version,
            arrival_mode=args.arrival_mode,
        )
        return


if __name__ == "__main__":
    main()
