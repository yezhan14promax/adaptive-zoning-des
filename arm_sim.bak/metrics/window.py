import csv
import math


def compute_overload_ratio(
    queue_ts_by_zone,
    threshold,
    sample_dt=1.0,
    warmup_s=0.0,
    scope="hotspot_only",
):
    if not queue_ts_by_zone:
        return []
    if isinstance(queue_ts_by_zone, dict):
        zone_ids = sorted(queue_ts_by_zone.keys())
        series = [queue_ts_by_zone[zone_id] for zone_id in zone_ids]
        num_steps = max((len(values) for values in series), default=0)
        queue_ts_by_zone = []
        for index in range(num_steps):
            queues = []
            for values in series:
                queues.append(values[index] if index < len(values) else 0.0)
            queue_ts_by_zone.append(queues)
    if queue_ts_by_zone and not isinstance(queue_ts_by_zone[0], (list, tuple)):
        queue_ts_by_zone = [queue_ts_by_zone]
    ratios = []
    for index, queues in enumerate(queue_ts_by_zone):
        t_end = (index + 1) * sample_dt
        if warmup_s and t_end <= warmup_s:
            ratios.append(None)
            continue
        if not queues:
            ratios.append(0.0)
            continue
        count = sum(1 for value in queues if value > threshold)
        ratios.append(count / len(queues))
    return ratios


def compute_any_overload_ratio(
    queue_ts_by_zone,
    threshold,
    sample_dt=1.0,
    warmup_s=0.0,
):
    if not queue_ts_by_zone:
        return []
    if queue_ts_by_zone and not isinstance(queue_ts_by_zone[0], (list, tuple)):
        queue_ts_by_zone = [queue_ts_by_zone]
    ratios = []
    for index, queues in enumerate(queue_ts_by_zone):
        t_end = (index + 1) * sample_dt
        if warmup_s and t_end <= warmup_s:
            ratios.append(None)
            continue
        if not queues:
            ratios.append(0.0)
            continue
        ratios.append(1.0 if any(value > threshold for value in queues) else 0.0)
    return ratios


def compute_window_kpis(
    recorder,
    duration_s,
    window_s,
    overload_ratios,
    violation_ms,
    overload_ratios_q500=None,
    overload_ratios_q1000=None,
    overload_ratios_q1500=None,
    hotspot_overload_ratios_q1000=None,
    queue_max_per_window=None,
    queue_min_nonhot_per_window=None,
    hotspot_zone_ids=None,
    warmup_s=0.0,
    overload_thresholds=(500, 1000, 1500),
):
    if window_s <= 0:
        return []
    num_windows = int(math.ceil(duration_s / window_s))
    windows = [
        {
            "t_start": i * window_s,
            "t_end": min((i + 1) * window_s, duration_s),
            "latencies": [],
            "generated": 0,
            "edge_completed": 0,
            "central_completed": 0,
            "completed": 0,
        }
        for i in range(num_windows)
    ]

    hotspot_zone_ids = (
        {int(z) for z in hotspot_zone_ids} if hotspot_zone_ids else None
    )
    for record in recorder.records.values():
        emit_time = record.get("emit_time")
        emit_zone = record.get("emit_zone_id")
        home_zone = record.get("home_zone_id")
        if home_zone is None:
            home_zone = emit_zone
        if hotspot_zone_ids is not None:
            if home_zone is None or int(home_zone) not in hotspot_zone_ids:
                continue
        arrive_zone = record.get("arrive_zone")
        cohort_time = arrive_zone if arrive_zone is not None else emit_time
        if cohort_time is None:
            continue
        final_done_time = record.get("final_done_time")
        final_done_stage = record.get("final_done_stage")

        index = int(cohort_time // window_s)
        if index >= num_windows:
            index = num_windows - 1
        windows[index]["generated"] += 1
        if final_done_time is not None:
            windows[index]["completed"] += 1
            if final_done_stage == "edge":
                windows[index]["edge_completed"] += 1
            elif final_done_stage == "central":
                windows[index]["central_completed"] += 1
            windows[index]["latencies"].append((final_done_time - cohort_time) * 1000.0)

    rows = []
    for i, window in enumerate(windows):
        latencies = sorted(window["latencies"])
        central_completed = window["central_completed"]
        edge_completed = window["edge_completed"]
        completed = window["completed"]
        total_completed = edge_completed + central_completed
        if completed > 0:
            mean_ms = sum(latencies) / completed
            p95_index = int(math.ceil(0.95 * completed) - 1)
            p95_ms = latencies[p95_index]
            violation_count = sum(1 for value in latencies if value > violation_ms)
            violation_rate = violation_count / completed
        else:
            mean_ms = float("nan")
            p95_ms = float("nan")
            violation_rate = float("nan")

        overload_ratio = 0.0
        if overload_ratios and i < len(overload_ratios) and overload_ratios[i] is not None:
            overload_ratio = overload_ratios[i]
        overload_ratio_q500 = None
        if (
            overload_ratios_q500
            and i < len(overload_ratios_q500)
            and overload_ratios_q500[i] is not None
        ):
            overload_ratio_q500 = overload_ratios_q500[i]
        overload_ratio_q1000 = None
        if (
            overload_ratios_q1000
            and i < len(overload_ratios_q1000)
            and overload_ratios_q1000[i] is not None
        ):
            overload_ratio_q1000 = overload_ratios_q1000[i]
        overload_ratio_q1500 = None
        if (
            overload_ratios_q1500
            and i < len(overload_ratios_q1500)
            and overload_ratios_q1500[i] is not None
        ):
            overload_ratio_q1500 = overload_ratios_q1500[i]
        hotspot_overload_ratio_q1000 = None
        if (
            hotspot_overload_ratios_q1000
            and i < len(hotspot_overload_ratios_q1000)
            and hotspot_overload_ratios_q1000[i] is not None
        ):
            hotspot_overload_ratio_q1000 = hotspot_overload_ratios_q1000[i]
        queue_max = None
        if queue_max_per_window and i < len(queue_max_per_window):
            queue_max = queue_max_per_window[i]
        queue_min_nonhot = None
        if queue_min_nonhot_per_window and i < len(queue_min_nonhot_per_window):
            queue_min_nonhot = queue_min_nonhot_per_window[i]

        rows.append(
            {
                "t_start": window["t_start"],
                "t_end": window["t_end"],
                "completed": completed,
                "edge_completed": edge_completed,
                "central_completed": central_completed,
                "total_completed": total_completed,
                "generated": window["generated"],
                "mean_ms": mean_ms,
                "p95_ms": p95_ms,
                "overload_ratio": overload_ratio,
                "overload_ratio_q500": overload_ratio_q500,
                "overload_ratio_q1000": overload_ratio_q1000,
                "overload_ratio_q1500": overload_ratio_q1500,
                "hotspot_overload_ratio_q1000": hotspot_overload_ratio_q1000,
                "violation_rate": violation_rate,
                "queue_max": queue_max,
                "queue_min_nonhot": queue_min_nonhot,
            }
        )

    return rows


def write_window_kpis_csv(path, scheme, seed, rows, extra_fields=None):
    extra_fields = extra_fields or {}
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "scheme",
                "seed",
                "t_start",
                "t_end",
                "completed",
                "total_completed",
                "edge_completed",
                "central_completed",
                "generated",
                "global_completed",
                "global_completed_total",
                "global_completion_ratio",
                "global_total_completed",
                "global_generated",
                "hotspot_completed_fixed",
                "fixed_hotspot_completed_total",
                "fixed_hotspot_completion_ratio",
                "fixed_hotspot_completed",
                "hotspot_generated_fixed",
                "mean_ms",
                "p95_ms",
                "overload_ratio",
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
                    "queue_max",
                    "queue_min_nonhot",
                    "hotspot_start_s",
                    "hotspot_end_s",
                    "hotspot_zone_id",
                    "hotspot_zone_ids",
                    "hotspot_ratio",
                    "hotspot_multiplier",
                    "arrivals_first_60s",
                    "violation_rate",
                    "queue_mean_hot",
                    "queue_p95_hot",
                    "migrated_weight_window",
                    "reconfig_actions_window",
                    "policy_reassign_ops",
                    "reconfig_action_count",
                    "migrated_robots_total",
                    "migrated_weight_total",
                    "feasible_ratio_mean",
                    "feasible_ratio_count",
                "rejected_no_feasible_target",
                "rejected_budget",
                "rejected_safety",
                "fallback_attempts",
                "fallback_success",
                "dmax_rejects",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    scheme,
                    seed,
                    f"{row['t_start']:.1f}",
                    f"{row['t_end']:.1f}",
                    row["completed"],
                    row.get("total_completed", ""),
                    row.get("edge_completed", ""),
                    row.get("central_completed", ""),
                    row["generated"],
                    row.get("global_completed", ""),
                    row.get("global_completed_total", ""),
                    row.get("global_completion_ratio", ""),
                    row.get("global_total_completed", ""),
                    row.get("global_generated", ""),
                    row.get("hotspot_completed_fixed", ""),
                    row.get("fixed_hotspot_completed_total", ""),
                    row.get("fixed_hotspot_completion_ratio", ""),
                    row.get("fixed_hotspot_completed", ""),
                    row.get("hotspot_generated_fixed", ""),
                    f"{row['mean_ms']:.3f}",
                    f"{row['p95_ms']:.3f}",
                    f"{row['overload_ratio']:.3f}",
                    ""
                    if row.get("overload_ratio_q500") is None
                    else f"{row['overload_ratio_q500']:.3f}",
                    ""
                    if row.get("overload_ratio_q1000") is None
                    else f"{row['overload_ratio_q1000']:.3f}",
                    ""
                    if row.get("overload_ratio_q1500") is None
                    else f"{row['overload_ratio_q1500']:.3f}",
                    ""
                    if row.get("fixed_hotspot_overload_ratio_q500") is None
                    else f"{row['fixed_hotspot_overload_ratio_q500']:.3f}",
                    ""
                    if row.get("fixed_hotspot_overload_ratio_q1000") is None
                    else f"{row['fixed_hotspot_overload_ratio_q1000']:.3f}",
                    ""
                    if row.get("fixed_hotspot_overload_ratio_q1500") is None
                    else f"{row['fixed_hotspot_overload_ratio_q1500']:.3f}",
                    ""
                    if row.get("hotspot_overload_ratio_q1000") is None
                    else f"{row['hotspot_overload_ratio_q1000']:.3f}",
                    ""
                    if row.get("serving_hotspot_overload_ratio_q500") is None
                    else f"{row['serving_hotspot_overload_ratio_q500']:.3f}",
                    ""
                    if row.get("serving_hotspot_overload_ratio_q1000") is None
                    else f"{row['serving_hotspot_overload_ratio_q1000']:.3f}",
                    ""
                    if row.get("serving_hotspot_overload_ratio_q1500") is None
                    else f"{row['serving_hotspot_overload_ratio_q1500']:.3f}",
                    ""
                    if row.get("global_topk_overload_ratio_q500") is None
                    else f"{row['global_topk_overload_ratio_q500']:.3f}",
                    ""
                    if row.get("global_topk_overload_ratio_q1000") is None
                    else f"{row['global_topk_overload_ratio_q1000']:.3f}",
                    ""
                    if row.get("global_topk_overload_ratio_q1500") is None
                    else f"{row['global_topk_overload_ratio_q1500']:.3f}",
                    ""
                    if row.get("queue_max") is None
                    else f"{row['queue_max']:.3f}",
                    ""
                    if row.get("queue_min_nonhot") is None
                    else f"{row['queue_min_nonhot']:.3f}",
                    extra_fields.get("hotspot_start_s", ""),
                    extra_fields.get("hotspot_end_s", ""),
                    extra_fields.get("hotspot_zone_id", ""),
                    extra_fields.get("hotspot_zone_ids", ""),
                    extra_fields.get("hotspot_ratio", ""),
                    extra_fields.get("hotspot_multiplier", ""),
                    extra_fields.get("arrivals_first_60s", ""),
                    f"{row['violation_rate']:.3f}",
                    ""
                    if row.get("queue_mean_hot") is None
                    else f"{row['queue_mean_hot']:.3f}",
                    ""
                    if row.get("queue_p95_hot") is None
                    else f"{row['queue_p95_hot']:.3f}",
                    ""
                    if row.get("migrated_weight_window") is None
                    else f"{row['migrated_weight_window']:.3f}",
                    ""
                    if row.get("reconfig_actions_window") is None
                    else f"{row['reconfig_actions_window']:.3f}",
                    extra_fields.get("policy_reassign_ops", 0),
                    extra_fields.get("reconfig_action_count", 0),
                    extra_fields.get("migrated_robots_total", 0),
                    f"{extra_fields.get('migrated_weight_total', 0.0):.3f}",
                    ""
                    if extra_fields.get("feasible_ratio_mean") is None
                    else f"{extra_fields.get('feasible_ratio_mean', 0.0):.6f}",
                    extra_fields.get("feasible_ratio_count", 0),
                    extra_fields.get("rejected_no_feasible_target", 0),
                    extra_fields.get("rejected_budget", 0),
                    extra_fields.get("rejected_safety", 0),
                    extra_fields.get("fallback_attempts", 0),
                    extra_fields.get("fallback_success", 0),
                    extra_fields.get("dmax_rejects", 0),
                ]
            )
