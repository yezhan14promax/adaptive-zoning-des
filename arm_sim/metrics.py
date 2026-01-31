from __future__ import annotations

import math
from typing import Dict, Iterable, List, Tuple


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return float("nan")
    values_sorted = sorted(values)
    if len(values_sorted) == 1:
        return values_sorted[0]
    k = (len(values_sorted) - 1) * pct
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return values_sorted[int(k)]
    d0 = values_sorted[int(f)] * (c - k)
    d1 = values_sorted[int(c)] * (k - f)
    return d0 + d1


def compute_overload_ratio(
    events_by_zone: Dict[int, List[Tuple[float, int]]],
    hotspot_zones: List[int],
    hotspot_window: Tuple[int, int],
    sim_duration_s: int,
    threshold: int,
) -> float:
    hot_start, hot_end = hotspot_window
    hot_start = max(0, int(hot_start))
    hot_end = min(sim_duration_s, int(hot_end))
    if hot_end <= hot_start:
        return 0.0
    horizon = hot_end - hot_start
    if not hotspot_zones:
        return 0.0

    total_overloaded = 0
    for zone in hotspot_zones:
        events = sorted(events_by_zone.get(zone, []), key=lambda e: e[0])
        q = 0
        idx = 0
        overloaded_seconds = 0
        for t in range(hot_start, hot_end):
            sample_time = t + 0.5
            while idx < len(events) and events[idx][0] <= sample_time:
                q += events[idx][1]
                idx += 1
            if q > threshold:
                overloaded_seconds += 1
        total_overloaded += overloaded_seconds

    return total_overloaded / float(horizon * len(hotspot_zones))


def compute_overload_ratios(
    events_by_zone: Dict[int, List[Tuple[float, int]]],
    hotspot_zones: List[int],
    hotspot_window: Tuple[int, int],
    sim_duration_s: int,
    thresholds: List[int],
) -> Dict[str, float]:
    results: Dict[str, float] = {}
    for threshold in thresholds:
        key = f"overload_q{int(threshold)}"
        results[key] = compute_overload_ratio(
            events_by_zone=events_by_zone,
            hotspot_zones=hotspot_zones,
            hotspot_window=hotspot_window,
            sim_duration_s=sim_duration_s,
            threshold=int(threshold),
        )
    return results


def compute_latency_metrics(latencies: Iterable[float]) -> Dict[str, float]:
    lat_list = list(latencies)
    if not lat_list:
        return {"p50": float("nan"), "p95": float("nan"), "mean": float("nan")}
    return {
        "p50": percentile(lat_list, 0.50),
        "p95": percentile(lat_list, 0.95),
        "mean": sum(lat_list) / len(lat_list),
    }


def compute_reconfig_cost_total(requests: Iterable[Dict]) -> float:
    total = 0.0
    for req in requests:
        if req.get("offloaded"):
            total += 1.0
    return total
