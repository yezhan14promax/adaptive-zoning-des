from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .metrics import compute_latency_metrics, compute_overload_ratios, compute_reconfig_cost_total
from .windowing import compute_window_metrics


@dataclass
class SimResult:
    scheme: str
    profile_id: str
    cost: float
    overload_q500: float
    overload_q1000: float
    overload_q1500: float
    p95_latency_hotspot: float
    p50_latency_hotspot: float
    mean_latency_hotspot: float
    completion_ratio_max: float
    reconfig_cost_total: float
    global_generated: int
    fixed_hotspot_generated: int
    requests: List[Dict]


@dataclass
class SimDebug:
    window_metrics: List
    events_by_zone: Dict[int, List[Tuple[float, int]]]


def simulate(
    arrivals,
    policy,
    rng_streams,
    N: int,
    hotspot_zones: List[int],
    hotspot_window: Tuple[int, int],
    sim_duration_s: int,
    mu_zone: float,
    mu_cloud: float,
    edge_cost: float,
    cloud_cost: float,
    queue_overload_threshold: int,
    overload_thresholds: List[int],
    counts: Dict[str, int],
    window_s: int,
) -> Tuple[SimResult, SimDebug]:
    cloud_zone = N
    available_time = [0.0 for _ in range(N + 1)]
    events_by_zone: Dict[int, List[Tuple[float, int]]] = {z: [] for z in hotspot_zones}

    requests: List[Dict] = []
    hotspot_set = set(hotspot_zones)

    for arrival in arrivals:
        admit_zone, offloaded = policy.route(arrival, rng_streams.policy_rng, hotspot_zones)
        service_rate = mu_cloud if offloaded else mu_zone
        service_time = rng_streams.service_rng.expovariate(service_rate)
        start_time = arrival.arrival_time
        completion_time = max(start_time, available_time[admit_zone]) + service_time
        available_time[admit_zone] = completion_time
        latency = completion_time - start_time

        req = {
            "arrival_time": arrival.arrival_time,
            "completion_time": completion_time,
            "home_zone": arrival.home_zone,
            "admit_zone": admit_zone,
            "offloaded": offloaded,
            "latency": latency,
        }
        requests.append(req)

        if admit_zone in hotspot_set:
            events_by_zone[admit_zone].append((arrival.arrival_time, 1))
            events_by_zone[admit_zone].append((completion_time, -1))

    hotspot_latencies = [
        r["latency"]
        for r in requests
        if (r["home_zone"] in hotspot_set and hotspot_window[0] <= r["arrival_time"] < hotspot_window[1])
    ]
    latency_stats = compute_latency_metrics(hotspot_latencies)

    thresholds = overload_thresholds or [queue_overload_threshold]
    overloads = compute_overload_ratios(
        events_by_zone=events_by_zone,
        hotspot_zones=hotspot_zones,
        hotspot_window=hotspot_window,
        sim_duration_s=sim_duration_s,
        thresholds=thresholds,
    )
    overload_q500 = overloads.get("overload_q500", overloads.get(f"overload_q{queue_overload_threshold}", 0.0))
    overload_q1000 = overloads.get("overload_q1000", overloads.get(f"overload_q{queue_overload_threshold}", 0.0))
    overload_q1500 = overloads.get("overload_q1500", overloads.get(f"overload_q{queue_overload_threshold}", 0.0))

    total_cost = 0.0
    for r in requests:
        total_cost += cloud_cost if r["offloaded"] else edge_cost
    cost = total_cost / len(requests) if requests else float("nan")
    reconfig_cost_total = compute_reconfig_cost_total(requests)

    window_metrics = compute_window_metrics(requests, window_s=window_s, sim_duration_s=sim_duration_s)
    completion_ratio_max = max((wm.completion_ratio for wm in window_metrics), default=0.0)
    if completion_ratio_max > 1.0:
        raise ValueError(f"completion_ratio_max>1 ({completion_ratio_max:.4f})")

    result = SimResult(
        scheme=policy.name,
        profile_id=getattr(policy, "profile", None).profile_id if getattr(policy, "profile", None) else "static",
        cost=cost,
        overload_q500=overload_q500,
        overload_q1000=overload_q1000,
        overload_q1500=overload_q1500,
        p95_latency_hotspot=latency_stats["p95"],
        p50_latency_hotspot=latency_stats["p50"],
        mean_latency_hotspot=latency_stats["mean"],
        completion_ratio_max=completion_ratio_max,
        reconfig_cost_total=reconfig_cost_total,
        global_generated=counts["global_generated"],
        fixed_hotspot_generated=counts["fixed_hotspot_generated"],
        requests=requests,
    )

    debug = SimDebug(window_metrics=window_metrics, events_by_zone=events_by_zone)
    return result, debug
