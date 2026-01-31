from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class ArrivalRecord:
    req_id: int
    arrival_time: float
    home_zone: int


def _poisson_times(rate: float, start: float, end: float, rng) -> List[float]:
    if rate <= 0:
        return []
    t = start
    arrivals = []
    while True:
        t += rng.expovariate(rate)
        if t >= end:
            break
        arrivals.append(t)
    return arrivals


def generate_arrivals(
    rng,
    N: int,
    sim_duration_s: int,
    hotspot_window: Tuple[int, int],
    hotspot_zones: List[int],
    lambda_mean: float,
    lambda_hot: float,
    lambda_cold: float,
) -> Tuple[List[ArrivalRecord], Dict[str, int]]:
    hot_start, hot_end = hotspot_window
    arrivals: List[ArrivalRecord] = []
    req_id = 0
    hotspot_set = set(hotspot_zones)

    for zone in range(N):
        zone_arrivals = []
        zone_arrivals.extend(_poisson_times(lambda_mean, 0.0, hot_start, rng))
        zone_rate = lambda_hot if zone in hotspot_set else lambda_cold
        zone_arrivals.extend(_poisson_times(zone_rate, hot_start, hot_end, rng))
        zone_arrivals.extend(_poisson_times(lambda_mean, hot_end, sim_duration_s, rng))
        for t in zone_arrivals:
            arrivals.append(ArrivalRecord(req_id=req_id, arrival_time=t, home_zone=zone))
            req_id += 1

    arrivals.sort(key=lambda r: r.arrival_time)

    global_generated = len(arrivals)
    fixed_hotspot_generated = sum(
        1
        for r in arrivals
        if (r.home_zone in hotspot_set and hot_start <= r.arrival_time < hot_end)
    )

    counts = {
        "global_generated": global_generated,
        "fixed_hotspot_generated": fixed_hotspot_generated,
    }
    return arrivals, counts
