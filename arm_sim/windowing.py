from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass
class WindowMetrics:
    window_index: int
    generated: int
    completed: int
    completion_ratio: float
    mean_latency: float


def compute_window_metrics(requests: Iterable[Dict], window_s: int, sim_duration_s: int) -> List[WindowMetrics]:
    buckets: Dict[int, Dict[str, float]] = {}
    for req in requests:
        window_index = int(req["arrival_time"] // window_s)
        bucket = buckets.setdefault(window_index, {"generated": 0, "completed": 0, "latency_sum": 0.0, "latency_count": 0})
        bucket["generated"] += 1
        if req["completion_time"] <= sim_duration_s:
            bucket["completed"] += 1
            bucket["latency_sum"] += req["latency"]
            bucket["latency_count"] += 1

    results: List[WindowMetrics] = []
    for window_index, bucket in sorted(buckets.items()):
        generated = int(bucket["generated"])
        completed = int(bucket["completed"])
        completion_ratio = completed / generated if generated > 0 else 0.0
        mean_latency = bucket["latency_sum"] / bucket["latency_count"] if bucket["latency_count"] > 0 else 0.0
        results.append(
            WindowMetrics(
                window_index=window_index,
                generated=generated,
                completed=completed,
                completion_ratio=completion_ratio,
                mean_latency=mean_latency,
            )
        )
    return results
