from __future__ import annotations

import csv
import hashlib
import os
from typing import Dict, List, Tuple

from .arrival import ArrivalRecord, generate_arrivals
from .config_schema import ExperimentConfig, compute_hotspot_params
from .policy_s2 import PolicyS2, S2Profile
from .policy_static import StaticEdgePolicy
from .rng import make_rng_streams
from .simulator import simulate


def _write_debug_arrival_trace(path: str, arrivals: List[ArrivalRecord]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["req_id", "arrival_time", "home_zone"])
        writer.writeheader()
        for r in arrivals:
            writer.writerow({"req_id": r.req_id, "arrival_time": r.arrival_time, "home_zone": r.home_zone})


def _write_window_kpis(path: str, window_metrics) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["window_index", "generated", "completed", "completion_ratio", "mean_latency"]
        )
        writer.writeheader()
        for wm in window_metrics:
            writer.writerow(
                {
                    "window_index": wm.window_index,
                    "generated": wm.generated,
                    "completed": wm.completed,
                    "completion_ratio": wm.completion_ratio,
                    "mean_latency": wm.mean_latency,
                }
            )


def _write_debug_windows(path: str, events_by_zone: Dict[int, List[Tuple[float, int]]], hotspot_window: Tuple[int, int]) -> None:
    hot_start, hot_end = hotspot_window
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["zone_id", "t", "q"])
        writer.writeheader()
        for zone, events in events_by_zone.items():
            events_sorted = sorted(events, key=lambda e: e[0])
            q = 0
            idx = 0
            for t in range(int(hot_start), int(hot_end)):
                sample_time = t + 0.5
                while idx < len(events_sorted) and events_sorted[idx][0] <= sample_time:
                    q += events_sorted[idx][1]
                    idx += 1
                writer.writerow({"zone_id": zone, "t": t, "q": q})


def _arrival_trace_hash(arrivals: List[ArrivalRecord]) -> str:
    digest = hashlib.sha256()
    for r in arrivals:
        digest.update(f"{r.arrival_time:.6f},{r.home_zone}\n".encode("utf-8"))
    return digest.hexdigest()


def _load_selected_profiles(selected_path: str) -> Dict[str, S2Profile]:
    with open(selected_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        selected = {}
        for row in reader:
            selected[row["scheme"]] = S2Profile(
                profile_id=row["profile_id"],
                budget_ratio=float(row["budget_ratio"]),
                weight_scale=float(row["weight_scale"]),
                cooldown_s=int(float(row["cooldown_s"])),
            )
        return selected


def _seed_selector(seed: int):
    import random

    rng = random.Random(seed + 71)

    def selector(N: int, k_hot: int):
        return sorted(rng.sample(list(range(N)), k_hot))

    return selector


def run_main(
    config: ExperimentConfig,
    selected_profiles_path: str,
    emit_debug: bool = False,
    debug_dir: str | None = None,
) -> Tuple[List[Dict], Dict[int, Dict], Dict[str, Dict[int, str]]]:
    selected_profiles = _load_selected_profiles(selected_profiles_path)
    schemes = [
        ("Static-edge", StaticEdgePolicy(config.N)),
        ("Lite", PolicyS2(config.N, selected_profiles["Lite"], cloud_zone=config.N)),
        ("Balanced", PolicyS2(config.N, selected_profiles["Balanced"], cloud_zone=config.N)),
        ("Strong", PolicyS2(config.N, selected_profiles["Strong"], cloud_zone=config.N)),
    ]

    summary_rows: List[Dict] = []
    derived_hotspot: Dict[int, Dict] = {}
    arrival_hashes: Dict[str, Dict[int, str]] = {scheme: {} for scheme, _ in schemes}
    metrics_by_scheme: Dict[str, Dict[str, List[float]]] = {
        scheme_name: {
            "cost": [],
            "overload_q500": [],
            "overload_q1000": [],
            "overload_q1500": [],
            "p95_latency_hotspot": [],
            "p50_latency_hotspot": [],
            "mean_latency_hotspot": [],
            "completion_ratio_max": [],
            "global_generated": [],
            "fixed_hotspot_generated": [],
            "reconfig_cost_total": [],
        }
        for scheme_name, _ in schemes
    }

    for seed in config.seeds:
        derived = compute_hotspot_params(
            N=config.N,
            phi=config.hotspot.phi,
            hotspot_skew=config.hotspot.hotspot_skew,
            lambda_mean=config.arrival.lambda_mean,
            hotspot_zone_selector=_seed_selector(seed),
        )
        derived_hotspot[seed] = {
            "N": derived.N,
            "k_hot": derived.k_hot,
            "phi_actual": derived.phi_actual,
            "lambda_hot": derived.lambda_hot,
            "lambda_cold": derived.lambda_cold,
            "hotspot_zones": derived.hotspot_zones,
        }

        arrival_rng = make_rng_streams(seed).arrival_rng
        arrivals, counts = generate_arrivals(
            arrival_rng,
            N=config.N,
            sim_duration_s=config.arrival.sim_duration_s,
            hotspot_window=config.hotspot.hotspot_window,
            hotspot_zones=derived.hotspot_zones,
            lambda_mean=config.arrival.lambda_mean,
            lambda_hot=derived.lambda_hot,
            lambda_cold=derived.lambda_cold,
        )
        arrival_hash = _arrival_trace_hash(arrivals)

        per_scheme_counts: Dict[str, Dict[str, int]] = {}

        for scheme_name, policy in schemes:
            rng_streams = make_rng_streams(seed)
            result, debug = simulate(
                arrivals=arrivals,
                policy=policy,
                rng_streams=rng_streams,
                N=config.N,
                hotspot_zones=derived.hotspot_zones,
                hotspot_window=config.hotspot.hotspot_window,
                sim_duration_s=config.arrival.sim_duration_s,
                mu_zone=config.service.mu_zone,
                mu_cloud=config.service.mu_cloud,
                edge_cost=config.service.edge_cost,
                cloud_cost=config.service.cloud_cost,
                queue_overload_threshold=config.windowing.queue_overload_threshold,
                overload_thresholds=config.windowing.queue_overload_thresholds,
                counts=counts,
                window_s=config.windowing.window_s,
            )
            if emit_debug and debug_dir and seed == config.seeds[0]:
                if scheme_name == "Static-edge":
                    _write_debug_arrival_trace(os.path.join(debug_dir, "arrival_trace.csv"), arrivals)
                _write_window_kpis(
                    os.path.join(debug_dir, f"window_kpis_{scheme_name}.csv"), debug.window_metrics
                )
                _write_debug_windows(
                    os.path.join(debug_dir, f"debug_windows_{scheme_name}.csv"),
                    debug.events_by_zone,
                    config.hotspot.hotspot_window,
                )
            metrics_by_scheme[scheme_name]["cost"].append(result.cost)
            metrics_by_scheme[scheme_name]["overload_q500"].append(result.overload_q500)
            metrics_by_scheme[scheme_name]["overload_q1000"].append(result.overload_q1000)
            metrics_by_scheme[scheme_name]["overload_q1500"].append(result.overload_q1500)
            metrics_by_scheme[scheme_name]["p95_latency_hotspot"].append(result.p95_latency_hotspot)
            metrics_by_scheme[scheme_name]["p50_latency_hotspot"].append(result.p50_latency_hotspot)
            metrics_by_scheme[scheme_name]["mean_latency_hotspot"].append(result.mean_latency_hotspot)
            metrics_by_scheme[scheme_name]["completion_ratio_max"].append(result.completion_ratio_max)
            metrics_by_scheme[scheme_name]["global_generated"].append(result.global_generated)
            metrics_by_scheme[scheme_name]["fixed_hotspot_generated"].append(result.fixed_hotspot_generated)
            metrics_by_scheme[scheme_name]["reconfig_cost_total"].append(result.reconfig_cost_total)
            arrival_hashes[scheme_name][seed] = arrival_hash
            per_scheme_counts[scheme_name] = {
                "global_generated": result.global_generated,
                "fixed_hotspot_generated": result.fixed_hotspot_generated,
            }

        global_vals = {vals["global_generated"] for vals in per_scheme_counts.values()}
        hotspot_vals = {vals["fixed_hotspot_generated"] for vals in per_scheme_counts.values()}
        if len(global_vals) > 1 or len(hotspot_vals) > 1:
            raise ValueError(f"Fairness violation seed={seed}: {per_scheme_counts}")

    for scheme_name, policy in schemes:
        metrics_accum = metrics_by_scheme[scheme_name]
        summary_rows.append(
            {
                "scheme": scheme_name,
                "profile_id": getattr(policy, "profile", None).profile_id if getattr(policy, "profile", None) else "static",
                "cost": sum(metrics_accum["cost"]) / len(metrics_accum["cost"]),
                "overload_q500": sum(metrics_accum["overload_q500"]) / len(metrics_accum["overload_q500"]),
                "overload_q1000": sum(metrics_accum["overload_q1000"]) / len(metrics_accum["overload_q1000"]),
                "overload_q1500": sum(metrics_accum["overload_q1500"]) / len(metrics_accum["overload_q1500"]),
                "p95_latency_hotspot": sum(metrics_accum["p95_latency_hotspot"]) / len(metrics_accum["p95_latency_hotspot"]),
                "p50_latency_hotspot": sum(metrics_accum["p50_latency_hotspot"]) / len(metrics_accum["p50_latency_hotspot"]),
                "mean_latency_hotspot": sum(metrics_accum["mean_latency_hotspot"]) / len(metrics_accum["mean_latency_hotspot"]),
                "completion_ratio_max": max(metrics_accum["completion_ratio_max"]),
                "reconfig_cost_total": sum(metrics_accum["reconfig_cost_total"]) / len(metrics_accum["reconfig_cost_total"]),
                "global_generated": sum(metrics_accum["global_generated"]) / len(metrics_accum["global_generated"]),
                "fixed_hotspot_generated": sum(metrics_accum["fixed_hotspot_generated"]) / len(metrics_accum["fixed_hotspot_generated"]),
                "overload_method": "metrics.compute_overload_ratios",
            }
        )

    return summary_rows, derived_hotspot, arrival_hashes


def run_fig4(
    config: ExperimentConfig,
    selected_profiles_path: str,
) -> Tuple[List[Dict], Dict[int, Dict], Dict[str, Dict[int, Dict[int, str]]]]:
    selected_profiles = _load_selected_profiles(selected_profiles_path)
    schemes = [
        ("Static-edge", StaticEdgePolicy),
        ("Lite", lambda n: PolicyS2(n, selected_profiles["Lite"], cloud_zone=n)),
        ("Balanced", lambda n: PolicyS2(n, selected_profiles["Balanced"], cloud_zone=n)),
        ("Strong", lambda n: PolicyS2(n, selected_profiles["Strong"], cloud_zone=n)),
    ]

    summary_rows: List[Dict] = []
    derived_hotspot: Dict[int, Dict] = {}
    arrival_hashes: Dict[str, Dict[int, Dict[int, str]]] = {scheme: {} for scheme, _ in schemes}

    for N in config.scale_Ns:
        for seed in config.seeds:
            derived = compute_hotspot_params(
                N=N,
                phi=config.hotspot.phi,
                hotspot_skew=config.hotspot.hotspot_skew,
                lambda_mean=config.arrival.lambda_mean,
                hotspot_zone_selector=_seed_selector(seed),
            )
            derived_hotspot.setdefault(N, {})[seed] = {
                "N": derived.N,
                "k_hot": derived.k_hot,
                "phi_actual": derived.phi_actual,
                "lambda_hot": derived.lambda_hot,
                "lambda_cold": derived.lambda_cold,
                "hotspot_zones": derived.hotspot_zones,
            }

        arrivals_by_seed: Dict[int, Tuple[List[ArrivalRecord], Dict[str, int], str]] = {}
        for seed in config.seeds:
            derived = derived_hotspot[N][seed]
            arrival_rng = make_rng_streams(seed).arrival_rng
            arrivals, counts = generate_arrivals(
                arrival_rng,
                N=N,
                sim_duration_s=config.arrival.sim_duration_s,
                hotspot_window=config.hotspot.hotspot_window,
                hotspot_zones=derived["hotspot_zones"],
                lambda_mean=config.arrival.lambda_mean,
                lambda_hot=derived["lambda_hot"],
                lambda_cold=derived["lambda_cold"],
            )
            arrivals_by_seed[seed] = (arrivals, counts, _arrival_trace_hash(arrivals))

        counts_by_seed: Dict[int, Dict[str, Dict[str, int]]] = {seed: {} for seed in config.seeds}

        for scheme_name, policy_factory in schemes:
            metrics_accum = {
                "cost": [],
                "overload_q500": [],
                "overload_q1000": [],
                "overload_q1500": [],
                "p95_latency_hotspot": [],
                "global_generated": [],
                "fixed_hotspot_generated": [],
            }
            for seed in config.seeds:
                derived = derived_hotspot[N][seed]
                arrivals, counts, arrival_hash = arrivals_by_seed[seed]
                rng_streams = make_rng_streams(seed)
                policy = policy_factory(N) if callable(policy_factory) else policy_factory(N)
                result, _ = simulate(
                    arrivals=arrivals,
                    policy=policy,
                    rng_streams=rng_streams,
                    N=N,
                    hotspot_zones=derived["hotspot_zones"],
                    hotspot_window=config.hotspot.hotspot_window,
                    sim_duration_s=config.arrival.sim_duration_s,
                    mu_zone=config.service.mu_zone,
                    mu_cloud=config.service.mu_cloud,
                    edge_cost=config.service.edge_cost,
                    cloud_cost=config.service.cloud_cost,
                    queue_overload_threshold=config.windowing.queue_overload_threshold,
                    overload_thresholds=config.windowing.queue_overload_thresholds,
                    counts=counts,
                    window_s=config.windowing.window_s,
                )
                metrics_accum["cost"].append(result.cost)
                metrics_accum["overload_q500"].append(result.overload_q500)
                metrics_accum["overload_q1000"].append(result.overload_q1000)
                metrics_accum["overload_q1500"].append(result.overload_q1500)
                metrics_accum["p95_latency_hotspot"].append(result.p95_latency_hotspot)
                metrics_accum["global_generated"].append(result.global_generated)
                metrics_accum["fixed_hotspot_generated"].append(result.fixed_hotspot_generated)
                arrival_hashes[scheme_name].setdefault(N, {})[seed] = arrival_hash
                counts_by_seed[seed][scheme_name] = {
                    "global_generated": result.global_generated,
                    "fixed_hotspot_generated": result.fixed_hotspot_generated,
                }

            summary_rows.append(
                {
                    "N": N,
                    "scheme": scheme_name,
                    "cost": sum(metrics_accum["cost"]) / len(metrics_accum["cost"]),
                    "overload_q500": sum(metrics_accum["overload_q500"]) / len(metrics_accum["overload_q500"]),
                    "overload_q1000": sum(metrics_accum["overload_q1000"]) / len(metrics_accum["overload_q1000"]),
                    "overload_q1500": sum(metrics_accum["overload_q1500"]) / len(metrics_accum["overload_q1500"]),
                    "p95_latency_hotspot": sum(metrics_accum["p95_latency_hotspot"]) / len(metrics_accum["p95_latency_hotspot"]),
                    "global_generated": sum(metrics_accum["global_generated"]) / len(metrics_accum["global_generated"]),
                    "fixed_hotspot_generated": sum(metrics_accum["fixed_hotspot_generated"]) / len(metrics_accum["fixed_hotspot_generated"]),
                    "overload_method": "metrics.compute_overload_ratios",
                }
            )

        for seed, scheme_counts in counts_by_seed.items():
            global_vals = {vals["global_generated"] for vals in scheme_counts.values()}
            hotspot_vals = {vals["fixed_hotspot_generated"] for vals in scheme_counts.values()}
            if len(global_vals) > 1 or len(hotspot_vals) > 1:
                raise ValueError(f"Fairness violation N={N} seed={seed}: {scheme_counts}")

    return summary_rows, derived_hotspot, arrival_hashes
