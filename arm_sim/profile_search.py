from __future__ import annotations

from typing import Dict, List, Tuple

from .arrival import generate_arrivals
from .config_schema import ExperimentConfig, compute_hotspot_params
from .policy_s2 import PolicyS2, S2Profile
from .rng import make_rng_streams
from .simulator import simulate


def build_profiles(config: ExperimentConfig) -> List[S2Profile]:
    profiles = []
    idx = 0
    for budget in config.profile_grid.budget_ratios:
        for weight in config.profile_grid.weight_scales:
            for cooldown in config.profile_grid.cooldown_s:
                profile_id = f"p{idx:03d}"
                profiles.append(
                    S2Profile(
                        profile_id=profile_id,
                        budget_ratio=budget,
                        weight_scale=weight,
                        cooldown_s=cooldown,
                    )
                )
                idx += 1
    return profiles


def evaluate_profiles(
    config: ExperimentConfig,
    seed_to_hotspot_selector,
) -> Tuple[List[Dict], Dict[int, Dict]]:
    profiles = build_profiles(config)
    summary_rows: List[Dict] = []
    derived_hotspot: Dict[int, Dict] = {}

    for seed in config.seeds:
        derived = compute_hotspot_params(
            N=config.N,
            phi=config.hotspot.phi,
            hotspot_skew=config.hotspot.hotspot_skew,
            lambda_mean=config.arrival.lambda_mean,
            hotspot_zone_selector=seed_to_hotspot_selector(seed),
        )
        derived_hotspot[seed] = {
            "N": derived.N,
            "k_hot": derived.k_hot,
            "phi_actual": derived.phi_actual,
            "lambda_hot": derived.lambda_hot,
            "lambda_cold": derived.lambda_cold,
            "hotspot_zones": derived.hotspot_zones,
        }

    for profile in profiles:
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
            derived = derived_hotspot[seed]
            rng_streams = make_rng_streams(seed)
            arrivals, counts = generate_arrivals(
                rng_streams.arrival_rng,
                N=config.N,
                sim_duration_s=config.arrival.sim_duration_s,
                hotspot_window=config.hotspot.hotspot_window,
                hotspot_zones=derived["hotspot_zones"],
                lambda_mean=config.arrival.lambda_mean,
                lambda_hot=derived["lambda_hot"],
                lambda_cold=derived["lambda_cold"],
            )
            policy = PolicyS2(config.N, profile, cloud_zone=config.N)
            result, _ = simulate(
                arrivals=arrivals,
                policy=policy,
                rng_streams=rng_streams,
                N=config.N,
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

        row = {
            "profile_id": profile.profile_id,
            "budget_ratio": profile.budget_ratio,
            "weight_scale": profile.weight_scale,
            "cooldown_s": profile.cooldown_s,
            "cost": sum(metrics_accum["cost"]) / len(metrics_accum["cost"]),
            "overload_q500": sum(metrics_accum["overload_q500"]) / len(metrics_accum["overload_q500"]),
            "overload_q1000": sum(metrics_accum["overload_q1000"]) / len(metrics_accum["overload_q1000"]),
            "overload_q1500": sum(metrics_accum["overload_q1500"]) / len(metrics_accum["overload_q1500"]),
            "p95_latency_hotspot": sum(metrics_accum["p95_latency_hotspot"]) / len(metrics_accum["p95_latency_hotspot"]),
            "global_generated": sum(metrics_accum["global_generated"]) / len(metrics_accum["global_generated"]),
            "fixed_hotspot_generated": sum(metrics_accum["fixed_hotspot_generated"]) / len(metrics_accum["fixed_hotspot_generated"]),
        }
        summary_rows.append(row)

    return summary_rows, derived_hotspot
