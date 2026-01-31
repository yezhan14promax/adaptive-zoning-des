import os
import argparse

from arm_sim.experiments.output_utils import normalize_output_root

from arm_sim.experiments.run_hotspot import run_scheme


def _print_summary(tag, scheme, dmax_ms, result):
    (
        output_path,
        mean_p95,
        _overload_duration,
        _overload_duration_q100,
        overload_duration_q1000,
        _hotspot_max_central_q,
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
        _migrate_penalty_ms,
        _q_high,
        _q_low,
        _move_k,
        _cooldown_s,
        _beta_capacity,
        _budget_gamma,
        _candidate_sample_m,
        _p2c_k,
    ) = result
    print(
        f"tag={tag} scheme={scheme} dmax_ms={dmax_ms:.1f} "
        f"hotspot_p95_mean_ms={mean_p95:.3f} "
        f"overload_duration_q1000_s={overload_duration_q1000:.1f} "
        f"max_zone_q={hotspot_max_zone_q} accepted_migrations={policy_reassign_ops} "
        f"migrated_robots_total={migrated_robots_total} "
        f"migrated_weight_total={migrated_weight_total:.3f} "
        f"rejected_budget={rejected_budget} rejected_safety={rejected_safety} "
        f"rejected_no_feasible_target={rejected_no_feasible_target} "
        f"fallback_attempts={fallback_attempts} fallback_success={fallback_success} "
        f"dmax_rejects={dmax_rejects}"
    )
    return output_path


def _run(tag, scheme, dmax_ms, seed, output_dir, state_rate_hz, zone_rate, **kwargs):
    result = run_scheme(
        scheme,
        seed,
        output_dir,
        state_rate_hz=state_rate_hz,
        zone_service_rate_msgs_s=zone_rate,
        write_csv=True,
        dmax_ms=dmax_ms,
        tag=tag,
        **kwargs,
    )
    return _print_summary(tag, scheme, dmax_ms, result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", default=None)
    args = parser.parse_args()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    output_dir = normalize_output_root(base_dir, args.output_root, prefix="figure")

    seed = 123
    state_rate_hz = 10
    zone_rate = 200

    output_paths = []

    main_runs = [
        ("Main_unconstrained", 1.0e9),
        ("Main_constrained", 30.0),
    ]
    for tag, dmax in main_runs:
        for scheme in ["S0", "S1", "S2"]:
            output_paths.append(
                _run(tag, scheme, dmax, seed, output_dir, state_rate_hz, zone_rate)
            )

    ablations = [
        ("Abl_NoFallback", {"disable_fallback": True}),
        ("Abl_NoBudget", {"disable_budget": True}),
        ("Abl_FixedK", {"fixed_k": True, "move_k_fixed": 5}),
    ]
    for base_tag, kwargs in ablations:
        for dmax, suffix in [(1.0e9, "unconstrained"), (30.0, "constrained")]:
            tag = f"{base_tag}_{suffix}"
            output_paths.append(
                _run(tag, "S2", dmax, seed, output_dir, state_rate_hz, zone_rate, **kwargs)
            )

    for candidate_m in [5, 10, 20]:
        tag = f"Sens_candidate_m{candidate_m}"
        output_paths.append(
            _run(
                tag,
                "S2",
                30.0,
                seed,
                output_dir,
                state_rate_hz,
                zone_rate,
                candidate_sample_m_override=candidate_m,
            )
        )

    return output_paths


if __name__ == "__main__":
    main()
