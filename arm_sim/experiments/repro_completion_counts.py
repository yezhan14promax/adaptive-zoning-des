import json
import os
import argparse

from arm_sim.experiments.output_utils import normalize_output_root

from arm_sim.experiments.run_hotspot import (
    _collect_seed_summary,
    _load_profile_overrides,
    _s2_profile_overrides,
    _window_has_columns,
    run_scheme,
)


def _load_main_config(figures_dir):
    path = os.path.join(figures_dir, "run_config.json")
    if not os.path.isfile(path):
        return {}
    with open(path, "r") as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", default=None)
    args = parser.parse_args()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    output_dir = normalize_output_root(base_dir, args.output_root, prefix="figure")
    figures_dir = os.path.join(output_dir, "figure_z16_noc")
    os.makedirs(output_dir, exist_ok=True)

    main_cfg = _load_main_config(figures_dir)
    arrival_mode = (main_cfg.get("arrival_mode") or main_cfg.get("arrival_mode_main") or "robot_emit")
    deterministic_hotspots = bool(main_cfg.get("deterministic_hotspots", False))
    hotspot_zones = main_cfg.get("deterministic_hotspot_zones") or []

    n_zones = int(main_cfg.get("N", 16))
    runs = main_cfg.get("runs") or []
    n_robots = int(runs[0].get("n_robots", 200)) if runs else 200
    state_rate_hz_base = (
        main_cfg.get("arrival", {}).get("state_rate_hz_base") if main_cfg.get("arrival") else 10
    )
    standard_load = float(main_cfg.get("standard_load", 2.0))
    calib_multiplier = float(main_cfg.get("calibration_multiplier", 1.0))
    state_rate_hz = state_rate_hz_base * standard_load * calib_multiplier

    service_cfg = main_cfg.get("service", {})
    zone_rate = float(service_cfg.get("zone_service_rate_msgs_s", 200))
    central_override = float(service_cfg.get("central_service_rate_msgs_s", n_zones * zone_rate))

    overrides_path = os.path.join(figures_dir, "s2_profile_overrides.json")
    base_profiles = _load_profile_overrides(overrides_path)
    overrides = _s2_profile_overrides(base_profiles=base_profiles)

    seed = 0
    schemes = [
        ("S1", "Static-edge", None),
        ("S2", "Ours-Lite", overrides.get("S2_conservative", {})),
        ("S2", "Ours-Balanced", overrides.get("S2_neutral", {})),
        ("S2", "Ours-Strong", overrides.get("S2_aggressive", {})),
    ]

    rows = []
    for scheme_key, label, override in schemes:
        tag = f"ReproCompletion_{label.replace(' ', '')}_seed{seed}"
        path = os.path.join(output_dir, f"window_kpis_{tag}_s{scheme_key[-1].lower()}.csv")
        if not _window_has_columns(path):
            run_scheme(
                scheme_key,
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
                arrival_mode=arrival_mode,
                hotspot_target_zones=hotspot_zones if deterministic_hotspots else None,
                hotspot_ratio_override=0.5 if deterministic_hotspots else None,
                **(override or {}),
            )
        row = _collect_seed_summary(
            output_dir,
            tag,
            scheme_key,
            label,
            "baseline" if scheme_key == "S1" else label,
            seed,
            n_zones,
            calib_multiplier,
        )
        rows.append(row)

    print("scheme,global_generated,fixed_hotspot_generated,total_completed")
    for row in rows:
        print(
            f"{row.get('scheme')},"
            f"{row.get('global_generated_requests')},"
            f"{row.get('hotspot_generated_requests_fixed')},"
            f"{row.get('total_completed')}"
        )


if __name__ == "__main__":
    main()
