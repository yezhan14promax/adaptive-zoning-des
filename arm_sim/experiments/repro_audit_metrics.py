import os
import csv
import argparse

from arm_sim.experiments import run_hotspot
from arm_sim.experiments.output_utils import normalize_output_root


def _mean(values):
    if not values:
        return float("nan")
    return sum(values) / len(values)


def _parse_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_rows(path):
    if not os.path.isfile(path):
        raise RuntimeError(f"Missing window_kpis CSV: {path}")
    with open(path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _hotspot_mean(rows, t_hot_start, t_hot_end, key):
    values = []
    for row in rows:
        t_start = _parse_float(row.get("t_start"))
        t_end = _parse_float(row.get("t_end"))
        if t_start is None or t_end is None:
            continue
        if t_start >= t_hot_start and t_end <= t_hot_end:
            value = _parse_float(row.get(key))
            if value is not None:
                values.append(value)
    return _mean(values) if values else float("nan")


def _print_debug_tail(debug_path, limit=20):
    if not os.path.isfile(debug_path):
        print(f"Missing debug windows: {debug_path}")
        return
    with open(debug_path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    print("Debug windows (top rows):")
    for row in rows[:limit]:
        print(
            f"t={row.get('t_start')}-{row.get('t_end')} "
            f"queue_p95_hot={row.get('queue_p95_hot')} "
            f"serving_hotspot_overload_q1000={row.get('serving_hotspot_overload_ratio_q1000')}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", default=None)
    args = parser.parse_args()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    output_root = normalize_output_root(base_dir, args.output_root, prefix="figure")
    output_dir = os.path.join(output_root, "figure_z16_noc")
    os.makedirs(output_dir, exist_ok=True)

    main_config = run_hotspot._load_main_run_config(output_root, output_dir)
    arrival = main_config["arrival"]
    service = main_config["service"]
    hotspot = main_config["hotspot"]
    state_rate_hz = (
        main_config["state_rate_hz_base"]
        * main_config["standard_load"]
        * (main_config.get("calibration_multiplier") or 1.0)
    )
    zone_rate = main_config["zone_rate"]
    n_zones = main_config["n_zones_main"]
    n_robots = main_config["n_robots_main"]
    central_override, _ = run_hotspot._central_rate_override(
        state_rate_hz, n_robots, n_zones, zone_rate
    )

    deterministic_hotspots = bool(main_config.get("deterministic_hotspots", True))
    hotspot_target_zones = main_config.get("deterministic_hotspot_zones")
    hotspot_ratio_override = main_config.get("hotspot_ratio_override")
    if hotspot_ratio_override is None:
        hotspot_ratio_override = run_hotspot.HOTSPOT_RATIO_BASE

    overrides_path = os.path.join(output_dir, "s2_profile_overrides.json")
    profile_overrides = run_hotspot._load_profile_overrides(overrides_path) or {}

    profiles = [
        ("S1", "baseline", None),
        ("S2", "conservative", profile_overrides.get("conservative") or {}),
        ("S2", "neutral", profile_overrides.get("neutral") or {}),
        ("S2", "aggressive", profile_overrides.get("aggressive") or {}),
    ]

    seed = 0
    summary_rows = []
    generated_vals = []
    hotspot_generated_vals = []
    completion_ratios = []
    for scheme_key, profile_name, overrides in profiles:
        overrides = overrides or {}
        tag = f"Audit_{scheme_key}_{profile_name}"
        run_hotspot.run_scheme(
            scheme_key,
            seed,
            output_dir,
            state_rate_hz=state_rate_hz,
            zone_service_rate_msgs_s=zone_rate,
            write_csv=True,
            n_zones_override=n_zones,
            n_robots_override=n_robots,
            central_service_rate_msgs_s_override=central_override,
            tag=tag,
            arrival_mode="zone_poisson",
            hotspot_target_zones=hotspot_target_zones if deterministic_hotspots else None,
            hotspot_ratio_override=hotspot_ratio_override,
            weight_scale_override=overrides.get("weight_scale_override"),
            budget_gamma_override=overrides.get("budget_gamma_override"),
            cooldown_s_override=overrides.get("cooldown_s_override"),
        )
        path = os.path.join(output_dir, f"window_kpis_{tag}_{scheme_key.lower()}.csv")
        rows = _load_rows(path)
        if not rows:
            raise RuntimeError(f"No rows for {scheme_key} {profile_name}")
        t_hot_start = _parse_float(rows[0].get("hotspot_start_s")) or 30.0
        t_hot_end = _parse_float(rows[0].get("hotspot_end_s")) or 80.0

        p95_ms = _hotspot_mean(rows, t_hot_start, t_hot_end, "p95_ms")
        fixed_over_q1000 = _hotspot_mean(rows, t_hot_start, t_hot_end, "fixed_hotspot_overload_ratio_q1000")
        serving_over_q1000 = _hotspot_mean(rows, t_hot_start, t_hot_end, "serving_hotspot_overload_ratio_q1000")
        global_topk_over_q1000 = _hotspot_mean(rows, t_hot_start, t_hot_end, "global_topk_overload_ratio_q1000")
        global_generated = sum(
            int(float(row.get("global_generated", 0) or 0)) for row in rows
        )
        hotspot_generated = sum(
            int(
                float(
                    row.get("fixed_hotspot_generated_requests")
                    or row.get("hotspot_generated_fixed")
                    or 0
                )
            )
            for row in rows
        )
        migrated_weight_total = _parse_float(rows[0].get("migrated_weight_total")) or 0.0
        reconfig_action_count = _parse_float(rows[0].get("reconfig_action_count")) or 0.0
        horizon_s = 0.0
        for row in rows:
            t_end = _parse_float(row.get("t_end"))
            if t_end is not None and t_end > horizon_s:
                horizon_s = t_end
        cost_per_request = (
            migrated_weight_total / global_generated if global_generated > 0 else float("nan")
        )
        reconfig_actions_per_s = (
            reconfig_action_count / horizon_s if horizon_s > 0 else float("nan")
        )
        completion_ratio = _parse_float(rows[0].get("global_completion_ratio"))
        fixed_completion_ratio = _parse_float(
            rows[0].get("fixed_hotspot_completion_ratio")
        )
        if completion_ratio is not None and not (0.0 <= completion_ratio <= 1.0):
            raise RuntimeError(
                "Invalid global_completion_ratio "
                f"{completion_ratio}; edge_completed={rows[0].get('edge_completed')}, "
                f"central_completed={rows[0].get('central_completed')}, "
                f"total_completed={rows[0].get('total_completed')}, "
                f"completed={rows[0].get('completed')}"
            )
        if fixed_completion_ratio is not None and not (0.0 <= fixed_completion_ratio <= 1.0):
            raise RuntimeError(
                "Invalid fixed_hotspot_completion_ratio "
                f"{fixed_completion_ratio}; edge_completed={rows[0].get('edge_completed')}, "
                f"central_completed={rows[0].get('central_completed')}, "
                f"total_completed={rows[0].get('total_completed')}, "
                f"completed={rows[0].get('completed')}"
            )

        summary_rows.append(
            {
                "scheme": "Static-edge" if scheme_key == "S1" else f"Ours-{profile_name.capitalize()}",
                "p95_ms": p95_ms,
                "fixed_over_q1000": fixed_over_q1000,
                "serving_over_q1000": serving_over_q1000,
                "global_topk_over_q1000": global_topk_over_q1000,
                "cost_per_request": cost_per_request,
                "reconfig_actions_per_s": reconfig_actions_per_s,
            }
        )
        generated_vals.append(global_generated)
        hotspot_generated_vals.append(hotspot_generated)
        if completion_ratio is not None:
            completion_ratios.append(completion_ratio)

    if len(set(generated_vals)) != 1:
        raise RuntimeError(f"global_generated mismatch across schemes: {generated_vals}")
    if len(set(hotspot_generated_vals)) != 1:
        raise RuntimeError(
            f"fixed_hotspot_generated mismatch across schemes: {hotspot_generated_vals}"
        )
    if completion_ratios and min(completion_ratios) < 0.99:
        print("Completion ratio below 0.99; dumping debug windows.")
        for scheme_key, profile_name, _ in profiles:
            tag = f"Audit_{scheme_key}_{profile_name}"
            debug_path = os.path.join(output_dir, f"debug_windows_{tag}_{scheme_key.lower()}.csv")
            _print_debug_tail(debug_path)
        raise RuntimeError(f"Completion ratio below 0.99: {completion_ratios}")

    for row in summary_rows:
        print(
            f"{row['scheme']} "
            f"p95_latency_hotspot_ms={row['p95_ms']} "
            f"fixed_hotspot_overload_q1000={row['fixed_over_q1000']} "
            f"serving_hotspot_overload_q1000={row['serving_over_q1000']} "
            f"global_topk_overload_q1000={row['global_topk_over_q1000']} "
            f"cost_per_request={row['cost_per_request']} "
            f"reconfig_actions_per_s={row['reconfig_actions_per_s']}"
        )

    print("Assertions PASS: generated counts consistent, completion_ratio_global >= 0.99.")


if __name__ == "__main__":
    main()
