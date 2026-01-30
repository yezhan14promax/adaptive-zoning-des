import argparse
import csv
import json
import math
import os
import time

from arm_sim.experiments import run_hotspot
from arm_sim.experiments.output_utils import normalize_output_root


def _load_csv(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path, "r", newline="") as handle:
        return list(csv.DictReader(handle))


def _parse_float(value):
    return run_hotspot._parse_float(value)


def _parse_int(value):
    return run_hotspot._parse_int(value)


def _load_baseline_targets(baseline_dir):
    summary_path = os.path.join(baseline_dir, "summary_main_and_ablations.csv")
    rows = _load_csv(summary_path)
    targets = {}
    for row in rows:
        if row.get("scheme") != "S2":
            continue
        profile = row.get("profile")
        if profile not in ("conservative", "neutral", "aggressive"):
            continue
        p95_s = _parse_float(row.get("hotspot_p95_mean_s"))
        if p95_s is None:
            p95_ms = _parse_float(row.get("hotspot_p95_mean_ms"))
            p95_s = p95_ms / 1000.0 if p95_ms is not None else None
        overload = _parse_float(row.get("fixed_hotspot_overload_ratio_q1000"))
        if overload is None:
            overload = _parse_float(row.get("overload_ratio_q1000"))
        targets[profile] = {
            "hotspot_p95_s": p95_s,
            "overload_q1000": overload,
            "migrated_weight_total": _parse_float(row.get("migrated_weight_total")),
            "reconfig_action_count": _parse_float(row.get("reconfig_action_count")),
        }
    if not targets:
        raise RuntimeError(f"No S2 profile rows found in {summary_path}")
    return targets


def _load_main_multiplier(baseline_dir):
    summary_path = os.path.join(baseline_dir, "summary_main_and_ablations.csv")
    rows = _load_csv(summary_path)
    for row in rows:
        if row.get("scheme") == "S1" and row.get("profile") == "baseline":
            multiplier = _parse_float(row.get("load_multiplier"))
            if multiplier is not None:
                return multiplier
    for row in rows:
        multiplier = _parse_float(row.get("load_multiplier"))
        if multiplier is not None:
            return multiplier
    raise RuntimeError(f"Missing load_multiplier in {summary_path}")


def _hotspot_window(rows):
    t_hot_start = 30.0
    t_hot_end = 80.0
    if rows:
        start = _parse_float(rows[0].get("hotspot_start_s"))
        end = _parse_float(rows[0].get("hotspot_end_s"))
        if start is not None and end is not None:
            t_hot_start = start
            t_hot_end = end
    return t_hot_start, t_hot_end


def _summarize_seed(path, scheme_label, seed):
    rows = run_hotspot._load_window_rows(path)
    t_hot_start, t_hot_end = _hotspot_window(rows)
    summary = run_hotspot._summarize_window_rows(rows, t_hot_start, t_hot_end, scheme_label, seed)
    p95_s = summary.get("hotspot_p95_mean_s")
    overload = summary.get("fixed_hotspot_overload_ratio_q1000")
    if overload is None:
        overload = summary.get("overload_ratio_q1000")
    return {
        "hotspot_p95_s": p95_s,
        "overload_q1000": overload,
        "migrated_weight_total": summary.get("migrated_weight_total"),
        "reconfig_action_count": summary.get("reconfig_action_count"),
    }


def _aggregate(values):
    vals = [v for v in values if v is not None and not math.isnan(v)]
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


def _fit_profile(
    baseline_dir,
    profile_label,
    target,
    seeds,
    weight_scales,
    budget_gammas,
    cooldown_vals,
    arrival_mode,
    deterministic_hotspots,
    hotspot_frac_zones,
):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    output_dir = normalize_output_root(base_dir, None, prefix="figure")
    state_rate_hz_base = 10
    standard_load = 2.0
    zone_rate = 200
    n_zones = 16
    base_n_zones = 8
    base_n_robots = 100
    robots_per_zone = base_n_robots / float(base_n_zones)
    n_robots = int(round(robots_per_zone * n_zones))
    multiplier = _load_main_multiplier(baseline_dir)
    state_rate_hz = state_rate_hz_base * standard_load * multiplier
    central_override, _ = run_hotspot._central_rate_override(
        state_rate_hz, n_robots, n_zones, zone_rate
    )
    hotspot_zones_count = max(
        1, int(round((hotspot_frac_zones or (1.0 / n_zones)) * n_zones))
    )
    hotspot_targets = run_hotspot._deterministic_hotspot_zones(
        n_zones, hotspot_zones_count, center_zone=0
    )
    extra_robots_per_zone = None
    if deterministic_hotspots:
        base_per_zone = n_robots / float(n_zones) if n_zones else 0.0
        desired_total = run_hotspot.HOTSPOT_RATIO_BASE * n_robots
        baseline_total = hotspot_zones_count * base_per_zone
        extra_total = max(0.0, desired_total - baseline_total)
        extra_robots_per_zone = (
            int(round(extra_total / hotspot_zones_count))
            if hotspot_zones_count > 0
            else 0
        )

    best = None
    for weight_scale in weight_scales:
        for budget_gamma in budget_gammas:
            for cooldown_s in cooldown_vals:
                metrics = []
                for seed in seeds:
                    tag = (
                        f"Fit_{profile_label}_w{weight_scale:.2f}_b{budget_gamma:.3f}_c{cooldown_s:.2f}_s{seed}"
                    ).replace(".", "p")
                    path = os.path.join(output_dir, f"window_kpis_{tag}_s2.csv")
                    if not os.path.isfile(path):
                        run_hotspot.run_scheme(
                            "S2",
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
                            hotspot_target_zones=hotspot_targets if deterministic_hotspots else None,
                            extra_robots_per_zone=extra_robots_per_zone if deterministic_hotspots else None,
                            arrival_mode=arrival_mode,
                            weight_scale_override=weight_scale,
                            budget_gamma_override=budget_gamma,
                            cooldown_s_override=cooldown_s,
                        )
                    metrics.append(_summarize_seed(path, "S2", seed))

                p95_mean = _aggregate([m.get("hotspot_p95_s") for m in metrics])
                overload_mean = _aggregate([m.get("overload_q1000") for m in metrics])
                cost_mean = _aggregate([m.get("migrated_weight_total") for m in metrics])
                action_mean = _aggregate([m.get("reconfig_action_count") for m in metrics])

                def _rel_err(val, target_val):
                    if val is None or target_val is None or math.isnan(val) or math.isnan(target_val):
                        return 1.0
                    if target_val == 0:
                        return abs(val - target_val)
                    return (val - target_val) / target_val

                err = 0.0
                err += _rel_err(p95_mean, target.get("hotspot_p95_s")) ** 2
                err += _rel_err(overload_mean, target.get("overload_q1000")) ** 2
                err += _rel_err(cost_mean, target.get("migrated_weight_total")) ** 2
                err += _rel_err(action_mean, target.get("reconfig_action_count")) ** 2

                candidate = {
                    "weight_scale_override": weight_scale,
                    "budget_gamma_override": budget_gamma,
                    "cooldown_s_override": cooldown_s,
                    "error": err,
                    "hotspot_p95_s": p95_mean,
                    "overload_q1000": overload_mean,
                    "migrated_weight_total": cost_mean,
                    "reconfig_action_count": action_mean,
                }
                if best is None or candidate["error"] < best["error"]:
                    best = candidate

    if best is None:
        raise RuntimeError(f"Failed to fit profile {profile_label}")
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_dir", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--arrival_mode", default="robot_emit")
    parser.add_argument("--deterministic_hotspots", action="store_true")
    parser.add_argument("--hotspot_frac_zones", type=float, default=None)
    parser.add_argument("--weight_scales", default="0.6,1.0,2.5")
    parser.add_argument("--budget_gammas", default="0.03,0.04,0.05,0.06,0.07,0.08,0.10")
    parser.add_argument("--cooldown_vals", default="0.5,1.0")
    args = parser.parse_args()

    seeds = run_hotspot._parse_seeds(args.seeds)
    weight_scales = [float(v) for v in args.weight_scales.split(",") if v.strip()]
    budget_gammas = [float(v) for v in args.budget_gammas.split(",") if v.strip()]
    cooldown_vals = [float(v) for v in args.cooldown_vals.split(",") if v.strip()]

    targets = _load_baseline_targets(args.baseline_dir)

    overrides = {}
    for profile_label, target in targets.items():
        best = _fit_profile(
            args.baseline_dir,
            profile_label,
            target,
            seeds,
            weight_scales,
            budget_gammas,
            cooldown_vals,
            args.arrival_mode,
            args.deterministic_hotspots,
            args.hotspot_frac_zones,
        )
        overrides[profile_label] = {
            "weight_scale_override": best["weight_scale_override"],
            "budget_gamma_override": best["budget_gamma_override"],
            "cooldown_s_override": best["cooldown_s_override"],
        }
        print(
            f"{profile_label}: w={best['weight_scale_override']} "
            f"b={best['budget_gamma_override']} c={best['cooldown_s_override']} "
            f"err={best['error']:.4f}"
        )

    payload = {"profiles": overrides, "timestamp": int(time.time())}
    output_path = args.output or os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "outputs")),
        "s2_profile_overrides_p28.json",
    )
    with open(output_path, "w", newline="") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    print(output_path)


if __name__ == "__main__":
    main()
