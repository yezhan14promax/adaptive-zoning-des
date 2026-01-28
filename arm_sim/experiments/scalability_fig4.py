import csv
import math
import os

from arm_sim.experiments.run_hotspot import run_scheme


HOT_START = 30.0
HOT_END = 80.0
WINDOW_S = 1.0
LOAD_SCALES = [1, 2, 3, 4]


def _parse_float(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _load_window_rows(path):
    if not os.path.isfile(path):
        raise RuntimeError(f"Missing window_kpis CSV: {path}")
    with open(path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        required = {
            "t_start",
            "t_end",
            "p95_ms",
            "mean_ms",
            "overload_ratio_q500",
            "overload_ratio_q1000",
            "overload_ratio_q1500",
            "generated",
            "completed",
            "hotspot_overload_ratio_q1000",
        }
        missing = [col for col in required if col not in headers]
        if missing:
            raise RuntimeError(
                f"Missing required columns in {path}: {', '.join(sorted(missing))}"
            )
        return list(reader)


def _window_has_columns(path):
    if not os.path.isfile(path):
        return False
    with open(path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = set(reader.fieldnames or [])
        required = {
            "t_start",
            "t_end",
            "p95_ms",
            "mean_ms",
            "overload_ratio_q500",
            "overload_ratio_q1000",
            "overload_ratio_q1500",
            "generated",
            "completed",
            "hotspot_overload_ratio_q1000",
        }
        return required.issubset(headers)


def _hotspot_mean(rows, key):
    values = []
    for row in rows:
        t_start = _parse_float(row.get("t_start"))
        t_end = _parse_float(row.get("t_end"))
        if t_start is None or t_end is None:
            continue
        if t_start >= HOT_START and t_end <= HOT_END:
            value = _parse_float(row.get(key))
            if value is not None and not math.isnan(value):
                values.append(value)
    if not values:
        return float("nan"), 0
    return sum(values) / len(values), len(values)


def _mean_overall(rows, key):
    values = []
    for row in rows:
        value = _parse_float(row.get(key))
        if value is None or math.isnan(value):
            raise RuntimeError(f"Invalid {key} value.")
        values.append(value)
    if not values:
        raise RuntimeError(f"No {key} values found.")
    return sum(values) / len(values)


def _summary_from_windows(path, scheme_label, seed, n_zones, load_scale):
    rows = _load_window_rows(path)
    hotspot_p95, p95_count = _hotspot_mean(rows, "p95_ms")
    hotspot_mean, mean_count = _hotspot_mean(rows, "mean_ms")
    hotspot_latency_n = sum(
        int(_parse_float(row.get("completed")) or 0)
        for row in rows
        if _parse_float(row.get("t_start")) is not None
        and _parse_float(row.get("t_end")) is not None
        and _parse_float(row.get("t_start")) >= HOT_START
        and _parse_float(row.get("t_end")) <= HOT_END
    )
    overload_ratio = _mean_overall(rows, "overload_ratio_q1000")
    overload_ratio_q500 = _mean_overall(rows, "overload_ratio_q500")
    overload_ratio_q1500 = _mean_overall(rows, "overload_ratio_q1500")
    hotspot_overload_ratio_q1000, _ = _hotspot_mean(rows, "hotspot_overload_ratio_q1000")
    generated_requests = sum(int(_parse_float(row.get("generated")) or 0) for row in rows)
    completed_requests = sum(int(_parse_float(row.get("completed")) or 0) for row in rows)
    completion_ratio = (
        completed_requests / generated_requests if generated_requests > 0 else float("nan")
    )
    first = rows[0] if rows else {}
    migrated_weight_total = _parse_float(first.get("migrated_weight_total")) or 0.0
    accepted_migrations = int(_parse_float(first.get("policy_reassign_ops")) or 0)
    if overload_ratio > 0.2 and hotspot_latency_n == 0:
        print(
            f"WARNING: overload_ratio_q1000={overload_ratio:.3f} but hotspot_latency_n=0 "
            f"for N={n_zones} scheme={scheme_label}"
        )
        hotspot_p95 = float("nan")
        hotspot_mean = float("nan")
    return {
        "N": n_zones,
        "load_scale": load_scale,
        "scheme": scheme_label,
        "hotspot_p95_mean_ms": hotspot_p95,
        "hotspot_mean_ms": hotspot_mean,
        "overload_ratio_q1000": overload_ratio,
        "overload_ratio_q500": overload_ratio_q500,
        "overload_ratio_q1500": overload_ratio_q1500,
        "hotspot_overload_ratio_q1000": hotspot_overload_ratio_q1000,
        "migrated_weight_total": migrated_weight_total,
        "accepted_migrations": accepted_migrations,
        "hotspot_latency_n": hotspot_latency_n,
        "generated_requests": generated_requests,
        "completed_requests": completed_requests,
        "completion_ratio": completion_ratio,
        "seed": seed,
    }


def _expected_tag(n_zones, label):
    return f"ScaleN{n_zones}_{label}"


def _profile_overrides():
    return {
        "Ours-Lite": {
            "q_high_override": 40,
            "q_low_override": 20,
            "cooldown_s_override": 2.0,
            "move_k_override": 6,
            "candidate_sample_m_override": 8,
            "p2c_k_override": 2,
            "beta_capacity_override": 0.90,
        },
        "Ours-Balanced": {
            "q_high_override": 30,
            "q_low_override": 15,
            "cooldown_s_override": 1.0,
            "move_k_override": 10,
            "candidate_sample_m_override": 10,
            "p2c_k_override": 2,
            "beta_capacity_override": 0.80,
        },
        "Ours-Strong": {
            "q_high_override": 20,
            "q_low_override": 10,
            "cooldown_s_override": 0.5,
            "move_k_override": 15,
            "candidate_sample_m_override": 20,
            "p2c_k_override": 3,
            "beta_capacity_override": 0.75,
        },
    }


def _summary_complete(rows, expected_keys):
    remaining = set(expected_keys)
    for row in rows:
        key = (int(row["N"]), float(row["load_scale"]), row["scheme"], int(row["seed"]))
        remaining.discard(key)
    return not remaining


def _summary_complete_scaled(rows, expected_keys):
    remaining = set(expected_keys)
    for row in rows:
        key = (int(row["N"]), row["scheme"], int(row["seed"]))
        remaining.discard(key)
    return not remaining


def run_scalability_experiment(output_dir=None):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    output_dir = output_dir or os.path.join(base_dir, "outputs")
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    summary_path = os.path.join(figures_dir, "summary_scalability_multiload.csv")
    diag_path = os.path.join(figures_dir, "summary_scalability_diag.csv")

    seeds = [123]
    n_values = [8, 16, 32, 64]
    schemes = ["Static-edge", "Ours-Lite", "Ours-Balanced", "Ours-Strong"]

    expected_keys = [
        (n, float(load_scale), scheme, seed)
        for load_scale in LOAD_SCALES
        for n in n_values
        for scheme in schemes
        for seed in seeds
    ]

    if os.path.isfile(summary_path) and os.path.isfile(diag_path):
        with open(summary_path, "r", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        if rows and _summary_complete(rows, expected_keys):
            return summary_path

    state_rate_hz_base = 10
    zone_rate = 200
    base_n_zones = 8
    base_n_robots = 100
    robots_per_zone = base_n_robots / float(base_n_zones)
    central_rate_per_zone = (10.0 * zone_rate) / float(base_n_zones)

    profile_overrides = _profile_overrides()
    summary_rows = []

    for load_scale in LOAD_SCALES:
        state_rate_hz = state_rate_hz_base * load_scale
        for n_zones in n_values:
            n_robots = int(round(robots_per_zone * n_zones))
            central_rate = central_rate_per_zone * n_zones
            for seed in seeds:
                tag = _expected_tag(n_zones, f"static_l{load_scale}")
                path = os.path.join(output_dir, f"window_kpis_{tag}_s1.csv")
                if not _window_has_columns(path):
                    run_scheme(
                        "S1",
                        seed,
                        output_dir,
                        state_rate_hz=state_rate_hz,
                        zone_service_rate_msgs_s=zone_rate,
                        write_csv=True,
                        n_zones_override=n_zones,
                        n_robots_override=n_robots,
                        central_service_rate_msgs_s_override=central_rate,
                        dmax_ms=30.0,
                        tag=tag,
                    )
                summary_rows.append(
                    _summary_from_windows(
                        path, "Static-edge", seed, n_zones, load_scale
                    )
                )

                for label, overrides in profile_overrides.items():
                    tag = _expected_tag(
                        n_zones, f"{label.replace('Ours-', '').lower()}_l{load_scale}"
                    )
                    path = os.path.join(output_dir, f"window_kpis_{tag}_s2.csv")
                    if not _window_has_columns(path):
                        run_scheme(
                            "S2",
                            seed,
                            output_dir,
                            state_rate_hz=state_rate_hz,
                            zone_service_rate_msgs_s=zone_rate,
                            write_csv=True,
                            n_zones_override=n_zones,
                            n_robots_override=n_robots,
                            central_service_rate_msgs_s_override=central_rate,
                            dmax_ms=30.0,
                            tag=tag,
                            **overrides,
                        )
                    summary_rows.append(
                        _summary_from_windows(
                            path, label, seed, n_zones, load_scale
                        )
                    )

    with open(summary_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "strategy",
                "scheme",
                "N",
                "load_scale",
                "hotspot_p95_latency",
                "hotspot_p95_mean_ms",
                "hotspot_mean_ms",
                "overload_ratio_q1000",
                "migrated_weight_total",
                "accepted_migrations",
                "seed",
            ]
        )
        for row in summary_rows:
            writer.writerow(
                [
                    row["scheme"],
                    row["scheme"],
                    row["N"],
                    f"{row['load_scale']:.1f}",
                    f"{row['hotspot_p95_mean_ms']:.3f}",
                    f"{row['hotspot_p95_mean_ms']:.3f}",
                    f"{row['hotspot_mean_ms']:.3f}",
                    f"{row['overload_ratio_q1000']:.6f}",
                    f"{row['migrated_weight_total']:.6f}",
                    row["accepted_migrations"],
                    row["seed"],
                ]
            )
    with open(diag_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "scheme",
                "N",
                "load_scale",
                "hotspot_p95_mean_ms",
                "hotspot_mean_ms",
                "hotspot_latency_n",
                "overload_ratio_q500",
                "overload_ratio_q1000",
                "overload_ratio_q1500",
                "hotspot_overload_ratio_q1000",
                "migrated_weight_total",
                "accepted_migrations",
                "generated_requests",
                "completed_requests",
                "completion_ratio",
                "seed",
            ]
        )
        for row in summary_rows:
            writer.writerow(
                [
                    row["scheme"],
                    row["N"],
                    f"{row['load_scale']:.1f}",
                    f"{row['hotspot_p95_mean_ms']:.3f}",
                    f"{row['hotspot_mean_ms']:.3f}",
                    row["hotspot_latency_n"],
                    f"{row['overload_ratio_q500']:.6f}",
                    f"{row['overload_ratio_q1000']:.6f}",
                    f"{row['overload_ratio_q1500']:.6f}",
                    f"{row['hotspot_overload_ratio_q1000']:.6f}",
                    f"{row['migrated_weight_total']:.6f}",
                    row["accepted_migrations"],
                    row["generated_requests"],
                    row["completed_requests"],
                    f"{row['completion_ratio']:.6f}",
                    row["seed"],
                ]
            )
    return summary_path


def _expected_scaled_tag(n_zones):
    return f"ScaleN{n_zones}_scaled"


def run_scaled_load_experiment(output_dir=None):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    output_dir = output_dir or os.path.join(base_dir, "outputs")
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    summary_path = os.path.join(figures_dir, "summary_fig4_scaled_load.csv")

    seeds = [123]
    n_values = [8, 16, 32, 64]
    schemes = ["Static-edge", "Ours-Lite", "Ours-Balanced", "Ours-Strong"]
    expected_keys = [(n, scheme, seed) for n in n_values for scheme in schemes for seed in seeds]

    if os.path.isfile(summary_path):
        with open(summary_path, "r", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        if rows and _summary_complete_scaled(rows, expected_keys):
            return summary_path

    state_rate_hz_base = 10
    zone_rate = 200
    base_n_zones = 8
    base_n_robots = 100
    robots_per_zone = base_n_robots / float(base_n_zones)
    central_rate_per_zone = (10.0 * zone_rate) / float(base_n_zones)

    profile_overrides = _profile_overrides()
    summary_rows = []

    for n_zones in n_values:
        load_scale = n_zones / 16.0
        state_rate_hz = state_rate_hz_base * load_scale
        n_robots = int(round(robots_per_zone * n_zones))
        central_rate = central_rate_per_zone * n_zones
        for seed in seeds:
            tag = _expected_scaled_tag(n_zones)
            path = os.path.join(output_dir, f"window_kpis_{tag}_s1.csv")
            if not _window_has_columns(path):
                run_scheme(
                    "S1",
                    seed,
                    output_dir,
                    state_rate_hz=state_rate_hz,
                    zone_service_rate_msgs_s=zone_rate,
                    write_csv=True,
                    n_zones_override=n_zones,
                    n_robots_override=n_robots,
                    central_service_rate_msgs_s_override=central_rate,
                    dmax_ms=30.0,
                    tag=tag,
                )
            summary_rows.append(
                _summary_from_windows(path, "Static-edge", seed, n_zones, load_scale)
            )

            for label, overrides in profile_overrides.items():
                tag = _expected_tag(n_zones, f"{label.replace('Ours-', '').lower()}_scaled")
                path = os.path.join(output_dir, f"window_kpis_{tag}_s2.csv")
                if not _window_has_columns(path):
                    run_scheme(
                        "S2",
                        seed,
                        output_dir,
                        state_rate_hz=state_rate_hz,
                        zone_service_rate_msgs_s=zone_rate,
                        write_csv=True,
                        n_zones_override=n_zones,
                        n_robots_override=n_robots,
                        central_service_rate_msgs_s_override=central_rate,
                        dmax_ms=30.0,
                        tag=tag,
                        **overrides,
                    )
                summary_rows.append(
                    _summary_from_windows(path, label, seed, n_zones, load_scale)
                )

    with open(summary_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "N",
                "load_multiplier",
                "load_scale",
                "method",
                "scheme",
                "hotspot_p95_s",
                "hotspot_p95_mean_ms",
                "hotspot_mean_ms",
                "overload_ratio_q1000",
                "migrated_weight_total",
                "accepted_migrations",
                "seed",
            ]
        )
        for row in summary_rows:
            writer.writerow(
                [
                    row["N"],
                    f"{row['load_scale']:.3f}",
                    f"{row['load_scale']:.3f}",
                    row["scheme"],
                    row["scheme"],
                    f"{row['hotspot_p95_mean_ms'] / 1000.0:.6f}"
                    if not math.isnan(row["hotspot_p95_mean_ms"])
                    else "nan",
                    f"{row['hotspot_p95_mean_ms']:.3f}"
                    if not math.isnan(row["hotspot_p95_mean_ms"])
                    else "nan",
                    f"{row['hotspot_mean_ms']:.3f}"
                    if not math.isnan(row["hotspot_mean_ms"])
                    else "nan",
                    f"{row['overload_ratio_q1000']:.6f}",
                    f"{row['migrated_weight_total']:.6f}",
                    row["accepted_migrations"],
                    row["seed"],
                ]
            )
    return summary_path


def main():
    summary_path = run_scalability_experiment()
    print(summary_path)


if __name__ == "__main__":
    main()
