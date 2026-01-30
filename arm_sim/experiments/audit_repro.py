import argparse
import csv
import hashlib
import os

from arm_sim.experiments import run_hotspot
from arm_sim.experiments.output_utils import normalize_output_root


def _sha256(path, block_size=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(block_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _load_window_rows(path):
    with open(path, "r", newline="") as handle:
        return list(csv.DictReader(handle))


def _hot_window(rows):
    t_hot_start = 30.0
    t_hot_end = 80.0
    if rows:
        start = run_hotspot._parse_float(rows[0].get("hotspot_start_s"))
        end = run_hotspot._parse_float(rows[0].get("hotspot_end_s"))
        if start is not None and end is not None:
            t_hot_start = start
            t_hot_end = end
    return t_hot_start, t_hot_end


def _summarize(path, scheme_label, seed):
    rows = _load_window_rows(path)
    t_hot_start, t_hot_end = _hot_window(rows)
    summary = run_hotspot._summarize_window_rows(rows, t_hot_start, t_hot_end, scheme_label, seed)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--arrival_mode", default="zone_poisson")
    parser.add_argument("--deterministic_hotspots", action="store_true")
    args = parser.parse_args()

    seeds = run_hotspot._parse_seeds(args.seeds)
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    output_dir = normalize_output_root(base_dir, args.output_root, prefix="figure")
    figures_dir = os.path.join(output_dir, "figure_z16_noc")

    profile_version = "audit"
    run_hotspot.run_main_pipeline(
        output_dir,
        figures_dir,
        seeds,
        profile_version,
        arrival_mode=args.arrival_mode,
        deterministic_hotspots=args.deterministic_hotspots,
        auto_profile_search=False,
    )

    summary_path = os.path.join(figures_dir, "summary_main_and_ablations.csv")
    if not os.path.isfile(summary_path):
        raise RuntimeError(f"Missing summary: {summary_path}")

    for seed in seeds:
        print(f"Seed {seed}")
        for scheme_key, label, profile in (
            ("S1", "Static-edge", "baseline"),
            ("S2", "Ours-Lite", "conservative"),
            ("S2", "Ours-Balanced", "neutral"),
            ("S2", "Ours-Strong", "aggressive"),
        ):
            tag = None
            # infer tag from summary rows
            with open(summary_path, "r", newline="") as handle:
                for row in csv.DictReader(handle):
                    if row.get("scheme") == scheme_key and row.get("profile") == profile:
                        tag = row.get("tag")
                        break
            if not tag:
                raise RuntimeError(f"Missing tag for {label}")
            seed_tag = run_hotspot._seeded_tag(tag, seed)
            window_path = os.path.join(output_dir, f"window_kpis_{seed_tag}_{scheme_key.lower()}.csv")
            trace_path = os.path.join(output_dir, f"arrival_trace_{seed_tag}_{scheme_key.lower()}.csv")
            if not os.path.isfile(window_path):
                raise RuntimeError(f"Missing window_kpis: {window_path}")
            if not os.path.isfile(trace_path):
                raise RuntimeError(f"Missing arrival_trace: {trace_path}")
            summary = _summarize(window_path, label, seed)
            arrival_hash = _sha256(trace_path)
            fixed_gen = summary.get("fixed_hotspot_generated_requests")
            fixed_comp = summary.get("fixed_hotspot_completed_requests")
            fixed_over = summary.get("fixed_hotspot_overload_ratio_q1000")
            p95 = summary.get("hotspot_p95_mean_s")
            cost = summary.get("migrated_weight_total")
            cost_per_s = None
            horizon = summary.get("horizon_s") or 120.0
            if cost is not None:
                cost_per_s = cost / float(horizon)
            print(
                f"{label}: arrival_hash={arrival_hash} "
                f"fixed_gen={fixed_gen} fixed_comp={fixed_comp} "
                f"fixed_over_q1000={fixed_over} p95_s={p95} "
                f"migrated_weight_total={cost} migrated_weight_per_s={cost_per_s}"
            )


if __name__ == "__main__":
    main()
