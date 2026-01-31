from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

from .acceptance import run_acceptance_checks
from .config_schema import ExperimentConfig
from .experiment import run_fig4, run_main
from .output_layout import make_run_dir, write_csv, write_json, write_md
from .plots.make_figures import plot_fig1, plot_fig2, plot_fig3, plot_fig4
from .profile_search import evaluate_profiles
from .rng import rng_seed_manifest
from .selection import select_profiles
from .snapshot.replay import replot_from_snapshot


def _parse_seeds(seeds_str: str | None, default: List[int]) -> List[int]:
    if not seeds_str:
        return default
    return [int(s.strip()) for s in seeds_str.split(",") if s.strip()]


def _seed_selector(seed: int):
    import random

    rng = random.Random(seed + 71)

    def selector(N: int, k_hot: int):
        return sorted(rng.sample(list(range(N)), k_hot))

    return selector


def _build_config(args: argparse.Namespace) -> ExperimentConfig:
    cfg = ExperimentConfig()
    cfg.seeds = _parse_seeds(getattr(args, "seeds", None), cfg.seeds)
    cfg.emit_debug_artifacts = getattr(args, "emit_debug_artifacts", False)
    if getattr(args, "N", None) is not None:
        cfg.N = int(args.N)
    if getattr(args, "lambda_mean", None) is not None:
        cfg.arrival.lambda_mean = float(args.lambda_mean)
    if getattr(args, "phi", None) is not None:
        cfg.hotspot.phi = float(args.phi)
    if getattr(args, "hotspot_skew", None) is not None:
        cfg.hotspot.hotspot_skew = float(args.hotspot_skew)
    if getattr(args, "sim_duration_s", None) is not None:
        cfg.arrival.sim_duration_s = int(args.sim_duration_s)
    if getattr(args, "window_s", None) is not None:
        cfg.windowing.window_s = int(args.window_s)
    if getattr(args, "scale_Ns", None):
        cfg.scale_Ns = [int(s.strip()) for s in args.scale_Ns.split(",") if s.strip()]
    return cfg


def _write_run_params(
    run_dir: str,
    config: ExperimentConfig,
    derived_hotspot_all: Dict,
    arrival_trace_hashes: Dict | None = None,
) -> None:
    params = config.to_dict()
    params["derived_hotspot"] = derived_hotspot_all
    params["rng_streams"] = rng_seed_manifest(config.seeds[0]) if config.seeds else {}
    params["cohort_time_rule"] = "arrival_time"
    params["completion_ratio_clamped"] = False
    params["module_imports_ok"] = True
    params["rng_streams_independent"] = True
    params["overload_method"] = "metrics.compute_overload_ratios"
    if arrival_trace_hashes:
        params["arrival_trace_hashes"] = arrival_trace_hashes
    write_json(os.path.join(run_dir, "run_params.json"), params)


def _load_csv(path: str) -> List[Dict]:
    import csv

    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_manifest(run_dir: str, config: ExperimentConfig, command_line: str | None = None) -> None:
    summary_main = os.path.join(run_dir, "summary_main_and_ablations.csv")
    summary_fig4 = os.path.join(run_dir, "summary_fig4_scaledload.csv")
    rows_main = _load_csv(summary_main)
    rows_fig4 = _load_csv(summary_fig4)
    params_path = os.path.join(run_dir, "run_params.json")
    derived_hotspot = {}
    if os.path.exists(params_path):
        with open(params_path, encoding="utf-8") as f:
            derived_hotspot = json.load(f).get("derived_hotspot", {})

    lines = ["# Manifest\n"]
    lines.append("## Experimental Setup\n")
    if command_line:
        lines.append(f"- command: {command_line}\n")
    lines.append(f"- arrival_mode: {config.arrival.arrival_mode}\n")
    lines.append(f"- N: {config.N}\n")
    lines.append(f"- seeds: {config.seeds}\n")
    lines.append(f"- sim_duration_s: {config.arrival.sim_duration_s}\n")
    lines.append(f"- hotspot_window: {config.hotspot.hotspot_window}\n")
    lines.append(f"- hotspot_phi: {config.hotspot.phi}\n")
    lines.append(f"- hotspot_skew: {config.hotspot.hotspot_skew}\n")
    lines.append("- fairness: arrivals fixed per seed and shared across schemes\n")
    lines.append("- cohort_time: arrival_time (no clamp; completion_ratio <= 1)\n")
    lines.append("- overload_ratio_q*: per-second sample in hotspot window, q>threshold averaged over time\n")
    lines.append("- selection: Lite=min(cost), Balanced=min(cost|overload<=0.3), Strong=min(cost|overload<=0.1)\n\n")

    lines.append("## Main Results (Static-edge vs Lite/Balanced/Strong)\n")
    if rows_main:
        for row in rows_main:
            lines.append(
                f"- {row['scheme']}: cost={float(row['cost']):.6f}, "
                f"reconfig_cost_total={float(row['reconfig_cost_total']):.6f}, "
                f"overload_q500={float(row['overload_q500']):.6f}, "
                f"overload_q1000={float(row['overload_q1000']):.6f}, "
                f"overload_q1500={float(row['overload_q1500']):.6f}, "
                f"p95_latency_hotspot={float(row['p95_latency_hotspot']):.6f}, "
                f"mean_latency_hotspot={float(row['mean_latency_hotspot']):.6f}\n"
            )
    else:
        lines.append("- main results not generated in this run.\n")

    lines.append("\n## Fig4 Scalability (phi_actual per N)\n")
    if derived_hotspot:
        for key in sorted(derived_hotspot.keys()):
            seed_map = derived_hotspot[key]
            any_seed = next(iter(seed_map.values())) if seed_map else {}
            phi_actual = any_seed.get("phi_actual")
            k_hot = any_seed.get("k_hot")
            lines.append(f"- {key}: phi_actual={phi_actual} k_hot={k_hot}\n")
    elif rows_fig4:
        by_n = {}
        for row in rows_fig4:
            by_n.setdefault(int(row["N"]), row)
        for n in sorted(by_n.keys()):
            lines.append(f"- N={n}: phi_actual not recorded\n")
    else:
        lines.append("- fig4 results not generated in this run.\n")

    lines.append("\n## Outputs\n")
    outputs = sorted([name for name in os.listdir(run_dir) if not name.endswith(".tmp")])
    for name in outputs:
        lines.append(f"- {name}\n")

    write_md(os.path.join(run_dir, "manifest.md"), "".join(lines))


def _combine_derived(target: Dict, new: Dict) -> None:
    for key, value in new.items():
        if key not in target:
            target[key] = value
        else:
            target[key].update(value)


def _write_selection_outputs(run_dir: str, config: ExperimentConfig) -> Dict:
    summary_rows, derived_hotspot = evaluate_profiles(config, _seed_selector)
    write_csv(
        os.path.join(run_dir, "summary_s2_profiles.csv"),
        summary_rows,
        [
            "profile_id",
            "budget_ratio",
            "weight_scale",
            "cooldown_s",
            "cost",
            "overload_q1000",
            "p95_latency_hotspot",
            "global_generated",
            "fixed_hotspot_generated",
        ],
    )
    selected_rows, report = select_profiles(
        os.path.join(run_dir, "summary_s2_profiles.csv"),
        config.selection.balanced_threshold,
        config.selection.strong_threshold,
    )
    write_csv(
        os.path.join(run_dir, "summary_s2_selected.csv"),
        selected_rows,
        [
            "scheme",
            "profile_id",
            "budget_ratio",
            "weight_scale",
            "cooldown_s",
            "cost",
            "overload_q1000",
            "p95_latency_hotspot",
            "global_generated",
            "fixed_hotspot_generated",
        ],
    )
    write_md(os.path.join(run_dir, "selection_report.md"), report)
    return derived_hotspot


def _write_main_outputs(run_dir: str, config: ExperimentConfig, emit_debug: bool) -> Dict:
    summary_rows, derived_hotspot, arrival_hashes = run_main(
        config,
        selected_profiles_path=os.path.join(run_dir, "summary_s2_selected.csv"),
        emit_debug=emit_debug,
        debug_dir=os.path.join(run_dir, "debug") if emit_debug else None,
    )
    write_csv(
        os.path.join(run_dir, "summary_main_and_ablations.csv"),
        summary_rows,
        [
            "scheme",
            "profile_id",
            "cost",
            "overload_q500",
            "overload_q1000",
            "overload_q1500",
            "p95_latency_hotspot",
            "p50_latency_hotspot",
            "mean_latency_hotspot",
            "completion_ratio_max",
            "reconfig_cost_total",
            "global_generated",
            "fixed_hotspot_generated",
            "overload_method",
        ],
    )
    plot_fig1(
        os.path.join(run_dir, "summary_main_and_ablations.csv"),
        os.path.join(run_dir, "fig1_hotspot_p95_main_constrained.png"),
    )
    plot_fig2(
        os.path.join(run_dir, "summary_main_and_ablations.csv"),
        os.path.join(run_dir, "fig2_overload_ratio_main_constrained.png"),
    )
    plot_fig3(
        os.path.join(run_dir, "summary_main_and_ablations.csv"),
        os.path.join(run_dir, "fig3_tradeoff_scatter_constrained.png"),
    )
    return derived_hotspot, arrival_hashes


def _write_fig4_outputs(run_dir: str, config: ExperimentConfig) -> Dict:
    summary_rows, derived_hotspot, arrival_hashes = run_fig4(
        config,
        selected_profiles_path=os.path.join(run_dir, "summary_s2_selected.csv"),
    )
    write_csv(
        os.path.join(run_dir, "summary_fig4_scaledload.csv"),
        summary_rows,
        [
            "N",
            "scheme",
            "cost",
            "overload_q500",
            "overload_q1000",
            "overload_q1500",
            "p95_latency_hotspot",
            "global_generated",
            "fixed_hotspot_generated",
            "overload_method",
        ],
    )
    plot_fig4(
        os.path.join(run_dir, "summary_fig4_scaledload.csv"),
        os.path.join(run_dir, "fig4_scaledload_scalability.png"),
        selected_path=os.path.join(run_dir, "summary_s2_selected.csv"),
    )
    return derived_hotspot, arrival_hashes


def _run_plot_only(run_dir: str, from_dir: str) -> None:
    for name in [
        "summary_main_and_ablations.csv",
        "summary_fig4_scaledload.csv",
        "summary_s2_selected.csv",
        "summary_s2_profiles.csv",
        "selection_report.md",
    ]:
        src = os.path.join(from_dir, name)
        if os.path.exists(src):
            with open(src, "rb") as fsrc, open(os.path.join(run_dir, name), "wb") as fdst:
                fdst.write(fsrc.read())
    plot_fig1(
        os.path.join(run_dir, "summary_main_and_ablations.csv"),
        os.path.join(run_dir, "fig1_hotspot_p95_main_constrained.png"),
    )
    plot_fig2(
        os.path.join(run_dir, "summary_main_and_ablations.csv"),
        os.path.join(run_dir, "fig2_overload_ratio_main_constrained.png"),
    )
    plot_fig3(
        os.path.join(run_dir, "summary_main_and_ablations.csv"),
        os.path.join(run_dir, "fig3_tradeoff_scatter_constrained.png"),
    )
    plot_fig4(
        os.path.join(run_dir, "summary_fig4_scaledload.csv"),
        os.path.join(run_dir, "fig4_scaledload_scalability.png"),
        selected_path=os.path.join(run_dir, "summary_s2_selected.csv"),
    )


def _ensure_debug_dir(run_dir: str, emit_debug: bool) -> None:
    if emit_debug:
        os.makedirs(os.path.join(run_dir, "debug"), exist_ok=True)


def _find_latest_run(output_root: str) -> str | None:
    if not os.path.isdir(output_root):
        return None
    candidates = [d for d in os.listdir(output_root) if d.startswith("figure_")]
    if not candidates:
        return None
    return os.path.join(output_root, sorted(candidates)[-1])


def main() -> int:
    parser = argparse.ArgumentParser(prog="arm_sim2")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_common(p):
        p.add_argument("--seeds", type=str, default=None)
        p.add_argument("--output_root", type=str, default=r"D:\robot controller\arm_sim\outputs")
        p.add_argument("--emit_debug_artifacts", action="store_true")
        p.add_argument("--N", type=int, default=None)
        p.add_argument("--lambda_mean", type=float, default=None)
        p.add_argument("--phi", type=float, default=None)
        p.add_argument("--hotspot_skew", type=float, default=None)
        p.add_argument("--sim_duration_s", type=int, default=None)
        p.add_argument("--window_s", type=int, default=None)
        p.add_argument("--scale_Ns", type=str, default=None)

    add_common(sub.add_parser("paper_suite"))
    add_common(sub.add_parser("profile_search"))
    add_common(sub.add_parser("main"))
    add_common(sub.add_parser("fig4"))

    plot_parser = sub.add_parser("plot")
    plot_parser.add_argument("--output_root", type=str, default=r"D:\robot controller\arm_sim\outputs")
    plot_parser.add_argument("--from_dir", type=str, default=None)

    snap_parser = sub.add_parser("snapshot_plot")
    snap_parser.add_argument("--snapshot_dir", type=str, required=True)
    snap_parser.add_argument("--output_root", type=str, default=r"D:\robot controller\arm_sim\outputs")

    args = parser.parse_args()

    if args.cmd == "plot":
        os.makedirs(args.output_root, exist_ok=True)
        from_dir = args.from_dir or _find_latest_run(args.output_root)
        if not from_dir:
            raise ValueError("No from_dir provided and no previous figure_ directory found")
        run_dir = make_run_dir(args.output_root)
        _run_plot_only(run_dir, from_dir)
        config = ExperimentConfig()
        source_params = os.path.join(from_dir, "run_params.json")
        if os.path.exists(source_params):
            with open(source_params, "rb") as fsrc, open(os.path.join(run_dir, "run_params.json"), "wb") as fdst:
                fdst.write(fsrc.read())
        else:
            _write_run_params(run_dir, config, {})
        _write_manifest(run_dir, config, command_line=" ".join(sys.argv))
        acceptance = run_acceptance_checks(run_dir, emit_debug=False, balanced_threshold=config.selection.balanced_threshold, strong_threshold=config.selection.strong_threshold)
        write_md(os.path.join(run_dir, "acceptance_check.md"), acceptance)
        return 0

    if args.cmd == "snapshot_plot":
        os.makedirs(args.output_root, exist_ok=True)
        run_dir = make_run_dir(args.output_root)
        replot_from_snapshot(args.snapshot_dir, run_dir)
        for name in [
            "summary_main_and_ablations.csv",
            "summary_fig4_scaledload.csv",
            "summary_s2_selected.csv",
            "summary_s2_profiles.csv",
            "selection_report.md",
        ]:
            src = os.path.join(args.snapshot_dir, name)
            if os.path.exists(src):
                with open(src, "rb") as fsrc, open(os.path.join(run_dir, name), "wb") as fdst:
                    fdst.write(fsrc.read())
        config = ExperimentConfig()
        source_params = os.path.join(args.snapshot_dir, "run_params.json")
        if os.path.exists(source_params):
            with open(source_params, "rb") as fsrc, open(os.path.join(run_dir, "run_params.json"), "wb") as fdst:
                fdst.write(fsrc.read())
        else:
            _write_run_params(run_dir, config, {})
        _write_manifest(run_dir, config, command_line=" ".join(sys.argv))
        acceptance = run_acceptance_checks(run_dir, emit_debug=False, balanced_threshold=config.selection.balanced_threshold, strong_threshold=config.selection.strong_threshold)
        write_md(os.path.join(run_dir, "acceptance_check.md"), acceptance)
        return 0

    config = _build_config(args)
    os.makedirs(args.output_root, exist_ok=True)
    run_dir = make_run_dir(args.output_root)
    _ensure_debug_dir(run_dir, config.emit_debug_artifacts)

    derived_all: Dict = {}
    arrival_hashes_all: Dict = {}

    if args.cmd in {"paper_suite", "profile_search", "main", "fig4"}:
        derived = _write_selection_outputs(run_dir, config)
        _combine_derived(derived_all, {"N{}".format(config.N): {str(k): v for k, v in derived.items()}})

    if args.cmd in {"paper_suite", "main"}:
        derived, arrival_hashes = _write_main_outputs(run_dir, config, emit_debug=config.emit_debug_artifacts)
        _combine_derived(derived_all, {"N{}".format(config.N): {str(k): v for k, v in derived.items()}})
        arrival_hashes_all["main"] = {scheme: {str(seed): h for seed, h in seed_map.items()} for scheme, seed_map in arrival_hashes.items()}

    if args.cmd in {"paper_suite", "fig4"}:
        derived, arrival_hashes = _write_fig4_outputs(run_dir, config)
        for n, seed_map in derived.items():
            _combine_derived(derived_all, {"N{}".format(n): {str(k): v for k, v in seed_map.items()}})
        arrival_hashes_all["fig4"] = {
            scheme: {str(n): {str(seed): h for seed, h in seed_map.items()} for n, seed_map in n_map.items()}
            for scheme, n_map in arrival_hashes.items()
        }

    _write_run_params(run_dir, config, derived_all, arrival_trace_hashes=arrival_hashes_all)
    _write_manifest(run_dir, config, command_line=" ".join(sys.argv))

    acceptance = run_acceptance_checks(
        run_dir,
        emit_debug=config.emit_debug_artifacts,
        balanced_threshold=config.selection.balanced_threshold,
        strong_threshold=config.selection.strong_threshold,
    )
    write_md(os.path.join(run_dir, "acceptance_check.md"), acceptance)
    return 0


if __name__ == "__main__":
    sys.exit(main())
