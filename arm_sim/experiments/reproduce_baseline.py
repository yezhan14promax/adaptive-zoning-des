import argparse
import json
import os
import shutil
import time

from arm_sim.experiments import run_hotspot
from arm_sim.experiments.repro_stamp import write_repro_stamp
from arm_sim.experiments.output_utils import normalize_output_root


def _load_manifest(path):
    with open(path, "r") as handle:
        return json.load(handle)


def _timestamped_dir(base_dir):
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(base_dir, f"repro_{stamp}")
    os.makedirs(path, exist_ok=True)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--arrival_mode", default=None)
    args = parser.parse_args()

    manifest = _load_manifest(args.manifest)
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    output_dir = normalize_output_root(base_dir, args.output_root, prefix="figure")
    figures_dir_main = os.path.join(output_dir, "figure_z16_noc")
    figures_dir_fig4 = os.path.join(output_dir, "figures")
    repro_dir = _timestamped_dir(os.path.join(output_dir, "reproduced"))

    seeds = manifest.get("main", {}).get("seeds") or [0, 1, 2]
    arrival_mode = args.arrival_mode or manifest.get("inferred", {}).get("arrival_mode", {}).get("value") or "robot_emit"

    run_hotspot.run_main_pipeline(
        output_dir,
        figures_dir_main,
        seeds,
        profile_version="baseline_repro",
        arrival_mode=arrival_mode,
        deterministic_hotspots=True,
        auto_profile_search=False,
    )
    run_hotspot.run_fig4_calibrate(
        output_dir,
        figures_dir_fig4,
        seeds,
        profile_version="baseline_repro",
        arrival_mode=arrival_mode,
        exclude_small_n=True,
        force_rerun=True,
    )
    calib_path = os.path.join(figures_dir_fig4, "fig4_calibration.csv")
    run_hotspot.run_fig4_sweep(
        output_dir,
        figures_dir_fig4,
        seeds,
        profile_version="baseline_repro",
        calibration_path=calib_path,
        enforce_n16_match=False,
        arrival_mode=arrival_mode,
        deterministic_hotspots=True,
        exclude_small_n=True,
        force_rerun=True,
    )

    for name in (
        "summary_main_and_ablations.csv",
        "summary_fig4_scaledload.csv",
        "summary_s2_profiles.csv",
        "summary_s2_selected.csv",
    ):
        src = os.path.join(figures_dir_main if "main" in name else figures_dir_fig4, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(repro_dir, name))

    for name in (
        "fig1_hotspot_p95_main_constrained.png",
        "fig2_overload_ratio_main_constrained.png",
        "fig3_tradeoff_scatter_constrained.png",
        "fig4_scaledload_scalability.png",
    ):
        src = os.path.join(figures_dir_main if "fig4" not in name else figures_dir_fig4, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(repro_dir, name))

    stamp_path = write_repro_stamp(
        repro_dir,
        {
            "manifest": os.path.abspath(args.manifest),
            "arrival_mode": arrival_mode,
            "seeds": seeds,
        },
    )
    print(f"Reproduction outputs: {repro_dir}")
    print(f"Repro stamp: {stamp_path}")


if __name__ == "__main__":
    main()
