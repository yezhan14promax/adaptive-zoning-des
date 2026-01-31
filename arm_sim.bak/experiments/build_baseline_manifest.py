import argparse
import csv
import json
import os
from datetime import datetime


def _load_csv(path):
    if not os.path.isfile(path):
        return None
    with open(path, "r", newline="") as handle:
        return list(csv.DictReader(handle))


def _parse_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_list(value):
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    return [v.strip() for v in text.split(",") if v.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_dir")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    baseline_dir = os.path.abspath(args.baseline_dir)
    manifest = {
        "baseline_dir": baseline_dir,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "files_present": sorted(os.listdir(baseline_dir)) if os.path.isdir(baseline_dir) else [],
        "inferred": {},
    }

    summary_main = _load_csv(os.path.join(baseline_dir, "summary_main_and_ablations.csv"))
    summary_fig4 = _load_csv(os.path.join(baseline_dir, "summary_fig4_scaledload.csv"))
    summary_profiles = _load_csv(os.path.join(baseline_dir, "summary_s2_profiles.csv"))
    summary_selected = _load_csv(os.path.join(baseline_dir, "summary_s2_selected.csv"))

    manifest["summary_paths"] = {
        "summary_main_and_ablations.csv": bool(summary_main),
        "summary_fig4_scaledload.csv": bool(summary_fig4),
        "summary_s2_profiles.csv": bool(summary_profiles),
        "summary_s2_selected.csv": bool(summary_selected),
    }

    if summary_main:
        baseline_main = next(
            (row for row in summary_main if row.get("scheme") == "S1" and row.get("profile") == "baseline"),
            summary_main[0],
        )
        manifest["main"] = {
            "N": _parse_float(baseline_main.get("N")),
            "load_multiplier": _parse_float(baseline_main.get("load_multiplier")),
            "hotspot_p95_s": _parse_float(baseline_main.get("hotspot_p95_mean_s")),
            "hotspot_mean_s": _parse_float(baseline_main.get("hotspot_mean_s")),
            "fixed_hotspot_overload_ratio_q1000": _parse_float(
                baseline_main.get("overload_ratio_q1000")
                or baseline_main.get("fixed_hotspot_overload_ratio_q1000")
            ),
            "seeds": _parse_list(baseline_main.get("seeds")),
        }
        manifest["inferred"]["standard_load"] = {"value": 2.0, "source": "tag std2p0", "inferred": True}
        manifest["inferred"]["arrival_mode"] = {"value": "unknown", "source": "no config in folder", "inferred": True}
        manifest["inferred"]["hotspot_window"] = {"value": "30-80s (likely)", "inferred": True}
    else:
        manifest["main"] = {}

    if summary_profiles:
        manifest["s2_profiles"] = [
            {
                "tag": row.get("tag"),
                "profile": row.get("profile"),
                "load_multiplier": _parse_float(row.get("load_multiplier")),
                "hotspot_p95_s": _parse_float(row.get("hotspot_p95_mean_s")),
                "hotspot_mean_s": _parse_float(row.get("hotspot_mean_s")),
                "fixed_hotspot_overload_ratio_q1000": _parse_float(
                    row.get("overload_ratio_q1000")
                    or row.get("fixed_hotspot_overload_ratio_q1000")
                ),
                "migrated_weight_total": _parse_float(row.get("migrated_weight_total")),
            }
            for row in summary_profiles
        ]
    else:
        manifest["s2_profiles"] = []

    if summary_selected:
        manifest["s2_selected"] = summary_selected
    else:
        manifest["s2_selected"] = []

    if summary_fig4:
        fig4_entries = []
        for row in summary_fig4:
            fig4_entries.append(
                {
                    "N": _parse_float(row.get("N")),
                    "scheme": row.get("scheme"),
                    "profile": row.get("profile"),
                    "load_multiplier": _parse_float(row.get("load_multiplier")),
                    "fixed_hotspot_overload_ratio_q1000": _parse_float(
                        row.get("overload_ratio_q1000")
                        or row.get("fixed_hotspot_overload_ratio_q1000")
                    ),
                    "hotspot_p95_s": _parse_float(row.get("hotspot_p95_mean_s")),
                }
            )
        manifest["fig4"] = fig4_entries
    else:
        manifest["fig4"] = []

    output_path = args.output or os.path.join(baseline_dir, "baseline_manifest.json")
    with open(output_path, "w", newline="") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    print(output_path)


if __name__ == "__main__":
    main()
