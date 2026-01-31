import argparse
import csv
import math
import os


def _load_csv(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path, "r", newline="") as handle:
        return list(csv.DictReader(handle))


def _parse_float(value):
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(value)
    except ValueError:
        return None


def _key(row):
    return (
        row.get("scheme"),
        row.get("profile"),
        row.get("N"),
    )


def _compare_sets(baseline_rows, new_rows, fields, label, tolerances):
    baseline = { _key(r): r for r in baseline_rows }
    new = { _key(r): r for r in new_rows }
    diffs = []
    for key, base_row in baseline.items():
        if key not in new:
            diffs.append((label, key, "missing", None))
            continue
        new_row = new[key]
        for field in fields:
            base_val = _parse_float(base_row.get(field))
            new_val = _parse_float(new_row.get(field))
            if base_val is None or new_val is None:
                continue
            if field in tolerances:
                tol = tolerances[field]
                if isinstance(tol, tuple):
                    mode, thresh = tol
                else:
                    mode, thresh = "abs", tol
                if mode == "rel":
                    denom = base_val if base_val != 0 else 1.0
                    delta = abs(new_val - base_val) / abs(denom)
                else:
                    delta = abs(new_val - base_val)
                if delta > thresh:
                    diffs.append((label, key, field, delta))
            else:
                if new_val != base_val:
                    diffs.append((label, key, field, abs(new_val - base_val)))
    return diffs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot_dir", required=True)
    parser.add_argument("--new_dir", required=True)
    parser.add_argument("--tolerance_p95_rel", type=float, default=0.05)
    parser.add_argument("--tolerance_overload_abs", type=float, default=0.03)
    parser.add_argument("--tolerance_cost_rel", type=float, default=0.10)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    baseline_main = _load_csv(os.path.join(args.snapshot_dir, "summary_main_and_ablations.csv"))
    baseline_fig4 = _load_csv(os.path.join(args.snapshot_dir, "summary_fig4_scaledload.csv"))
    new_main = _load_csv(os.path.join(args.new_dir, "summary_main_and_ablations.csv"))
    new_fig4 = _load_csv(os.path.join(args.new_dir, "summary_fig4_scaledload.csv"))

    tolerances = {
        "hotspot_p95_mean_s": ("rel", args.tolerance_p95_rel),
        "hotspot_p95_mean_ms": ("rel", args.tolerance_p95_rel),
        "fixed_hotspot_overload_ratio_q1000": ("abs", args.tolerance_overload_abs),
        "overload_ratio_q1000": ("abs", args.tolerance_overload_abs),
        "migrated_weight_total": ("rel", args.tolerance_cost_rel),
    }

    fields = [
        "hotspot_p95_mean_s",
        "hotspot_p95_mean_ms",
        "fixed_hotspot_overload_ratio_q1000",
        "overload_ratio_q1000",
        "migrated_weight_total",
    ]

    diffs = []
    diffs.extend(_compare_sets(baseline_main, new_main, fields, "main", tolerances))
    diffs.extend(_compare_sets(baseline_fig4, new_fig4, fields, "fig4", tolerances))

    report_lines = []
    report_lines.append("# Compare to Snapshot Report\n")
    report_lines.append(f"snapshot_dir: {args.snapshot_dir}\n")
    report_lines.append(f"new_dir: {args.new_dir}\n\n")
    if diffs:
        diffs.sort(key=lambda item: (item[0], str(item[1]), str(item[2])))
        report_lines.append("## Differences beyond tolerance\n")
        for label, key, field, delta in diffs[:100]:
            report_lines.append(f"- {label} {key} {field} delta={delta}\n")
    else:
        report_lines.append("## Status\n- All compared fields within tolerance.\n")

    # Heuristic cause hints
    report_lines.append("\n## Likely causes (heuristic)\n")
    load_mismatch = False
    for base_row in baseline_main:
        if base_row.get("scheme") == "S1" and base_row.get("profile") == "baseline":
            base_load = _parse_float(base_row.get("load_multiplier"))
            break
    else:
        base_load = None
    new_load = None
    for new_row in new_main:
        if new_row.get("scheme") == "S1" and new_row.get("profile") == "baseline":
            new_load = _parse_float(new_row.get("load_multiplier"))
            break
    if base_load is not None and new_load is not None:
        if abs(base_load - new_load) > 1.0e-4:
            load_mismatch = True
            report_lines.append(f"- load_multiplier differs: baseline={base_load} new={new_load}\n")
    if not load_mismatch:
        report_lines.append("- load_multiplier appears consistent.\n")
    report_lines.append("- Check arrival_mode / deterministic_hotspots if metrics diverge with same multipliers.\n")
    report_lines.append("- Check profile overrides (S2) if migrated_weight_total or overload diverge.\n")

    report_path = args.report or os.path.join(args.new_dir, "compare_report.md")
    with open(report_path, "w", newline="") as handle:
        handle.write("".join(report_lines))

    print(report_path)
    if diffs:
        raise SystemExit(1)
    print("Snapshot comparison within tolerance.")


if __name__ == "__main__":
    main()
