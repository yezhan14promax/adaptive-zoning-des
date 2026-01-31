import argparse
import csv
import os
from collections import defaultdict


def _load_summary(path):
    if not os.path.isfile(path):
        raise RuntimeError(f"Missing summary CSV: {path}")
    with open(path, "r", newline="") as handle:
        return list(csv.DictReader(handle))


def _parse_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _key_for_main(row):
    return (row.get("scheme"), row.get("profile"))


def _key_for_fig4(row):
    return (row.get("N"), row.get("scheme"), row.get("profile"))


def _diff_rows(base_rows, new_rows, keys, fields):
    base_map = {keys(row): row for row in base_rows}
    new_map = {keys(row): row for row in new_rows}
    diffs = []
    for key in sorted(set(base_map) | set(new_map)):
        base = base_map.get(key, {})
        new = new_map.get(key, {})
        for field in fields:
            b_val = _parse_float(base.get(field))
            n_val = _parse_float(new.get(field))
            if b_val is None or n_val is None:
                continue
            diffs.append((abs(n_val - b_val), key, field, b_val, n_val))
    diffs.sort(reverse=True, key=lambda item: item[0])
    return diffs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_dir")
    parser.add_argument("new_dir")
    parser.add_argument("--report", default=None)
    parser.add_argument("--tol_overload", type=float, default=0.05)
    parser.add_argument("--tol_latency_s", type=float, default=0.5)
    args = parser.parse_args()

    baseline_main = _load_summary(os.path.join(args.baseline_dir, "summary_main_and_ablations.csv"))
    new_main = _load_summary(os.path.join(args.new_dir, "summary_main_and_ablations.csv"))
    baseline_fig4 = _load_summary(os.path.join(args.baseline_dir, "summary_fig4_scaledload.csv"))
    new_fig4 = _load_summary(os.path.join(args.new_dir, "summary_fig4_scaledload.csv"))

    main_fields = [
        "hotspot_mean_s",
        "hotspot_p95_s",
        "fixed_hotspot_overload_ratio_q500",
        "fixed_hotspot_overload_ratio_q1000",
        "fixed_hotspot_overload_ratio_q1500",
        "migrated_weight_total",
        "cost_per_request",
        "reconfig_actions_per_s",
    ]
    fig4_fields = list(main_fields)

    main_diffs = _diff_rows(baseline_main, new_main, _key_for_main, main_fields)
    fig4_diffs = _diff_rows(baseline_fig4, new_fig4, _key_for_fig4, fig4_fields)

    lines = []
    lines.append("Top 5 diffs (Main):")
    for diff in main_diffs[:5]:
        delta, key, field, b_val, n_val = diff
        lines.append(f"{key} {field}: baseline={b_val} new={n_val} diff={delta}")

    lines.append("Top 5 diffs (Fig4):")
    for diff in fig4_diffs[:5]:
        delta, key, field, b_val, n_val = diff
        lines.append(f"{key} {field}: baseline={b_val} new={n_val} diff={delta}")

    report_text = "\n".join(lines)
    print(report_text)

    if args.report:
        with open(args.report, "w", newline="") as handle:
            handle.write(report_text)

    violations = []
    for diff in main_diffs:
        delta, key, field, _, _ = diff
        if field in ("hotspot_p95_s", "hotspot_mean_s") and delta > args.tol_latency_s:
            violations.append((key, field, delta))
        if "overload_ratio_q1000" in field and delta > args.tol_overload:
            violations.append((key, field, delta))
    for diff in fig4_diffs:
        delta, key, field, _, _ = diff
        if field in ("hotspot_p95_s", "hotspot_mean_s") and delta > args.tol_latency_s:
            violations.append((key, field, delta))
        if "overload_ratio_q1000" in field and delta > args.tol_overload:
            violations.append((key, field, delta))

    if violations:
        raise SystemExit(f"Comparison failed tolerance: {violations[:5]}")


if __name__ == "__main__":
    main()
