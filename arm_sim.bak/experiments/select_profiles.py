import argparse
import csv
import os

from arm_sim.experiments.artifacts import write_selection_report
def _parse_float(value):
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _resolve_metric_key(headers):
    # Single source of truth for constraints: fixed_hotspot_overload_ratio_q1000 if present,
    # otherwise overload_ratio_q1000. No other fallback to avoid drift.
    if "fixed_hotspot_overload_ratio_q1000" in headers:
        return "fixed_hotspot_overload_ratio_q1000"
    if "overload_ratio_q1000" in headers:
        return "overload_ratio_q1000"
    return None


def _select_min_cost(rows, metric_key, cost_key, threshold=None):
    best_feasible = None
    best_infeasible = None
    best_infeasible_violation = None
    feasible_count = 0
    missing_metric = 0
    missing_cost = 0
    for row in rows:
        metric = _parse_float(row.get(metric_key))
        cost = _parse_float(row.get(cost_key))
        if metric is None:
            missing_metric += 1
            continue
        if cost is None:
            missing_cost += 1
            continue
        if threshold is not None and metric > threshold:
            violation = metric - threshold
            if best_infeasible is None:
                best_infeasible = row
                best_infeasible_violation = violation
            elif violation < best_infeasible_violation:
                best_infeasible = row
                best_infeasible_violation = violation
            elif violation == best_infeasible_violation and cost < _parse_float(best_infeasible.get(cost_key)):
                best_infeasible = row
            continue
        feasible_count += 1
        if best_feasible is None or cost < _parse_float(best_feasible.get(cost_key)):
            best_feasible = row
    if best_feasible is not None:
        return best_feasible, True, 0.0, feasible_count, missing_metric, missing_cost
    return best_infeasible, False, best_infeasible_violation, feasible_count, missing_metric, missing_cost


def select_profiles(candidates_rows, metric_key=None, cost_key="migrated_weight_total"):
    if not candidates_rows:
        raise RuntimeError("No candidate rows provided for profile selection.")
    headers = candidates_rows[0].keys()
    metric_key = metric_key or _resolve_metric_key(headers)
    if metric_key is None:
        raise RuntimeError("Missing overload metric column: fixed_hotspot_overload_ratio_q1000 or overload_ratio_q1000")
    rules_text = (
        "- Lite: minimal cost without overload constraint\n"
        "- Balanced: minimal cost with overload_q1000 <= 0.3\n"
        "- Strong: minimal cost with overload_q1000 <= 0.1\n"
        "- If constraint infeasible: choose closest to threshold (min violation), then min cost\n"
    )

    selected = []
    lite, lite_feasible, lite_violation, lite_count, lite_missing_metric, lite_missing_cost = _select_min_cost(
        candidates_rows, metric_key, cost_key, threshold=None
    )
    balanced, balanced_feasible, balanced_violation, balanced_count, balanced_missing_metric, balanced_missing_cost = _select_min_cost(
        candidates_rows, metric_key, cost_key, threshold=0.3
    )
    strong, strong_feasible, strong_violation, strong_count, strong_missing_metric, strong_missing_cost = _select_min_cost(
        candidates_rows, metric_key, cost_key, threshold=0.1
    )

    def _emit(label, row, threshold, feasible, violation):
        if row is None:
            return None
        return {
            "label": label,
            "constraint_q1000": threshold if threshold is not None else "",
            "chosen_profile": row.get("tag") or row.get("profile"),
            "profile": row.get("profile"),
            "scheme": row.get("scheme"),
            "N": row.get("N"),
            "load_multiplier": row.get("load_multiplier"),
            "cost": _parse_float(row.get(cost_key)),
            "overload_q1000": _parse_float(row.get(metric_key)),
            "feasible": bool(feasible),
            "slack_or_violation": 0.0 if feasible else (violation or 0.0),
            "fallback_used": "" if feasible else "closest_to_threshold_then_min_cost",
        }

    selected.append(_emit("Lite", lite, None, lite_feasible, lite_violation))
    selected.append(_emit("Balanced", balanced, 0.3, balanced_feasible, balanced_violation))
    selected.append(_emit("Strong", strong, 0.1, strong_feasible, strong_violation))

    selected = [row for row in selected if row]
    total_candidates = len(candidates_rows)
    counts = {
        "candidates_total": len(candidates_rows),
        "lite_candidates": len(candidates_rows),
        "balanced_candidates": balanced_count,
        "strong_candidates": strong_count,
        "lite_missing_metric": lite_missing_metric,
        "lite_missing_cost": lite_missing_cost,
        "balanced_missing_metric": balanced_missing_metric,
        "balanced_missing_cost": balanced_missing_cost,
        "strong_missing_metric": strong_missing_metric,
        "strong_missing_cost": strong_missing_cost,
    }
    balanced_threshold_fail = max(
        0,
        total_candidates - balanced_count - balanced_missing_metric - balanced_missing_cost,
    )
    strong_threshold_fail = max(
        0,
        total_candidates - strong_count - strong_missing_metric - strong_missing_cost,
    )
    rules_text += (
        f"\nCandidates: total={counts['candidates_total']} "
        f"balanced_feasible={counts['balanced_candidates']} "
        f"strong_feasible={counts['strong_candidates']}"
    )
    rules_text += (
        f"\nThreshold failures: balanced={balanced_threshold_fail} strong={strong_threshold_fail}"
    )
    rules_text += (
        f"\nMissing metrics: lite={counts['lite_missing_metric']} "
        f"balanced={counts['balanced_missing_metric']} "
        f"strong={counts['strong_missing_metric']}; "
        f"missing costs: lite={counts['lite_missing_cost']} "
        f"balanced={counts['balanced_missing_cost']} "
        f"strong={counts['strong_missing_cost']}"
    )
    # Hard assertions: if feasible set exists, selected must satisfy threshold.
    def _assert_feasible(label, threshold, feasible_count, chosen):
        if threshold is None:
            return
        if feasible_count > 0 and chosen is not None:
            metric = _parse_float(chosen.get(metric_key))
            if metric is None or metric > threshold:
                raise RuntimeError(
                    f"{label} selection violated threshold {threshold} with metric={metric}; "
                    f"feasible_count={feasible_count}. Check candidate rows."
                )

    _assert_feasible("Balanced", 0.3, balanced_count, balanced)
    _assert_feasible("Strong", 0.1, strong_count, strong)

    return selected, rules_text, metric_key, cost_key


def write_selection(out_dir, candidates_path, output_path):
    with open(candidates_path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    selected, rules_text, metric_key, cost_key = select_profiles(rows)
    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0].keys()))
        writer.writeheader()
        for row in selected:
            writer.writerow(row)
    report_path = write_selection_report(out_dir, rows, selected, rules_text)
    return output_path, report_path, metric_key, cost_key


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--output_csv", default=None)
    args = parser.parse_args()

    output_csv = args.output_csv or os.path.join(args.out_dir, "summary_s2_selected.csv")
    write_selection(args.out_dir, args.candidates_csv, output_csv)


if __name__ == "__main__":
    main()
