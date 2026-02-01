from __future__ import annotations

import csv
from typing import Dict, List, Tuple


def _load_profiles(path: str) -> List[Dict]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            parsed = {
                **row,
                "cost": float(row["cost"]),
                "p95_latency_hotspot": float(row["p95_latency_hotspot"]),
            }
            for key in ["overload_q500", "overload_q1000", "overload_q1500"]:
                if key in row:
                    parsed[key] = float(row[key])
            rows.append(parsed)
        return rows


def _select_by_threshold(rows: List[Dict], overload_key: str, threshold: float) -> Tuple[Dict, bool, str]:
    feasible = [r for r in rows if r[overload_key] <= threshold]
    if feasible:
        chosen = min(feasible, key=lambda r: (r["cost"], r[overload_key]))
        return chosen, False, "feasible"

    # infeasible fallback
    sorted_rows = sorted(rows, key=lambda r: (abs(r[overload_key] - threshold), r[overload_key], r["cost"]))
    chosen = sorted_rows[0]
    return chosen, True, "infeasible_fallback"


def select_profiles(
    summary_path: str,
    balanced_threshold: float,
    strong_threshold: float,
    selection_overload_q: int,
) -> Tuple[List[Dict], str]:
    rows = _load_profiles(summary_path)
    if not rows:
        raise ValueError("summary_s2_profiles.csv is empty")

    overload_key = f"overload_q{int(selection_overload_q)}"
    if overload_key not in rows[0]:
        raise ValueError(f"selection_overload_q column missing: {overload_key}")

    lite = min(rows, key=lambda r: (r["cost"], r[overload_key]))
    balanced, balanced_infeasible, balanced_reason = _select_by_threshold(rows, overload_key, balanced_threshold)
    strong, strong_infeasible, strong_reason = _select_by_threshold(rows, overload_key, strong_threshold)

    selected = [
        {"scheme": "Lite", **lite},
        {"scheme": "Balanced", **balanced},
        {"scheme": "Strong", **strong},
    ]
    for row in selected:
        row["selection_overload_q"] = int(selection_overload_q)

    report_lines = []
    report_lines.append("# Selection Report\n")
    report_lines.append("## Thresholds\n")
    report_lines.append(f"- selection_overload_q: {selection_overload_q}\n")
    report_lines.append(f"- Balanced threshold (overload_q{selection_overload_q}): {balanced_threshold}\n")
    report_lines.append(f"- Strong threshold (overload_q{selection_overload_q}): {strong_threshold}\n")
    report_lines.append("Selection constraint uses overload_q{sel_q} only; q500/q1500 are shown for sensitivity.\n".format(sel_q=selection_overload_q))
    report_lines.append("\n")
    report_lines.append("## Candidate Table (top 10 by cost)\n")
    report_lines.append("|profile_id|cost|overload_q500|overload_q1000|overload_q1500|p95_latency_hotspot|budget_ratio|weight_scale|cooldown_s|\n")
    report_lines.append("|---|---|---|---|---|---|---|---|---|\n")
    for row in sorted(rows, key=lambda r: r["cost"])[:10]:
        report_lines.append(
            f"|{row['profile_id']}|{row['cost']:.4f}|{row.get('overload_q500', float('nan')):.4f}|"
            f"{row.get('overload_q1000', float('nan')):.4f}|{row.get('overload_q1500', float('nan')):.4f}|"
            f"{row['p95_latency_hotspot']:.4f}|{row['budget_ratio']}|{row['weight_scale']}|{row['cooldown_s']}|\n"
        )

    report_lines.append("\n## Decisions\n")
    report_lines.append(f"- Lite: profile {lite['profile_id']} (min cost).\n")
    report_lines.append(
        f"- Balanced: profile {balanced['profile_id']} ({balanced_reason}, infeasible={balanced_infeasible}).\n"
    )
    report_lines.append(
        f"- Strong: profile {strong['profile_id']} ({strong_reason}, infeasible={strong_infeasible}).\n"
    )

    report_lines.append("\n## Selected Profiles\n")
    for row in selected:
        report_lines.append(
            f"- {row['scheme']}: profile {row['profile_id']} | cost={row['cost']:.4f} | "
            f"{overload_key}={row[overload_key]:.4f} | p95_latency_hotspot={row['p95_latency_hotspot']:.4f}\n"
        )

    report = "".join(report_lines)
    return selected, report
