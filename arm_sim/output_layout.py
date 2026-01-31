from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Iterable, List


DEFAULT_FILES = {
    "run_params.json",
    "manifest.md",
    "acceptance_check.md",
    "summary_s2_profiles.csv",
    "summary_s2_selected.csv",
    "summary_main_and_ablations.csv",
    "summary_fig4_scaledload.csv",
    "selection_report.md",
    "fig1_hotspot_p95_main_constrained.png",
    "fig2_overload_ratio_main_constrained.png",
    "fig3_tradeoff_scatter_constrained.png",
    "fig4_scaledload_scalability.png",
}

SNAPSHOT_ONLY_FILES = {"snapshot_used.txt"}

DEBUG_FILES_PREFIX = {
    "arrival_trace.csv",
}


def make_run_dir(output_root: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_root, f"figure_{timestamp}")
    os.makedirs(run_dir, exist_ok=False)
    return run_dir


def write_json(path: str, payload) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def write_md(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def write_csv(path: str, rows: Iterable[dict], fieldnames: List[str]) -> None:
    import csv

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def allowed_files(emit_debug: bool, include_snapshot: bool) -> List[str]:
    allowed = set(DEFAULT_FILES)
    if include_snapshot:
        allowed |= SNAPSHOT_ONLY_FILES
    if emit_debug:
        allowed.add("debug")
    return sorted(allowed)


def assert_only_whitelist(run_dir: str, emit_debug: bool, include_snapshot: bool) -> List[str]:
    allowed = set(allowed_files(emit_debug, include_snapshot))
    actual = set(os.listdir(run_dir))
    unexpected = sorted(actual - allowed)
    missing = sorted(allowed - actual)
    return unexpected, missing
