from __future__ import annotations

import json
import os
import tempfile
from typing import Dict, List, Tuple
import re

from .output_layout import DEFAULT_FILES, SNAPSHOT_ONLY_FILES
from .selection import select_profiles
from .snapshot.replay import replot_from_snapshot


def _load_csv(path: str) -> List[Dict]:
    import csv

    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _check_output_structure(run_dir: str, emit_debug: bool) -> Tuple[bool, str]:
    if not os.path.basename(run_dir).startswith("figure_"):
        return False, "output dir name does not start with figure_"

    allowed = set(DEFAULT_FILES)
    if emit_debug:
        allowed.add("debug")
    actual = set(os.listdir(run_dir))
    # acceptance_check.md is generated after checks run
    if "acceptance_check.md" not in actual:
        allowed.discard("acceptance_check.md")
    unexpected = sorted(actual - allowed)
    missing = sorted(allowed - actual)
    if unexpected:
        return False, f"unexpected files: {unexpected}"
    if missing:
        return False, f"missing required files: {missing}"
    return True, ""


def _check_fairness(summary_main: str) -> Tuple[bool, str]:
    if not os.path.exists(summary_main):
        return False, "summary_main_and_ablations.csv missing"
    rows = _load_csv(summary_main)
    if not rows:
        return False, "summary_main_and_ablations.csv empty"
    global_vals = {float(r.get("global_generated", "nan")) for r in rows}
    hotspot_vals = {float(r.get("fixed_hotspot_generated", "nan")) for r in rows}
    if len(global_vals) > 1:
        return False, f"global_generated mismatch: {sorted(global_vals)}"
    if len(hotspot_vals) > 1:
        return False, f"fixed_hotspot_generated mismatch: {sorted(hotspot_vals)}"
    return True, ""


def _check_cohort(run_params_path: str, summary_main: str) -> Tuple[bool, str]:
    if not os.path.exists(run_params_path):
        return False, "run_params.json missing"
    with open(run_params_path, encoding="utf-8") as f:
        params = json.load(f)
    if params.get("cohort_time_rule") != "arrival_time":
        return False, f"cohort_time_rule={params.get('cohort_time_rule')}"
    if params.get("completion_ratio_clamped") is True:
        return False, "completion_ratio was clamped"

    if not os.path.exists(summary_main):
        return False, "summary_main_and_ablations.csv missing"
    rows = _load_csv(summary_main)
    max_ratio = max(float(r.get("completion_ratio_max", "0")) for r in rows)
    if max_ratio > 1.0:
        return False, f"completion_ratio_max>1 ({max_ratio:.4f})"
    return True, ""


def _check_metric_consistency(summary_main: str, summary_fig4: str) -> Tuple[bool, str]:
    if not os.path.exists(summary_main) or not os.path.exists(summary_fig4):
        return False, "summary_main_and_ablations.csv or summary_fig4_scaledload.csv missing"
    rows_main = _load_csv(summary_main)
    rows_fig4 = _load_csv(summary_fig4)
    if not rows_main or not rows_fig4:
        return False, "summary file missing"
    method_main = {r.get("overload_method") for r in rows_main}
    method_fig4 = {r.get("overload_method") for r in rows_fig4}
    if method_main != method_fig4:
        return False, f"overload_method mismatch {method_main} vs {method_fig4}"
    if method_main != {"metrics.compute_overload_ratios"}:
        return False, f"overload_method unexpected {method_main}"
    return True, ""


def _check_required_columns(summary_main: str, summary_fig4: str) -> Tuple[bool, str]:
    rows_main = _load_csv(summary_main)
    rows_fig4 = _load_csv(summary_fig4)
    if not rows_main:
        return False, "summary_main_and_ablations.csv empty"
    if not rows_fig4:
        return False, "summary_fig4_scaledload.csv empty"

    required_main = [
        "p95_latency_hotspot",
        "mean_latency_hotspot",
        "overload_q500",
        "overload_q1000",
        "overload_q1500",
        "reconfig_cost_total",
    ]
    required_fig4 = [
        "p95_latency_hotspot",
        "overload_q500",
        "overload_q1000",
        "overload_q1500",
    ]
    missing_main = [c for c in required_main if c not in rows_main[0]]
    missing_fig4 = [c for c in required_fig4 if c not in rows_fig4[0]]
    if missing_main or missing_fig4:
        return False, f"missing columns main={missing_main} fig4={missing_fig4}"
    return True, ""


def _parse_manifest_numbers(manifest_path: str) -> Dict[str, Dict[str, float]]:
    if not os.path.exists(manifest_path):
        return {}
    with open(manifest_path, encoding="utf-8") as f:
        text = f.read().splitlines()
    pattern = re.compile(r"^-\s+(?P<scheme>[^:]+):\s+cost=(?P<cost>[-+e0-9.]+),\s+reconfig_cost_total=(?P<reconfig>[-+e0-9.]+),\s+overload_q500=(?P<q500>[-+e0-9.]+),\s+overload_q1000=(?P<q1000>[-+e0-9.]+),\s+overload_q1500=(?P<q1500>[-+e0-9.]+),\s+p95_latency_hotspot=(?P<p95>[-+e0-9.]+),\s+mean_latency_hotspot=(?P<mean>[-+e0-9.]+)$")
    parsed: Dict[str, Dict[str, float]] = {}
    for line in text:
        match = pattern.match(line.strip())
        if match:
            scheme = match.group("scheme")
            parsed[scheme] = {
                "cost": float(match.group("cost")),
                "reconfig_cost_total": float(match.group("reconfig")),
                "overload_q500": float(match.group("q500")),
                "overload_q1000": float(match.group("q1000")),
                "overload_q1500": float(match.group("q1500")),
                "p95_latency_hotspot": float(match.group("p95")),
                "mean_latency_hotspot": float(match.group("mean")),
            }
    return parsed


def _check_manifest_consistency(manifest_path: str, summary_main: str) -> Tuple[bool, str]:
    rows = _load_csv(summary_main)
    if not rows:
        return False, "summary_main_and_ablations.csv empty"
    parsed = _parse_manifest_numbers(manifest_path)
    if not parsed:
        return False, "manifest missing numeric scheme lines"

    tolerance = 1e-6
    missing = []
    mismatches = []
    for row in rows:
        scheme = row.get("scheme")
        if scheme not in parsed:
            missing.append(scheme)
            continue
        expected = parsed[scheme]
        for key in [
            "cost",
            "reconfig_cost_total",
            "overload_q500",
            "overload_q1000",
            "overload_q1500",
            "p95_latency_hotspot",
            "mean_latency_hotspot",
        ]:
            actual = float(row.get(key))
            if abs(actual - expected[key]) > tolerance:
                mismatches.append(f"{scheme}:{key} actual={actual} manifest={expected[key]}")
    if missing or mismatches:
        return False, f"manifest mismatch missing={missing} mismatches={mismatches}"
    return True, ""


def _check_selection(summary_profiles: str, summary_selected: str, balanced_threshold: float, strong_threshold: float) -> Tuple[bool, str]:
    if not os.path.exists(summary_profiles) or not os.path.exists(summary_selected):
        return False, "summary_s2_profiles.csv or summary_s2_selected.csv missing"
    expected, _ = select_profiles(summary_profiles, balanced_threshold, strong_threshold)
    expected_map = {row["scheme"]: row["profile_id"] for row in expected}
    selected_rows = _load_csv(summary_selected)
    if not selected_rows:
        return False, "summary_s2_selected.csv empty"
    actual_map = {row["scheme"]: row["profile_id"] for row in selected_rows}
    if expected_map != actual_map:
        return False, f"selected mismatch expected {expected_map} actual {actual_map}"
    return True, ""


def _check_fig4_nonempty(summary_fig4: str, fig4_path: str) -> Tuple[bool, str]:
    if not os.path.exists(summary_fig4):
        return False, "summary_fig4_scaledload.csv missing"
    if not os.path.exists(fig4_path):
        return False, "fig4 file missing"
    rows = _load_csv(summary_fig4)
    if not rows:
        return False, "summary_fig4_scaledload.csv empty"
    Ns = sorted({int(r["N"]) for r in rows})
    schemes = {r["scheme"] for r in rows}
    by_scheme = {s: {int(r["N"]) for r in rows if r["scheme"] == s} for s in schemes}
    missing_map = {s: [n for n in Ns if n not in by_scheme[s]] for s in schemes}
    points_map = {s: len(by_scheme[s]) for s in schemes}
    drawable = [s for s, nset in by_scheme.items() if len(nset) >= 2]
    if not drawable:
        return False, f"no drawable curves; Ns={Ns} schemes={schemes} missing={missing_map} points={points_map}"
    return True, f"points={points_map} missing={missing_map}"


def _check_arrival_hashes(run_params_path: str) -> Tuple[bool, str]:
    if not os.path.exists(run_params_path):
        return False, "run_params.json missing"
    with open(run_params_path, encoding="utf-8") as f:
        params = json.load(f)
    hashes = params.get("arrival_trace_hashes", {})
    if not hashes:
        return False, "arrival_trace_hashes missing"

    issues = []
    main_hashes = hashes.get("main", {})
    if main_hashes:
        schemes = list(main_hashes.keys())
        seeds = set()
        for scheme in schemes:
            seeds |= set(main_hashes.get(scheme, {}).keys())
        for seed in seeds:
            vals = {main_hashes.get(scheme, {}).get(seed) for scheme in schemes}
            if len(vals) > 1:
                issues.append(f"main seed={seed} hashes={main_hashes}")

    fig4_hashes = hashes.get("fig4", {})
    if fig4_hashes:
        schemes = list(fig4_hashes.keys())
        Ns = set()
        for scheme in schemes:
            Ns |= set(fig4_hashes.get(scheme, {}).keys())
        for n in Ns:
            seeds = set()
            for scheme in schemes:
                seeds |= set(fig4_hashes.get(scheme, {}).get(n, {}).keys())
            for seed in seeds:
                vals = {fig4_hashes.get(scheme, {}).get(n, {}).get(seed) for scheme in schemes}
                if len(vals) > 1:
                    issues.append(f"fig4 N={n} seed={seed} hashes={fig4_hashes}")

    if issues:
        return False, f"arrival hash mismatch: {issues}"
    return True, ""


def _check_decoupling(run_params_path: str) -> Tuple[bool, str]:
    if not os.path.exists(run_params_path):
        return False, "run_params.json missing"
    with open(run_params_path, encoding="utf-8") as f:
        params = json.load(f)
    rng_streams = params.get("rng_streams", {})
    seeds = {rng_streams.get("arrival_seed"), rng_streams.get("service_seed"), rng_streams.get("policy_seed")}
    if len(seeds) != 3:
        return False, "RNG streams not independent"
    if params.get("module_imports_ok") is not True:
        return False, "module imports failed"
    return True, ""


def _check_snapshot_replay(run_dir: str) -> Tuple[bool, str]:
    required = [
        os.path.join(run_dir, "summary_main_and_ablations.csv"),
        os.path.join(run_dir, "summary_fig4_scaledload.csv"),
        os.path.join(run_dir, "summary_s2_selected.csv"),
    ]
    if any(not os.path.exists(p) for p in required):
        return False, "snapshot inputs missing"
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "figure_snapshot_test")
            os.makedirs(output_dir, exist_ok=False)
            replot_from_snapshot(run_dir, output_dir)
            used_path = os.path.join(output_dir, "snapshot_used.txt")
            if not os.path.exists(used_path):
                return False, "snapshot_used.txt not generated"
    except Exception as exc:
        return False, f"snapshot_plot failed: {exc}"
    return True, ""


def run_acceptance_checks(run_dir: str, emit_debug: bool, balanced_threshold: float, strong_threshold: float) -> str:
    summary_main = os.path.join(run_dir, "summary_main_and_ablations.csv")
    summary_fig4 = os.path.join(run_dir, "summary_fig4_scaledload.csv")
    summary_profiles = os.path.join(run_dir, "summary_s2_profiles.csv")
    summary_selected = os.path.join(run_dir, "summary_s2_selected.csv")
    run_params_path = os.path.join(run_dir, "run_params.json")
    fig4_path = os.path.join(run_dir, "fig4_scaledload_scalability.png")
    manifest_path = os.path.join(run_dir, "manifest.md")

    checks = [
        ("1) Output dir structure", _check_output_structure(run_dir, emit_debug)),
        ("2) Fairness", _check_fairness(summary_main)),
        ("3) Cohort semantics", _check_cohort(run_params_path, summary_main)),
        ("4) Metric consistency", _check_metric_consistency(summary_main, summary_fig4)),
        ("5) Required columns", _check_required_columns(summary_main, summary_fig4)),
        ("6) Selection rules", _check_selection(summary_profiles, summary_selected, balanced_threshold, strong_threshold)),
        ("7) Fig4 non-empty", _check_fig4_nonempty(summary_fig4, fig4_path)),
        ("8) Decoupling", _check_decoupling(run_params_path)),
        ("9) Manifest consistency", _check_manifest_consistency(manifest_path, summary_main)),
        ("10) Arrival hash consistency", _check_arrival_hashes(run_params_path)),
        ("11) Snapshot replay", _check_snapshot_replay(run_dir)),
    ]

    lines = ["# Acceptance Check\n"]
    for label, (passed, reason) in checks:
        status = "PASS" if passed else "FAIL"
        if reason:
            lines.append(f"- {label}: {status} ({reason})\n")
        else:
            lines.append(f"- {label}: {status}\n")
    return "".join(lines)
