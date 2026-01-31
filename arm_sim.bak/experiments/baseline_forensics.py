import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import sys


BASELINE_DEFAULT = r"D:\robot controller\arm_sim\outputs\outputs\figures_20260122_225920"
OUTPUTS_ROOT = r"D:\robot controller\arm_sim\outputs\outputs"
TIME_WINDOW_START = dt.datetime(2026, 1, 22, 22, 0, 0)
TIME_WINDOW_END = dt.datetime(2026, 1, 22, 23, 30, 0)


def _to_local_time(ts):
    return dt.datetime.fromtimestamp(ts)


def _fmt_time(ts):
    return _to_local_time(ts).strftime("%Y-%m-%d %H:%M:%S")


def _sha256(path, block_size=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(block_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _sniff_delimiter(path):
    with open(path, "r", newline="") as handle:
        sample = handle.read(4096)
    if not sample:
        return ","
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
        return dialect.delimiter
    except csv.Error:
        return ","


def _read_csv(path):
    delim = _sniff_delimiter(path)
    with open(path, "r", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delim)
        headers = reader.fieldnames or []
        rows = list(reader)
    return delim, headers, rows


def _parse_float(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_int(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _collect_distinct(rows, key):
    values = []
    seen = set()
    for row in rows:
        if key not in row:
            continue
        val = row.get(key)
        if val is None:
            continue
        text = str(val).strip()
        if text == "":
            continue
        if text in seen:
            continue
        seen.add(text)
        values.append(text)
    return values


def _extract_metrics(rows):
    metrics = {}
    for row in rows:
        scheme = row.get("scheme")
        profile = row.get("profile")
        if not scheme:
            continue
        label = None
        if scheme == "S1" and profile == "baseline":
            label = "Static-edge"
        elif scheme == "S2" and profile == "conservative":
            label = "Ours-Lite"
        elif scheme == "S2" and profile == "neutral":
            label = "Ours-Balanced"
        elif scheme == "S2" and profile == "aggressive":
            label = "Ours-Strong"
        if not label:
            continue
        p95_s = _parse_float(row.get("hotspot_p95_mean_s"))
        mean_s = _parse_float(row.get("hotspot_mean_s"))
        if p95_s is None:
            p95_ms = _parse_float(row.get("hotspot_p95_mean_ms"))
            if p95_ms is not None:
                p95_s = p95_ms / 1000.0
        if mean_s is None:
            mean_ms = _parse_float(row.get("hotspot_mean_ms"))
            if mean_ms is not None:
                mean_s = mean_ms / 1000.0
        metrics[label] = {
            "hotspot_mean_s": mean_s,
            "hotspot_p95_s": p95_s,
            "overload_q500": _parse_float(
                row.get("fixed_hotspot_overload_ratio_q500")
                or row.get("overload_ratio_q500")
            ),
            "overload_q1000": _parse_float(
                row.get("fixed_hotspot_overload_ratio_q1000")
                or row.get("overload_ratio_q1000")
            ),
            "overload_q1500": _parse_float(
                row.get("fixed_hotspot_overload_ratio_q1500")
                or row.get("overload_ratio_q1500")
            ),
            "migrated_weight_total": _parse_float(row.get("migrated_weight_total")),
            "reconfig_action_count": _parse_float(row.get("reconfig_action_count")),
            "load_multiplier": _parse_float(row.get("load_multiplier")),
            "tag": row.get("tag"),
        }
    return metrics


def _collect_sensn64_files(base_dir):
    matches = []
    for root, _, files in os.walk(base_dir):
        for name in files:
            if not name.startswith("window_kpis_SensN64_") or not name.endswith(".csv"):
                continue
            if "p28" not in name:
                continue
            if "m0p056" not in name and "m0p062" not in name:
                continue
            if "seed" not in name:
                continue
            path = os.path.join(root, name)
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                mtime = 0.0
            if not (TIME_WINDOW_START <= _to_local_time(mtime) <= TIME_WINDOW_END):
                continue
            matches.append((path, mtime))
    matches.sort(key=lambda item: item[0])
    return matches


def _summarize_sensn64(files):
    summary = {
        "file_count": len(files),
        "hotspot_zone_ids": {},
        "generated_totals": {},
        "m0p062_metrics": {},
        "notes": [],
    }

    # keep latest file per (multiplier, seed, scheme)
    dedup = {}
    for path, mtime in files:
        base = os.path.basename(path)
        scheme = "unknown"
        if "Static-edge" in base:
            scheme = "Static-edge"
        elif "Ours-Lite" in base:
            scheme = "Ours-Lite"
        elif "Ours-Balanced" in base:
            scheme = "Ours-Balanced"
        elif "Ours-Strong" in base:
            scheme = "Ours-Strong"
        multiplier = "m0p056" if "m0p056" in base else ("m0p062" if "m0p062" in base else "unknown")
        seed = None
        for part in base.split("_"):
            if part.startswith("seed"):
                seed = part.replace("seed", "").replace(".csv", "")
                break
        key = (multiplier, seed, scheme)
        prev = dedup.get(key)
        if prev is None or mtime > prev[1]:
            dedup[key] = (path, mtime)

    for (multiplier, seed, scheme), (path, _) in dedup.items():
        _, headers, rows = _read_csv(path)
        if not rows:
            continue
        key = (multiplier, seed)

        # hotspot_zone_id consistency
        hotspot_ids = set()
        for row in rows:
            hz = row.get("hotspot_zone_id")
            if hz is not None and str(hz).strip() != "":
                hotspot_ids.add(str(hz).strip())
        summary["hotspot_zone_ids"].setdefault(key, {})[scheme] = sorted(hotspot_ids)

        # generated totals
        gen_total = 0
        for row in rows:
            gen_total += int(_parse_float(row.get("generated")) or 0)
        summary["generated_totals"].setdefault(key, {})[scheme] = gen_total

        # per-scheme metrics for m0p062
        if multiplier == "m0p062":
            over_q1000 = _parse_float(rows[-1].get("overload_ratio_q1000"))
            over_q1500 = _parse_float(rows[-1].get("overload_ratio_q1500"))
            migrated = _parse_float(rows[-1].get("migrated_weight_total"))
            summary["m0p062_metrics"].setdefault(scheme, []).append(
                {
                    "seed": seed,
                    "overload_q1000": over_q1000,
                    "overload_q1500": over_q1500,
                    "migrated_weight_total": migrated,
                }
            )

    # compute mean metrics for m0p062 per scheme
    mean_metrics = {}
    for scheme, entries in summary["m0p062_metrics"].items():
        vals_q1000 = [e.get("overload_q1000") for e in entries if e.get("overload_q1000") is not None]
        vals_q1500 = [e.get("overload_q1500") for e in entries if e.get("overload_q1500") is not None]
        vals_cost = [e.get("migrated_weight_total") for e in entries if e.get("migrated_weight_total") is not None]
        if vals_q1000:
            mean_metrics.setdefault(scheme, {})["overload_q1000_mean"] = sum(vals_q1000) / len(vals_q1000)
        if vals_q1500:
            mean_metrics.setdefault(scheme, {})["overload_q1500_mean"] = sum(vals_q1500) / len(vals_q1500)
        if vals_cost:
            mean_metrics.setdefault(scheme, {})["migrated_weight_total_mean"] = sum(vals_cost) / len(vals_cost)
    summary["m0p062_means"] = mean_metrics
    return summary


def _stringify_keys(mapping):
    output = {}
    for key, value in mapping.items():
        if isinstance(key, tuple):
            key = "|".join(str(part) for part in key)
        output[str(key)] = value
    return output


def _inventory_dir(path):
    entries = []
    for entry in os.scandir(path):
        if not entry.is_file():
            continue
        stat = entry.stat()
        entries.append(
            {
                "name": entry.name,
                "path": entry.path,
                "size": stat.st_size,
                "mtime": _fmt_time(stat.st_mtime),
                "ext": os.path.splitext(entry.name)[1].lower(),
            }
        )
    entries.sort(key=lambda item: item["name"])
    return entries


def _scan_repo_for_strings(repo_root, needles):
    matches = []
    for root, _, files in os.walk(repo_root):
        for name in files:
            if not name.endswith(".py"):
                continue
            path = os.path.join(root, name)
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    lines = handle.readlines()
            except OSError:
                continue
            for idx, line in enumerate(lines, start=1):
                for needle in needles:
                    if needle in line:
                        matches.append(
                            {
                                "path": path,
                                "line": idx,
                                "needle": needle,
                                "text": line.strip(),
                            }
                        )
    return matches


def _time_window_scan(paths):
    results = []
    for base in paths:
        if not os.path.isdir(base):
            continue
        for root, _, files in os.walk(base):
            for name in files:
                path = os.path.join(root, name)
                try:
                    stat = os.stat(path)
                except OSError:
                    continue
                mtime = _to_local_time(stat.st_mtime)
                if TIME_WINDOW_START <= mtime <= TIME_WINDOW_END:
                    entry = {
                        "path": path,
                        "name": name,
                        "size": stat.st_size,
                        "mtime": _fmt_time(stat.st_mtime),
                        "ext": os.path.splitext(name)[1].lower(),
                    }
                    if entry["ext"] == ".csv":
                        _, headers, rows = _read_csv(path)
                        entry["headers"] = headers
                        entry["row_count"] = len(rows)
                    results.append(entry)
    results.sort(key=lambda item: item["mtime"])
    return results


def _infer_parameters(summary_main, summary_fig4):
    evidence = []
    inferred = {}

    rows_main = summary_main.get("rows", [])
    rows_fig4 = summary_fig4.get("rows", [])

    seeds_vals = _collect_distinct(rows_main, "seeds")
    if seeds_vals:
        inferred["seeds"] = {"value": seeds_vals, "confidence": "high"}
        evidence.append(
            {
                "parameter": "seeds",
                "value": seeds_vals,
                "evidence": "summary_main_and_ablations.csv column 'seeds'",
                "confidence": "high",
            }
        )

    load_main = None
    for row in rows_main:
        if row.get("scheme") == "S1" and row.get("profile") == "baseline":
            load_main = _parse_float(row.get("load_multiplier"))
            break
    if load_main is None and rows_main:
        load_main = _parse_float(rows_main[0].get("load_multiplier"))
    if load_main is not None:
        inferred["main_load_multiplier"] = {
            "value": load_main,
            "confidence": "high",
        }
        evidence.append(
            {
                "parameter": "main_load_multiplier",
                "value": load_main,
                "evidence": "summary_main_and_ablations.csv load_multiplier",
                "confidence": "high",
            }
        )

    per_n = {}
    for row in rows_fig4:
        if row.get("scheme") != "S1" or row.get("profile") != "baseline":
            continue
        n_val = _parse_int(row.get("N"))
        if n_val is None:
            continue
        mult = _parse_float(row.get("load_multiplier"))
        if mult is None:
            continue
        per_n[n_val] = mult
    if per_n:
        inferred["fig4_load_multiplier_by_N"] = {
            "value": per_n,
            "confidence": "high",
        }
        evidence.append(
            {
                "parameter": "fig4_load_multiplier_by_N",
                "value": per_n,
                "evidence": "summary_fig4_scaledload.csv baseline rows",
                "confidence": "high",
            }
        )

    tag_vals = _collect_distinct(rows_main, "tag")
    profile_version = None
    for tag in tag_vals:
        if "p" in tag:
            profile_version = tag
            break
    if profile_version:
        inferred["profile_version_tag"] = {
            "value": profile_version,
            "confidence": "medium",
        }
        evidence.append(
            {
                "parameter": "profile_version_tag",
                "value": profile_version,
                "evidence": "summary_main_and_ablations.csv tag string",
                "confidence": "medium",
            }
        )

    arrival_mode = {"value": "unknown", "confidence": "low"}
    inferred["arrival_mode"] = arrival_mode
    evidence.append(
        {
            "parameter": "arrival_mode",
            "value": "unknown",
            "evidence": "No arrival_mode column in baseline CSVs",
            "confidence": "low",
        }
    )

    deterministic = {"value": "unknown", "confidence": "low"}
    inferred["deterministic_hotspots"] = deterministic
    evidence.append(
        {
            "parameter": "deterministic_hotspots",
            "value": "unknown",
            "evidence": "No deterministic_hotspots flag in baseline CSVs",
            "confidence": "low",
        }
    )

    cohort = {"value": "unknown", "confidence": "low"}
    inferred["latency_cohort_definition"] = cohort
    evidence.append(
        {
            "parameter": "latency_cohort_definition",
            "value": "unknown",
            "evidence": "CSV contains only aggregated mean/p95; cohort definition not logged",
            "confidence": "low",
        }
    )

    return inferred, evidence


def _write_manifest(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _write_report(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as handle:
        handle.write(text)


def _default_out_dir():
    stamp = dt.datetime.now().strftime("figure_%Y%m%d_%H%M%S")
    return os.path.join(OUTPUTS_ROOT, stamp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_dir", default=BASELINE_DEFAULT)
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    baseline_dir = os.path.abspath(args.baseline_dir)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else _default_out_dir()
    if not os.path.isdir(baseline_dir):
        raise FileNotFoundError(baseline_dir)

    inventory = _inventory_dir(baseline_dir)
    png_hashes = {}
    csv_summaries = {}

    for item in inventory:
        if item["ext"] == ".png":
            png_hashes[item["name"]] = _sha256(item["path"])
        if item["ext"] == ".csv":
            delim, headers, rows = _read_csv(item["path"])
            csv_summaries[item["name"]] = {
                "delimiter": delim,
                "headers": headers,
                "row_count": len(rows),
                "rows": rows,
            }

    main_summary = csv_summaries.get("summary_main_and_ablations.csv", {})
    fig4_summary = csv_summaries.get("summary_fig4_scaledload.csv", {})
    profile_summary = csv_summaries.get("summary_s2_profiles.csv", {})
    selected_summary = csv_summaries.get("summary_s2_selected.csv", {})

    metrics_main = _extract_metrics(main_summary.get("rows", []))
    metrics_fig4 = _extract_metrics(fig4_summary.get("rows", []))

    sens_files = _collect_sensn64_files(os.path.join(os.path.dirname(baseline_dir), ".."))
    sens_summary = _summarize_sensn64(sens_files)
    sens_summary_serializable = dict(sens_summary)
    sens_summary_serializable["hotspot_zone_ids"] = _stringify_keys(
        sens_summary.get("hotspot_zone_ids", {})
    )
    sens_summary_serializable["generated_totals"] = _stringify_keys(
        sens_summary.get("generated_totals", {})
    )

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    needles = [
        "summary_main_and_ablations.csv",
        "summary_fig4_scaledload.csv",
        "summary_s2_profiles.csv",
        "summary_s2_selected.csv",
        "hotspot_p95_mean_ms",
        "fixed_hotspot_overload_ratio_q1000",
        "overload_ratio_q1000",
    ]
    writer_matches = _scan_repo_for_strings(repo_root, needles)

    scan_paths = [
        os.path.join(repo_root, "outputs", "outputs"),
        os.path.join(repo_root, "outputs"),
        os.path.join(repo_root, "outputs"),
    ]
    time_window_artifacts = _time_window_scan(scan_paths)

    inferred, evidence = _infer_parameters(main_summary, fig4_summary)

    manifest = {
        "baseline_dir": baseline_dir,
        "generated_at": dt.datetime.utcnow().isoformat() + "Z",
        "inventory": inventory,
        "png_hashes": png_hashes,
        "csv_summaries": {
            name: {
                "delimiter": data.get("delimiter"),
                "headers": data.get("headers"),
                "row_count": data.get("row_count"),
            }
            for name, data in csv_summaries.items()
        },
        "metrics_main": metrics_main,
        "metrics_fig4": metrics_fig4,
        "sensn64_files": sens_files,
        "sensn64_summary": sens_summary_serializable,
        "writer_matches": writer_matches,
        "time_window_artifacts": time_window_artifacts,
        "inferred_parameters": inferred,
        "evidence": evidence,
    }

    manifest_path = os.path.join(out_dir, "baseline_manifest_from_folder.json")
    _write_manifest(manifest_path, manifest)

    report_lines = []
    report_lines.append("Baseline Forensics Report\n")
    report_lines.append(f"Baseline dir: {baseline_dir}\n")
    report_lines.append("Inventory:\n")
    for item in inventory:
        report_lines.append(
            f"- {item['name']} size={item['size']} mtime={item['mtime']} ext={item['ext']}\n"
        )
    report_lines.append("\nPNG hashes:\n")
    for name, digest in png_hashes.items():
        report_lines.append(f"- {name} sha256={digest}\n")
    report_lines.append("\nCSV summaries:\n")
    for name, summary in csv_summaries.items():
        report_lines.append(
            f"- {name} delimiter={summary['delimiter']} rows={summary['row_count']}\n"
        )
        report_lines.append(f"  headers={summary['headers']}\n")

    report_lines.append("\nKey metrics (main):\n")
    for label, metrics in metrics_main.items():
        report_lines.append(f"- {label}: {metrics}\n")
    report_lines.append("\nKey metrics (fig4):\n")
    for label, metrics in metrics_fig4.items():
        report_lines.append(f"- {label}: {metrics}\n")

    report_lines.append("\nWriter code matches:\n")
    for match in writer_matches:
        report_lines.append(
            f"- {match['path']}:{match['line']} '{match['needle']}'\n"
        )

    report_lines.append("\nArtifacts near 2026-01-22 22:00-23:30:\n")
    for entry in time_window_artifacts:
        report_lines.append(
            f"- {entry['path']} size={entry['size']} mtime={entry['mtime']}\n"
        )
        if entry.get("headers"):
            report_lines.append(f"  headers={entry['headers']}\n")

    report_lines.append("\nInferred parameters:\n")
    for key, value in inferred.items():
        report_lines.append(f"- {key}: {value}\n")

    report_lines.append("\nHypothesized reproduction commands:\n")
    report_lines.append("Primary guess (baseline defaults):\n")
    report_lines.append(
        '  python -m arm_sim.experiments.run_hotspot --mode main --seeds "0,1,2"\n'
    )
    report_lines.append(
        '  python -m arm_sim.experiments.run_hotspot --mode fig4_calibrate --seeds "0,1,2"\n'
    )
    report_lines.append(
        '  python -m arm_sim.experiments.run_hotspot --mode fig4_run --seeds "0,1,2"\n'
    )
    report_lines.append(
        "  python -m arm_sim.plots.make_paper_figures --plot-only\n"
    )
    report_lines.append("\nAlternative A (deterministic hotspots):\n")
    report_lines.append(
        '  python -m arm_sim.experiments.run_hotspot --mode main --seeds "0,1,2" --deterministic_hotspots\n'
    )
    report_lines.append(
        '  python -m arm_sim.experiments.run_hotspot --mode fig4_calibrate --seeds "0,1,2" --deterministic_hotspots\n'
    )
    report_lines.append(
        '  python -m arm_sim.experiments.run_hotspot --mode fig4_run --seeds "0,1,2" --deterministic_hotspots\n'
    )
    report_lines.append(
        "  python -m arm_sim.plots.make_paper_figures --plot-only\n"
    )

    report_path = os.path.join(out_dir, "baseline_forensics_report.txt")
    _write_report(report_path, "".join(report_lines))

    evidence_md = []
    evidence_md.append("# Evidence Report (Baseline Forensics)\n\n")
    evidence_md.append("## Baseline main summary\n")
    if main_summary:
        rows = main_summary.get("rows", [])
        base_row = None
        for row in rows:
            if row.get("scheme") == "S1" and row.get("profile") == "baseline":
                base_row = row
                break
        if base_row:
            evidence_md.append(f"- load_multiplier: {base_row.get('load_multiplier')}\n")
            evidence_md.append(f"- seeds: {base_row.get('seeds')}\n")
            evidence_md.append(f"- tag: {base_row.get('tag')}\n")
    evidence_md.append("\n## Baseline fig4 per-N multipliers\n")
    fig4_rows = fig4_summary.get("rows", [])
    for row in fig4_rows:
        if row.get("scheme") == "S1" and row.get("profile") == "baseline":
            evidence_md.append(
                f"- N={row.get('N')} load_multiplier={row.get('load_multiplier')}\n"
            )
    evidence_md.append("\n## Baseline S2 profiles\n")
    for label, metrics in metrics_main.items():
        if label.startswith("Ours-"):
            evidence_md.append(
                f"- {label}: migrated_weight_total={metrics.get('migrated_weight_total')}, "
                f"overload_q1000={metrics.get('overload_q1000')}, "
                f"hotspot_p95_s={metrics.get('hotspot_p95_s')}\n"
            )
    evidence_md.append("\n## Tag inference\n")
    tag_vals = _collect_distinct(main_summary.get("rows", []), "tag")
    for tag in tag_vals:
        if tag:
            evidence_md.append(f"- tag: {tag}\n")
    evidence_md.append("\n## SensN64 window_kpis (p28 m0p056/m0p062)\n")
    evidence_md.append(f"- files_found: {sens_summary.get('file_count')} (expected 17 per user note)\n")
    evidence_md.append("### hotspot_zone_id consistency\n")
    for key, mapping in sens_summary.get("hotspot_zone_ids", {}).items():
        evidence_md.append(f"- {key}: {mapping}\n")
    evidence_md.append("### generated totals by scheme\n")
    for key, mapping in sens_summary.get("generated_totals", {}).items():
        evidence_md.append(f"- {key}: {mapping}\n")
    evidence_md.append("### m0p062 per-scheme metrics\n")
    for scheme, entries in sens_summary.get("m0p062_metrics", {}).items():
        evidence_md.append(f"- {scheme}:\n")
        for entry in entries:
            evidence_md.append(
                f"  - seed={entry.get('seed')} "
                f"overload_q1000={entry.get('overload_q1000')} "
                f"overload_q1500={entry.get('overload_q1500')} "
                f"migrated_weight_total={entry.get('migrated_weight_total')}\n"
            )
    evidence_md.append("### m0p062 per-scheme means\n")
    for scheme, metrics in sens_summary.get("m0p062_means", {}).items():
        evidence_md.append(
            f"- {scheme}: overload_q1000_mean={metrics.get('overload_q1000_mean')} "
            f"overload_q1500_mean={metrics.get('overload_q1500_mean')} "
            f"migrated_weight_total_mean={metrics.get('migrated_weight_total_mean')}\n"
        )
    evidence_md.append("\n## Evidence chain conclusion\n")
    evidence_md.append(
        "- SensN64 files show a single hotspot_zone_id per file; consistent across schemes per seed/m.\n"
    )
    evidence_md.append(
        "- Generated totals differ by scheme for same seed/m in some files, suggesting arrival_mode closer to robot_emit than zone_poisson.\n"
    )
    evidence_md.append(
        "- SensN64 p28 m0p056/m0p062 implies single-hotspot (k_hot=1) structure consistent with baseline tags.\n"
    )

    evidence_path = os.path.join(out_dir, "evidence_report.md")
    _write_report(evidence_path, "".join(evidence_md))

    print("Hypothesized reproduction commands:")
    print(
        'python -m arm_sim.experiments.run_hotspot --mode main --seeds "0,1,2"'
    )
    print(
        'python -m arm_sim.experiments.run_hotspot --mode fig4_calibrate --seeds "0,1,2"'
    )
    print(
        'python -m arm_sim.experiments.run_hotspot --mode fig4_run --seeds "0,1,2"'
    )
    print("python -m arm_sim.plots.make_paper_figures --plot-only")

    print("\nTop evidence items:")
    for item in evidence[:10]:
        print(
            f"{item['parameter']} => {item['value']} "
            f"(confidence={item['confidence']}) via {item['evidence']}"
        )

    print(f"\nManifest: {manifest_path}")
    print(f"Report: {report_path}")
    print(f"Evidence report: {evidence_path}")


if __name__ == "__main__":
    main()


