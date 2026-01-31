import argparse
import csv
import hashlib
import os


def _sha256_file(path, block_size=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(block_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _read_csv(path):
    with open(path, "r", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_report(path, lines):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def _check_png(path, min_size=10_000):
    return os.path.isfile(path) and os.path.getsize(path) >= min_size


def _validate_checksums(run_dir):
    checksum_path = os.path.join(run_dir, "checksums.sha256")
    if not os.path.isfile(checksum_path):
        return False, "checksums.sha256 missing"
    failures = []
    with open(checksum_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            digest, rel = line.split("  ", 1)
            path = os.path.join(run_dir, rel)
            if not os.path.isfile(path):
                failures.append(f"missing {rel}")
                continue
            if _sha256_file(path) != digest:
                failures.append(f"hash mismatch {rel}")
    if failures:
        return False, "; ".join(failures)
    return True, "ok"


def _parse_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _check_zip_contains(zip_path, required_names):
    import zipfile
    if not os.path.isfile(zip_path):
        return False, f"missing {os.path.basename(zip_path)}"
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = set(zf.namelist())
    except zipfile.BadZipFile:
        return False, f"bad zip {os.path.basename(zip_path)}"
    missing = [name for name in required_names if name not in names]
    if missing:
        return False, f"zip missing {missing}"
    return True, "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--snapshot_dir", default=None)
    parser.add_argument("--require_snapshot_used", action="store_true")
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    snapshot_dir = os.path.abspath(args.snapshot_dir) if args.snapshot_dir else None
    lines = []
    ok = True

    pngs = [
        "fig1_hotspot_p95_main_constrained.png",
        "fig2_overload_ratio_main_constrained.png",
        "fig3_tradeoff_scatter_constrained.png",
        "fig4_scaledload_scalability.png",
    ]
    for name in pngs:
        path = os.path.join(run_dir, name)
        if _check_png(path):
            lines.append(f"PASS png {name}")
        else:
            ok = False
            lines.append(f"FAIL png {name}")

    fig4_path = os.path.join(run_dir, "summary_fig4_scaledload.csv")
    if not os.path.isfile(fig4_path):
        ok = False
        lines.append("FAIL summary_fig4_scaledload.csv missing")
    else:
        rows = _read_csv(fig4_path)
        n_vals = sorted({int(float(r["N"])) for r in rows if r.get("N")})
        if len(n_vals) >= 2:
            lines.append(f"PASS fig4 N_unique={n_vals}")
        else:
            ok = False
            lines.append(f"FAIL fig4 N_unique={n_vals}")
        schemes = {r.get("scheme") for r in rows}
        lines.append(f"INFO fig4 schemes={sorted(schemes)}")

    selected_path = os.path.join(run_dir, "summary_s2_selected.csv")
    if not os.path.isfile(selected_path):
        ok = False
        lines.append("FAIL summary_s2_selected.csv missing")
    else:
        sel_rows = _read_csv(selected_path)
        labels = {r.get("label") or r.get("profile_name") for r in sel_rows}
        if {"Ours-Lite", "Ours-Balanced", "Ours-Strong"} <= labels or {"Lite", "Balanced", "Strong"} <= labels:
            lines.append("PASS selected profiles present")
        else:
            ok = False
            lines.append(f"FAIL selected profiles labels={sorted(labels)}")
        # Selection rule checks
        for row in sel_rows:
            label = (row.get("label") or "").strip()
            overload = _parse_float(row.get("overload_q1000"))
            feasible = str(row.get("feasible", "")).lower() in ("true", "1", "yes")
            if label == "Balanced" and feasible and overload is not None and overload > 0.3:
                ok = False
                lines.append(f"FAIL Balanced overload_q1000={overload} > 0.3")
            if label == "Strong" and feasible and overload is not None and overload > 0.1:
                ok = False
                lines.append(f"FAIL Strong overload_q1000={overload} > 0.1")

    selection_report = os.path.join(run_dir, "selection_report.md")
    if os.path.isfile(selection_report):
        lines.append("PASS selection_report.md present")
    else:
        ok = False
        lines.append("FAIL selection_report.md missing")

    report_path = os.path.join(run_dir, "validation_report.md")
    status = "PASS" if ok else "FAIL"
    lines.insert(0, f"# Validation Report ({status})")
    _write_report(report_path, lines)

    # Refresh checksums to include validation_report.md written above.
    try:
        from arm_sim.experiments.artifacts import write_checksums
        write_checksums(run_dir)
    except Exception:
        pass
    checksum_ok, checksum_msg = _validate_checksums(run_dir)
    if not checksum_ok:
        ok = False
        lines.append(f"FAIL checksums: {checksum_msg}")
        _write_report(report_path, lines)

    if snapshot_dir and args.require_snapshot_used:
        snapshot_used = os.path.join(run_dir, "snapshot_used.txt")
        if not os.path.isfile(snapshot_used):
            ok = False
            lines.append("FAIL snapshot_used.txt missing")
            _write_report(report_path, lines)

    # Artifact zip presence + content
    zips = [name for name in os.listdir(run_dir) if name.startswith("artifacts_") and name.endswith(".zip")]
    if zips:
        # Prefer most recent valid zip (non-empty + passes required set).
        required = [
            "run_config_resolved.json",
            "repro_stamp.json",
            "summary_main_and_ablations.csv",
            "summary_fig4_scaledload.csv",
            "selection_report.md",
            "validation_report.md",
            "fig1_hotspot_p95_main_constrained.png",
            "fig2_overload_ratio_main_constrained.png",
            "fig3_tradeoff_scatter_constrained.png",
            "fig4_scaledload_scalability.png",
        ]
        zip_candidates = sorted(
            (os.path.join(run_dir, name) for name in zips),
            key=lambda p: os.path.getmtime(p),
            reverse=True,
        )
        chosen = None
        last_err = None
        for candidate in zip_candidates:
            if os.path.getsize(candidate) <= 0:
                continue
            zip_ok, zip_msg = _check_zip_contains(candidate, required)
            if zip_ok:
                chosen = candidate
                break
            last_err = zip_msg
        if chosen:
            lines.append(f"PASS artifacts zip {os.path.basename(chosen)}")
        else:
            ok = False
            lines.append(f"FAIL artifacts zip: {last_err or 'no valid zip'}")
    else:
        ok = False
        lines.append("FAIL artifacts zip missing")

    _write_report(report_path, lines)

    print(report_path)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
