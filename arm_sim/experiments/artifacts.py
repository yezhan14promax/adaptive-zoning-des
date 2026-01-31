import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import zipfile


def _sha256_file(path, block_size=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(block_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _safe_run(cmd):
    try:
        result = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True, text=True)
        return result.strip()
    except Exception:
        return None


def write_run_config_resolved(out_dir, args=None, derived_dict=None):
    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args) if args is not None else {},
        "derived": derived_dict or {},
    }
    path = os.path.join(out_dir, "run_config_resolved.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path


def write_repro_stamp(out_dir, args=None):
    git_commit = _safe_run("git rev-parse HEAD")
    git_status = _safe_run("git status --porcelain")
    pip_freeze = _safe_run("python -m pip freeze")
    pip_hash = hashlib.sha256(pip_freeze.encode("utf-8")).hexdigest() if pip_freeze else None
    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hostname": platform.node(),
        "os": platform.platform(),
        "python_version": sys.version,
        "pip_freeze_sha256": pip_hash,
        "git_commit": git_commit,
        "git_dirty": bool(git_status),
        "command_line": sys.argv,
        "seeds": getattr(args, "seeds", None) if args is not None else None,
        "repo_root": os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
    }
    path = os.path.join(out_dir, "repro_stamp.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path


def write_checksums(out_dir):
    lines = []
    for root, _, files in os.walk(out_dir):
        for name in files:
            if not name.lower().endswith((".csv", ".png", ".json", ".md")):
                continue
            path = os.path.join(root, name)
            rel = os.path.relpath(path, out_dir)
            digest = _sha256_file(path)
            lines.append(f"{digest}  {rel}")
    path = os.path.join(out_dir, "checksums.sha256")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(sorted(lines)))
    return path


def write_selection_report(out_dir, candidates_rows, selected_rows, rules_text):
    lines = ["# Selection Report\n", "\n## Rules\n", rules_text, "\n\n## Candidates\n"]
    if not candidates_rows:
        lines.append("(no candidates)\n")
    else:
        headers = list(candidates_rows[0].keys())
        lines.append("| " + " | ".join(headers) + " |\n")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in candidates_rows:
            lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |\n")
    lines.append("\n## Selected\n")
    if not selected_rows:
        lines.append("(no selections)\n")
    else:
        headers = list(selected_rows[0].keys())
        lines.append("| " + " | ".join(headers) + " |\n")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in selected_rows:
            lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |\n")
    path = os.path.join(out_dir, "selection_report.md")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("".join(lines))
    return path


def pack_artifacts(out_dir):
    run_id = time.strftime("%Y%m%d_%H%M%S")
    zip_path = os.path.join(out_dir, f"artifacts_{run_id}.zip")
    required_names = [
        "run_config_resolved.json",
        "run_config.json",
        "repro_stamp.json",
        "checksums.sha256",
        "selection_report.md",
        "validation_report.md",
        "summary_main_and_ablations.csv",
        "summary_fig4_scaledload.csv",
        "summary_s2_selected.csv",
        "fig1_hotspot_p95_main_constrained.png",
        "fig2_overload_ratio_main_constrained.png",
        "fig3_tradeoff_scatter_constrained.png",
        "fig4_scaledload_scalability.png",
        "fig1_hotspot_p95_main_constrained.pdf",
        "fig2_overload_ratio_main_constrained.pdf",
        "fig3_tradeoff_scatter_constrained.pdf",
        "fig4_scaledload_scalability.pdf",
    ]
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in required_names:
            path = os.path.join(out_dir, name)
            if not os.path.isfile(path):
                continue
            rel = os.path.relpath(path, out_dir)
            zf.write(path, rel)
    return zip_path
