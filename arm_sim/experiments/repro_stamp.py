import json
import os
import subprocess
import sys
import time


def _run_git(cmd, base_dir):
    try:
        result = subprocess.run(
            cmd,
            cwd=base_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def write_repro_stamp(output_dir, run_config, cmdline=None, snapshot_root=None, artifacts=None):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    stamp = {
        "timestamp": int(time.time()),
        "cmdline": cmdline or " ".join(sys.argv),
        "python_version": sys.version,
        "git_commit": _run_git(["git", "rev-parse", "HEAD"], base_dir),
        "git_status": _run_git(["git", "status", "--porcelain"], base_dir),
        "git_diff_stat": _run_git(["git", "diff", "--stat"], base_dir),
        "run_config": run_config,
    }
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "repro_stamp.json")
    with open(path, "w", newline="") as handle:
        json.dump(stamp, handle, indent=2, sort_keys=True)
    if snapshot_root:
        artifacts = artifacts or {}
        snap_name = time.strftime("%Y%m%d_%H%M%S")
        snap_dir = os.path.join(snapshot_root, f"config_{snap_name}")
        os.makedirs(snap_dir, exist_ok=True)
        with open(os.path.join(snap_dir, "repro_stamp.json"), "w", newline="") as handle:
            json.dump(stamp, handle, indent=2, sort_keys=True)
        for label, src in artifacts.items():
            if src and os.path.isfile(src):
                dst = os.path.join(snap_dir, os.path.basename(src))
                try:
                    with open(src, "rb") as r, open(dst, "wb") as w:
                        w.write(r.read())
                except Exception:
                    pass
        stamp["snapshot_dir"] = snap_dir
    return path
