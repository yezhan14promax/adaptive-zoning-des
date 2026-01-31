from __future__ import annotations

import csv
import os
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_csv(path: str) -> List[Dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_fig1(summary_path: str, output_path: str) -> None:
    rows = _load_csv(summary_path)
    if not rows:
        raise ValueError("summary_main_and_ablations.csv is empty (Fig1)")
    required = ["scheme", "p95_latency_hotspot", "mean_latency_hotspot"]
    missing = [c for c in required if c not in rows[0]]
    if missing:
        raise ValueError(f"Fig1 missing columns: {missing}")
    schemes = [r["scheme"] for r in rows]
    p95_values = [float(r["p95_latency_hotspot"]) for r in rows]
    mean_values = [float(r["mean_latency_hotspot"]) for r in rows]
    x = list(range(len(schemes)))
    width = 0.35
    plt.figure(figsize=(7, 4.5))
    bars1 = plt.bar([i - width / 2 for i in x], p95_values, width=width, label="P95", color="#4C78A8")
    bars2 = plt.bar([i + width / 2 for i in x], mean_values, width=width, label="Mean", color="#F58518")
    plt.xticks(x, schemes)
    plt.ylabel("Hotspot latency (s)")
    plt.title("Fig1: Hotspot latency (P95 vs Mean)")
    plt.legend()
    for bars in [bars1, bars2]:
        for bar in bars:
            value = bar.get_height()
            label = f"{value:.1f}s" if value >= 1e-3 else "<1e-3s"
            plt.text(bar.get_x() + bar.get_width() / 2, value, label, ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_fig2(summary_path: str, output_path: str) -> None:
    rows = _load_csv(summary_path)
    if not rows:
        raise ValueError("summary_main_and_ablations.csv is empty (Fig2)")
    required = ["scheme", "overload_q500", "overload_q1000", "overload_q1500"]
    missing = [c for c in required if c not in rows[0]]
    if missing:
        raise ValueError(f"Fig2 missing columns: {missing}")
    schemes = [r["scheme"] for r in rows]
    values_500 = [float(r["overload_q500"]) for r in rows]
    values_1000 = [float(r["overload_q1000"]) for r in rows]
    values_1500 = [float(r["overload_q1500"]) for r in rows]
    x = list(range(len(schemes)))
    width = 0.25
    plt.figure(figsize=(7, 4.5))
    bars1 = plt.bar([i - width for i in x], values_500, width=width, label="Overload ratio (queue > 500)", color="#4C78A8")
    bars2 = plt.bar(x, values_1000, width=width, label="Overload ratio (queue > 1000)", color="#F58518")
    bars3 = plt.bar([i + width for i in x], values_1500, width=width, label="Overload ratio (queue > 1500)", color="#54A24B")
    plt.xticks(x, schemes)
    plt.ylabel("Overload ratio")
    plt.title("Fig2: Hotspot overload ratio")
    plt.legend(fontsize=8)
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            value = bar.get_height()
            label = f"{value:.3f}" if value >= 1e-3 else "<1e-3"
            plt.text(bar.get_x() + bar.get_width() / 2, value, label, ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_fig3(summary_path: str, output_path: str) -> None:
    rows = _load_csv(summary_path)
    if not rows:
        raise ValueError("summary_main_and_ablations.csv is empty (Fig3)")
    required = ["scheme", "reconfig_cost_total", "p95_latency_hotspot"]
    missing = [c for c in required if c not in rows[0]]
    if missing:
        raise ValueError(f"Fig3 missing columns: {missing}")
    plt.figure(figsize=(6, 4))
    for row in rows:
        x = float(row["reconfig_cost_total"])
        y = float(row["p95_latency_hotspot"])
        plt.scatter(x, y, label=row["scheme"], s=60)
        plt.text(x, y, row["scheme"], fontsize=9, ha="left")
    plt.xlabel("Reconfiguration cost total")
    plt.ylabel("Hotspot p95 latency (s)")
    plt.title("Fig3: Reconfig cost vs latency")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_fig4(summary_path: str, output_path: str, selected_path: str | None = None) -> None:
    rows = _load_csv(summary_path)
    if not rows:
        raise ValueError("summary_fig4_scaledload.csv is empty")

    required = ["N", "scheme", "p95_latency_hotspot", "overload_q1000"]
    missing_cols = [c for c in required if c not in rows[0]]
    if missing_cols:
        raise ValueError(f"Fig4 missing columns: {missing_cols}")

    Ns = sorted({int(r["N"]) for r in rows})
    schemes = sorted({r["scheme"] for r in rows})
    by_scheme: Dict[str, Dict[int, Dict]] = {s: {} for s in schemes}
    for r in rows:
        by_scheme[r["scheme"]][int(r["N"])] = r

    plt.figure(figsize=(6.5, 4.5))
    plotted = 0
    missing_map = {}
    points_map = {}
    for scheme in schemes:
        xs = []
        ys = []
        missing = []
        for n in Ns:
            if n in by_scheme[scheme]:
                xs.append(n)
                ys.append(float(by_scheme[scheme][n]["p95_latency_hotspot"]))
            else:
                missing.append(n)
        missing_map[scheme] = missing
        points_map[scheme] = len(xs)
        if len(xs) < 2:
            continue
        plt.plot(xs, ys, marker="o", label=scheme)
        plotted += 1

    if any(missing_map.values()) or any(count < 2 for count in points_map.values()):
        diagnostics = {
            "N values": Ns,
            "schemes present": schemes,
            "missing N per scheme": missing_map,
            "points per scheme": points_map,
        }
        print(f"Fig4 diagnostics: {diagnostics}")

    if plotted == 0:
        diagnostics = {
            "N values": Ns,
            "schemes present": schemes,
            "selected mapping loaded": bool(selected_path and os.path.exists(selected_path)),
            "row count": len(rows),
            "missing columns": missing_cols,
            "missing N per scheme": missing_map,
            "points per scheme": points_map,
        }
        msg = "Fig4 has no drawable curves. Diagnostics:\n"
        for key, value in diagnostics.items():
            msg += f"{key}: {value}\n"
        raise ValueError(msg)

    plt.xlabel("N (zones)")
    plt.ylabel("Hotspot p95 latency (s)")
    plt.title("Fig4: Scaled-load scalability")
    plt.legend()
    plt.xticks(Ns, Ns)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
