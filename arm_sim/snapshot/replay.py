from __future__ import annotations

import os

from ..plots.make_figures import plot_fig1, plot_fig2, plot_fig3, plot_fig4
from ..output_layout import write_md


def replot_from_snapshot(snapshot_dir: str, output_dir: str) -> None:
    summary_main = os.path.join(snapshot_dir, "summary_main_and_ablations.csv")
    summary_fig4 = os.path.join(snapshot_dir, "summary_fig4_scaledload.csv")
    selected = os.path.join(snapshot_dir, "summary_s2_selected.csv")

    plot_fig1(summary_main, os.path.join(output_dir, "fig1_hotspot_p95_main_constrained.png"))
    plot_fig2(summary_main, os.path.join(output_dir, "fig2_overload_ratio_main_constrained.png"))
    plot_fig3(summary_main, os.path.join(output_dir, "fig3_tradeoff_scatter_constrained.png"))
    plot_fig4(summary_fig4, os.path.join(output_dir, "fig4_scaledload_scalability.png"), selected_path=selected)

    write_md(os.path.join(output_dir, "snapshot_used.txt"), f"snapshot_dir: {snapshot_dir}\n")
