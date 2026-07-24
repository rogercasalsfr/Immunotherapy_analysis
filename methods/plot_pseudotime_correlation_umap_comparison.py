#!/usr/bin/env python3
"""Plot supplementary pseudotime correlation heatmaps."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).with_name(".mplconfig")))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


RESULT_DIR = Path("results/pseudotime_correlation_umap_comparison")
FIXED_CSV = RESULT_DIR / "fixed_external_umap_spearman_correlations.csv"
RECOMPUTED_CSV = RESULT_DIR / "recomputed_condition_umap_spearman_correlations.csv"

METHOD_LABELS = {
    "monocle3_pseudotime": "Monocle3",
    "paga_dpt_pseudotime": "PAGA/DPT",
    "slingshot_pseudotime": "Slingshot",
}

ROW_ORDER = ["CD8 pre", "CD8 post", "CD4 pre", "CD4 post"]
COL_ORDER = ["Monocle3 vs PAGA/DPT", "Monocle3 vs Slingshot", "PAGA/DPT vs Slingshot"]


def format_p_value(value: float) -> str:
    if pd.isna(value):
        return "p=NA"
    if value == 0:
        return "p<1e-300"
    if value < 0.001:
        return f"p={value:.1e}"
    return f"p={value:.3f}"


def load_correlations(path: Path, strategy: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    data["strategy"] = strategy
    data["condition"] = data["lineage"].astype(str) + " " + data["treatment"].astype(str)
    data["method_pair"] = (
        data["method_a"].map(METHOD_LABELS) + " vs " + data["method_b"].map(METHOD_LABELS)
    )
    data["annotation"] = data.apply(
        lambda row: f"rho={row['spearman_rho']:.2f}\n{format_p_value(row['p_value'])}",
        axis=1,
    )
    return data


def draw_panel(ax, data: pd.DataFrame, title: str, show_ylabels: bool) -> None:
    heatmap_data = data.pivot(index="condition", columns="method_pair", values="spearman_rho")
    heatmap_data = heatmap_data.reindex(index=ROW_ORDER, columns=COL_ORDER)

    annotations = data.pivot(index="condition", columns="method_pair", values="annotation")
    annotations = annotations.reindex(index=ROW_ORDER, columns=COL_ORDER)

    sns.heatmap(
        heatmap_data,
        ax=ax,
        cmap="viridis",
        vmin=-1,
        vmax=1,
        annot=annotations,
        fmt="",
        linewidths=0.8,
        linecolor="white",
        cbar=False,
        square=False,
        annot_kws={"fontsize": 7.5, "linespacing": 1.15},
    )
    ax.set_title(title, fontsize=11, pad=10)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels(
        ["Monocle3\nvs PAGA/DPT", "Monocle3\nvs Slingshot", "PAGA/DPT\nvs Slingshot"],
        rotation=0,
        ha="center",
    )
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelrotation=0, labelsize=9)
    if not show_ylabels:
        ax.set_yticklabels([])


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    fixed = load_correlations(FIXED_CSV, "Fixed external UMAP")
    recomputed = load_correlations(RECOMPUTED_CSV, "Recomputed condition UMAP")
    combined = pd.concat([fixed, recomputed], ignore_index=True)
    combined.to_csv(RESULT_DIR / "pseudotime_correlation_plot_table.csv", index=False)

    sns.set_theme(style="white", context="paper")
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9.6, 4.6),
        gridspec_kw={"width_ratios": [1, 1], "wspace": 0.08},
        constrained_layout=False,
    )

    draw_panel(axes[0], fixed, "Fixed external UMAP positions", show_ylabels=True)
    draw_panel(axes[1], recomputed, "UMAP recomputed per condition", show_ylabels=False)

    cbar_ax = fig.add_axes([0.92, 0.27, 0.015, 0.48])
    norm = plt.Normalize(vmin=-1, vmax=1)
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Spearman rho", fontsize=9)
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle(
        "Agreement between pseudotime methods across embedding strategies",
        fontsize=12,
        y=0.98,
    )
    fig.text(
        0.02,
        0.035,
        "Rows: lineage and treatment. Cells show Spearman rho and two-sided p-value. "
        "p<1e-300 indicates numerical underflow.",
        fontsize=7.5,
    )
    fig.subplots_adjust(left=0.12, right=0.9, top=0.86, bottom=0.19)

    fig.savefig(RESULT_DIR / "pseudotime_correlation_umap_strategy_heatmap.pdf", bbox_inches="tight")
    fig.savefig(RESULT_DIR / "pseudotime_correlation_umap_strategy_heatmap.png", dpi=450, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
