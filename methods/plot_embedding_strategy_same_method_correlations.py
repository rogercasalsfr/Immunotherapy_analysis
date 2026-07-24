#!/usr/bin/env python3
"""Correlate same-method pseudotime across UMAP embedding strategies."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).with_name(".mplconfig")))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr


RESULT_DIR = Path("results/pseudotime_correlation_umap_comparison")
FIXED_PSEUDOTIME = Path(
    "results/trajectory_robustness_external_umap/"
    "pseudotime_correlations_cd8post_mem_root/external_umap_per_cell_pseudotime.csv"
)
RECOMPUTED_PSEUDOTIME = Path(
    "results/trajectory_robustness_condition_umap/"
    "pseudotime_correlations_cd8post_mem_root/condition_umap_per_cell_pseudotime.csv"
)

METHODS = [
    ("monocle3_pseudotime", "Monocle3"),
    ("paga_dpt_pseudotime", "PAGA/DPT"),
    ("slingshot_pseudotime", "Slingshot"),
]

ROW_ORDER = ["CD8 pre", "CD8 post", "CD4 pre", "CD4 post"]
METHOD_ORDER = ["Monocle3", "PAGA/DPT", "Slingshot"]


def format_p_value(value: float) -> str:
    if pd.isna(value):
        return "p=NA"
    if value == 0:
        return "p<1e-300"
    if value < 0.001:
        return f"p={value:.1e}"
    return f"p={value:.3f}"


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    fixed = pd.read_csv(FIXED_PSEUDOTIME)
    recomputed = pd.read_csv(RECOMPUTED_PSEUDOTIME)

    merge_keys = ["cell", "lineage", "treatment"]
    merged = fixed.merge(
        recomputed,
        on=merge_keys,
        suffixes=("_fixed_external_umap", "_recomputed_condition_umap"),
        how="inner",
    )

    rows = []
    for (lineage, treatment), group in merged.groupby(["lineage", "treatment"], sort=False):
        condition = f"{lineage} {treatment}"
        for column, method_label in METHODS:
            fixed_col = f"{column}_fixed_external_umap"
            recomputed_col = f"{column}_recomputed_condition_umap"
            pair = group[[fixed_col, recomputed_col]].dropna()
            if pair.shape[0] < 3:
                rho = float("nan")
                p_value = float("nan")
            else:
                result = spearmanr(pair[fixed_col], pair[recomputed_col])
                rho = float(result.statistic)
                p_value = float(result.pvalue)
            rows.append(
                {
                    "condition": condition,
                    "lineage": lineage,
                    "treatment": treatment,
                    "method": method_label,
                    "n_intersecting_cells": int(pair.shape[0]),
                    "spearman_rho": rho,
                    "p_value": p_value,
                }
            )

    correlations = pd.DataFrame(rows)
    correlations.to_csv(
        RESULT_DIR / "same_method_fixed_vs_recomputed_umap_spearman_correlations.csv",
        index=False,
    )

    merged.to_csv(
        RESULT_DIR / "same_method_fixed_vs_recomputed_umap_joined_pseudotime.csv",
        index=False,
    )

    plot_data = correlations.pivot(index="condition", columns="method", values="spearman_rho")
    plot_data = plot_data.reindex(index=ROW_ORDER, columns=METHOD_ORDER)
    annotations = correlations.copy()
    annotations["annotation"] = annotations.apply(
        lambda row: f"rho={row['spearman_rho']:.2f}\n{format_p_value(row['p_value'])}\nn={row['n_intersecting_cells']}",
        axis=1,
    )
    annotation_data = annotations.pivot(index="condition", columns="method", values="annotation")
    annotation_data = annotation_data.reindex(index=ROW_ORDER, columns=METHOD_ORDER)

    sns.set_theme(style="white", context="paper")
    fig, ax = plt.subplots(figsize=(5.4, 3.9))
    sns.heatmap(
        plot_data,
        ax=ax,
        cmap=sns.color_palette("mako", as_cmap=True),
        vmin=0,
        vmax=1,
        annot=annotation_data,
        fmt="",
        linewidths=0.8,
        linecolor="white",
        cbar_kws={"label": "Spearman rho"},
        annot_kws={"fontsize": 8, "linespacing": 1.05},
    )
    ax.set_title("Same-method pseudotime robustness to UMAP strategy", fontsize=11, pad=10)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelrotation=0, labelsize=8)
    ax.tick_params(axis="y", labelrotation=0, labelsize=9)
    fig.text(
        0.02,
        0.02,
        "Fixed external UMAP vs UMAP recomputed per condition; correlations use intersecting cells.",
        fontsize=7.5,
    )
    fig.subplots_adjust(left=0.19, right=0.96, top=0.86, bottom=0.2)
    fig.savefig(RESULT_DIR / "same_method_fixed_vs_recomputed_umap_heatmap.pdf", bbox_inches="tight")
    fig.savefig(RESULT_DIR / "same_method_fixed_vs_recomputed_umap_heatmap.png", dpi=450, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
