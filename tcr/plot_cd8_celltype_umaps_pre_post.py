#!/usr/bin/env python3
"""Plot CD8 pre/post UMAPs annotated by cell type."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).with_name(".mplconfig")))

import h5py
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from shared_tcr_analysis import DATASETS, read_obs_table


CD8_STATE_ORDER = ["CD8_naive", "CD8_mem", "CD8_eff", "CD8_act", "CD8_ex_act", "CD8_ex"]

CD8_PALETTE = {
    "CD8_naive": "skyblue",
    "CD8_mem": "green",
    "CD8_eff": "navy",
    "CD8_act": "gold",
    "CD8_ex_act": "purple",
    "CD8_ex": "red",
}


def read_umap(path: Path) -> pd.DataFrame:
    with h5py.File(path, "r") as handle:
        umap = handle["obsm"]["X_umap"][:]
    return pd.DataFrame({"umap_1": umap[:, 0], "umap_2": umap[:, 1]})


def load_cd8() -> pd.DataFrame:
    config = DATASETS["cd8"]
    obs = read_obs_table(config.adata_path).reset_index(drop=True)
    umap = read_umap(config.adata_path)

    data = pd.concat([obs, umap], axis=1)
    data["celltype"] = data["celltype"].astype(str)
    data["treatment"] = data["treatment"].astype(str)
    return data


def padded_limits(values: pd.Series, pad_fraction: float = 0.04) -> tuple[float, float]:
    low = float(values.min())
    high = float(values.max())
    pad = (high - low) * pad_fraction
    return low - pad, high + pad


def draw_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    treatment: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    panel = data.loc[data["treatment"] == treatment].copy()

    for state in CD8_STATE_ORDER:
        subset = panel.loc[panel["celltype"] == state]
        if subset.empty:
            continue
        ax.scatter(
            subset["umap_1"],
            subset["umap_2"],
            s=5,
            c=CD8_PALETTE[state],
            alpha=0.72,
            linewidths=0,
            rasterized=True,
        )

    for state in CD8_STATE_ORDER:
        subset = panel.loc[panel["celltype"] == state]
        if subset.empty:
            continue
        ax.text(
            subset["umap_1"].median(),
            subset["umap_2"].median(),
            state,
            ha="center",
            va="center",
            fontsize=9,
            weight="bold",
            color="black",
            path_effects=[pe.withStroke(linewidth=3.5, foreground="white")],
        )

    ax.set_title(f"CD8 {treatment}  n={len(panel):,}", fontsize=14, pad=8)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=CD8_PALETTE[state],
            markeredgecolor="none",
            markersize=7,
            label=state,
        )
        for state in CD8_STATE_ORDER
    ]


def plot_cd8_umaps(output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    data = load_cd8()

    xlim = padded_limits(data["umap_1"])
    ylim = padded_limits(data["umap_2"])

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.5))
    draw_panel(axes[0], data, "pre", xlim, ylim)
    draw_panel(axes[1], data, "post", xlim, ylim)

    axes[1].legend(
        handles=legend_handles(),
        title="CD8 cell type",
        loc="center left",
        bbox_to_anchor=(1.03, 0.5),
        frameon=False,
        fontsize=9,
        title_fontsize=10,
    )

    fig.suptitle("CD8 cell-type annotated UMAPs before and after treatment", fontsize=16, y=0.98)
    fig.subplots_adjust(left=0.07, right=0.84, top=0.86, bottom=0.14, wspace=0.23)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300)
    fig.savefig(output_prefix.with_suffix(".pdf"))
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot CD8 pre/post UMAPs colored by cell type.")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("results/within_treatment_conservation/figures/cd8_celltype_umaps_pre_post"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_cd8_umaps(args.output_prefix)


if __name__ == "__main__":
    main()
