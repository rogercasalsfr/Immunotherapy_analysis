#!/usr/bin/env python3
"""CD8 clonotype conservation using an external UMAP coordinate table.

The external coordinate CSV is joined to the CD8 AnnData metadata by cell
barcode. The analysis is then run separately for pre and post treatment:

1. exact paired TRA+TRB clonotypes conserved across CD8 cell states;
2. state-pair sharing matrices and dot plots;
3. treatment-masked UMAP highlights of the top conserved clonotypes;
4. treatment-masked cell-type annotated UMAPs using the same coordinates.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).with_name(".mplconfig")))

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from clonotype_trajectory_conservation import (
    SIGNATURES,
    STATE_ORDERS,
    TreatmentResult,
    add_primary_clonotype_column,
    pairwise_state_sharing,
    plot_state_sharing_heatmap,
    plot_top_clonotype_dotplot,
    summarize_within_treatment,
    top_conserved_for_plot,
)
from shared_tcr_analysis import DATASETS, TCR_PATH, annotate_cells, prepare_tcr_table, read_obs_table


CD8_PALETTE = {
    "CD8_naive": "skyblue",
    "CD8_mem": "green",
    "CD8_eff": "navy",
    "CD8_act": "gold",
    "CD8_ex_act": "purple",
    "CD8_ex": "red",
}

TOP_CLONOTYPE_COLORS = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#7F3C8D",
    "#999933",
]

TOP_CLONOTYPE_LABEL_OFFSETS = [
    (0.00, -0.20),
    (-0.42, 0.45),
    (0.42, -0.45),
    (0.58, 0.38),
    (0.00, 0.55),
    (-0.48, -0.42),
    (0.42, 0.46),
    (0.52, -0.18),
]


def padded_limits(values: pd.Series, pad_fraction: float = 0.04) -> tuple[float, float]:
    low = float(values.min())
    high = float(values.max())
    pad = (high - low) * pad_fraction
    return low - pad, high + pad


def load_cd8_with_external_umap(coordinates_path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    config = DATASETS["cd8"]
    obs = read_obs_table(config.adata_path)
    tcr = prepare_tcr_table(TCR_PATH)
    annotated = annotate_cells(obs, tcr, config)
    annotated = add_primary_clonotype_column(annotated)
    annotated.index = annotated.index.astype(str)
    annotated.index.name = "cell_id"

    coords = pd.read_csv(coordinates_path)
    required = {"cell", "UMAP_1", "UMAP_2"}
    missing = required.difference(coords.columns)
    if missing:
        raise ValueError(f"{coordinates_path} is missing required columns: {sorted(missing)}")

    coords = coords[["cell", "UMAP_1", "UMAP_2"]].copy()
    coords["cell"] = coords["cell"].astype(str)
    coords = coords.drop_duplicates("cell", keep="first").set_index("cell")

    joined = annotated.join(coords, how="inner")
    joined = joined.rename(columns={"UMAP_1": "umap_1", "UMAP_2": "umap_2"})

    overlap = {
        "n_cd8_adata_cells": int(annotated.shape[0]),
        "n_external_umap_cells": int(coords.shape[0]),
        "n_matched_cells_used": int(joined.shape[0]),
        "n_external_umap_cells_without_cd8_metadata": int(coords.shape[0] - joined.shape[0]),
        "n_cd8_adata_cells_without_external_umap": int(annotated.shape[0] - joined.shape[0]),
    }
    return joined, overlap


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
        for state in STATE_ORDERS["cd8"]
    ]


def style_umap_axis(
    ax: plt.Axes,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    ylabel: bool = True,
    label_fontsize: float = 12,
    tick_fontsize: float = 10,
) -> None:
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("UMAP 1", fontsize=label_fontsize)
    ax.set_ylabel("UMAP 2" if ylabel else "", fontsize=label_fontsize)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=tick_fontsize)


def plot_celltype_points(
    ax: plt.Axes,
    panel: pd.DataFrame,
    alpha: float = 0.88,
    point_size: float = 6,
    label_states: bool = True,
    label_fontsize: float = 10,
) -> None:
    for state in STATE_ORDERS["cd8"]:
        subset = panel.loc[panel["celltype_label"].astype(str) == state]
        if subset.empty:
            continue
        ax.scatter(
            subset["umap_1"],
            subset["umap_2"],
            s=point_size,
            c=CD8_PALETTE[state],
            alpha=alpha,
            linewidths=0,
            rasterized=True,
        )

    if not label_states:
        return

    for state in STATE_ORDERS["cd8"]:
        subset = panel.loc[panel["celltype_label"].astype(str) == state]
        if subset.empty:
            continue
        ax.text(
            subset["umap_1"].median(),
            subset["umap_2"].median(),
            state,
            ha="center",
            va="center",
            fontsize=label_fontsize,
            weight="bold",
            color="black",
            path_effects=[pe.withStroke(linewidth=3.8, foreground="white")],
        )


def clone_color(rank: int) -> str:
    return TOP_CLONOTYPE_COLORS[(rank - 1) % len(TOP_CLONOTYPE_COLORS)]


def clone_legend_label(row: pd.Series) -> str:
    return f"{row['plot_label']}  ({int(row['n_cells']):,} cells, {int(row['n_celltypes'])} states)"


def plot_background(ax: plt.Axes, panel: pd.DataFrame, alpha: float = 0.26) -> None:
    ax.scatter(
        panel["umap_1"],
        panel["umap_2"],
        s=5,
        c="#cfcfcf",
        alpha=alpha,
        linewidths=0,
        rasterized=True,
    )


def plot_single_shared_clonotype_panel(
    ax: plt.Axes,
    panel: pd.DataFrame,
    top: pd.DataFrame,
    signature_col: str,
    title_fontsize: float = 13,
    legend_fontsize: float = 9,
) -> None:
    plot_background(ax, panel, alpha=0.23)

    signature = top.index[0]
    row = top.iloc[0]
    selected = panel.loc[panel[signature_col] == signature]

    handles = []
    for state in STATE_ORDERS["cd8"]:
        subset = selected.loc[selected["celltype_label"].astype(str) == state]
        if subset.empty:
            continue
        ax.scatter(
            subset["umap_1"],
            subset["umap_2"],
            s=42,
            c=CD8_PALETTE[state],
            alpha=0.96,
            edgecolors="white",
            linewidths=0.28,
            rasterized=True,
        )
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=CD8_PALETTE[state],
                markeredgecolor="none",
                markersize=7,
                label=f"{state} ({len(subset):,})",
            )
        )

    ax.set_title(
        f"Most shared clonotype\n{row['plot_label']} · {int(row['n_cells']):,} cells · {int(row['n_celltypes'])} states",
        fontsize=title_fontsize,
        pad=9,
    )
    if handles:
        ax.legend(
            handles=handles,
            loc="lower left",
            bbox_to_anchor=(0.0, 0.0),
            frameon=False,
            fontsize=legend_fontsize,
            borderaxespad=0.0,
            handletextpad=0.3,
        )


def plot_top_clonotypes_panel(
    ax: plt.Axes,
    panel: pd.DataFrame,
    top: pd.DataFrame,
    signature_col: str,
    show_legend: bool = True,
) -> None:
    plot_background(ax, panel, alpha=0.24)

    plotted_handles: list[Line2D] = []
    for rank, signature in reversed(list(enumerate(top.index, start=1))):
        row = top.loc[signature]
        selected = panel.loc[panel[signature_col] == signature]
        if selected.empty:
            continue
        color = clone_color(rank)
        ax.scatter(
            selected["umap_1"],
            selected["umap_2"],
            s=22,
            c=color,
            alpha=0.94,
            edgecolors="white",
            linewidths=0.22,
            rasterized=True,
        )
        label_dx, label_dy = TOP_CLONOTYPE_LABEL_OFFSETS[(rank - 1) % len(TOP_CLONOTYPE_LABEL_OFFSETS)]
        ax.text(
            selected["umap_1"].median() + label_dx,
            selected["umap_2"].median() + label_dy,
            f"C{rank:02d}",
            ha="center",
            va="center",
            fontsize=8,
            weight="bold",
            color=color,
            path_effects=[pe.withStroke(linewidth=3.0, foreground="white")],
        )
        plotted_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=color,
                markeredgecolor="none",
                markersize=6,
                label=clone_legend_label(row),
            )
        )

    ax.set_title(f"Top {len(top)} conserved paired clonotypes", fontsize=11, pad=7)
    if show_legend and plotted_handles:
        ax.legend(
            handles=plotted_handles[::-1],
            title="Ranked by shared states",
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            fontsize=7.2,
            title_fontsize=8.4,
            handletextpad=0.4,
        )


def plot_top_clonotype_transition_table(
    ax: plt.Axes,
    top: pd.DataFrame,
    state_order: list[str],
    title_fontsize: float = 13,
    row_label_fontsize: float = 9.5,
    tick_fontsize: float = 10,
    summary_fontsize: float = 9.2,
    count_fontsize: float = 8.5,
) -> None:
    state_cols = [state for state in state_order if state in top.columns]
    state_cols.extend([col for col in top.columns if col not in state_cols and col.startswith("CD")])
    if not state_cols:
        ax.axis("off")
        return

    labels = list(top["plot_label"])
    n_rows = len(labels)
    n_cols = len(state_cols)

    for row_idx, (_, row) in enumerate(top.iterrows()):
        if row_idx % 2:
            ax.axhspan(row_idx - 0.5, row_idx + 0.5, color="#f7f7f7", zorder=0)

        ax.text(
            n_cols + 0.32,
            row_idx,
            f"{int(row['n_cells']):,} cells · {int(row['n_celltypes'])} states",
            va="center",
            ha="left",
            fontsize=summary_fontsize,
            color="#333333",
        )

        for col_idx, state in enumerate(state_cols):
            count = int(row.get(state, 0))
            if count <= 0:
                continue

            size = 85 + (count**0.5) * 34
            ax.scatter(
                col_idx,
                row_idx,
                s=size,
                c=CD8_PALETTE.get(state, "#777777"),
                alpha=0.9,
                edgecolors="white",
                linewidths=0.55,
                zorder=3,
            )
            ax.text(
                col_idx,
                row_idx,
                str(count),
                ha="center",
                va="center",
                fontsize=count_fontsize,
                weight="bold",
                color="white" if count >= 10 else "#222222",
                path_effects=[pe.withStroke(linewidth=1.15, foreground="black" if count >= 10 else "white")],
                zorder=4,
            )

    for edge in [x - 0.5 for x in range(n_cols + 1)]:
        ax.axvline(edge, color="#dddddd", linewidth=0.55, zorder=1)
    for edge in [y - 0.5 for y in range(n_rows + 1)]:
        ax.axhline(edge, color="#dddddd", linewidth=0.55, zorder=1)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(state_cols, rotation=35, ha="right", fontsize=tick_fontsize)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(labels, fontsize=row_label_fontsize)
    ax.set_xlim(-0.5, n_cols + 2.65)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_title(
        "Top conserved paired clonotypes\ncounts across ordered CD8 states",
        fontsize=title_fontsize,
        pad=9,
    )
    ax.set_xlabel("CD8 state order", fontsize=tick_fontsize + 1)
    ax.set_ylabel("")
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_pre_post_transition_heatmaps(
    top_by_treatment: dict[str, pd.DataFrame],
    state_order: list[str],
    output_path: Path,
) -> None:
    """Plot separate pre/post clonotype-by-state heatmaps on a shared scale."""
    available = {
        treatment: top_by_treatment.get(treatment, pd.DataFrame())
        for treatment in ["pre", "post"]
    }
    available = {treatment: top for treatment, top in available.items() if not top.empty}
    if not available:
        return

    state_cols = [state for state in state_order if any(state in top.columns for top in available.values())]
    max_count = max(
        int(top.reindex(columns=state_cols, fill_value=0).to_numpy().max())
        for top in available.values()
    )
    vmax = float(np.log1p(max_count))

    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("white")

    for treatment in ["pre", "post"]:
        top = available.get(treatment, pd.DataFrame())
        if top.empty:
            continue

        fig, ax = plt.subplots(figsize=(15.5, 5.8))
        counts = top.reindex(columns=state_cols, fill_value=0).fillna(0).astype(int).to_numpy()
        log_counts = np.log1p(counts.astype(float))
        masked_log_counts = np.ma.masked_where(counts == 0, log_counts)
        image = ax.imshow(
            masked_log_counts,
            cmap=cmap,
            vmin=0,
            vmax=vmax,
            interpolation="none",
            aspect="auto",
        )

        for row_idx in range(counts.shape[0]):
            for col_idx in range(counts.shape[1]):
                count = counts[row_idx, col_idx]
                if count <= 1:
                    continue
                text_color = "white" if log_counts[row_idx, col_idx] <= 0.46 * vmax else "#111111"
                ax.text(
                    col_idx,
                    row_idx,
                    f"{count:,}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    weight="bold",
                    color=text_color,
                )

        row_labels = list(top["plot_label"])
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=9.5)
        ax.set_ylabel("Paired clonotype", fontsize=11)
        ax.set_xticks(range(len(state_cols)))
        ax.set_xticklabels(state_cols, fontsize=11)
        ax.set_xticks(np.arange(-0.5, len(state_cols), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.4)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.tick_params(axis="both", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

        fig.suptitle(
            f"CD8 {treatment}: conserved paired TCR clonotypes across CD8 states",
            fontsize=18,
            y=0.97,
        )
        colorbar_axis = fig.add_axes([0.925, 0.18, 0.015, 0.68])
        colorbar = fig.colorbar(image, cax=colorbar_axis)
        colorbar.set_label("log1p(number of cells)", fontsize=11)

        treatment_output = output_path.with_name(f"{output_path.name}_{treatment}")
        fig.subplots_adjust(left=0.25, right=0.90, top=0.86, bottom=0.13)
        fig.savefig(treatment_output.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(treatment_output.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)


def validate_top_counts(
    annotated: pd.DataFrame,
    top: pd.DataFrame,
    signature_col: str,
    treatment: str,
    state_order: list[str],
) -> pd.DataFrame:
    panel = annotated.loc[annotated["treatment_label"].astype(str) == treatment].copy()
    rows = []

    for rank, (signature, row) in enumerate(top.iterrows(), start=1):
        selected = panel.loc[panel[signature_col] == signature]
        state_counts = selected["celltype_label"].astype(str).value_counts()
        present_states = [state for state in state_order if int(state_counts.get(state, 0)) > 0]
        plotted_n_cells = int(selected.shape[0])
        table_n_cells = int(row["n_cells"])
        table_n_celltypes = int(row["n_celltypes"])
        table_celltypes = str(row["celltypes"])
        plotted_celltypes = "|".join(present_states)

        state_counts_match = True
        for state in state_order:
            if state in row.index and int(row[state]) != int(state_counts.get(state, 0)):
                state_counts_match = False
                break

        rows.append(
            {
                "treatment": treatment,
                "rank": rank,
                "paired_clonotype_signature_aa": signature,
                "plot_label": row["plot_label"],
                "table_n_cells": table_n_cells,
                "plotted_n_cells": plotted_n_cells,
                "n_cells_match": table_n_cells == plotted_n_cells,
                "table_n_celltypes": table_n_celltypes,
                "plotted_n_celltypes": len(present_states),
                "n_celltypes_match": table_n_celltypes == len(present_states),
                "table_celltypes": table_celltypes,
                "plotted_celltypes": plotted_celltypes,
                "celltypes_match": table_celltypes == plotted_celltypes,
                "state_counts_match": state_counts_match,
            }
        )

    validation = pd.DataFrame(rows)
    if not validation.empty:
        mismatch = validation.loc[
            ~(
                validation["n_cells_match"]
                & validation["n_celltypes_match"]
                & validation["celltypes_match"]
                & validation["state_counts_match"]
            )
        ]
        if not mismatch.empty:
            raise ValueError(
                "Top clonotype plot counts do not match the conservation table:\n"
                + mismatch.to_string(index=False)
            )
    return validation


def plot_celltype_umaps(annotated: pd.DataFrame, output_path: Path) -> None:
    xlim = padded_limits(annotated["umap_1"])
    ylim = padded_limits(annotated["umap_2"])

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.5))
    for ax, treatment in zip(axes, ["pre", "post"]):
        panel = annotated.loc[annotated["treatment_label"].astype(str) == treatment].copy()
        plot_celltype_points(ax, panel, alpha=0.88, point_size=5.5, label_states=True)

        ax.set_title(f"CD8 {treatment}  n={len(panel):,}", fontsize=14, pad=8)
        style_umap_axis(ax, xlim, ylim, ylabel=True)

    axes[1].legend(
        handles=legend_handles(),
        title="CD8 cell type",
        loc="center left",
        bbox_to_anchor=(1.03, 0.5),
        frameon=False,
        fontsize=9,
        title_fontsize=10,
    )
    fig.suptitle("CD8 cell-type annotated UMAPs using external coordinates", fontsize=16, y=0.98)
    fig.subplots_adjust(left=0.07, right=0.84, top=0.86, bottom=0.14, wspace=0.23)
    fig.savefig(output_path.with_suffix(".png"), dpi=300)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_umap_highlight_external(
    annotated: pd.DataFrame,
    top: pd.DataFrame,
    signature_col: str,
    treatment: str,
    output_path: Path,
) -> None:
    if top.empty:
        return

    panel = annotated.loc[annotated["treatment_label"].astype(str) == treatment].copy()
    xlim = padded_limits(annotated["umap_1"])
    ylim = padded_limits(annotated["umap_2"])

    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    plot_top_clonotypes_panel(ax, panel, top, signature_col, show_legend=True)
    ax.set_title(f"CD8 {treatment}: top conserved paired clonotypes", fontsize=13, pad=9)
    style_umap_axis(ax, xlim, ylim, ylabel=True)
    fig.subplots_adjust(left=0.08, right=0.61, top=0.9, bottom=0.12)
    fig.savefig(output_path.with_suffix(".png"), dpi=300)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_three_panel_external_umap(
    annotated: pd.DataFrame,
    top: pd.DataFrame,
    signature_col: str,
    treatment: str,
    output_path: Path,
) -> None:
    if top.empty:
        return

    panel = annotated.loc[annotated["treatment_label"].astype(str) == treatment].copy()
    xlim = padded_limits(annotated["umap_1"])
    ylim = padded_limits(annotated["umap_2"])

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(23.5, 7.0),
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.9]},
    )

    plot_celltype_points(axes[0], panel, alpha=0.9, point_size=6.4, label_states=True, label_fontsize=11)
    axes[0].set_title(f"Cell type\nn={len(panel):,}", fontsize=14, pad=9)

    plot_single_shared_clonotype_panel(axes[1], panel, top, signature_col, title_fontsize=14, legend_fontsize=9.5)
    plot_top_clonotype_transition_table(axes[2], top, STATE_ORDERS["cd8"])

    for idx, ax in enumerate(axes[:2]):
        style_umap_axis(ax, xlim, ylim, ylabel=idx == 0, label_fontsize=12, tick_fontsize=10.5)

    fig.suptitle(
        f"CD8 {treatment}: external UMAP TCR conservation",
        fontsize=17,
        y=0.98,
    )
    fig.subplots_adjust(left=0.05, right=0.985, top=0.82, bottom=0.18, wspace=0.27)
    fig.savefig(output_path.with_suffix(".png"), dpi=300)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_pre_post_three_panel_external_umap(
    annotated: pd.DataFrame,
    top_by_treatment: dict[str, pd.DataFrame],
    signature_col: str,
    output_path: Path,
) -> None:
    xlim = padded_limits(annotated["umap_1"])
    ylim = padded_limits(annotated["umap_2"])

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(23.5, 13.4),
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.9], "height_ratios": [1.0, 1.0]},
    )

    for row_idx, treatment in enumerate(["pre", "post"]):
        top = top_by_treatment.get(treatment, pd.DataFrame())
        if top.empty:
            continue

        panel = annotated.loc[annotated["treatment_label"].astype(str) == treatment].copy()
        row_axes = axes[row_idx]

        plot_celltype_points(
            row_axes[0],
            panel,
            alpha=0.9,
            point_size=6.2,
            label_states=True,
            label_fontsize=10.8,
        )
        row_axes[0].set_title(f"CD8 {treatment}: cell type\nn={len(panel):,}", fontsize=14, pad=9)

        plot_single_shared_clonotype_panel(
            row_axes[1],
            panel,
            top,
            signature_col,
            title_fontsize=14,
            legend_fontsize=9.3,
        )
        plot_top_clonotype_transition_table(
            row_axes[2],
            top,
            STATE_ORDERS["cd8"],
            title_fontsize=14,
            row_label_fontsize=10,
            tick_fontsize=10.5,
            summary_fontsize=9.5,
            count_fontsize=8.8,
        )

        for col_idx, ax in enumerate(row_axes[:2]):
            style_umap_axis(ax, xlim, ylim, ylabel=col_idx == 0, label_fontsize=12, tick_fontsize=10.2)

    fig.suptitle("CD8 external UMAP TCR conservation", fontsize=18, y=0.985)
    fig.subplots_adjust(left=0.052, right=0.986, top=0.91, bottom=0.075, wspace=0.27, hspace=0.42)
    fig.savefig(output_path.with_suffix(".png"), dpi=300)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def run_cd8_external_umap_analysis(
    coordinates_path: Path,
    output_dir: Path,
    min_cells: int,
    min_celltypes: int,
    top_n: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    annotated, overlap = load_cd8_with_external_umap(coordinates_path)
    pd.DataFrame([overlap]).to_csv(output_dir / "cd8_external_umap_overlap_summary.csv", index=False)

    annotation_cols = [
        "tcr_barcode",
        "celltype_label",
        "treatment_label",
        "patient_label",
        "has_tcr",
        "tra_signature_aa",
        "trb_signature_aa",
        "paired_clonotype_signature_aa",
        "umap_1",
        "umap_2",
    ]
    annotated[annotation_cols].to_csv(output_dir / "cd8_external_umap_cell_tcr_annotations.csv")

    plot_celltype_umaps(
        annotated,
        figures_dir / "cd8_external_umap_celltype_pre_post",
    )

    report_rows = []
    validation_rows = []
    top_by_treatment = {}
    state_order = STATE_ORDERS["cd8"]
    signature_name = "paired_clonotype_aa"
    signature_col = SIGNATURES[signature_name]

    for treatment in ["pre", "post"]:
        summary, conserved = summarize_within_treatment(
            annotated=annotated,
            signature_col=signature_col,
            state_order=state_order,
            treatment=treatment,
            min_cells=min_cells,
            min_celltypes=min_celltypes,
        )
        pairwise_long, pairwise_matrix = pairwise_state_sharing(
            annotated=annotated,
            signature_col=signature_col,
            state_order=state_order,
            treatment=treatment,
        )
        result = TreatmentResult(
            dataset="cd8",
            treatment=treatment,
            signature_name=signature_name,
            summary=summary,
            conserved=conserved,
            pairwise_long=pairwise_long,
            pairwise_matrix=pairwise_matrix,
        )

        prefix = f"cd8_external_umap_{treatment}_{signature_name}"
        result.summary.to_csv(output_dir / f"{prefix}_within_treatment_summary.csv")
        result.conserved.to_csv(output_dir / f"{prefix}_conserved_across_trajectory_states.csv")
        result.pairwise_long.to_csv(output_dir / f"{prefix}_state_pair_sharing_long.csv", index=False)
        result.pairwise_matrix.to_csv(output_dir / f"{prefix}_state_pair_sharing_matrix.csv")

        top = top_conserved_for_plot(result.conserved, top_n=top_n)
        top_by_treatment[treatment] = top
        validation = validate_top_counts(
            annotated=annotated,
            top=top,
            signature_col=signature_col,
            treatment=treatment,
            state_order=state_order,
        )
        if not validation.empty:
            validation_rows.append(validation)
        top.to_csv(output_dir / f"{prefix}_top_conserved_for_plots.csv")

        plot_state_sharing_heatmap(
            result.pairwise_matrix,
            title=f"CD8 {treatment}: paired clonotypes shared across states",
            output_path=figures_dir / f"{prefix}_state_pair_sharing_heatmap.png",
        )
        plot_top_clonotype_dotplot(
            top,
            state_order=state_order,
            title=f"CD8 {treatment}: top conserved paired clonotypes",
            output_path=figures_dir / f"{prefix}_top_conserved_dotplot.png",
        )
        plot_umap_highlight_external(
            annotated=annotated,
            top=top,
            signature_col=signature_col,
            treatment=treatment,
            output_path=figures_dir / f"{prefix}_umap_top_conserved",
        )
        plot_three_panel_external_umap(
            annotated=annotated,
            top=top,
            signature_col=signature_col,
            treatment=treatment,
            output_path=figures_dir / f"cd8_external_umap_{treatment}_three_panel_tcr_conservation",
        )

        report_rows.append(
            {
                "dataset": "cd8",
                "treatment": treatment,
                "coordinates": str(coordinates_path),
                "n_cells_used": int((annotated["treatment_label"].astype(str) == treatment).sum()),
                "n_unique_paired_clonotypes": int(summary.shape[0]),
                "n_conserved_paired_clonotypes": int(conserved.shape[0]),
                "n_conserved_cells": int(conserved["n_cells"].sum()) if not conserved.empty else 0,
                "top_signature": str(top.index[0]) if not top.empty else "",
                "top_signature_n_cells": int(top.iloc[0]["n_cells"]) if not top.empty else 0,
                "top_signature_n_celltypes": int(top.iloc[0]["n_celltypes"]) if not top.empty else 0,
                "top_signature_states": str(top.iloc[0]["celltypes"]) if not top.empty else "",
            }
        )

    report = pd.DataFrame(report_rows)
    report.to_csv(output_dir / "cd8_external_umap_within_treatment_overview.csv", index=False)
    plot_pre_post_three_panel_external_umap(
        annotated=annotated,
        top_by_treatment=top_by_treatment,
        signature_col=signature_col,
        output_path=figures_dir / "cd8_external_umap_pre_post_three_panel_tcr_conservation",
    )
    plot_pre_post_transition_heatmaps(
        top_by_treatment=top_by_treatment,
        state_order=state_order,
        output_path=figures_dir / "cd8_external_umap_tcr_transition_heatmap",
    )
    if validation_rows:
        pd.concat(validation_rows, ignore_index=True).to_csv(
            output_dir / "cd8_external_umap_three_panel_count_validation.csv",
            index=False,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CD8 clonotype conservation on external UMAP coordinates.")
    parser.add_argument(
        "--coordinates",
        type=Path,
        default=Path("/home/roger/Baixades/totscd8_umap_coordinates.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/cd8_external_umap_conservation"),
    )
    parser.add_argument("--min-cells", type=int, default=2)
    parser.add_argument("--min-celltypes", type=int, default=2)
    parser.add_argument("--top-n", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_cd8_external_umap_analysis(
        coordinates_path=args.coordinates,
        output_dir=args.output_dir,
        min_cells=args.min_cells,
        min_celltypes=args.min_celltypes,
        top_n=args.top_n,
    )


if __name__ == "__main__":
    main()
