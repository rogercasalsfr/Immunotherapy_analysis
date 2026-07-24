#!/usr/bin/env python3
"""Corroborate clonotype conservation within pre and post trajectories.

This pipeline starts from the same AnnData and TCR inputs used in
``shared_tcr_analysis.py`` and produces treatment-specific evidence that
clonotypes are conserved across trajectory-associated cell states.

Primary evidence is based on exact paired amino-acid clonotypes, requiring
both TRA and TRB chains to be present in a cell. TRB-only summaries are also
written as supplementary evidence.

Example
-------
python3 clonotype_trajectory_conservation.py --dataset all
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).with_name(".mplconfig")))

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from shared_tcr_analysis import (
    DATASETS,
    TCR_PATH,
    DatasetConfig,
    annotate_cells,
    prepare_tcr_table,
    read_obs_table,
)


TREATMENT_ORDER = ["pre", "post"]

STATE_ORDERS = {
    "cd8": ["CD8_naive", "CD8_mem", "CD8_eff", "CD8_act", "CD8_ex_act", "CD8_ex"],
    "cd4": ["Naive", "Tfh", "Th17", "Treg"],
}

SIGNATURES = {
    "paired_clonotype_aa": "paired_clonotype_signature_aa",
    "trb_aa": "trb_signature_aa",
}


@dataclass(frozen=True)
class TreatmentResult:
    dataset: str
    treatment: str
    signature_name: str
    summary: pd.DataFrame
    conserved: pd.DataFrame
    pairwise_long: pd.DataFrame
    pairwise_matrix: pd.DataFrame


def ordered_values(values: Iterable[object], order: list[str]) -> list[str]:
    present = {str(value) for value in pd.Series(values).dropna().unique()}
    ordered = [value for value in order if value in present]
    ordered.extend(sorted(present.difference(order)))
    return ordered


def ordered_join(values: Iterable[object], order: list[str]) -> str:
    return "|".join(ordered_values(values, order))


def read_umap_from_h5ad(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        if "obsm" not in handle or "X_umap" not in handle["obsm"]:
            raise KeyError(f"{path} does not contain obsm['X_umap']")
        return np.asarray(handle["obsm"]["X_umap"])


def add_primary_clonotype_column(annotated: pd.DataFrame) -> pd.DataFrame:
    annotated = annotated.copy()
    has_tra = annotated["tra_signature_aa"].notna() & annotated["tra_signature_aa"].astype(str).ne("")
    has_trb = annotated["trb_signature_aa"].notna() & annotated["trb_signature_aa"].astype(str).ne("")
    annotated["paired_clonotype_signature_aa"] = annotated["clonotype_signature_aa"].where(has_tra & has_trb)
    return annotated


def build_annotated_cells(config: DatasetConfig) -> pd.DataFrame:
    obs = read_obs_table(config.adata_path)
    tcr = prepare_tcr_table(TCR_PATH)
    annotated = annotate_cells(obs, tcr, config)
    annotated = add_primary_clonotype_column(annotated)

    umap = read_umap_from_h5ad(config.adata_path)
    annotated["umap_1"] = umap[:, 0]
    annotated["umap_2"] = umap[:, 1]
    annotated.index.name = "cell_id"
    return annotated


def add_trajectory_span_columns(summary: pd.DataFrame, state_cols: list[str]) -> pd.DataFrame:
    summary = summary.copy()
    state_position = {state: idx for idx, state in enumerate(state_cols)}
    adjacent_pairs = list(zip(state_cols[:-1], state_cols[1:]))

    starts: list[str | None] = []
    ends: list[str | None] = []
    spans: list[int] = []
    adjacent_counts: list[int] = []

    for _, row in summary[state_cols].iterrows():
        present_states = [state for state in state_cols if int(row.get(state, 0)) > 0]
        if not present_states:
            starts.append(None)
            ends.append(None)
            spans.append(0)
            adjacent_counts.append(0)
            continue

        positions = [state_position[state] for state in present_states]
        starts.append(state_cols[min(positions)])
        ends.append(state_cols[max(positions)])
        spans.append(max(positions) - min(positions) + 1)
        adjacent_counts.append(sum(int(row[left] > 0 and row[right] > 0) for left, right in adjacent_pairs))

    summary["trajectory_start_state"] = starts
    summary["trajectory_end_state"] = ends
    summary["trajectory_state_span"] = spans
    summary["n_adjacent_state_pairs_present"] = adjacent_counts
    return summary


def summarize_within_treatment(
    annotated: pd.DataFrame,
    signature_col: str,
    state_order: list[str],
    treatment: str,
    min_cells: int,
    min_celltypes: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    subset = annotated.loc[
        (annotated["treatment_label"].astype(str) == treatment) & annotated[signature_col].notna()
    ].copy()

    if subset.empty:
        empty = pd.DataFrame()
        return empty, empty

    grouped = subset.groupby(signature_col, dropna=True)
    summary = grouped.agg(
        n_cells=("tcr_barcode", "size"),
        n_celltypes=("celltype_label", pd.Series.nunique),
        n_patients=("patient_label", pd.Series.nunique),
    )
    summary["treatment"] = treatment
    summary["celltypes"] = grouped["celltype_label"].apply(lambda values: ordered_join(values, state_order))
    summary["patients"] = grouped["patient_label"].apply(
        lambda values: "|".join(sorted(pd.Series(values).dropna().astype(str).unique()))
    )

    if "tra_signature_aa" in subset.columns:
        summary["tra_signature_aa"] = grouped["tra_signature_aa"].first()
    if "trb_signature_aa" in subset.columns:
        summary["trb_signature_aa"] = grouped["trb_signature_aa"].first()

    counts = pd.crosstab(subset[signature_col], subset["celltype_label"])
    state_cols = [state for state in state_order if state in counts.columns]
    state_cols.extend(sorted(set(counts.columns).difference(state_cols)))
    counts = counts.reindex(columns=state_cols, fill_value=0)

    summary = summary.join(counts)
    summary = add_trajectory_span_columns(summary, state_cols)
    summary = summary.sort_values(
        ["n_celltypes", "n_adjacent_state_pairs_present", "trajectory_state_span", "n_cells"],
        ascending=[False, False, False, False],
    )

    conserved = summary.loc[
        (summary["n_cells"] >= min_cells) & (summary["n_celltypes"] >= min_celltypes)
    ].copy()
    return summary, conserved


def pairwise_state_sharing(
    annotated: pd.DataFrame,
    signature_col: str,
    state_order: list[str],
    treatment: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    subset = annotated.loc[
        (annotated["treatment_label"].astype(str) == treatment) & annotated[signature_col].notna()
    ].copy()

    if subset.empty:
        return pd.DataFrame(), pd.DataFrame()

    presence = pd.crosstab(subset[signature_col], subset["celltype_label"]).gt(0)
    states = [state for state in state_order if state in presence.columns]
    states.extend(sorted(set(presence.columns).difference(states)))
    presence = presence.reindex(columns=states, fill_value=False)

    presence_int = presence.astype(int)
    shared = presence_int.T.dot(presence_int)
    shared = shared.astype(int).reindex(index=states, columns=states)
    shared.index.name = "state"
    shared.columns.name = "state"

    rows = []
    adjacent = set(zip(states[:-1], states[1:]))
    for state_a in states:
        for state_b in states:
            both = int((presence[state_a] & presence[state_b]).sum())
            only_a_or_b = int((presence[state_a] | presence[state_b]).sum())
            n_a = int(presence[state_a].sum())
            n_b = int(presence[state_b].sum())
            rows.append(
                {
                    "treatment": treatment,
                    "state_a": state_a,
                    "state_b": state_b,
                    "n_shared_clonotypes": both,
                    "n_state_a_clonotypes": n_a,
                    "n_state_b_clonotypes": n_b,
                    "jaccard": both / only_a_or_b if only_a_or_b else np.nan,
                    "is_adjacent_trajectory_pair": (state_a, state_b) in adjacent or (state_b, state_a) in adjacent,
                }
            )

    return pd.DataFrame(rows), shared


def short_clone_label(signature: str, rank: int) -> str:
    trb = signature.split("TRB:", 1)[1] if "TRB:" in signature else signature
    trb = trb.split(";", 1)[0]
    if len(trb) > 24:
        trb = f"{trb[:21]}..."
    return f"C{rank:02d} TRB:{trb}"


def top_conserved_for_plot(conserved: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if conserved.empty:
        return conserved.copy()

    top = conserved.head(top_n).copy()
    top["plot_label"] = [
        short_clone_label(str(signature), rank) for rank, signature in enumerate(top.index, start=1)
    ]
    return top


def plot_state_sharing_heatmap(matrix: pd.DataFrame, title: str, output_path: Path) -> None:
    if matrix.empty:
        return

    width = max(4.5, 0.85 * matrix.shape[1] + 1.5)
    height = max(4.0, 0.75 * matrix.shape[0] + 1.2)
    fig, ax = plt.subplots(figsize=(width, height))
    sns.heatmap(matrix, cmap="viridis", annot=True, fmt="d", square=True, cbar_kws={"label": "Shared clonotypes"}, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_top_clonotype_dotplot(
    top: pd.DataFrame,
    state_order: list[str],
    title: str,
    output_path: Path,
) -> None:
    if top.empty:
        return

    state_cols = [state for state in state_order if state in top.columns]
    state_cols.extend([col for col in top.columns if col not in state_cols and col.startswith("CD")])
    if not state_cols:
        return

    plot_df = (
        top[["plot_label", *state_cols]]
        .set_index("plot_label")
        .stack()
        .rename("n_cells")
        .reset_index()
        .rename(columns={"level_1": "state"})
    )
    plot_df = plot_df.loc[plot_df["n_cells"] > 0].copy()
    if plot_df.empty:
        return

    x_pos = {state: idx for idx, state in enumerate(state_cols)}
    y_labels = list(top["plot_label"])[::-1]
    y_pos = {label: idx for idx, label in enumerate(y_labels)}

    fig, ax = plt.subplots(figsize=(max(6.5, len(state_cols) * 0.95), max(3.8, len(y_labels) * 0.42)))
    sizes = np.sqrt(plot_df["n_cells"].astype(float)) * 42
    sc = ax.scatter(
        plot_df["state"].map(x_pos),
        plot_df["plot_label"].map(y_pos),
        s=sizes,
        c=np.log1p(plot_df["n_cells"].astype(float)),
        cmap="magma",
        alpha=0.88,
        edgecolors="black",
        linewidths=0.25,
    )
    ax.set_xticks(range(len(state_cols)))
    ax.set_xticklabels(state_cols, rotation=35, ha="right")
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("log1p(cells)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_umap_highlight(
    annotated: pd.DataFrame,
    top: pd.DataFrame,
    signature_col: str,
    treatment: str,
    title: str,
    output_path: Path,
) -> None:
    if top.empty:
        return

    subset = annotated.loc[annotated["treatment_label"].astype(str) == treatment].copy()
    if subset.empty:
        return

    top_signatures = list(top.index)
    colors = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(8.8, 5.5))
    ax.scatter(subset["umap_1"], subset["umap_2"], s=4, c="#d0d0d0", alpha=0.45, linewidths=0)

    for idx, signature in enumerate(top_signatures):
        label = top.loc[signature, "plot_label"]
        selected = subset.loc[subset[signature_col] == signature]
        if selected.empty:
            continue
        ax.scatter(
            selected["umap_1"],
            selected["umap_2"],
            s=16,
            c=[colors(idx % 10)],
            alpha=0.9,
            linewidths=0,
            label=label,
        )

    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=9, markerscale=1.2)
    fig.subplots_adjust(left=0.08, right=0.63, top=0.9, bottom=0.12)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def write_cell_annotations(annotated: pd.DataFrame, dataset: str, output_dir: Path) -> None:
    keep = [
        "tcr_barcode",
        "celltype_label",
        "treatment_label",
        "patient_label",
        "has_tcr",
        "tra_signature_aa",
        "trb_signature_aa",
        "clonotype_signature_aa",
        "paired_clonotype_signature_aa",
        "umap_1",
        "umap_2",
    ]
    available = [col for col in keep if col in annotated.columns]
    annotated[available].to_csv(output_dir / f"{dataset}_cell_tcr_annotations.csv")


def write_treatment_outputs(
    result: TreatmentResult,
    output_dir: Path,
    figures_dir: Path,
    annotated: pd.DataFrame,
    state_order: list[str],
    top_n: int,
    make_plots: bool,
) -> pd.DataFrame:
    prefix = f"{result.dataset}_{result.treatment}_{result.signature_name}"
    result.summary.to_csv(output_dir / f"{prefix}_within_treatment_summary.csv")
    result.conserved.to_csv(output_dir / f"{prefix}_conserved_across_trajectory_states.csv")
    result.pairwise_long.to_csv(output_dir / f"{prefix}_state_pair_sharing_long.csv", index=False)
    result.pairwise_matrix.to_csv(output_dir / f"{prefix}_state_pair_sharing_matrix.csv")

    top = top_conserved_for_plot(result.conserved, top_n=top_n)
    if not top.empty:
        top.to_csv(output_dir / f"{prefix}_top_conserved_for_plots.csv")

    if make_plots and result.signature_name == "paired_clonotype_aa":
        plot_state_sharing_heatmap(
            result.pairwise_matrix,
            title=f"{result.dataset.upper()} {result.treatment}: paired clonotypes shared across states",
            output_path=figures_dir / f"{prefix}_state_pair_sharing_heatmap.png",
        )
        plot_top_clonotype_dotplot(
            top,
            state_order=state_order,
            title=f"{result.dataset.upper()} {result.treatment}: top conserved paired clonotypes",
            output_path=figures_dir / f"{prefix}_top_conserved_dotplot.png",
        )
        plot_umap_highlight(
            annotated,
            top,
            signature_col=SIGNATURES[result.signature_name],
            treatment=result.treatment,
            title=f"{result.dataset.upper()} {result.treatment}: conserved paired clonotypes",
            output_path=figures_dir / f"{prefix}_umap_top_conserved.png",
        )

    return top


def analyze_dataset(
    dataset: str,
    config: DatasetConfig,
    output_dir: Path,
    figures_dir: Path,
    min_cells: int,
    min_celltypes: int,
    top_n: int,
    make_plots: bool,
) -> list[dict[str, object]]:
    state_order = STATE_ORDERS[dataset]
    annotated = build_annotated_cells(config)
    write_cell_annotations(annotated, dataset, output_dir)

    report_rows: list[dict[str, object]] = []
    treatments = [
        treatment
        for treatment in TREATMENT_ORDER
        if treatment in set(annotated["treatment_label"].dropna().astype(str))
    ]

    for treatment in treatments:
        for signature_name, signature_col in SIGNATURES.items():
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
                dataset=dataset,
                treatment=treatment,
                signature_name=signature_name,
                summary=summary,
                conserved=conserved,
                pairwise_long=pairwise_long,
                pairwise_matrix=pairwise_matrix,
            )
            top = write_treatment_outputs(
                result=result,
                output_dir=output_dir,
                figures_dir=figures_dir,
                annotated=annotated,
                state_order=state_order,
                top_n=top_n,
                make_plots=make_plots,
            )
            report_rows.append(
                {
                    "dataset": dataset,
                    "treatment": treatment,
                    "signature_name": signature_name,
                    "n_cells_with_signature": int(annotated.loc[
                        (annotated["treatment_label"].astype(str) == treatment)
                        & annotated[signature_col].notna()
                    ].shape[0]),
                    "n_unique_signatures": int(summary.shape[0]),
                    "n_conserved_signatures": int(conserved.shape[0]),
                    "n_conserved_cells": int(conserved["n_cells"].sum()) if not conserved.empty else 0,
                    "top_signature": str(top.index[0]) if not top.empty else "",
                    "top_signature_n_cells": int(top.iloc[0]["n_cells"]) if not top.empty else 0,
                    "top_signature_n_celltypes": int(top.iloc[0]["n_celltypes"]) if not top.empty else 0,
                    "top_signature_states": str(top.iloc[0]["celltypes"]) if not top.empty else "",
                }
            )

    return report_rows


def write_report(report: pd.DataFrame, output_dir: Path, min_cells: int, min_celltypes: int) -> None:
    report.to_csv(output_dir / "within_treatment_conservation_overview.csv", index=False)

    primary = report.loc[report["signature_name"] == "paired_clonotype_aa"].copy()
    lines = [
        "# Within-treatment clonotype conservation report",
        "",
        "Primary evidence uses exact paired amino-acid clonotypes and requires both TRA and TRB chains.",
        "TRB-only outputs are supplementary because they are less strict than paired TRA+TRB clonotypes.",
        "",
        f"Conserved clonotypes are defined as signatures with at least {min_cells} cells and at least {min_celltypes} trajectory states within the same treatment.",
        "",
        "## Primary paired-clonotype summary",
        "",
    ]

    if primary.empty:
        lines.append("No primary paired-clonotype rows were generated.")
    else:
        for row in primary.itertuples(index=False):
            lines.extend(
                [
                    f"### {row.dataset.upper()} {row.treatment}",
                    "",
                    f"- Cells with paired clonotype: {row.n_cells_with_signature}",
                    f"- Unique paired clonotypes: {row.n_unique_signatures}",
                    f"- Conserved paired clonotypes across states: {row.n_conserved_signatures}",
                    f"- Cells belonging to conserved paired clonotypes: {row.n_conserved_cells}",
                    f"- Top conserved clonotype: `{row.top_signature}`",
                    f"- Top conserved clonotype cells/states: {row.top_signature_n_cells} cells across {row.top_signature_n_celltypes} states ({row.top_signature_states})",
                    "",
                ]
            )

    lines.extend(
        [
            "## Main outputs",
            "",
            "- `*_paired_clonotype_aa_conserved_across_trajectory_states.csv`: the key evidence table for each dataset and treatment.",
            "- `*_paired_clonotype_aa_state_pair_sharing_matrix.csv`: number of exact paired clonotypes shared between trajectory states within one treatment.",
            "- `figures/*_paired_clonotype_aa_state_pair_sharing_heatmap.png`: heatmap version of the state-pair matrix.",
            "- `figures/*_paired_clonotype_aa_top_conserved_dotplot.png`: distribution of top conserved clonotypes across ordered trajectory states.",
            "- `figures/*_paired_clonotype_aa_umap_top_conserved.png`: UMAP highlighting top conserved clonotypes within the selected treatment.",
            "- `*_trb_aa_*`: supplementary TRB-only versions of the same tables.",
            "",
            "## Interpretation guardrails",
            "",
            "- Treat the paired TRA+TRB tables as the primary evidence.",
            "- Interpret TRB-only sharing as supportive, not definitive.",
            "- Keep pre and post separate when discussing trajectory conservation.",
            "- For figures, use only a small number of top clonotypes so the UMAP remains readable.",
            "",
        ]
    )

    (output_dir / "within_treatment_conservation_report.md").write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantify and plot clonotype conservation across trajectory states within pre and post separately."
    )
    parser.add_argument("--dataset", choices=sorted(DATASETS.keys()) + ["all"], default="all")
    parser.add_argument("--output-dir", type=Path, default=Path("results/within_treatment_conservation"))
    parser.add_argument("--min-cells", type=int, default=2)
    parser.add_argument("--min-celltypes", type=int, default=2)
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--no-plots", action="store_true", help="Write tables only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    dataset_names = sorted(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    report_rows: list[dict[str, object]] = []
    for dataset in dataset_names:
        report_rows.extend(
            analyze_dataset(
                dataset=dataset,
                config=DATASETS[dataset],
                output_dir=args.output_dir,
                figures_dir=figures_dir,
                min_cells=args.min_cells,
                min_celltypes=args.min_celltypes,
                top_n=args.top_n,
                make_plots=not args.no_plots,
            )
        )

    report = pd.DataFrame(report_rows)
    write_report(report, args.output_dir, min_cells=args.min_cells, min_celltypes=args.min_celltypes)


if __name__ == "__main__":
    main()
