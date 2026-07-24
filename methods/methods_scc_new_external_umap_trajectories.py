#!/usr/bin/env python3
"""SCC trajectory panels using the new per-condition external UMAP coordinates.

Inputs from /home/roger/Baixades:
- precd8_umap_coordinates.csv   -> SCC CD8 pre
- postcd8_umap_coordinates.csv  -> SCC CD8 post
- pre_umap_coordinates.csv      -> SCC CD4 pre
- post_umap_coordinates.csv     -> SCC CD4 post

The coordinates are used as supplied. For each row, trajectory methods are
computed on the cells in that external embedding.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).with_name(".mplconfig")))
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(__file__).with_name(".numba_cache")))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import patchworklib as pw
import scanpy as sc

import methods as mt


SCC_METADATA_PATH = Path(
    "/home/roger/protocol/cancer-pseudotime-grn-workflow/scripts/data/GSE123813/"
    "GSE123813_scc_metadata.txt.gz"
)

COORDINATE_SPECS = [
    {
        "key": "scc_cd8_pre",
        "row_name": "SCC CD8 pre",
        "path": Path("/home/roger/Baixades/precd8_umap_coordinates.csv"),
        "expected_treatment": "pre",
        "lineage": "CD8",
        "root_celltype": "CD8_mem",
    },
    {
        "key": "scc_cd8_post",
        "row_name": "SCC CD8 post",
        "path": Path("/home/roger/Baixades/postcd8_umap_coordinates.csv"),
        "expected_treatment": "post",
        "lineage": "CD8",
        "root_celltype": "CD8_mem",
    },
    {
        "key": "scc_cd4_pre",
        "row_name": "SCC CD4 pre",
        "path": Path("/home/roger/Baixades/pre_umap_coordinates.csv"),
        "expected_treatment": "pre",
        "lineage": "CD4",
        "root_celltype": "Naive",
    },
    {
        "key": "scc_cd4_post",
        "row_name": "SCC CD4 post",
        "path": Path("/home/roger/Baixades/post_umap_coordinates.csv"),
        "expected_treatment": "post",
        "lineage": "CD4",
        "root_celltype": "Naive",
    },
]

CD8_PALETTE = {
    "CD8_naive": "skyblue",
    "CD8_mem": "green",
    "CD8_ex": "red",
    "CD8_ex_act": "purple",
    "CD8_eff": "navy",
    "CD8_act": "gold",
}

CD4_PALETTE = {
    "Naive": "#4C72B0",
    "Tfh": "#DD8452",
    "Th17": "#55A868",
    "Treg": "#C44E52",
}


def load_scc_metadata(path: Path) -> pd.DataFrame:
    metadata = pd.read_csv(path, sep="\t")
    required = {"cell.id", "patient", "treatment", "cluster"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    metadata = metadata.drop_duplicates("cell.id", keep="first").set_index("cell.id")
    metadata["celltype"] = metadata["cluster"].astype(str)
    return metadata


def load_coordinates(path: Path) -> pd.DataFrame:
    coords = pd.read_csv(path)
    required = {"cell", "UMAP_1", "UMAP_2"}
    missing = required.difference(coords.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return coords.drop_duplicates("cell", keep="first").set_index("cell")[["UMAP_1", "UMAP_2"]]


def make_coordinate_adata(metadata: pd.DataFrame, spec: dict[str, object]) -> sc.AnnData:
    coords = load_coordinates(spec["path"])
    common = coords.index.intersection(metadata.index)
    if common.empty:
        raise ValueError(f"No SCC metadata cells matched {spec['path']}.")

    obs = metadata.loc[common].copy()
    expected_treatment = str(spec["expected_treatment"])
    unexpected_treatments = sorted(set(obs["treatment"].astype(str)) - {expected_treatment})
    if unexpected_treatments:
        raise ValueError(
            f"{spec['path']} contains treatments outside {expected_treatment!r}: {unexpected_treatments}"
        )

    umap = coords.loc[obs.index].to_numpy(dtype=float)
    adata = sc.AnnData(X=umap.copy(), obs=obs)
    adata.obsm["X_umap"] = umap
    adata.obsm["X_pca"] = umap
    return adata


def palette_for_lineage(lineage: str) -> dict[str, str]:
    if lineage == "CD8":
        return CD8_PALETTE
    if lineage == "CD4":
        return CD4_PALETTE
    raise ValueError(f"Unknown lineage: {lineage}")


def plot_coordinate_overview(specs: list[dict[str, object]], output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    axes = axes.ravel()

    for ax, spec in zip(axes, specs):
        adata = spec["adata"]
        palette = spec["palette"]
        colors = adata.obs["celltype"].astype(str).map(palette).fillna("#808080")
        umap = adata.obsm["X_umap"]

        ax.scatter(umap[:, 0], umap[:, 1], c=colors, s=3, linewidths=0, alpha=0.85)
        ax.set_title(f"{spec['row_name']} ({adata.n_obs:,} cells)")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_aspect("equal", adjustable="datalim")

        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=color,
                markeredgecolor="none",
                markersize=6,
                label=celltype,
            )
            for celltype, color in palette.items()
            if celltype in set(adata.obs["celltype"].astype(str))
        ]
        ax.legend(handles=handles, frameon=False, fontsize=8, loc="best")

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "scc_new_coordinates_umap_overview.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "scc_new_coordinates_umap_overview.pdf", bbox_inches="tight")
    plt.close(fig)


def build_trajectory_row(
    adata_subset: sc.AnnData,
    palette: dict[str, str],
    root_celltype: str,
    row_name: str,
    label_prefix: str,
    brick_figsize: tuple[float, float] = (3.8, 3.4),
):
    monocle_ax = pw.Brick(f"{label_prefix}_monocle3", figsize=brick_figsize)
    paga_ax = pw.Brick(f"{label_prefix}_paga", figsize=brick_figsize)
    slingshot_ax = pw.Brick(f"{label_prefix}_slingshot", figsize=brick_figsize)

    mt.monocle3(
        adata_subset.copy(),
        palette_cd8=palette,
        ax=monocle_ax,
        title=f"{row_name} | Monocle3",
        show=False,
    )
    mt.run_paga(
        adata_subset.copy(),
        root_celltype=root_celltype,
        ax=paga_ax,
        title=f"{row_name} | PAGA",
        show=False,
    )
    mt.run_slingshot(
        adata_subset.copy(),
        start_node=root_celltype,
        ax=slingshot_ax,
        title=f"{row_name} | Slingshot",
        show=False,
    )

    return monocle_ax | paga_ax | slingshot_ax


def build_trajectory_canvas(specs: list[dict[str, object]]):
    pw.clear()
    pw.param["margin"] = 0.3

    rows = []
    for spec in specs:
        rows.append(
            build_trajectory_row(
                adata_subset=spec["adata"],
                palette=spec["palette"],
                root_celltype=spec["root_celltype"],
                row_name=spec["row_name"],
                label_prefix=spec["key"],
            )
        )

    canvas = rows[0]
    for row in rows[1:]:
        canvas = canvas / row
    return canvas


def make_summary(specs: list[dict[str, object]]) -> pd.DataFrame:
    rows = []
    for spec in specs:
        adata = spec["adata"]
        counts = adata.obs["celltype"].astype(str).value_counts().to_dict()
        rows.append(
            {
                "row_name": spec["row_name"],
                "coordinate_file": str(spec["path"]),
                "n_cells": int(adata.n_obs),
                "lineage": spec["lineage"],
                "treatments": "|".join(sorted(adata.obs["treatment"].astype(str).unique())),
                "celltypes": "|".join(sorted(adata.obs["celltype"].astype(str).unique())),
                **{f"n_{key}": int(value) for key, value in counts.items()},
            }
        )
    return pd.DataFrame(rows)


def run(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_scc_metadata(SCC_METADATA_PATH)
    resolved_specs = []
    for spec in COORDINATE_SPECS:
        adata = make_coordinate_adata(metadata, spec)
        resolved = dict(spec)
        resolved["adata"] = adata
        resolved["palette"] = palette_for_lineage(str(spec["lineage"]))
        resolved_specs.append(resolved)

    make_summary(resolved_specs).to_csv(output_dir / "scc_new_coordinates_summary.csv", index=False)
    plot_coordinate_overview(resolved_specs, output_dir)

    canvas = build_trajectory_canvas(resolved_specs)
    canvas.savefig(str(output_dir / "scc_new_coordinates_trajectory_12panel.png"), dpi=300)
    canvas.savefig(str(output_dir / "scc_new_coordinates_trajectory_12panel.pdf"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SCC trajectories using new external UMAP coordinates.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/scc_new_external_umap_trajectories"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.output_dir)


if __name__ == "__main__":
    main()
