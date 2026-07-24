#!/usr/bin/env python3
"""Trajectory robustness figures using external CD4/CD8 UMAP coordinates.

This reproduces the methods_diff trajectory layout with a fixed external UMAP
embedding. Trajectory methods are recomputed per subset, while the UMAP
coordinates come from:

- /home/roger/Baixades/tots_cd4_umap_coordinates.csv
- /home/roger/Baixades/totscd8_umap_coordinates.csv

Outputs:
- 12-panel pre/post-separated figure: 4 biological subsets x 3 methods.
- 6-panel all-treatment figure: CD4 all and CD8 all x 3 methods.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).with_name(".mplconfig")))
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(__file__).with_name(".numba_cache")))

import pandas as pd
import patchworklib as pw
import scanpy as sc

import methods as mt


ADATA_PATH = Path("/home/roger/data_itx/adata_all.h5ad")
UMAP_CD4_PATH = Path("/home/roger/Baixades/tots_cd4_umap_coordinates.csv")
UMAP_CD8_PATH = Path("/home/roger/Baixades/totscd8_umap_coordinates.csv")

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

CD8_CELLTYPES = ["CD8_ex", "CD8_mem", "CD8_eff", "CD8_naive", "CD8_ex_act", "CD8_act"]
CD4_CELLTYPES = ["Naive", "Tfh", "Th17", "Treg"]


def load_external_umap(path: Path) -> pd.DataFrame:
    umap = pd.read_csv(path)
    required = {"cell", "UMAP_1", "UMAP_2"}
    missing = required.difference(umap.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return umap.drop_duplicates("cell", keep="first").set_index("cell")[["UMAP_1", "UMAP_2"]]


def subset_with_external_umap(adata: sc.AnnData, umap: pd.DataFrame, celltypes: list[str]) -> sc.AnnData:
    common = adata.obs_names.intersection(umap.index)
    subset = adata[common].copy()
    subset = subset[subset.obs["celltype"].astype(str).isin(celltypes)].copy()
    subset.obsm["X_umap"] = umap.loc[subset.obs_names].to_numpy()
    return subset


def build_trajectory_row(
    adata_subset: sc.AnnData,
    palette: dict[str, str],
    root_celltype: str,
    start_node: str,
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
        start_node=start_node,
        ax=slingshot_ax,
        title=f"{row_name} | Slingshot",
        show=False,
    )

    return monocle_ax | paga_ax | slingshot_ax


def save_canvas(canvas, output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    canvas.savefig(str(output_prefix.with_suffix(".png")), dpi=300)
    canvas.savefig(str(output_prefix.with_suffix(".pdf")))


def build_canvas(specs: list[dict[str, object]], margin: float = 0.3):
    pw.clear()
    pw.param["margin"] = margin

    rows = []
    for spec in specs:
        row = build_trajectory_row(
            adata_subset=spec["adata"],
            palette=spec["palette"],
            root_celltype=spec["root_celltype"],
            start_node=spec["start_node"],
            row_name=spec["row_name"],
            label_prefix=spec["label_prefix"],
            brick_figsize=spec.get("brick_figsize", (3.8, 3.4)),
        )
        rows.append(row)

    canvas = rows[0]
    for row in rows[1:]:
        canvas = canvas / row
    return canvas


def make_summary(specs: list[dict[str, object]]) -> pd.DataFrame:
    rows = []
    for spec in specs:
        adata_subset = spec["adata"]
        counts = adata_subset.obs["celltype"].astype(str).value_counts().to_dict()
        rows.append(
            {
                "row_name": spec["row_name"],
                "n_cells": int(adata_subset.n_obs),
                "celltypes": "|".join(sorted(adata_subset.obs["celltype"].astype(str).unique())),
                "treatments": "|".join(sorted(adata_subset.obs["treatment"].astype(str).unique())),
                **{f"n_{key}": int(value) for key, value in counts.items()},
            }
        )
    return pd.DataFrame(rows)


def run(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(ADATA_PATH)
    umap_cd4 = load_external_umap(UMAP_CD4_PATH)
    umap_cd8 = load_external_umap(UMAP_CD8_PATH)

    adata_cd4 = subset_with_external_umap(adata, umap_cd4, CD4_CELLTYPES)
    adata_cd8 = subset_with_external_umap(adata, umap_cd8, CD8_CELLTYPES)

    pre_post_specs = [
        {
            "row_name": "CD8 pre",
            "label_prefix": "ext_cd8_pre",
            "adata": adata_cd8[adata_cd8.obs["treatment"].astype(str) == "pre"].copy(),
            "palette": CD8_PALETTE,
            "root_celltype": "CD8_naive",
            "start_node": "CD8_naive",
        },
        {
            "row_name": "CD8 post",
            "label_prefix": "ext_cd8_post",
            "adata": adata_cd8[adata_cd8.obs["treatment"].astype(str) == "post"].copy(),
            "palette": CD8_PALETTE,
            "root_celltype": "CD8_naive",
            "start_node": "CD8_naive",
        },
        {
            "row_name": "CD4 pre",
            "label_prefix": "ext_cd4_pre",
            "adata": adata_cd4[adata_cd4.obs["treatment"].astype(str) == "pre"].copy(),
            "palette": CD4_PALETTE,
            "root_celltype": "Naive",
            "start_node": "Naive",
        },
        {
            "row_name": "CD4 post",
            "label_prefix": "ext_cd4_post",
            "adata": adata_cd4[adata_cd4.obs["treatment"].astype(str) == "post"].copy(),
            "palette": CD4_PALETTE,
            "root_celltype": "Naive",
            "start_node": "Naive",
        },
    ]

    all_treatment_specs = [
        {
            "row_name": "CD8 pre+post",
            "label_prefix": "ext_cd8_all",
            "adata": adata_cd8.copy(),
            "palette": CD8_PALETTE,
            "root_celltype": "CD8_naive",
            "start_node": "CD8_naive",
        },
        {
            "row_name": "CD4 pre+post",
            "label_prefix": "ext_cd4_all",
            "adata": adata_cd4.copy(),
            "palette": CD4_PALETTE,
            "root_celltype": "Naive",
            "start_node": "Naive",
        },
    ]

    make_summary(pre_post_specs).to_csv(output_dir / "external_umap_pre_post_12panel_summary.csv", index=False)
    make_summary(all_treatment_specs).to_csv(output_dir / "external_umap_all_together_summary.csv", index=False)

    pre_post_canvas = build_canvas(pre_post_specs)
    save_canvas(pre_post_canvas, output_dir / "external_umap_pre_post_12panel")

    all_canvas = build_canvas(all_treatment_specs)
    save_canvas(all_canvas, output_dir / "external_umap_pre_post_all_together")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate methods_diff robustness figures using external UMAP coordinates.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/trajectory_robustness_external_umap"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.output_dir)


if __name__ == "__main__":
    main()
