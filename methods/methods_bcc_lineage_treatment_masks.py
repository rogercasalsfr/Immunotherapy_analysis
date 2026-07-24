#!/usr/bin/env python3
"""Run trajectory-method panels on BCC CD4/CD8 lineages.

This mirrors the lineage definitions in:
    Itx_anl/preprocess/preprocess_bcc.ipynb

The BCC metadata already contains UMAP1/UMAP2 coordinates. We use that single
joint embedding for both treatments, then mask pre or post cells for each row.
This keeps the comparison simple: the coordinate system is fixed, while each
trajectory method is run on the masked treatment subset.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).with_name(".mplconfig")))
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(__file__).with_name(".numba_cache")))

import numpy as np
import pandas as pd
import patchworklib as pw
import scanpy as sc

import methods as mt


BCC_METADATA_PATH = Path(
    "/home/roger/protocol/cancer-pseudotime-grn-workflow/scripts/data/GSE123813/"
    "GSE123813_bcc_all_metadata.txt.gz"
)

CD4_LINEAGE = ["CD4_T_cells", "Tregs"]
CD8_LINEAGE = ["CD8_mem_T_cells", "CD8_ex_T_cells", "CD8_act_T_cells"]

CD4_PALETTE = {
    "CD4_T_cells": "#4C72B0",
    "Tregs": "#C44E52",
}

CD8_PALETTE = {
    "CD8_mem_T_cells": "green",
    "CD8_ex_T_cells": "red",
    "CD8_act_T_cells": "gold",
}


def load_bcc_metadata(path: Path) -> pd.DataFrame:
    metadata = pd.read_csv(path, sep="\t")
    required = {"cell.id", "patient", "treatment", "cluster", "UMAP1", "UMAP2"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    metadata = metadata.drop_duplicates("cell.id", keep="first").set_index("cell.id")
    metadata["celltype"] = metadata["cluster"].astype(str)
    return metadata


def make_lineage_adata(metadata: pd.DataFrame, celltypes: list[str]) -> sc.AnnData:
    obs = metadata[metadata["celltype"].isin(celltypes)].copy()
    obs = obs.dropna(subset=["UMAP1", "UMAP2"])

    umap = obs[["UMAP1", "UMAP2"]].to_numpy(dtype=float)
    adata = sc.AnnData(
        X=umap.copy(),
        obs=obs,
    )
    adata.obsm["X_umap"] = umap
    adata.obsm["X_pca"] = umap
    return adata


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


def build_canvas(specs: list[dict[str, object]], margin: float = 0.3):
    pw.clear()
    pw.param["margin"] = margin

    rows = []
    for spec in specs:
        row = build_trajectory_row(
            adata_subset=spec["adata"],
            palette=spec["palette"],
            root_celltype=spec["root_celltype"],
            row_name=spec["row_name"],
            label_prefix=spec["label_prefix"],
        )
        rows.append(row)

    canvas = rows[0]
    for row in rows[1:]:
        canvas = canvas / row
    return canvas


def save_canvas(canvas, output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    canvas.savefig(str(output_prefix.with_suffix(".png")), dpi=300)
    canvas.savefig(str(output_prefix.with_suffix(".pdf")))


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

    metadata = load_bcc_metadata(BCC_METADATA_PATH)
    adata_cd4 = make_lineage_adata(metadata, CD4_LINEAGE)
    adata_cd8 = make_lineage_adata(metadata, CD8_LINEAGE)

    specs = [
        {
            "row_name": "BCC CD8 pre",
            "label_prefix": "bcc_cd8_pre",
            "adata": adata_cd8[adata_cd8.obs["treatment"].astype(str) == "pre"].copy(),
            "palette": CD8_PALETTE,
            "root_celltype": "CD8_mem_T_cells",
        },
        {
            "row_name": "BCC CD8 post",
            "label_prefix": "bcc_cd8_post",
            "adata": adata_cd8[adata_cd8.obs["treatment"].astype(str) == "post"].copy(),
            "palette": CD8_PALETTE,
            "root_celltype": "CD8_mem_T_cells",
        },
        {
            "row_name": "BCC CD4 pre",
            "label_prefix": "bcc_cd4_pre",
            "adata": adata_cd4[adata_cd4.obs["treatment"].astype(str) == "pre"].copy(),
            "palette": CD4_PALETTE,
            "root_celltype": "CD4_T_cells",
        },
        {
            "row_name": "BCC CD4 post",
            "label_prefix": "bcc_cd4_post",
            "adata": adata_cd4[adata_cd4.obs["treatment"].astype(str) == "post"].copy(),
            "palette": CD4_PALETTE,
            "root_celltype": "CD4_T_cells",
        },
    ]

    empty_specs = [spec["row_name"] for spec in specs if spec["adata"].n_obs == 0]
    if empty_specs:
        raise ValueError(f"These treatment masks have no cells: {empty_specs}")

    summary = make_summary(specs)
    summary.to_csv(output_dir / "bcc_lineage_joint_umap_treatment_masks_summary.csv", index=False)

    canvas = build_canvas(specs)
    save_canvas(canvas, output_dir / "bcc_lineage_joint_umap_treatment_masks_12panel")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run BCC CD4/CD8 trajectory-method panels on one joint UMAP with pre/post masks."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/bcc_lineage_joint_umap_treatment_masks"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.output_dir)


if __name__ == "__main__":
    main()
