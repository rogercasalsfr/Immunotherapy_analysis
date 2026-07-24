#!/usr/bin/env python3
"""Compute pseudotime correlations after recomputing UMAP per condition.

Root rules:
- CD8 pre  -> CD8_naive
- CD8 post -> CD8_mem
- CD4 pre/post -> Naive
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).with_name(".mplconfig")))
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(__file__).with_name(".numba_cache")))

import pandas as pd
import scanpy as sc

from methods_external_umap_pseudotime_correlations import (
    ADATA_PATH,
    CD4_CELLTYPES,
    CD8_CELLTYPES,
    compute_pseudotime_table,
    compute_spearman,
)


def recompute_condition_umap(
    adata: sc.AnnData,
    n_top_genes: int,
    n_pcs: int,
    random_state: int,
) -> sc.AnnData:
    work = adata.copy()
    if "X_pca" not in work.obsm:
        raise KeyError("The input AnnData does not contain obsm['X_pca']; cannot recompute condition UMAP.")
    usable_pcs = min(n_pcs, work.obsm["X_pca"].shape[1])
    sc.pp.neighbors(work, use_rep="X_pca", n_pcs=usable_pcs, random_state=random_state)
    sc.tl.umap(work, random_state=random_state)
    return work


def build_specs(n_top_genes: int, n_pcs: int, random_state: int) -> list[dict[str, object]]:
    adata = sc.read_h5ad(ADATA_PATH)

    raw_specs = [
        {
            "row_name": "CD8 pre condition UMAP",
            "lineage": "CD8",
            "treatment": "pre",
            "celltypes": CD8_CELLTYPES,
            "root_celltype": "CD8_naive",
        },
        {
            "row_name": "CD8 post condition UMAP",
            "lineage": "CD8",
            "treatment": "post",
            "celltypes": CD8_CELLTYPES,
            "root_celltype": "CD8_mem",
        },
        {
            "row_name": "CD4 pre condition UMAP",
            "lineage": "CD4",
            "treatment": "pre",
            "celltypes": CD4_CELLTYPES,
            "root_celltype": "Naive",
        },
        {
            "row_name": "CD4 post condition UMAP",
            "lineage": "CD4",
            "treatment": "post",
            "celltypes": CD4_CELLTYPES,
            "root_celltype": "Naive",
        },
    ]

    specs = []
    for spec in raw_specs:
        subset = adata[
            (adata.obs["treatment"].astype(str) == spec["treatment"])
            & adata.obs["celltype"].astype(str).isin(spec["celltypes"])
        ].copy()
        subset = recompute_condition_umap(
            subset,
            n_top_genes=n_top_genes,
            n_pcs=n_pcs,
            random_state=random_state,
        )
        resolved = dict(spec)
        resolved["adata"] = subset
        del resolved["celltypes"]
        specs.append(resolved)
    return specs


def save_umap_coordinates(specs: list[dict[str, object]], output_dir: Path) -> None:
    frames = []
    for spec in specs:
        adata = spec["adata"]
        coords = pd.DataFrame(
            adata.obsm["X_umap"],
            index=adata.obs_names,
            columns=["UMAP_1", "UMAP_2"],
        )
        coords.insert(0, "cell", coords.index)
        coords["row_name"] = spec["row_name"]
        coords["lineage"] = spec["lineage"]
        coords["treatment"] = spec["treatment"]
        coords["celltype"] = adata.obs["celltype"].astype(str).to_numpy()
        frames.append(coords)
    pd.concat(frames, ignore_index=True).to_csv(output_dir / "condition_umap_coordinates.csv", index=False)


def run(output_dir: Path, n_top_genes: int, n_pcs: int, random_state: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    tables = []
    audits = []
    specs = build_specs(n_top_genes=n_top_genes, n_pcs=n_pcs, random_state=random_state)
    save_umap_coordinates(specs, output_dir)

    for spec in specs:
        if spec["adata"].n_obs == 0:
            raise ValueError(f"No cells found for {spec['row_name']}.")
        table, audit = compute_pseudotime_table(spec)
        tables.append(table)
        audits.append(audit)

    pseudotime = pd.concat(tables, ignore_index=True)
    correlations = compute_spearman(pseudotime)

    pseudotime.to_csv(output_dir / "condition_umap_per_cell_pseudotime.csv", index=False)
    correlations.to_csv(output_dir / "condition_umap_pseudotime_spearman_correlations.csv", index=False)
    pd.DataFrame(audits).to_csv(output_dir / "condition_umap_pseudotime_root_audit.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recompute UMAP per condition and correlate pseudotime methods.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/trajectory_robustness_condition_umap/pseudotime_correlations_cd8post_mem_root"),
    )
    parser.add_argument("--n-top-genes", type=int, default=2000)
    parser.add_argument("--n-pcs", type=int, default=30)
    parser.add_argument("--random-state", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        output_dir=args.output_dir,
        n_top_genes=args.n_top_genes,
        n_pcs=args.n_pcs,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
