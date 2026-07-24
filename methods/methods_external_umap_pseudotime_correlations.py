#!/usr/bin/env python3
"""Compute pseudotime correlations for the earlier ITx external UMAP analysis.

This uses the older external coordinate files:
- /home/roger/Baixades/tots_cd4_umap_coordinates.csv
- /home/roger/Baixades/totscd8_umap_coordinates.csv

For each treatment/lineage subset, it computes per-cell pseudotime from:
- Monocle3-style graph ordering
- PAGA/DPT
- Slingshot unified pseudotime

Then it reports pairwise Spearman correlations and p-values.
"""

from __future__ import annotations

import argparse
import os
from itertools import combinations
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).with_name(".mplconfig")))
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(__file__).with_name(".numba_cache")))

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import spearmanr

from py_monocle import learn_graph, order_cells
from pyslingshot import Slingshot

import methods as mt


ADATA_PATH = Path("/home/roger/data_itx/adata_all.h5ad")
UMAP_CD4_PATH = Path("/home/roger/Baixades/tots_cd4_umap_coordinates.csv")
UMAP_CD8_PATH = Path("/home/roger/Baixades/totscd8_umap_coordinates.csv")

CD8_CELLTYPES = ["CD8_ex", "CD8_mem", "CD8_eff", "CD8_naive", "CD8_ex_act", "CD8_act"]
CD4_CELLTYPES = ["Naive", "Tfh", "Th17", "Treg"]

PSEUDOTIME_COLUMNS = ["monocle3_pseudotime", "paga_dpt_pseudotime", "slingshot_pseudotime"]


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


def first_cell_index_for_celltype(adata: sc.AnnData, root_celltype: str) -> int:
    matches = np.flatnonzero(adata.obs["celltype"].astype(str).to_numpy() == root_celltype)
    if matches.size == 0:
        raise ValueError(f"root_celltype {root_celltype!r} was not found.")
    return int(matches[0])


def monocle3_pseudotime(adata: sc.AnnData, root_celltype: str) -> np.ndarray:
    projected_points, mst, centroids = learn_graph(
        matrix=adata.obsm["X_umap"],
        clusters=adata.obs["celltype"],
    )
    root_cell = first_cell_index_for_celltype(adata, root_celltype)
    return order_cells(
        adata.obsm["X_umap"],
        centroids,
        mst=mst,
        projected_points=projected_points,
        root_cells=root_cell,
    )


def paga_dpt_pseudotime(adata: sc.AnnData, root_celltype: str) -> np.ndarray:
    work = adata.copy()
    if "X_pca" in work.obsm:
        sc.pp.neighbors(work, use_rep="X_pca")
    else:
        sc.pp.neighbors(work)

    sc.tl.diffmap(work)
    sc.pp.neighbors(work, use_rep="X_diffmap")
    sc.tl.paga(work, groups="celltype")

    work.uns["iroot"] = first_cell_index_for_celltype(work, root_celltype)
    sc.tl.dpt(work)
    return work.obs["dpt_pseudotime"].to_numpy(dtype=float)


def slingshot_root_mapping(adata: sc.AnnData, root_celltype: str) -> tuple[int, dict[str, int]]:
    categories = np.unique(adata.obs["celltype"].astype(str).to_numpy())
    mapping = {celltype: int(idx) for idx, celltype in enumerate(categories)}
    if root_celltype not in mapping:
        raise ValueError(f"root_celltype {root_celltype!r} was not found in Slingshot mapping.")
    return mapping[root_celltype], mapping


def slingshot_pseudotime(adata: sc.AnnData, root_celltype: str) -> tuple[np.ndarray, dict[str, object]]:
    work = adata.copy()
    start_node, mapping = slingshot_root_mapping(work, root_celltype)
    slingshot = Slingshot(
        work,
        celltype_key="celltype",
        obsm_key="X_umap",
        start_node=start_node,
        is_debugging=False,
    )
    slingshot.fit(num_epochs=1)
    audit = {
        "root_celltype": root_celltype,
        "slingshot_start_node": int(start_node),
        "slingshot_cluster_mapping": "|".join(f"{key}:{value}" for key, value in mapping.items()),
    }
    return np.asarray(slingshot.unified_pseudotime, dtype=float), audit


def compute_pseudotime_table(spec: dict[str, object]) -> tuple[pd.DataFrame, dict[str, object]]:
    adata = spec["adata"]
    root_celltype = str(spec["root_celltype"])

    table = pd.DataFrame(
        {
            "cell": adata.obs_names,
            "row_name": spec["row_name"],
            "lineage": spec["lineage"],
            "treatment": spec["treatment"],
            "root_celltype": root_celltype,
            "celltype": adata.obs["celltype"].astype(str).to_numpy(),
        }
    )
    table["monocle3_pseudotime"] = monocle3_pseudotime(adata.copy(), root_celltype)
    table["paga_dpt_pseudotime"] = paga_dpt_pseudotime(adata.copy(), root_celltype)
    slingshot_values, slingshot_audit = slingshot_pseudotime(adata.copy(), root_celltype)
    table["slingshot_pseudotime"] = slingshot_values

    audit = {
        "row_name": spec["row_name"],
        "lineage": spec["lineage"],
        "treatment": spec["treatment"],
        "n_cells": int(adata.n_obs),
        **slingshot_audit,
    }
    table["slingshot_start_node"] = audit["slingshot_start_node"]
    return table, audit


def compute_spearman(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (row_name, lineage, treatment), group in table.groupby(["row_name", "lineage", "treatment"], sort=False):
        for method_a, method_b in combinations(PSEUDOTIME_COLUMNS, 2):
            pair = group[["cell", method_a, method_b]].replace([np.inf, -np.inf], np.nan).dropna()
            if pair.shape[0] < 3:
                rho = np.nan
                p_value = np.nan
            else:
                result = spearmanr(pair[method_a], pair[method_b])
                rho = float(result.statistic)
                p_value = float(result.pvalue)

            rows.append(
                {
                    "row_name": row_name,
                    "lineage": lineage,
                    "treatment": treatment,
                    "method_a": method_a,
                    "method_b": method_b,
                    "n_cells": int(pair.shape[0]),
                    "spearman_rho": rho,
                    "p_value": p_value,
                }
            )
    return pd.DataFrame(rows)


def build_specs() -> list[dict[str, object]]:
    adata = sc.read_h5ad(ADATA_PATH)
    umap_cd4 = load_external_umap(UMAP_CD4_PATH)
    umap_cd8 = load_external_umap(UMAP_CD8_PATH)

    adata_cd4 = subset_with_external_umap(adata, umap_cd4, CD4_CELLTYPES)
    adata_cd8 = subset_with_external_umap(adata, umap_cd8, CD8_CELLTYPES)

    return [
        {
            "row_name": "CD8 pre",
            "lineage": "CD8",
            "treatment": "pre",
            "adata": adata_cd8[adata_cd8.obs["treatment"].astype(str) == "pre"].copy(),
            "root_celltype": "CD8_naive",
        },
        {
            "row_name": "CD8 post",
            "lineage": "CD8",
            "treatment": "post",
            "adata": adata_cd8[adata_cd8.obs["treatment"].astype(str) == "post"].copy(),
            "root_celltype": "CD8_mem",
        },
        {
            "row_name": "CD4 pre",
            "lineage": "CD4",
            "treatment": "pre",
            "adata": adata_cd4[adata_cd4.obs["treatment"].astype(str) == "pre"].copy(),
            "root_celltype": "Naive",
        },
        {
            "row_name": "CD4 post",
            "lineage": "CD4",
            "treatment": "post",
            "adata": adata_cd4[adata_cd4.obs["treatment"].astype(str) == "post"].copy(),
            "root_celltype": "Naive",
        },
    ]


def run(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    tables = []
    audits = []
    for spec in build_specs():
        if spec["adata"].n_obs == 0:
            raise ValueError(f"No cells found for {spec['row_name']}.")
        table, audit = compute_pseudotime_table(spec)
        tables.append(table)
        audits.append(audit)

    pseudotime = pd.concat(tables, ignore_index=True)
    correlations = compute_spearman(pseudotime)

    pseudotime.to_csv(output_dir / "external_umap_per_cell_pseudotime.csv", index=False)
    correlations.to_csv(output_dir / "external_umap_pseudotime_spearman_correlations.csv", index=False)
    pd.DataFrame(audits).to_csv(output_dir / "external_umap_pseudotime_root_audit.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-cell pseudotime and method correlations for ITx external UMAP trajectories."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/trajectory_robustness_external_umap/pseudotime_correlations_cd8post_mem_root"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.output_dir)


if __name__ == "__main__":
    main()
