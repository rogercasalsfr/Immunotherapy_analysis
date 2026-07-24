#!/usr/bin/env python3
"""BCC CD8 trajectory-method panels after first-pass QC.

Workflow:
1. Load the BCC metadata/count matrix used in preprocess_bcc.ipynb.
2. Keep only the BCC CD8 lineage:
   CD8_mem_T_cells, CD8_ex_T_cells, CD8_act_T_cells.
3. Run a simple first-pass QC.
4. Compute one CD8 UMAP using pre and post cells together.
5. Mask that same UMAP into pre/post rows and run Monocle3, PAGA, Slingshot.
"""

from __future__ import annotations

import argparse
import gc
import gzip
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).with_name(".mplconfig")))
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(__file__).with_name(".numba_cache")))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import patchworklib as pw
import scanpy as sc
from scipy import sparse

import methods as mt


BCC_DATA_DIR = Path("/home/roger/protocol/cancer-pseudotime-grn-workflow/scripts/data/GSE123813")
BCC_METADATA_PATH = BCC_DATA_DIR / "GSE123813_bcc_all_metadata.txt.gz"
BCC_COUNTS_PATH = BCC_DATA_DIR / "GSE123813_bcc_scRNA_counts.txt.gz"

CD8_LINEAGE = ["CD8_mem_T_cells", "CD8_ex_T_cells", "CD8_act_T_cells"]
CD8_ROOT_CELLTYPE = "CD8_mem_T_cells"

CD8_PALETTE = {
    "CD8_mem_T_cells": "green",
    "CD8_ex_T_cells": "red",
    "CD8_act_T_cells": "gold",
}


def load_bcc_metadata(path: Path) -> pd.DataFrame:
    metadata = pd.read_csv(path, sep="\t")
    required = {"cell.id", "patient", "treatment", "cluster"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    metadata = metadata.drop_duplicates("cell.id", keep="first").set_index("cell.id")
    metadata["celltype"] = metadata["cluster"].astype(str)
    return metadata


def load_cd8_counts(metadata: pd.DataFrame) -> sc.AnnData:
    cd8_obs = metadata[metadata["celltype"].isin(CD8_LINEAGE)].copy()
    cd8_cells = set(cd8_obs.index.astype(str))

    with gzip.open(BCC_COUNTS_PATH, "rb") as handle:
        header = handle.readline().decode().rstrip("\n").split("\t")

        selected_positions = []
        selected_cells = []
        for idx, cell in enumerate(header):
            if cell in cd8_cells:
                selected_positions.append(idx)
                selected_cells.append(cell)

        if not selected_cells:
            raise ValueError("No CD8 cells from the metadata were found in the BCC count matrix.")

        selected_positions = np.asarray(selected_positions, dtype=np.int64)
        gene_names = []
        row_chunks = []
        col_chunks = []
        data_chunks = []

        for gene_idx, line in enumerate(handle):
            gene_name, values_text = line.rstrip(b"\n").split(b"\t", 1)
            values = np.fromstring(values_text.decode(), sep="\t", dtype=np.float32)
            selected_values = values[selected_positions]
            nonzero = np.flatnonzero(selected_values)

            gene_names.append(gene_name.decode())
            if nonzero.size:
                row_chunks.append(nonzero.astype(np.int32, copy=False))
                col_chunks.append(np.full(nonzero.size, gene_idx, dtype=np.int32))
                data_chunks.append(selected_values[nonzero].astype(np.float32, copy=False))

    rows = np.concatenate(row_chunks) if row_chunks else np.array([], dtype=np.int32)
    cols = np.concatenate(col_chunks) if col_chunks else np.array([], dtype=np.int32)
    data = np.concatenate(data_chunks) if data_chunks else np.array([], dtype=np.float32)

    X = sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(len(selected_cells), len(gene_names)),
        dtype=np.float32,
    )
    obs = cd8_obs.loc[selected_cells].copy()
    var = pd.DataFrame(index=pd.Index(gene_names, dtype=str))

    del rows, cols, data, row_chunks, col_chunks, data_chunks
    gc.collect()

    adata = sc.AnnData(X=X, obs=obs, var=var)
    adata.var_names_make_unique()
    adata.layers["counts"] = adata.X.copy()
    return adata


def write_qc_summary(adata: sc.AnnData, output_path: Path, label: str) -> pd.DataFrame:
    metrics = ["n_genes_by_counts", "total_counts"]
    if "pct_counts_mt" in adata.obs:
        metrics.append("pct_counts_mt")

    rows = []
    for treatment, obs in adata.obs.groupby("treatment", observed=True):
        row = {
            "label": label,
            "treatment": treatment,
            "n_cells": int(obs.shape[0]),
        }
        for metric in metrics:
            values = obs[metric].astype(float)
            row.update(
                {
                    f"{metric}_median": float(values.median()),
                    f"{metric}_q05": float(values.quantile(0.05)),
                    f"{metric}_q95": float(values.quantile(0.95)),
                }
            )
        rows.append(row)

    rows.append(
        {
            "label": label,
            "treatment": "all",
            "n_cells": int(adata.n_obs),
            **{
                f"{metric}_{stat}": float(getattr(adata.obs[metric].astype(float), stat)())
                for metric in metrics
                for stat in ["median"]
            },
        }
    )
    summary = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    return summary


def save_qc_plots(adata: sc.AnnData, output_dir: Path, prefix: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    qc_keys = ["n_genes_by_counts", "total_counts"]
    if "pct_counts_mt" in adata.obs:
        qc_keys.append("pct_counts_mt")

    sc.pl.violin(
        adata,
        qc_keys,
        groupby="treatment",
        rotation=45,
        show=False,
        multi_panel=True,
    )
    plt.savefig(output_dir / f"{prefix}_qc_violin.png", dpi=200, bbox_inches="tight")
    plt.close("all")

    color = "pct_counts_mt" if "pct_counts_mt" in adata.obs else None
    sc.pl.scatter(
        adata,
        x="total_counts",
        y="n_genes_by_counts",
        color=color,
        show=False,
    )
    plt.savefig(output_dir / f"{prefix}_qc_counts_vs_genes.png", dpi=200, bbox_inches="tight")
    plt.close("all")


def first_pass_qc(
    adata: sc.AnnData,
    output_dir: Path,
    min_genes: int,
    min_cells: int,
    max_pct_mt: float,
    max_genes_quantile: float,
) -> sc.AnnData:
    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
    qc_vars = ["mt"] if bool(adata.var["mt"].any()) else None
    sc.pp.calculate_qc_metrics(adata, qc_vars=qc_vars, inplace=True, percent_top=None)

    write_qc_summary(adata, output_dir / "bcc_cd8_qc_before.csv", "before_qc")
    save_qc_plots(adata, output_dir, "before_qc")

    keep = adata.obs["n_genes_by_counts"] >= min_genes
    max_genes = float(adata.obs["n_genes_by_counts"].quantile(max_genes_quantile))
    keep &= adata.obs["n_genes_by_counts"] <= max_genes
    if "pct_counts_mt" in adata.obs:
        keep &= adata.obs["pct_counts_mt"] <= max_pct_mt

    filtered = adata[keep].copy()
    sc.pp.filter_genes(filtered, min_cells=min_cells)
    sc.pp.calculate_qc_metrics(filtered, qc_vars=qc_vars, inplace=True, percent_top=None)

    write_qc_summary(filtered, output_dir / "bcc_cd8_qc_after.csv", "after_qc")
    save_qc_plots(filtered, output_dir, "after_qc")

    pd.DataFrame(
        [
            {
                "min_genes": min_genes,
                "min_cells": min_cells,
                "max_pct_mt": max_pct_mt,
                "max_genes_quantile": max_genes_quantile,
                "max_genes_by_counts": max_genes,
                "n_cells_before": int(adata.n_obs),
                "n_cells_after": int(filtered.n_obs),
                "n_genes_before": int(adata.n_vars),
                "n_genes_after": int(filtered.n_vars),
            }
        ]
    ).to_csv(output_dir / "bcc_cd8_qc_thresholds.csv", index=False)

    return filtered


def compute_joint_umap(
    adata: sc.AnnData,
    n_top_genes: int,
    n_pcs: int,
    random_state: int,
) -> sc.AnnData:
    adata = adata.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor="seurat")
    if "highly_variable" in adata.var:
        adata = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(adata, max_value=10, zero_center=False)
    sc.tl.pca(adata, n_comps=n_pcs, random_state=random_state)
    sc.pp.neighbors(adata, n_pcs=n_pcs, random_state=random_state)
    sc.tl.umap(adata, random_state=random_state)
    return adata


def save_joint_umap(adata: sc.AnnData, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    adata.obs["celltype"] = adata.obs["celltype"].astype("category")
    adata.uns["celltype_colors"] = [
        CD8_PALETTE.get(category, "#808080")
        for category in adata.obs["celltype"].cat.categories
    ]

    sc.pl.umap(
        adata,
        color=["celltype", "treatment"],
        frameon=False,
        show=False,
        wspace=0.35,
    )
    plt.savefig(output_dir / "bcc_cd8_joint_post_qc_umap.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "bcc_cd8_joint_post_qc_umap.pdf", bbox_inches="tight")
    plt.close("all")

    coords = pd.DataFrame(
        adata.obsm["X_umap"],
        index=adata.obs_names,
        columns=["UMAP_1", "UMAP_2"],
    )
    coords.insert(0, "cell", coords.index)
    coords.to_csv(output_dir / "bcc_cd8_joint_post_qc_umap_coordinates.csv", index=False)


def build_trajectory_row(
    adata_subset: sc.AnnData,
    row_name: str,
    label_prefix: str,
    brick_figsize: tuple[float, float] = (3.8, 3.4),
):
    monocle_ax = pw.Brick(f"{label_prefix}_monocle3", figsize=brick_figsize)
    paga_ax = pw.Brick(f"{label_prefix}_paga", figsize=brick_figsize)
    slingshot_ax = pw.Brick(f"{label_prefix}_slingshot", figsize=brick_figsize)

    mt.monocle3(
        adata_subset.copy(),
        palette_cd8=CD8_PALETTE,
        ax=monocle_ax,
        title=f"{row_name} | Monocle3",
        show=False,
    )
    mt.run_paga(
        adata_subset.copy(),
        root_celltype=CD8_ROOT_CELLTYPE,
        ax=paga_ax,
        title=f"{row_name} | PAGA",
        show=False,
    )
    mt.run_slingshot(
        adata_subset.copy(),
        start_node=CD8_ROOT_CELLTYPE,
        ax=slingshot_ax,
        title=f"{row_name} | Slingshot",
        show=False,
    )

    return monocle_ax | paga_ax | slingshot_ax


def build_trajectory_canvas(adata: sc.AnnData):
    pw.clear()
    pw.param["margin"] = 0.3

    rows = []
    for treatment in ["pre", "post"]:
        subset = adata[adata.obs["treatment"].astype(str) == treatment].copy()
        if subset.n_obs == 0:
            raise ValueError(f"No BCC CD8 cells remain for treatment={treatment!r}.")
        rows.append(
            build_trajectory_row(
                subset,
                row_name=f"BCC CD8 {treatment}",
                label_prefix=f"bcc_cd8_qc_{treatment}",
            )
        )

    return rows[0] / rows[1]


def save_trajectory_summary(adata: sc.AnnData, output_dir: Path) -> None:
    rows = []
    for treatment in ["pre", "post"]:
        subset = adata[adata.obs["treatment"].astype(str) == treatment]
        counts = subset.obs["celltype"].astype(str).value_counts().to_dict()
        rows.append(
            {
                "row_name": f"BCC CD8 {treatment}",
                "n_cells": int(subset.n_obs),
                "n_genes": int(subset.n_vars),
                "celltypes": "|".join(sorted(subset.obs["celltype"].astype(str).unique())),
                "treatments": treatment,
                **{f"n_{key}": int(value) for key, value in counts.items()},
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "bcc_cd8_qc_joint_umap_treatment_masks_summary.csv", index=False)


def run(
    output_dir: Path,
    min_genes: int,
    min_cells: int,
    max_pct_mt: float,
    max_genes_quantile: float,
    n_top_genes: int,
    n_pcs: int,
    random_state: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_bcc_metadata(BCC_METADATA_PATH)
    adata = load_cd8_counts(metadata)
    adata = first_pass_qc(
        adata,
        output_dir=output_dir,
        min_genes=min_genes,
        min_cells=min_cells,
        max_pct_mt=max_pct_mt,
        max_genes_quantile=max_genes_quantile,
    )
    adata = compute_joint_umap(
        adata,
        n_top_genes=n_top_genes,
        n_pcs=n_pcs,
        random_state=random_state,
    )
    adata.write_h5ad(output_dir / "bcc_cd8_post_qc_joint_umap.h5ad", compression="gzip")

    save_joint_umap(adata, output_dir)
    save_trajectory_summary(adata, output_dir)

    canvas = build_trajectory_canvas(adata)
    canvas.savefig(str(output_dir / "bcc_cd8_qc_joint_umap_treatment_masks_6panel.png"), dpi=300)
    canvas.savefig(str(output_dir / "bcc_cd8_qc_joint_umap_treatment_masks_6panel.pdf"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run BCC CD8 QC, joint UMAP, and pre/post masked trajectory panels."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/bcc_cd8_qc_joint_umap_treatment_masks"),
    )
    parser.add_argument("--min-genes", type=int, default=200)
    parser.add_argument("--min-cells", type=int, default=3)
    parser.add_argument("--max-pct-mt", type=float, default=20.0)
    parser.add_argument("--max-genes-quantile", type=float, default=0.99)
    parser.add_argument("--n-top-genes", type=int, default=2000)
    parser.add_argument("--n-pcs", type=int, default=30)
    parser.add_argument("--random-state", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        output_dir=args.output_dir,
        min_genes=args.min_genes,
        min_cells=args.min_cells,
        max_pct_mt=args.max_pct_mt,
        max_genes_quantile=args.max_genes_quantile,
        n_top_genes=args.n_top_genes,
        n_pcs=args.n_pcs,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
