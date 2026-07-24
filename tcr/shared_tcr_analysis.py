from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re

import h5py
import pandas as pd


TCR_PATH = Path("/home/roger/Github/Immunotherapy/GSE123813_scc_tcr.txt.gz")


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    adata_path: Path
    barcode_col: str | None
    celltype_col: str
    treatment_col: str
    patient_col: str | None = None
    sample_col: str | None = None


DATASETS = {
    "cd8": DatasetConfig(
        name="cd8",
        adata_path=Path("/home/roger/Itx_anl/Itx_anl/preprocess/adata_cd8_integrated.h5ad"),
        barcode_col=None,
        celltype_col="celltype",
        treatment_col="treatment",
        patient_col="patient",
    ),
    "cd4": DatasetConfig(
        name="cd4",
        adata_path=Path("/home/roger/Github/Master_thesis/python/totscd4/adata.h5ad"),
        barcode_col="barcode",
        celltype_col="cluster_redefined",
        treatment_col="treatment",
        sample_col="orig.ident",
    ),
}


def decode_h5_value(value):
    if isinstance(value, bytes):
        return value.decode()
    return value


def read_obs_value(obj) -> pd.Series:
    if isinstance(obj, h5py.Dataset):
        values = [decode_h5_value(item) for item in obj[:]]
        return pd.Series(values)

    categories = [decode_h5_value(item) for item in obj["categories"][:]]
    codes = obj["codes"][:]
    values = [categories[code] if code >= 0 else None for code in codes]
    return pd.Series(values)


def read_obs_table(adata_path: Path) -> pd.DataFrame:
    with h5py.File(adata_path, "r") as handle:
        obs_group = handle["obs"]
        obs_data = {key: read_obs_value(obs_group[key]) for key in obs_group.keys()}

    obs = pd.DataFrame(obs_data)
    if "_index" in obs.columns:
        obs.index = obs["_index"]
    return obs


def infer_patient(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None

    match = re.search(r"(su\d+)", str(value))
    return match.group(1) if match else None


def extract_chains(seq_string: object, chain: str) -> list[str]:
    if pd.isna(seq_string):
        return []

    chains = []
    for part in str(seq_string).split(";"):
        if part.startswith(f"{chain}:"):
            chains.append(part.split(":", 1)[1])
    return chains


def normalize_chain_list(chain_list: list[str]) -> tuple[str, ...]:
    return tuple(sorted(set(chain_list)))


def chain_signature(chain_list: tuple[str, ...]) -> str | None:
    return "|".join(chain_list) if chain_list else None


def clonotype_signature(tra_list: tuple[str, ...], trb_list: tuple[str, ...]) -> str | None:
    if not tra_list and not trb_list:
        return None

    tra_sig = "|".join(tra_list) if tra_list else "-"
    trb_sig = "|".join(trb_list) if trb_list else "-"
    return f"TRA:{tra_sig};TRB:{trb_sig}"


def prepare_tcr_table(tcr_path: Path) -> pd.DataFrame:
    tcr = pd.read_csv(tcr_path, sep="\t", index_col=0)

    tcr["TRA_aa_list"] = tcr["cdr3s_aa"].apply(lambda value: normalize_chain_list(extract_chains(value, "TRA")))
    tcr["TRB_aa_list"] = tcr["cdr3s_aa"].apply(lambda value: normalize_chain_list(extract_chains(value, "TRB")))
    tcr["TRA_nt_list"] = tcr["cdr3s_nt"].apply(lambda value: normalize_chain_list(extract_chains(value, "TRA")))
    tcr["TRB_nt_list"] = tcr["cdr3s_nt"].apply(lambda value: normalize_chain_list(extract_chains(value, "TRB")))

    tcr["tra_signature_aa"] = tcr["TRA_aa_list"].apply(chain_signature)
    tcr["trb_signature_aa"] = tcr["TRB_aa_list"].apply(chain_signature)
    tcr["clonotype_signature_aa"] = tcr.apply(
        lambda row: clonotype_signature(row["TRA_aa_list"], row["TRB_aa_list"]),
        axis=1,
    )
    return tcr


def annotate_cells(obs: pd.DataFrame, tcr: pd.DataFrame, config: DatasetConfig) -> pd.DataFrame:
    annotated = obs.copy()

    if config.barcode_col:
        annotated["tcr_barcode"] = annotated[config.barcode_col]
    else:
        annotated["tcr_barcode"] = annotated.index

    tcr_subset = tcr[
        [
            "cdr3s_nt",
            "cdr3s_aa",
            "tra_signature_aa",
            "trb_signature_aa",
            "clonotype_signature_aa",
            "TRA_aa_list",
            "TRB_aa_list",
        ]
    ]

    annotated = annotated.join(tcr_subset, on="tcr_barcode")
    annotated["has_tcr"] = annotated["clonotype_signature_aa"].notna()
    annotated["celltype_label"] = annotated[config.celltype_col].astype("string")
    annotated["treatment_label"] = annotated[config.treatment_col].astype("string")

    if config.patient_col and config.patient_col in annotated.columns:
        annotated["patient_label"] = annotated[config.patient_col].astype("string")
    elif config.sample_col and config.sample_col in annotated.columns:
        annotated["patient_label"] = annotated[config.sample_col].apply(infer_patient).astype("string")
        annotated["sample_label"] = annotated[config.sample_col].astype("string")
    else:
        annotated["patient_label"] = pd.Series(index=annotated.index, dtype="string")

    return annotated


def summarize_signatures(
    annotated: pd.DataFrame,
    signature_col: str,
    config: DatasetConfig,
    min_cells: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    subset = annotated.loc[annotated[signature_col].notna()].copy()
    grouped = subset.groupby(signature_col, dropna=True)

    summary = grouped.agg(
        n_cells=("tcr_barcode", "size"),
        n_celltypes=("celltype_label", pd.Series.nunique),
        n_treatments=("treatment_label", pd.Series.nunique),
        n_patients=("patient_label", pd.Series.nunique),
    )

    summary["celltypes"] = grouped["celltype_label"].apply(
        lambda values: "|".join(sorted(pd.Series(values).dropna().unique()))
    )
    summary["treatments"] = grouped["treatment_label"].apply(
        lambda values: "|".join(sorted(pd.Series(values).dropna().unique()))
    )
    summary["patients"] = grouped["patient_label"].apply(
        lambda values: "|".join(sorted(pd.Series(values).dropna().unique()))
    )

    if signature_col == "clonotype_signature_aa":
        summary["tra_signature_aa"] = grouped["tra_signature_aa"].first()
        summary["trb_signature_aa"] = grouped["trb_signature_aa"].first()

    counts = (
        subset.groupby([signature_col, "celltype_label"])
        .size()
        .unstack(fill_value=0)
        .sort_index(axis=1)
    )

    if signature_col == "clonotype_signature_aa":
        celltype_columns = [column for column in counts.columns]
        summary = summary.join(counts[celltype_columns])

    shared = summary.loc[(summary["n_cells"] >= min_cells) & (summary["n_celltypes"] >= 2)].copy()
    shared = shared.sort_values(["n_celltypes", "n_cells"], ascending=[False, False])

    presence = counts.gt(0).astype(int)
    sharing_matrix = presence.T.dot(presence)
    sharing_matrix.index.name = "celltype"
    sharing_matrix.columns.name = "celltype"

    return summary.sort_values("n_cells", ascending=False), shared, sharing_matrix


def write_outputs(
    dataset_name: str,
    signature_name: str,
    summary: pd.DataFrame,
    shared: pd.DataFrame,
    sharing_matrix: pd.DataFrame,
    output_dir: Path,
) -> None:
    summary.to_csv(output_dir / f"{dataset_name}_{signature_name}_summary.csv")
    shared.to_csv(output_dir / f"{dataset_name}_{signature_name}_shared_across_celltypes.csv")
    sharing_matrix.to_csv(output_dir / f"{dataset_name}_{signature_name}_celltype_sharing_matrix.csv")


def run_analysis(dataset_name: str, min_cells: int, output_dir: Path) -> None:
    config = DATASETS[dataset_name]
    obs = read_obs_table(config.adata_path)
    tcr = prepare_tcr_table(TCR_PATH)
    annotated = annotate_cells(obs, tcr, config)

    coverage = {
        "dataset": dataset_name,
        "n_cells_total": len(annotated),
        "n_cells_with_tcr": int(annotated["has_tcr"].sum()),
        "tcr_coverage_fraction": round(float(annotated["has_tcr"].mean()), 4),
        "n_unique_clonotypes_aa": int(annotated["clonotype_signature_aa"].nunique(dropna=True)),
        "n_unique_trb_aa": int(annotated["trb_signature_aa"].nunique(dropna=True)),
    }
    pd.DataFrame([coverage]).to_csv(output_dir / f"{dataset_name}_coverage.csv", index=False)

    for signature_col, signature_name in [
        ("clonotype_signature_aa", "clonotype_aa"),
        ("trb_signature_aa", "trb_aa"),
    ]:
        summary, shared, sharing_matrix = summarize_signatures(
            annotated=annotated,
            signature_col=signature_col,
            config=config,
            min_cells=min_cells,
        )
        write_outputs(dataset_name, signature_name, summary, shared, sharing_matrix, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize shared TCRs across cell types.")
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS.keys()) + ["all"],
        default="all",
        help="Dataset to analyze.",
    )
    parser.add_argument(
        "--min-cells",
        type=int,
        default=2,
        help="Minimum number of cells required for a signature to be kept in the shared table.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory where CSV outputs will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset_names = sorted(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    for dataset_name in dataset_names:
        run_analysis(dataset_name=dataset_name, min_cells=args.min_cells, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
