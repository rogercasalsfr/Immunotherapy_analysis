**Immunotherapy trajectory inference**

This project investigates the dynamics of T cell differentiation in the tumor microenvironment (TME) using single-cell RNA sequencing data (GEO123813) from patients pre- and post-therapy. Through trajectory inference, functional characterization, TCR clonotype tracking, and regulon analysis, it aims to uncover key mechanisms and drivers of the immune T cell response to anti-PD1 therapy.

See graphical approach.
![image](https://github.com/user-attachments/assets/f7cf5462-edfe-4052-b075-171cc6faeee6)

This repository contains the scripts and notebooks required to reproduce the results presented in the study.

## Repository Layout
- **`preprocess/`** — Python (Scanpy) preprocessing, QC, and cell-type annotation for the SCC and BCC datasets (`preprocess.ipynb`, `preprocess_bcc.ipynb`), plus the original R/Seurat preprocessing (`Preprocess.Rmd`).
- **`trajectory_inference/`** — Trajectory and pseudotime inference. `methods.ipynb` runs the Python pipeline (Scanpy + Slingshot via `pyslingshot`); `TI_anl_CD4.Rmd` / `TI_anl_CD8.Rmd` contain the original R/Monocle3 analysis.
- **`methods/`** — Benchmarking and robustness checks across trajectory methods and UMAP embedding strategies (per-condition vs. external/fixed embeddings), including pseudotime correlation and reproducibility panels.
- **`tcr/`** — TCR clonotype conservation analysis: joins scRNA-seq cell states to paired TRA/TRB clonotypes and evaluates conservation across cell states and treatment (pre/post).
- **`functional_analysis/`** — Functional enrichment analysis of CD4/CD8 populations (R).
- **`regulon_inference/`** — Regulon / gene regulatory network inference with pySCENIC (`InferGRN.sh`, `InferGRN.ipynb`) and downstream network figures (R).

## Environment and Reproducibility
Analyses use both **Python** and **R**:
- **Python** (Scanpy, `pyslingshot`, pandas, h5py) — preprocessing, trajectory inference, method benchmarking, and TCR clonotype analysis.
- **R** (Seurat, Monocle3, SeuratWrappers) — the original preprocessing/trajectory pipeline and functional enrichment analysis.

For transparency and reproducibility, the full R package list and versions are provided in **`sessionInfo_Itx.txt`**.

## Analysis Workflow
- **Data preprocessing:** Scanpy (Python) and **Seurat** (R)
- **Trajectory inference:** Slingshot via `pyslingshot` (Python) and **Monocle3** (R)
- **TCR clonotype conservation:** custom Python pipeline joining scRNA-seq and TCR-seq data
- **Regulon activity inference:** carried out with **pySCENIC**

## Network Visualization
Transcription factor (**TF**) interaction networks were generated using **Cytoscape**.
Details of the Cytoscape environment are included in the corresponding **`session_cytoscape.cys`** file.
