import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:
    import patchworklib as pw
except ImportError:
    pw = None

try:
    from pyslingshot import Slingshot
except ImportError:
    Slingshot = None

from py_monocle import (
    learn_graph,
    order_cells,
    compute_cell_states,
    regression_analysis,
    differential_expression_genes,
)


def _resolve_condition_value(value, condition):
    if isinstance(value, dict):
        if condition not in value:
            raise KeyError(f"Missing value for condition '{condition}'.")
        return value[condition]
    return value


def _build_celltype_colors(adata, palette_cd8):
    if palette_cd8 is None:
        return None
    return adata.obs["celltype"].map(palette_cd8)


def _add_celltype_legend(ax, adata, palette_cd8):
    if not palette_cd8:
        return

    handles = []
    for celltype in adata.obs["celltype"].astype(str).unique():
        if celltype not in palette_cd8:
            continue
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=palette_cd8[celltype],
                markeredgecolor="none",
                markersize=5,
                label=celltype,
            )
        )

    if handles:
        ax.legend(
            handles=handles,
            title="celltype",
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            fontsize=8,
            title_fontsize=9,
        )


def monocle3(
    adata,
    palette_cd8=None,
    ax=None,
    root_cells=0,
    title="Monocle3",
    show=True,
    point_size=8,
    edge_linewidth=1,
):
    projected_points, mst, centroids = learn_graph(
        matrix=adata.obsm["X_umap"],
        clusters=adata.obs["celltype"],
    )
    pseudotime = order_cells(
        adata.obsm["X_umap"],
        centroids,
        mst=mst,
        projected_points=projected_points,
        root_cells=root_cells,
    )
    adata.obs["monocle3_pseudotime"] = pseudotime

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    umap = adata.obsm["X_umap"]
    color = _build_celltype_colors(adata, palette_cd8)
    if color is None:
        color = pseudotime
        scatter_kwargs = {"cmap": "viridis"}
    else:
        scatter_kwargs = {}

    ax.scatter(umap[:, 0], umap[:, 1], c=color, s=point_size, **scatter_kwargs)

    edges = np.array(mst.nonzero()).T
    for edge in edges:
        ax.plot(centroids[edge, 0], centroids[edge, 1], c="black", linewidth=edge_linewidth)

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    _add_celltype_legend(ax, adata, palette_cd8)

    if show:
        plt.show()
    return ax



def run_paga(
    adata,
    root_celltype,
    ax=None,
    title="PAGA",
    show=True,
):
    if "X_pca" in adata.obsm:
        sc.pp.neighbors(adata, use_rep="X_pca")
    else:
        sc.pp.neighbors(adata)

    sc.tl.diffmap(adata)
    sc.pp.neighbors(adata, use_rep="X_diffmap")
    sc.tl.paga(adata, groups="celltype")

    iroot = np.flatnonzero(adata.obs["celltype"] == root_celltype)
    if iroot.size == 0:
        raise ValueError(f"root_celltype '{root_celltype}' was not found in adata.obs['celltype'].")

    adata.uns["iroot"] = int(iroot[0])
    sc.tl.dpt(adata)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    sc.pl.paga(adata, color="celltype", ax=ax, title=title, show=False)

    if show:
        plt.show()
    return ax



def run_slingshot(
    adata,
    start_node,
    ax=None,
    title="Slingshot",
    show=True,
    fit_kwargs=None,
    line_color="black",
    line_width=2,
    line_alpha=0.9,
):
    if Slingshot is None:
        raise ImportError(
            "pyslingshot is required for run_slingshot(). Install it in the notebook environment first."
        )

    slingshot = Slingshot(
        adata,
        celltype_key="celltype",
        obsm_key="X_umap",
        start_node=start_node,
        is_debugging="verbose",
    )

    resolved_fit_kwargs = {"num_epochs": 1}
    if fit_kwargs is not None:
        resolved_fit_kwargs.update(fit_kwargs)
    slingshot.fit(**resolved_fit_kwargs)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    slingshot.plotter.clusters(
        ax,
        labels=np.arange(slingshot.num_clusters),
        s=6,
        alpha=0.6,
    )

    if slingshot.lineages is None:
        raise RuntimeError("Slingshot did not produce any lineages.")

    for lineage_idx, lineage in enumerate(slingshot.lineages):
        lineage_points = slingshot.cluster_centres[np.array(list(lineage), dtype=int)]
        ax.plot(
            lineage_points[:, 0],
            lineage_points[:, 1],
            color=line_color,
            linewidth=line_width,
            alpha=line_alpha,
            label=f"Lineage {lineage_idx}",
        )

    ax.legend(frameon=False, fontsize=8)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")

    if show:
        plt.show()
    return ax



def plot_trajectory_methods_by_condition(
    adata,
    condition_key,
    palette_cd8=None,
    root_celltype=None,
    start_node=0,
    conditions=None,
    brick_figsize=(3.8, 3.4),
    margin=0.3,
    fit_kwargs=None,
    show=True,
):
    if pw is None:
        raise ImportError(
            "patchworklib is required for plot_trajectory_methods_by_condition(). Install it in the notebook environment first."
        )

    if condition_key not in adata.obs:
        raise KeyError(f"'{condition_key}' was not found in adata.obs.")
    if root_celltype is None:
        raise ValueError("root_celltype is required. You can pass a single value or a dict keyed by condition.")

    if conditions is None:
        conditions = adata.obs[condition_key].astype(str).unique().tolist()

    pw.param["margin"] = margin
    canvas = None

    for condition in conditions:
        subset = adata[adata.obs[condition_key].astype(str) == str(condition)].copy()
        if subset.n_obs == 0:
            continue

        condition_root = _resolve_condition_value(root_celltype, condition)
        condition_start = _resolve_condition_value(start_node, condition)

        monocle_ax = pw.Brick(f"{condition}_monocle3", figsize=brick_figsize)
        paga_ax = pw.Brick(f"{condition}_paga", figsize=brick_figsize)
        slingshot_ax = pw.Brick(f"{condition}_slingshot", figsize=brick_figsize)

        monocle3(
            subset.copy(),
            palette_cd8=palette_cd8,
            ax=monocle_ax,
            title=f"{condition} | Monocle3",
            show=False,
        )
        run_paga(
            subset.copy(),
            root_celltype=condition_root,
            ax=paga_ax,
            title=f"{condition} | PAGA",
            show=False,
        )
        run_slingshot(
            subset.copy(),
            start_node=condition_start,
            ax=slingshot_ax,
            title=f"{condition} | Slingshot",
            show=False,
            fit_kwargs=fit_kwargs,
        )

        row = monocle_ax | paga_ax | slingshot_ax
        canvas = row if canvas is None else canvas / row

    if canvas is None:
        raise ValueError("No conditions produced a plot. Check your condition names and filters.")

    if show:
        canvas.savefig()
    return canvas
