"""Niche feature matrix 构建。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import output_root
from .graph import build_spatial_knn, graph_to_edge_frame
from .manifest import load_manifest
from .utils import ensure_dir, get_logger, require_columns, save_json

LOGGER = get_logger(__name__)


def build_features(config: dict[str, Any], manifest_path: str | Path | None = None) -> list[Path]:
    """整合 abundance、空间邻域和表达 PCA，构建 Niche 特征。

    参数:
        config: pipeline 配置字典。
        manifest_path: 可选 manifest 路径。

    返回值:
        每个样本 feature CSV 路径列表。

    异常/边界:
        需要先完成 preprocess 与 deconvolve；缺少 abundance CSV 会抛出异常。
    """

    import scanpy as sc

    manifest = load_manifest(config, manifest_path)
    feature_cfg = config.get("features", {})
    out_root = output_root(config)
    feature_dir = ensure_dir(out_root / "features")
    graph_dir = ensure_dir(out_root / "graphs")
    gene_score_dir = ensure_dir(out_root / "features" / "gene_scores")

    written: list[Path] = []
    reports: list[dict[str, Any]] = []
    for row in manifest.itertuples(index=False):
        sample_id = str(row.sample_id)
        LOGGER.info("Building niche features for %s", sample_id)
        h5ad_path = out_root / "preprocessed" / f"{sample_id}.preprocessed.h5ad"
        abundance_path = out_root / "deconvolution" / "abundance" / f"{sample_id}.spot_celltype_abundance.csv"
        if not h5ad_path.exists():
            raise FileNotFoundError(f"Preprocessed h5ad does not exist: {h5ad_path}")
        if not abundance_path.exists():
            raise FileNotFoundError(f"Abundance CSV does not exist: {abundance_path}")

        adata = sc.read_h5ad(h5ad_path)
        abundance = pd.read_csv(abundance_path)
        require_columns(abundance, ["sample_id", "spot_id"], "abundance")
        abundance = abundance.loc[abundance["sample_id"].astype(str) == sample_id].copy()
        abundance = abundance.set_index("spot_id")
        abundance = abundance.reindex(adata.obs_names.astype(str))
        if abundance.isna().any().any():
            missing = int(abundance.isna().any(axis=1).sum())
            raise ValueError(f"{sample_id} abundance is missing {missing} preprocessed spots.")

        cell_type_cols = [column for column in abundance.columns if column != "sample_id"]
        comp = abundance[cell_type_cols].astype(float)
        if bool(feature_cfg.get("normalize_abundance_to_composition", True)):
            row_sums = comp.sum(axis=1).replace(0, np.nan)
            comp = comp.div(row_sums, axis=0).fillna(0.0)

        coordinates = _spot_coordinates(adata)
        k = int(feature_cfg.get("spatial_k", 12))
        graph = build_spatial_knn(list(adata.obs_names.astype(str)), coordinates, k=k, include_self=True)
        edge_frame = graph_to_edge_frame(graph, sample_id)
        edge_path = graph_dir / f"{sample_id}.spatial_knn_k{k}.csv"
        edge_frame.to_csv(edge_path, index=False, encoding="utf-8")

        comp_values = comp.to_numpy(dtype=float)
        smooth_values = comp_values[graph.neighbors].mean(axis=1)
        smooth = pd.DataFrame(
            smooth_values,
            index=comp.index,
            columns=[f"neighborhood_{column}" for column in comp.columns],
        )
        comp.columns = [f"composition_{column}" for column in comp.columns]

        pieces = [comp, smooth]
        if bool(feature_cfg.get("include_expression_pcs", True)):
            pcs = _expression_pcs(adata, int(feature_cfg.get("n_expression_pcs", 30)))
            pieces.append(pcs)

        gene_scores = _gene_set_scores(adata, feature_cfg.get("gene_sets", {}))
        if not gene_scores.empty:
            gene_scores.to_csv(gene_score_dir / f"{sample_id}.gene_scores.csv", encoding="utf-8")
            pieces.append(gene_scores)

        features = pd.concat(pieces, axis=1)
        features.insert(0, "spot_id", adata.obs_names.astype(str))
        features.insert(0, "sample_id", sample_id)
        out_path = feature_dir / f"{sample_id}.niche_features.csv"
        features.to_csv(out_path, index=False, encoding="utf-8")
        written.append(out_path)
        reports.append(
            {
                "sample_id": sample_id,
                "n_spots": int(features.shape[0]),
                "n_features": int(features.shape[1] - 2),
                "graph_edges": int(edge_frame.shape[0]),
                "feature_path": str(out_path),
            }
        )

    save_json({"samples": reports}, feature_dir / "feature_summary.json")
    return written


def _spot_coordinates(adata: Any) -> np.ndarray:
    """选择用于邻域图的 spot 坐标。

    参数:
        adata: 预处理后的 AnnData。

    返回值:
        ``n_spots x 2`` 坐标矩阵。

    异常/边界:
        优先使用 Visium array row/col，因为它更适合固定阵列邻域；缺失时退回 fullres pixel 坐标。
    """

    if {"array_row", "array_col"}.issubset(set(adata.obs.columns)):
        return adata.obs[["array_row", "array_col"]].to_numpy(dtype=float)
    if "spatial" in adata.obsm.keys():
        return np.asarray(adata.obsm["spatial"], dtype=float)
    raise ValueError("AnnData must contain array_row/array_col or obsm['spatial'] coordinates.")


def _expression_pcs(adata: Any, n_pcs: int) -> pd.DataFrame:
    """提取表达 PCA 特征。

    参数:
        adata: 预处理后的 AnnData。
        n_pcs: 需要保留的 PC 数。

    返回值:
        spot x PC DataFrame。

    异常/边界:
        若预处理阶段没有 PCA，则返回空表而不是临时重算，避免隐藏流程错误。
    """

    if "X_pca" not in adata.obsm.keys():
        return pd.DataFrame(index=adata.obs_names)
    pcs = np.asarray(adata.obsm["X_pca"])
    keep = min(n_pcs, pcs.shape[1])
    return pd.DataFrame(pcs[:, :keep], index=adata.obs_names, columns=[f"expression_pc_{idx + 1}" for idx in range(keep)])


def _gene_set_scores(adata: Any, gene_sets: dict[str, list[str]] | None) -> pd.DataFrame:
    """计算可选 gene set score。

    参数:
        adata: 预处理后的 AnnData。
        gene_sets: ``名称 -> 基因列表`` 的字典。

    返回值:
        spot x gene set score DataFrame；无配置时为空表。

    异常/边界:
        每个 gene set 至少需要 2 个可匹配基因；不满足时跳过该 gene set。
    """

    if not gene_sets:
        return pd.DataFrame(index=adata.obs_names)

    import scanpy as sc

    scores = pd.DataFrame(index=adata.obs_names)
    available = set(map(str, adata.var_names))
    for name, genes in gene_sets.items():
        matched = [gene for gene in genes if gene in available]
        if len(matched) < 2:
            LOGGER.warning("Skipping gene set %s because fewer than 2 genes are present.", name)
            continue
        score_name = f"geneset_{name}"
        sc.tl.score_genes(adata, gene_list=matched, score_name=score_name, use_raw=False)
        scores[score_name] = adata.obs[score_name].to_numpy()
    return scores
