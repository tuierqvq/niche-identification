"""Niche 解释表与 marker 基因摘要。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import output_root
from .manifest import load_manifest
from .utils import ensure_dir, get_logger, require_columns, save_json

LOGGER = get_logger(__name__)


def annotate_niches(config: dict[str, Any], manifest_path: str | Path | None = None) -> dict[str, Path]:
    """生成 Niche 注释摘要、样本比例和 marker 表。

    参数:
        config: pipeline 配置字典。
        manifest_path: 可选 manifest 路径。

    返回值:
        关键输出文件路径字典。

    异常/边界:
        marker 基因分析依赖预处理 h5ad；若某样本缺少足够标签类别则跳过该样本 marker。
    """

    manifest = load_manifest(config, manifest_path)
    out_root = output_root(config)
    label_path = out_root / "clusters" / "spot_global_niche_labels.csv"
    if not label_path.exists():
        raise FileNotFoundError(f"Global niche labels do not exist: {label_path}")

    labels = pd.read_csv(label_path)
    require_columns(labels, ["sample_id", "spot_id", "global_niche", "local_niche"], "global labels")
    annotation_dir = ensure_dir(out_root / "annotation")
    summary = _composition_summary(config, labels)
    summary_path = annotation_dir / "niche_annotation_table.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8")

    proportion = (
        labels.groupby(["sample_id", "global_niche"]).size().rename("n_spots").reset_index()
    )
    totals = proportion.groupby("sample_id")["n_spots"].transform("sum")
    proportion["proportion"] = proportion["n_spots"] / totals
    proportion_path = annotation_dir / "sample_niche_proportion.csv"
    proportion.to_csv(proportion_path, index=False, encoding="utf-8")

    marker_path = _write_marker_tables(config, manifest, labels, annotation_dir)
    outputs = {
        "annotation_table": summary_path,
        "sample_niche_proportion": proportion_path,
        "marker_genes": marker_path,
    }
    save_json({key: str(value) for key, value in outputs.items()}, annotation_dir / "annotation_outputs.json")
    return outputs


def _composition_summary(config: dict[str, Any], labels: pd.DataFrame) -> pd.DataFrame:
    """按 global niche 汇总细胞组成和置信度。"""

    out_root = output_root(config)
    feature_frames: list[pd.DataFrame] = []
    for sample_id in sorted(labels["sample_id"].astype(str).unique()):
        feature_path = out_root / "features" / f"{sample_id}.niche_features.csv"
        if not feature_path.exists():
            continue
        feature = pd.read_csv(feature_path)
        feature_frames.append(feature)
    if not feature_frames:
        raise FileNotFoundError("No niche feature CSV files are available for annotation.")

    features = pd.concat(feature_frames, axis=0, ignore_index=True)
    merged = labels.merge(features, on=["sample_id", "spot_id"], how="left")
    comp_cols = [column for column in merged.columns if column.startswith("composition_")]
    rows: list[dict[str, object]] = []
    for niche, group in merged.groupby("global_niche"):
        means = group[comp_cols].mean(numeric_only=True).sort_values(ascending=False)
        top = [column.replace("composition_", "") for column in means.head(3).index]
        rows.append(
            {
                "global_niche": niche,
                "n_spots": int(group.shape[0]),
                "n_samples": int(group["sample_id"].nunique()),
                "top_cell_types": ";".join(top),
                "mean_neighbor_agreement": float(group["neighbor_agreement"].mean()) if "neighbor_agreement" in group else np.nan,
                "low_confidence_fraction": float((group.get("confidence", "high") == "low").mean())
                if "confidence" in group
                else 0.0,
                **{f"mean_{column}": float(value) for column, value in means.items()},
            }
        )
    return pd.DataFrame(rows).sort_values(["n_samples", "n_spots"], ascending=[False, False])


def _write_marker_tables(
    config: dict[str, Any],
    manifest: pd.DataFrame,
    labels: pd.DataFrame,
    annotation_dir: Path,
) -> Path:
    """对每个样本计算 global niche marker 基因。

    参数:
        config: pipeline 配置字典。
        manifest: 样本清单。
        labels: spot 级 global niche 标签。
        annotation_dir: 输出目录。

    返回值:
        marker CSV 路径。

    异常/边界:
        单样本有效 niche 数小于 2 时跳过；所有样本都跳过时写出空表。
    """

    import scanpy as sc

    out_root = output_root(config)
    rows: list[dict[str, object]] = []
    for row in manifest.itertuples(index=False):
        sample_id = str(row.sample_id)
        h5ad_path = out_root / "preprocessed" / f"{sample_id}.preprocessed.h5ad"
        if not h5ad_path.exists():
            LOGGER.warning("Skipping marker genes for %s because h5ad is missing.", sample_id)
            continue
        adata = sc.read_h5ad(h5ad_path)
        sample_labels = labels.loc[labels["sample_id"].astype(str) == sample_id].set_index("spot_id")
        common = [spot for spot in adata.obs_names.astype(str) if spot in sample_labels.index]
        if len(common) < 10:
            continue
        adata = adata[common].copy()
        adata.obs["global_niche"] = sample_labels.loc[common, "global_niche"].astype(str).to_numpy()
        if adata.obs["global_niche"].nunique() < 2:
            continue
        sc.tl.rank_genes_groups(adata, groupby="global_niche", method="wilcoxon", n_genes=30)
        result = adata.uns["rank_genes_groups"]
        for group in result["names"].dtype.names:
            names = result["names"][group]
            scores = result["scores"][group]
            pvals_adj = result["pvals_adj"][group]
            if "logfoldchanges" in result and group in result["logfoldchanges"].dtype.names:
                logfoldchanges = result["logfoldchanges"][group]
            else:
                logfoldchanges = [np.nan] * len(names)
            for rank, gene in enumerate(names):
                rows.append(
                    {
                        "sample_id": sample_id,
                        "global_niche": group,
                        "rank": rank + 1,
                        "gene": str(gene),
                        "score": float(scores[rank]),
                        "logfoldchange": float(logfoldchanges[rank]) if np.isfinite(logfoldchanges[rank]) else np.nan,
                        "pval_adj": float(pvals_adj[rank]),
                    }
                )

    marker_path = annotation_dir / "niche_marker_genes.csv"
    pd.DataFrame(rows).to_csv(marker_path, index=False, encoding="utf-8")
    return marker_path
