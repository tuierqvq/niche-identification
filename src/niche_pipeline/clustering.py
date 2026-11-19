"""生态位聚类入口：默认 ISCHIA，保留两级聚类作为备选。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

from .config import output_root
from .manifest import load_manifest
from .utils import ensure_dir, get_logger, require_columns, save_json

LOGGER = get_logger(__name__)


def cluster_niches(config: dict[str, Any], manifest_path: str | Path | None = None) -> dict[str, Path]:
    """按 clustering.method 选择 ISCHIA 或原有两级聚类，返回兼容的标签路径。"""

    method = str(config.get("clustering", {}).get("method", "ischia")).lower()
    if method == "ischia":
        from .ischia import run_ischia

        return run_ischia(config, manifest_path)
    if method == "two_stage":
        return cluster_two_stage_niches(config, manifest_path)
    raise ValueError(f"Unsupported clustering.method={method!r}. Use 'ischia' or 'two_stage'.")


def cluster_two_stage_niches(config: dict[str, Any], manifest_path: str | Path | None = None) -> dict[str, Path]:
    """执行单样本 local niche 聚类与跨样本 global 对齐。

    参数:
        config: pipeline 配置字典。
        manifest_path: 可选 manifest 路径。

    返回值:
        关键输出文件路径字典。

    异常/边界:
        需要 scanpy/leidenalg；若 local cluster 数少于 2，则 global 对齐会退化为单类。
    """

    import anndata as ad
    import scanpy as sc

    manifest = load_manifest(config, manifest_path)
    cluster_cfg = config.get("clustering", {})
    out_root = output_root(config)
    cluster_dir = ensure_dir(out_root / "clusters")
    local_dir = ensure_dir(cluster_dir / "local")

    local_frames: list[pd.DataFrame] = []
    signature_frames: list[pd.DataFrame] = []
    for row in manifest.itertuples(index=False):
        sample_id = str(row.sample_id)
        LOGGER.info("Clustering local niches for %s", sample_id)
        features = _load_feature_frame(out_root / "features" / f"{sample_id}.niche_features.csv")
        numeric = _numeric_feature_matrix(features)
        scaled = StandardScaler().fit_transform(numeric)
        adata = ad.AnnData(X=scaled)
        sc.pp.neighbors(adata, n_neighbors=int(cluster_cfg.get("local_neighbors", 15)), use_rep="X")
        sc.tl.leiden(
            adata,
            resolution=float(cluster_cfg.get("local_resolution", 0.8)),
            key_added="local_niche",
            random_state=int(cluster_cfg.get("random_seed", 42)),
        )
        labels = pd.DataFrame(
            {
                "sample_id": sample_id,
                "spot_id": features["spot_id"].astype(str),
                "local_niche": [f"{sample_id}_Niche_{label}" for label in adata.obs["local_niche"].astype(str)],
            }
        )
        labels["neighbor_agreement"] = _neighbor_agreement(
            labels,
            out_root / "graphs" / f"{sample_id}.spatial_knn_k{int(config.get('features', {}).get('spatial_k', 12))}.csv",
        )
        labels.to_csv(local_dir / f"{sample_id}.local_niche_labels.csv", index=False, encoding="utf-8")
        local_frames.append(labels)

        signature = numeric.copy()
        signature["local_niche"] = labels["local_niche"].to_numpy()
        signature_frame = signature.groupby("local_niche").mean(numeric_only=True)
        signature_frame.insert(0, "sample_id", sample_id)
        signature_frame.insert(1, "local_niche", signature_frame.index.astype(str))
        signature_frames.append(signature_frame.reset_index(drop=True))

    all_local = pd.concat(local_frames, axis=0, ignore_index=True)
    signatures = pd.concat(signature_frames, axis=0, ignore_index=True)
    taxonomy = _align_global_niches(signatures, int(cluster_cfg.get("global_niches", 8)))
    all_labels = all_local.merge(taxonomy[["local_niche", "global_niche"]], on="local_niche", how="left")
    all_labels = _assign_confidence(all_labels, taxonomy, cluster_cfg)

    local_path = cluster_dir / "spot_local_niche_labels.csv"
    taxonomy_path = cluster_dir / "global_niche_taxonomy.csv"
    global_path = cluster_dir / "spot_global_niche_labels.csv"
    signature_path = cluster_dir / "local_niche_signatures.csv"
    all_local.to_csv(local_path, index=False, encoding="utf-8")
    signatures.to_csv(signature_path, index=False, encoding="utf-8")
    taxonomy.to_csv(taxonomy_path, index=False, encoding="utf-8")
    all_labels.to_csv(global_path, index=False, encoding="utf-8")
    save_json(
        {
            "local_labels": str(local_path),
            "global_labels": str(global_path),
            "taxonomy": str(taxonomy_path),
            "signatures": str(signature_path),
        },
        cluster_dir / "cluster_outputs.json",
    )
    return {
        "local_labels": local_path,
        "global_labels": global_path,
        "taxonomy": taxonomy_path,
        "signatures": signature_path,
    }


def _load_feature_frame(path: Path) -> pd.DataFrame:
    """读取 feature CSV 并检查核心列。"""

    if not path.exists():
        raise FileNotFoundError(f"Niche feature CSV does not exist: {path}")
    frame = pd.read_csv(path)
    require_columns(frame, ["sample_id", "spot_id"], "niche features")
    return frame


def _numeric_feature_matrix(features: pd.DataFrame) -> pd.DataFrame:
    """提取可用于聚类的数值特征。"""

    exclude = {"sample_id", "spot_id"}
    numeric = features.drop(columns=[column for column in exclude if column in features.columns]).select_dtypes(include=[np.number])
    if numeric.empty:
        raise ValueError("No numeric niche features are available for clustering.")
    return numeric.fillna(0.0)


def _align_global_niches(signatures: pd.DataFrame, requested_niches: int) -> pd.DataFrame:
    """将 local niche signature 聚类成 global niche。"""

    feature_cols = [column for column in signatures.columns if column not in {"sample_id", "local_niche"}]
    if len(signatures) == 1:
        taxonomy = signatures[["sample_id", "local_niche"]].copy()
        taxonomy["global_niche"] = "Global_Niche_0"
        return taxonomy

    n_clusters = max(1, min(requested_niches, len(signatures)))
    scaled = StandardScaler().fit_transform(signatures[feature_cols].fillna(0.0))
    model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(scaled)
    taxonomy = signatures[["sample_id", "local_niche"]].copy()
    taxonomy["global_niche"] = [f"Global_Niche_{label}" for label in labels]
    return taxonomy


def _neighbor_agreement(labels: pd.DataFrame, edge_path: Path) -> pd.Series:
    """计算每个 spot 邻居标签一致性。"""

    if not edge_path.exists():
        return pd.Series(np.nan, index=labels.index)
    edges = pd.read_csv(edge_path)
    label_map = labels.set_index("spot_id")["local_niche"].to_dict()
    agreements: dict[str, float] = {}
    for spot_id, group in edges.groupby("source_spot_id"):
        source_label = label_map.get(str(spot_id))
        neighbor_ids = [str(value) for value in group.loc[group["neighbor_rank"] > 0, "target_spot_id"]]
        neighbor_labels = [label_map.get(neighbor_id) for neighbor_id in neighbor_ids]
        neighbor_labels = [value for value in neighbor_labels if value is not None]
        agreements[str(spot_id)] = float(np.mean([value == source_label for value in neighbor_labels])) if neighbor_labels else np.nan
    return labels["spot_id"].astype(str).map(agreements)


def _assign_confidence(labels: pd.DataFrame, taxonomy: pd.DataFrame, cluster_cfg: dict[str, Any]) -> pd.DataFrame:
    """标记 low-confidence / mixed niche。

    参数:
        labels: spot 级标签。
        taxonomy: local 到 global 的映射。
        cluster_cfg: 聚类配置。

    返回值:
        增加 confidence 与 confidence_reason 的标签表。

    异常/边界:
        该标记是 MVP 的保守过滤，不替代人工病理/生物学命名。
    """

    del taxonomy
    min_spots = int(cluster_cfg.get("min_spots_per_local_niche", 30))
    min_agreement = float(cluster_cfg.get("min_neighbor_agreement", 0.25))
    sizes = labels["local_niche"].value_counts()
    labels = labels.copy()
    labels["confidence"] = "high"
    labels["confidence_reason"] = ""

    small_mask = labels["local_niche"].map(sizes) < min_spots
    labels.loc[small_mask, "confidence"] = "low"
    labels.loc[small_mask, "confidence_reason"] = "small_local_niche"

    agreement_mask = labels["neighbor_agreement"].notna() & (labels["neighbor_agreement"] < min_agreement)
    labels.loc[agreement_mask, "confidence"] = "low"
    labels.loc[agreement_mask & (labels["confidence_reason"] == ""), "confidence_reason"] = "low_neighbor_agreement"
    labels.loc[agreement_mask & (labels["confidence_reason"] != "low_neighbor_agreement"), "confidence_reason"] += ";low_neighbor_agreement"
    return labels
