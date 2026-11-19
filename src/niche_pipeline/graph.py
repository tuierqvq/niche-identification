"""空间邻域图构建工具。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class SpatialGraph:
    """保存 spot 邻域图的轻量结构。"""

    spot_ids: list[str]
    neighbors: np.ndarray
    distances: np.ndarray


def build_spatial_knn(
    spot_ids: list[str],
    coordinates: np.ndarray,
    k: int,
    include_self: bool = True,
) -> SpatialGraph:
    """根据二维空间坐标构建 kNN 邻域。

    参数:
        spot_ids: spot ID 列表，顺序必须与坐标一致。
        coordinates: ``n_spots x 2`` 坐标矩阵，通常为 Visium array row/col。
        k: 每个 spot 的邻居数量，不含自身。
        include_self: 是否把自身放在邻居列表第一列。

    返回值:
        ``SpatialGraph``，其中 neighbors 为 spot 下标矩阵。

    异常/边界:
        spot 数小于 2 或 k 非正时抛出异常；k 会被限制在 ``n_spots - 1``。
    """

    coords = np.asarray(coordinates, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coordinates must be an n_spots x 2 array.")
    if len(spot_ids) != coords.shape[0]:
        raise ValueError("spot_ids length must match coordinate rows.")
    if coords.shape[0] < 2:
        raise ValueError("At least two spots are required to build a graph.")
    if k <= 0:
        raise ValueError("k must be positive.")

    effective_k = min(k, coords.shape[0] - 1)
    query_k = effective_k + 1
    model = NearestNeighbors(n_neighbors=query_k, metric="euclidean")
    model.fit(coords)
    distances, indices = model.kneighbors(coords)

    if not include_self:
        indices = indices[:, 1:]
        distances = distances[:, 1:]
    return SpatialGraph(spot_ids=spot_ids, neighbors=indices, distances=distances)


def graph_to_edge_frame(graph: SpatialGraph, sample_id: str) -> pd.DataFrame:
    """将邻域图转换为边表。

    参数:
        graph: ``build_spatial_knn`` 输出。
        sample_id: 样本 ID。

    返回值:
        包含 source、target、rank、distance 的边表。

    异常/边界:
        自身边会保留 rank=0，便于复现邻域平均时的输入。
    """

    rows: list[dict[str, object]] = []
    for source_idx, source_id in enumerate(graph.spot_ids):
        for rank, target_idx in enumerate(graph.neighbors[source_idx]):
            rows.append(
                {
                    "sample_id": sample_id,
                    "source_spot_id": source_id,
                    "target_spot_id": graph.spot_ids[int(target_idx)],
                    "neighbor_rank": rank,
                    "distance": float(graph.distances[source_idx, rank]),
                }
            )
    return pd.DataFrame(rows)
