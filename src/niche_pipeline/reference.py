"""scRNA-seq reference 数据校验。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import pandas as pd

from .config import output_root, resolve_path
from .manifest import load_manifest
from .utils import ensure_dir, get_logger, save_json

LOGGER = get_logger(__name__)


def validate_reference(config: dict[str, Any], manifest_path: str | Path | None = None) -> dict[str, Any]:
    """校验 CARD/cell2location 所需的 scRNA-seq reference。

    参数:
        config: pipeline 配置字典。
        manifest_path: 可选 manifest 路径，用于统计与 ST 数据的基因交集。

    返回值:
        包含 reference 形状、cell type 数、共享基因数的报告字典。

    异常/边界:
        reference 路径为空、缺少 cell type 列、无共享基因或 counts 来源不可用时抛出异常。
    """

    ref_cfg = config.get("reference", {})
    ref_path = resolve_path(config, ref_cfg.get("h5ad"))
    if ref_path is None:
        raise ValueError(
            "reference.h5ad is required for CARD/cell2location. "
            "Please provide a GBM/glioma scRNA-seq reference .h5ad file."
        )
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference h5ad does not exist: {ref_path}")

    cell_type_key = str(ref_cfg.get("cell_type_key") or "cell_type")
    counts_layer = ref_cfg.get("counts_layer")
    gene_symbol_key = ref_cfg.get("gene_symbol_key")

    # read_h5ad(backed='r') 仍会加载 layers；大 reference 的 scaled layer 可能占用数十 GB。
    # 校验只需要元数据和基因名，不读取任何表达矩阵。
    with h5py.File(ref_path, "r") as handle:
        _, var_path = _count_source_paths(handle, counts_layer)
        obs = _read_element(handle["obs"])
        var = _read_element(handle["var"])
        count_var = var if var_path == "var" else _read_element(handle[var_path])

    if cell_type_key not in obs.columns:
        raise ValueError(f"Reference obs does not contain cell_type_key={cell_type_key!r}")
    if len(obs) < 10 or len(count_var) < 100:
        raise ValueError(f"Reference is unexpectedly small: {(len(obs), len(count_var))}")

    labels = obs[cell_type_key]
    if labels.isna().any() or labels.astype(str).str.strip().eq("").any():
        raise ValueError(f"Reference cell_type_key={cell_type_key!r} contains missing or empty labels.")
    batch_key = ref_cfg.get("batch_key")
    if batch_key and batch_key not in obs.columns:
        raise ValueError(f"Reference obs does not contain batch_key={batch_key!r}")
    cell_type_counts = labels.astype(str).value_counts()
    if cell_type_counts.empty:
        raise ValueError(f"Reference cell_type_key={cell_type_key!r} has no labels.")

    count_gene_names = _reference_gene_names(count_var, str(gene_symbol_key) if gene_symbol_key else None)
    shared_gene_report = _shared_gene_report(config, count_gene_names, manifest_path)
    report = {
        "reference_h5ad": str(ref_path),
        "n_cells": int(len(obs)),
        "n_genes": int(len(var)),
        "n_count_source_genes": int(len(count_gene_names)),
        "cell_type_key": cell_type_key,
        "gene_symbol_key": gene_symbol_key,
        "n_cell_types": int(cell_type_counts.shape[0]),
        "cell_type_counts": {str(k): int(v) for k, v in cell_type_counts.items()},
        "counts_source": counts_layer or ("raw.X" if var_path == "raw/var" else "X"),
        "shared_genes": shared_gene_report,
    }
    out_dir = ensure_dir(output_root(config) / "reference")
    save_json(report, out_dir / "reference_validation.json")
    LOGGER.info("Reference validation report written to %s", out_dir / "reference_validation.json")
    return report


def _shared_gene_report(
    config: dict[str, Any],
    reference_genes: pd.Index,
    manifest_path: str | Path | None,
) -> dict[str, Any]:
    """统计 reference 与每个 ST 样本的基因交集。

    参数:
        config: pipeline 配置字典。
        reference_genes: reference 基因名。
        manifest_path: 可选 manifest 路径。

    返回值:
        每个样本的 ST 基因数与共享基因数。

    异常/边界:
        manifest 不存在时返回空报告，让 prepare-reference 可在下载数据前单独检查 reference。
    """

    try:
        manifest = load_manifest(config, manifest_path)
    except FileNotFoundError:
        return {}

    ref_set = set(reference_genes.astype(str))
    report: dict[str, Any] = {}
    for row in manifest.itertuples(index=False):
        with h5py.File(row.st_path, "r") as handle:
            st_var = _read_element(handle["var"])
        st_genes = pd.Index(st_var.index.astype(str))
        shared = len(ref_set.intersection(st_genes))
        report[str(row.sample_id)] = {
            "st_genes": int(len(st_var)),
            "shared_genes": int(shared),
            "shared_fraction_of_st": float(shared / max(1, len(st_var))),
        }
        if shared < 100:
            raise ValueError(
                f"Reference shares too few genes with sample {row.sample_id}: {shared}. "
                "Check whether gene symbols match HEST var_names."
            )
    return report


def _reference_gene_names(var: pd.DataFrame, gene_symbol_key: str | None) -> pd.Index:
    """从计数来源的 var 中解析参考基因名；指定列不存在或基因名为空时失败。

    参数:
        var: 所选 count 来源的基因元数据。
        gene_symbol_key: 可选 ``var`` 列名，常用于 CELLxGENE 的 ``feature_name``。

    返回值:
        基因名 Index。

    异常/边界:
        若指定的 gene symbol 列不存在会抛出异常；重复 gene symbol 保留原值用于交集统计。
    """

    if gene_symbol_key:
        if gene_symbol_key not in var.columns:
            raise ValueError(f"Reference var does not contain gene_symbol_key={gene_symbol_key!r}")
        names = pd.Index(var[gene_symbol_key])
    else:
        names = pd.Index(var.index)
    if names.isna().any() or (names.astype(str).str.strip() == "").any():
        raise ValueError("Reference gene names must not be missing or empty.")
    return names.astype(str)


def _read_element(element: Any) -> Any:
    """读取单个 H5AD 元素；兼容 anndata 0.10 和 0.11+ 的公开读取接口。"""

    try:
        from anndata.io import read_elem
    except ImportError:  # anndata 0.10
        from anndata.experimental import read_elem
    return read_elem(element)


def _count_source_paths(handle: h5py.File, counts_layer: str | None) -> tuple[str, str]:
    """返回计数和对应 var 的 H5AD 路径；配置的 layer 优先，缺少来源时失败。"""

    if counts_layer:
        if "layers" not in handle or counts_layer not in handle["layers"]:
            raise ValueError(f"Reference does not contain counts_layer={counts_layer!r}")
        return f"layers/{counts_layer}", "var"
    if "raw" in handle and "X" in handle["raw"]:
        return "raw/X", "raw/var"
    if "X" not in handle:
        raise ValueError("Reference does not contain a count matrix in X or raw.X.")
    return "X", "var"


def load_reference_counts(config: dict[str, Any]) -> Any:
    """只读取 reference 所选的计数、obs 和对应 var，返回不含额外 layers 的 AnnData。

    输出的 X 已经是所选计数来源，供 CARD 导出；不会加载未使用的 scaled layer。
    """

    import anndata as ad

    ref_cfg = config.get("reference", {})
    path = resolve_path(config, ref_cfg.get("h5ad"))
    if path is None:
        raise ValueError("reference.h5ad is required for CARD/cell2location.")
    with h5py.File(path, "r") as handle:
        counts_path, var_path = _count_source_paths(handle, ref_cfg.get("counts_layer"))
        obs = _read_element(handle["obs"])
        var = _read_element(handle[var_path])
        counts = _read_element(handle[counts_path])
    return ad.AnnData(X=counts, obs=obs, var=var)
