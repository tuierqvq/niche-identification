"""Visium ST 表达矩阵预处理。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import output_root
from .manifest import load_manifest
from .utils import ensure_dir, get_logger, save_json

LOGGER = get_logger(__name__)


def preprocess_samples(config: dict[str, Any], manifest_path: str | Path | None = None) -> list[Path]:
    """对 manifest 中的 Visium 样本执行 Scanpy 标准预处理。

    参数:
        config: pipeline 配置字典。
        manifest_path: 可选 manifest 路径；为空时使用配置路径。

    返回值:
        预处理后 `.h5ad` 文件路径列表。

    异常/边界:
        需要安装 scanpy；当 QC 后 spot 或 gene 数不足以 PCA 时抛出异常。
    """

    import scanpy as sc

    manifest = load_manifest(config, manifest_path)
    params = config.get("preprocessing", {})
    out_root = output_root(config)
    h5ad_dir = ensure_dir(out_root / "preprocessed")
    qc_dir = ensure_dir(out_root / "qc")

    written: list[Path] = []
    summaries: list[dict[str, Any]] = []
    for row in manifest.itertuples(index=False):
        sample_id = str(row.sample_id)
        LOGGER.info("Preprocessing sample %s", sample_id)
        adata = sc.read_h5ad(row.st_path)
        before_shape = adata.shape

        if "counts" not in adata.layers:
            # 保留原始 counts，供 CARD/cell2location 使用；X 用于归一化表达分析。
            adata.layers["counts"] = adata.X.copy()

        adata.obs["sample_id"] = sample_id
        if "in_tissue" in adata.obs.columns:
            adata = adata[adata.obs["in_tissue"].astype(int) == 1].copy()

        _ensure_qc_metrics(adata)
        min_genes = int(params.get("min_genes_per_spot", 200))
        max_pct_mito = float(params.get("max_pct_mito", 25))
        keep_spots = (adata.obs["n_genes_by_counts"] >= min_genes) & (adata.obs["pct_counts_mito"] <= max_pct_mito)
        adata = adata[keep_spots].copy()

        min_cells = int(params.get("min_cells_per_gene", 3))
        sc.pp.filter_genes(adata, min_cells=min_cells)
        if "counts" not in adata.layers:
            raise RuntimeError("Internal error: counts layer was lost during filtering.")

        if adata.n_obs < 3 or adata.n_vars < 3:
            raise ValueError(f"Sample {sample_id} has too few spots/genes after QC: {adata.shape}")

        target_sum = float(params.get("target_sum", 10000))
        sc.pp.normalize_total(adata, target_sum=target_sum)
        sc.pp.log1p(adata)
        # 差异基因分析使用 log-normalized 表达；X 后续会被按基因缩放用于 PCA。
        # 原始整数计数继续保留在 layers['counts']，供 CARD 使用。
        adata.raw = adata.copy()

        n_top = min(int(params.get("n_top_genes", 3000)), max(1, adata.n_vars))
        if adata.n_vars > 1:
            sc.pp.highly_variable_genes(adata, n_top_genes=n_top, flavor="seurat", inplace=True)
        else:
            adata.var["highly_variable"] = True

        n_hvg = int(np.asarray(adata.var["highly_variable"]).sum())
        max_pcs = min(int(params.get("n_pcs", 50)), adata.n_obs - 1, max(1, n_hvg - 1))
        if max_pcs < 2:
            raise ValueError(f"Sample {sample_id} has too few HVGs for PCA after QC.")
        # Visium 矩阵较大，zero_center=False 可避免稀疏矩阵在本地或服务器上被意外 densify。
        sc.pp.scale(adata, max_value=10, zero_center=False)
        sc.tl.pca(adata, n_comps=max_pcs, mask_var="highly_variable", svd_solver="arpack")

        out_path = h5ad_dir / f"{sample_id}.preprocessed.h5ad"
        adata.write_h5ad(out_path)
        written.append(out_path)

        qc_csv = qc_dir / f"{sample_id}.spot_qc.csv"
        adata.obs.to_csv(qc_csv, encoding="utf-8")
        _write_qc_plot(adata, qc_dir / f"{sample_id}.qc_histograms.png")

        summaries.append(
            {
                "sample_id": sample_id,
                "spots_before": int(before_shape[0]),
                "genes_before": int(before_shape[1]),
                "spots_after": int(adata.n_obs),
                "genes_after": int(adata.n_vars),
                "n_hvg": n_hvg,
                "n_pcs": int(max_pcs),
                "is_targeted_panel": bool(getattr(row, "is_targeted_panel", False)),
            }
        )

    pd.DataFrame(summaries).to_csv(qc_dir / "preprocessing_summary.csv", index=False, encoding="utf-8")
    save_json({"preprocessed_h5ad": [str(path) for path in written]}, qc_dir / "preprocessing_outputs.json")
    return written


def _ensure_qc_metrics(adata: Any) -> None:
    """补齐 Scanpy QC 指标。

    参数:
        adata: AnnData 对象。

    返回值:
        无；直接在 ``obs`` 和 ``var`` 上补充 QC 列。

    异常/边界:
        当输入已包含 HEST 预计算 QC 列时仍会重算，保证过滤逻辑一致。
    """

    import scanpy as sc

    var_names = pd.Index(adata.var_names.astype(str))
    if "mito" not in adata.var.columns:
        adata.var["mito"] = var_names.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mito"], inplace=True, percent_top=None, log1p=True)


def _write_qc_plot(adata: Any, output_path: Path) -> None:
    """写出每个样本的 QC 直方图。

    参数:
        adata: 预处理后的 AnnData。
        output_path: PNG 输出路径。

    返回值:
        无。

    异常/边界:
        仅用于人工检查，绘图失败不会影响核心数据输出。
    """

    try:
        import matplotlib.pyplot as plt

        columns = ["total_counts", "n_genes_by_counts", "pct_counts_mito"]
        fig, axes = plt.subplots(1, len(columns), figsize=(12, 3))
        for axis, column in zip(axes, columns):
            axis.hist(np.asarray(adata.obs[column]), bins=40, color="#4477aa", alpha=0.85)
            axis.set_title(column)
            axis.set_xlabel(column)
            axis.set_ylabel("spots")
        fig.tight_layout()
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
    except Exception as exc:  # pragma: no cover - diagnostic artifact only
        LOGGER.warning("Could not write QC plot %s: %s", output_path, exc)
