"""反卷积入口与可选的 cell2location 实现；默认使用标准 CARD。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .config import output_root, resolve_path
from .manifest import load_manifest
from .reference import validate_reference
from .utils import ensure_dir, get_logger, save_json

LOGGER = get_logger(__name__)


def run_deconvolution(config: dict[str, Any], manifest_path: str | Path | None = None) -> list[Path]:
    """按配置选择反卷积后端。

    参数:
        config: pipeline 配置字典。
        manifest_path: 可选 manifest 路径。

    返回值:
        每个样本 abundance CSV 路径列表。

    异常/边界:
        ``card`` 与 ``cell2location`` 均需要带细胞类型注释的 scRNA reference；
        ``card`` 还需要 R/CARD 环境。
    """

    method = str(config.get("deconvolution", {}).get("method", "card")).lower()
    if method == "card":
        from .card import run_card

        return run_card(config, manifest_path)
    if method == "cell2location":
        return run_cell2location(config, manifest_path)
    raise ValueError(f"Unsupported deconvolution.method={method!r}. Use 'card' or 'cell2location'.")


def run_cell2location(config: dict[str, Any], manifest_path: str | Path | None = None) -> list[Path]:
    """运行 cell2location 并输出 spot x cell type abundance。

    参数:
        config: pipeline 配置字典。
        manifest_path: 可选 manifest 路径。

    返回值:
        每个样本 abundance CSV 的路径列表。

    异常/边界:
        需要安装 cell2location、scvi-tools 和 PyTorch；reference 缺失时会先在校验阶段失败。
    """

    try:
        import scanpy as sc
        from cell2location.models import Cell2location, RegressionModel
    except ImportError as exc:
        raise ImportError(
            "cell2location is required for deconvolution. "
            "Install requirements on the GPU server, then rerun this command."
        ) from exc

    validate_reference(config, manifest_path)
    manifest = load_manifest(config, manifest_path)
    ref_cfg = config.get("reference", {})
    deconv_cfg = config.get("deconvolution", {})
    ref_path = resolve_path(config, ref_cfg.get("h5ad"))
    if ref_path is None:
        raise ValueError("reference.h5ad is required.")

    out_dir = ensure_dir(output_root(config) / "deconvolution")
    model_dir = ensure_dir(out_dir / "models")
    abundance_dir = ensure_dir(out_dir / "abundance")
    h5ad_dir = ensure_dir(out_dir / "h5ad")

    LOGGER.info("Training cell2location reference regression model")
    adata_ref = sc.read_h5ad(ref_path)
    adata_ref = _select_counts_source(adata_ref, ref_cfg.get("counts_layer"))
    adata_ref = _apply_gene_symbol_key(adata_ref, ref_cfg.get("gene_symbol_key"))
    cell_type_key = str(ref_cfg.get("cell_type_key") or "cell_type")
    batch_key = ref_cfg.get("batch_key") or None

    RegressionModel.setup_anndata(adata=adata_ref, labels_key=cell_type_key, batch_key=batch_key)
    reg_model = RegressionModel(adata_ref)
    _train_cell2location_model(
        reg_model,
        deconv_cfg=deconv_cfg,
        max_epochs=int(deconv_cfg.get("regression_max_epochs", 250)),
        batch_size=int(deconv_cfg.get("regression_batch_size", 2500)),
        train_size=float(deconv_cfg.get("train_size", 1.0)),
        stage="reference regression",
    )
    adata_ref = reg_model.export_posterior(
        adata_ref,
        sample_kwargs={
            "num_samples": int(deconv_cfg.get("posterior_num_samples", 1000)),
            "batch_size": int(deconv_cfg.get("posterior_batch_size", 2500)),
        },
    )
    reg_model.save(str(model_dir / "reference_regression"), overwrite=True)
    cell_state_df = _extract_cell_state_signatures(adata_ref)

    written: list[Path] = []
    reports: list[dict[str, Any]] = []
    for row in manifest.itertuples(index=False):
        sample_id = str(row.sample_id)
        preprocessed_path = output_root(config) / "preprocessed" / f"{sample_id}.preprocessed.h5ad"
        if not preprocessed_path.exists():
            raise FileNotFoundError(f"Preprocessed h5ad does not exist: {preprocessed_path}")

        LOGGER.info("Running cell2location spatial mapping for %s", sample_id)
        adata_vis = sc.read_h5ad(preprocessed_path)
        if "counts" not in adata_vis.layers:
            raise ValueError(f"{preprocessed_path} does not contain layers['counts']. Run preprocess first.")

        shared_genes = sorted(set(adata_vis.var_names).intersection(cell_state_df.index))
        if len(shared_genes) < 100:
            raise ValueError(f"Too few shared genes for {sample_id}: {len(shared_genes)}")
        adata_vis = adata_vis[:, shared_genes].copy()
        sample_cell_state = cell_state_df.loc[shared_genes, :].copy()

        Cell2location.setup_anndata(adata=adata_vis, layer="counts")
        spatial_model = Cell2location(
            adata_vis,
            cell_state_df=sample_cell_state,
            N_cells_per_location=int(deconv_cfg.get("n_cells_per_location", 8)),
            detection_alpha=float(deconv_cfg.get("detection_alpha", 20)),
        )
        _train_cell2location_model(
            spatial_model,
            deconv_cfg=deconv_cfg,
            max_epochs=int(deconv_cfg.get("spatial_max_epochs", 30000)),
            batch_size=deconv_cfg.get("spatial_batch_size"),
            train_size=float(deconv_cfg.get("train_size", 1.0)),
            stage=f"spatial mapping {sample_id}",
        )
        adata_vis = spatial_model.export_posterior(
            adata_vis,
            sample_kwargs={
                "num_samples": int(deconv_cfg.get("posterior_num_samples", 1000)),
                "batch_size": int(deconv_cfg.get("posterior_batch_size", 2500)),
            },
        )
        spatial_model.save(str(model_dir / sample_id), overwrite=True)

        abundance = _extract_abundance(adata_vis)
        abundance.insert(0, "spot_id", adata_vis.obs_names.astype(str))
        abundance.insert(0, "sample_id", sample_id)
        abundance_path = abundance_dir / f"{sample_id}.spot_celltype_abundance.csv"
        abundance.to_csv(abundance_path, index=False, encoding="utf-8")
        adata_vis.write_h5ad(h5ad_dir / f"{sample_id}.cell2location.h5ad")
        written.append(abundance_path)
        reports.append({"sample_id": sample_id, "shared_genes": len(shared_genes), "abundance_path": str(abundance_path)})

    save_json({"samples": reports}, out_dir / "deconvolution_summary.json")
    return written


def _select_counts_source(adata: Any, counts_layer: str | None) -> Any:
    """把 reference 的 count 来源放到 ``X``。

    参数:
        adata: reference AnnData。
        counts_layer: 可选 counts layer 名称。

    返回值:
        使用 count 矩阵的 AnnData；必要时会从 ``raw`` 重建对象。

    异常/边界:
        若配置了不存在的 layer 则抛出异常；未配置时优先使用 raw.X。
    """

    if counts_layer:
        if counts_layer not in adata.layers:
            raise ValueError(f"Reference counts_layer={counts_layer!r} does not exist.")
        adata.X = adata.layers[counts_layer].copy()
        return adata
    if adata.raw is not None and adata.raw.shape == adata.shape:
        adata.X = adata.raw.X.copy()
        return adata
    if adata.raw is not None:
        raw = adata.raw.to_adata()
        raw.obs = adata.obs.copy()
        raw.uns = adata.uns.copy()
        return raw
    return adata


def _apply_gene_symbol_key(adata: Any, gene_symbol_key: str | None) -> Any:
    """将 reference 的 gene symbol 列设置为 ``var_names``。

    参数:
        adata: reference AnnData。
        gene_symbol_key: 可选 ``var`` 列名，例如 CELLxGENE 的 ``feature_name``。

    返回值:
        更新后的 AnnData。

    异常/边界:
        重复 gene symbol 会通过 ``var_names_make_unique`` 去重；这会保留可匹配的首个 symbol，
        并防止 AnnData/cell2location 因重复索引失败。
    """

    if not gene_symbol_key:
        return adata
    if gene_symbol_key not in adata.var.columns:
        raise ValueError(f"Reference var does not contain gene_symbol_key={gene_symbol_key!r}")
    adata.var["original_var_names"] = adata.var_names.astype(str)
    adata.var_names = adata.var[gene_symbol_key].astype(str)
    adata.var_names_make_unique()
    return adata


def _train_cell2location_model(
    model: Any,
    deconv_cfg: dict[str, Any],
    max_epochs: int,
    batch_size: Any,
    train_size: float,
    stage: str,
) -> None:
    """以兼容新版/旧版 scvi-tools 的方式训练 cell2location 模型。

    参数:
        model: ``RegressionModel`` 或 ``Cell2location`` 实例。
        deconv_cfg: 反卷积配置块。
        max_epochs: 最大训练轮数。
        batch_size: batch size；为空时不传递该参数。
        train_size: 训练集比例。
        stage: 日志中的训练阶段名称。

    返回值:
        无。

    异常/边界:
        scvi-tools 的 PyroSviTrainMixin 使用 ``accelerator``/``device``；旧版 cell2location 可能仍
        只接受 ``use_gpu``。当新版参数被拒绝时会自动回退到旧参数。
    """

    base_kwargs: dict[str, Any] = {"max_epochs": max_epochs, "train_size": train_size}
    if batch_size not in (None, ""):
        base_kwargs["batch_size"] = int(batch_size)

    modern_kwargs = {**base_kwargs, **_resolve_training_device_kwargs(deconv_cfg)}
    LOGGER.info("Training %s with device kwargs: %s", stage, {k: modern_kwargs[k] for k in modern_kwargs if k in {"accelerator", "device"}})
    try:
        model.train(**modern_kwargs)
        return
    except TypeError as exc:
        message = str(exc)
        if not _looks_like_device_kwarg_error(message):
            raise
        LOGGER.warning(
            "Training %s rejected accelerator/device kwargs (%s); retrying with legacy use_gpu.",
            stage,
            message,
        )

    legacy_kwargs = {**base_kwargs, "use_gpu": _resolve_use_gpu(deconv_cfg.get("use_gpu", "auto"))}
    model.train(**legacy_kwargs)


def _resolve_training_device_kwargs(deconv_cfg: dict[str, Any]) -> dict[str, Any]:
    """解析 scvi-tools/lightning 训练设备参数。

    参数:
        deconv_cfg: 反卷积配置块。

    返回值:
        传给 ``model.train`` 的 ``accelerator`` 与 ``device`` 参数。

    异常/边界:
        当配置为 ``auto`` 且检测到 CUDA 时，默认只使用第 0 张可见 GPU，避免在多卡
        服务器上意外启动多进程训练。
    """

    accelerator = deconv_cfg.get("accelerator")
    device = deconv_cfg.get("device")
    devices = deconv_cfg.get("devices")

    if accelerator in (None, ""):
        use_gpu = deconv_cfg.get("use_gpu", "auto")
        accelerator = "gpu" if _resolve_use_gpu(use_gpu) else "cpu"
    accelerator = str(accelerator).lower()
    if accelerator == "auto":
        use_gpu = deconv_cfg.get("use_gpu", "auto")
        accelerator = "gpu" if _resolve_use_gpu(use_gpu) else "cpu"
    if accelerator == "cuda":
        accelerator = "gpu"

    if device not in (None, ""):
        resolved_device = _normalise_device(device, accelerator=accelerator)
    elif devices not in (None, "", "auto"):
        resolved_device = _device_from_devices(devices)
    elif devices == "auto":
        resolved_device = "auto"
    else:
        if accelerator == "gpu":
            resolved_device = _normalise_device(deconv_cfg.get("gpu_device", 0), accelerator=accelerator)
        else:
            resolved_device = "auto"

    return {"accelerator": accelerator, "device": resolved_device}


def _normalise_device(device: Any, accelerator: str = "auto") -> Any:
    """标准化 YAML/CLI 中的单卡 device 配置。

    参数:
        device: 可以是整数、数字字符串或 ``auto``。
        accelerator: 训练加速器类型。

    返回值:
        适合传给 scvi PyroSviTrainMixin 的 device 值。

    异常/边界:
        GPU 下会把数字编号返回为单元素列表，例如 ``[0]``。这样可以避免某些
        scvi/lightning 组合把整数 0 或字符串 "0" 误解成无效的 ``Trainer(devices=0)``。
    """

    accelerator = str(accelerator).lower()
    if isinstance(device, int):
        return [device] if accelerator == "gpu" else device
    text = str(device).strip()
    if text.lower() == "auto":
        return "auto"
    try:
        value = int(text)
        return [value] if accelerator == "gpu" else value
    except ValueError:
        return text


def _device_from_devices(devices: Any) -> Any:
    """兼容旧配置中的 ``devices`` 字段并转换为单个 ``device``。

    参数:
        devices: 整数、单元素列表或逗号分隔字符串。

    返回值:
        单个 device 编号或 ``auto``。

    异常/边界:
        多个 GPU 会被保守地取第一个，因为 cell2location/pyro 训练路径不应在此处
        误传 Trainer 的 ``devices`` 参数。
    """

    if isinstance(devices, (list, tuple)):
        if not devices:
            return "auto"
        if len(devices) > 1:
            LOGGER.warning("cell2location Pyro training received multiple devices %s; using the first one only.", devices)
        return _normalise_device(devices[0], accelerator="gpu")
    text = str(devices).strip()
    if text.lower() == "auto":
        return "auto"
    if "," in text:
        parts = [part.strip() for part in text.split(",") if part.strip()]
        if len(parts) > 1:
            LOGGER.warning("cell2location Pyro training received multiple devices %s; using the first one only.", text)
        return _normalise_device(parts[0], accelerator="gpu") if parts else "auto"
    return _normalise_device(text, accelerator="gpu")


def _looks_like_device_kwarg_error(message: str) -> bool:
    """判断 TypeError 是否来自训练设备参数兼容问题。"""

    needles = [
        "unexpected keyword argument 'accelerator'",
        'unexpected keyword argument "accelerator"',
        "unexpected keyword argument 'devices'",
        'unexpected keyword argument "devices"',
        "unexpected keyword argument 'device'",
        'unexpected keyword argument "device"',
        "unexpected keyword argument 'use_gpu'",
        'unexpected keyword argument "use_gpu"',
    ]
    return any(needle in message for needle in needles)


def _resolve_use_gpu(value: Any) -> bool:
    """解析 GPU 使用策略。

    参数:
        value: ``auto``、布尔值或字符串布尔值。

    返回值:
        是否使用 GPU。

    异常/边界:
        ``auto`` 会检查 ``torch.cuda.is_available``；torch 不可导入时返回 False。
    """

    if isinstance(value, bool):
        return value
    text = str(value).lower()
    if text == "auto":
        try:
            import torch

            return bool(torch.cuda.is_available())
        except Exception:
            return False
    return text in {"1", "true", "yes", "y"}


def _extract_cell_state_signatures(adata_ref: Any) -> pd.DataFrame:
    """从 RegressionModel posterior 中提取 cell type signature。

    参数:
        adata_ref: 已导出 posterior 的 reference AnnData。

    返回值:
        基因 x cell type signature 表。

    异常/边界:
        cell2location 版本字段名存在差异，因此按常见位置依次查找。
    """

    factor_names = [str(name) for name in adata_ref.uns.get("mod", {}).get("factor_names", [])]
    if not factor_names:
        raise ValueError("Could not find cell2location factor names in reference posterior.")

    preferred = "means_per_cluster_mu_fg"
    if preferred in adata_ref.varm.keys():
        frame = pd.DataFrame(adata_ref.varm[preferred], index=adata_ref.var_names)
        if frame.shape[1] == len(factor_names):
            frame.columns = factor_names
            return frame

    columns = [f"{preferred}_{name}" for name in factor_names]
    if all(column in adata_ref.var.columns for column in columns):
        frame = adata_ref.var[columns].copy()
        frame.columns = factor_names
        return frame

    raise ValueError("Could not extract cell state signatures from cell2location RegressionModel output.")


def _extract_abundance(adata_vis: Any) -> pd.DataFrame:
    """从 Cell2location posterior 中提取 abundance 矩阵。

    参数:
        adata_vis: 已导出 posterior 的空间 AnnData。

    返回值:
        spot x cell type abundance 表。

    异常/边界:
        优先使用 posterior mean；若版本只提供 q05，则退回到 q05。
    """

    for key in ("means_cell_abundance_w_sf", "q05_cell_abundance_w_sf"):
        if key in adata_vis.obsm.keys():
            frame = pd.DataFrame(adata_vis.obsm[key], index=adata_vis.obs_names)
            factor_names = [str(name) for name in adata_vis.uns.get("mod", {}).get("factor_names", [])]
            if frame.shape[1] == len(factor_names):
                frame.columns = factor_names
            else:
                frame.columns = [str(column).replace(f"{key}_", "") for column in frame.columns]
            return frame.reset_index(drop=True)
    raise ValueError("Could not find cell abundance in adata_vis.obsm.")
