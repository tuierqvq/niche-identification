"""HEST 样本清单构建与校验。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import resolve_path
from .utils import ensure_dir, get_logger, require_columns

LOGGER = get_logger(__name__)


def build_manifest(config: dict[str, Any], output_path: str | Path | None = None) -> pd.DataFrame:
    """基于 HEST metadata 构建 MVP 样本清单。

    参数:
        config: pipeline 配置字典。
        output_path: 可选输出 CSV 路径；为空时使用配置中的 ``paths.manifest``。

    返回值:
        每行一个样本的 manifest，包含 ST、metadata、WSI 路径与质量标记。

    异常/边界:
        当 HEST metadata 或样本核心文件缺失时抛出异常，避免后续步骤产生空结果。
    """

    data_root = resolve_path(config, config.get("paths", {}).get("data_root"))
    if data_root is None or not data_root.exists():
        raise FileNotFoundError(f"HEST data_root does not exist: {data_root}")

    meta_csv = data_root / "HEST_v1_3_0.csv"
    if not meta_csv.exists():
        raise FileNotFoundError(f"HEST metadata CSV does not exist: {meta_csv}")

    meta = pd.read_csv(meta_csv)
    require_columns(meta, ["id", "st_technology", "oncotree_code", "nb_genes"], "HEST metadata")

    sample_ids = [str(sample_id) for sample_id in config.get("samples", [])]
    if not sample_ids:
        raise ValueError("Config field 'samples' must contain at least one sample id.")

    rows: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        matched = meta.loc[meta["id"].astype(str) == sample_id]
        if matched.empty:
            raise ValueError(f"Sample {sample_id} was not found in {meta_csv}")

        row = matched.iloc[0].to_dict()
        metadata_path = data_root / "metadata" / f"{sample_id}.json"
        st_path = data_root / "st" / f"{sample_id}.h5ad"
        image_name = str(row.get("image_filename") or f"{sample_id}.tif")
        wsi_path = data_root / "wsis" / image_name

        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata JSON does not exist: {metadata_path}")
        if not st_path.exists():
            raise FileNotFoundError(f"ST AnnData file does not exist: {st_path}")
        if not wsi_path.exists():
            raise FileNotFoundError(f"WSI file does not exist: {wsi_path}")

        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata_json = json.load(handle)

        st_technology = str(row.get("st_technology", ""))
        nb_genes = int(row.get("nb_genes") or metadata_json.get("nb_genes") or 0)
        is_targeted = sample_id == "TENX30" or nb_genes < 5000
        include_in_mvp = st_technology.lower() == "visium"

        rows.append(
            {
                "sample_id": sample_id,
                "dataset_title": row.get("dataset_title"),
                "organ": row.get("organ"),
                "tissue": row.get("tissue"),
                "disease_state": row.get("disease_state"),
                "oncotree_code": row.get("oncotree_code"),
                "species": row.get("species"),
                "st_technology": st_technology,
                "preservation_method": row.get("preservation_method"),
                "magnification": row.get("magnification"),
                "nb_genes": nb_genes,
                "spots_under_tissue": row.get("spots_under_tissue") or metadata_json.get("spots_under_tissue"),
                "pixel_size_um": row.get("pixel_size_um_embedded")
                if pd.notna(row.get("pixel_size_um_embedded"))
                else row.get("pixel_size_um_estimated"),
                "spot_diameter_um": row.get("spot_diameter") or metadata_json.get("spot_diameter"),
                "inter_spot_dist_um": row.get("inter_spot_dist") or metadata_json.get("inter_spot_dist"),
                "st_path": str(st_path),
                "metadata_path": str(metadata_path),
                "wsi_path": str(wsi_path),
                "is_targeted_panel": bool(is_targeted),
                "expression_confidence": "lower_targeted_panel" if is_targeted else "standard_wta_or_probe",
                "include_in_mvp": bool(include_in_mvp),
                "exclusion_reason": "" if include_in_mvp else f"st_technology={st_technology}",
            }
        )

    manifest = pd.DataFrame(rows)
    if not manifest["include_in_mvp"].all():
        excluded = manifest.loc[~manifest["include_in_mvp"], ["sample_id", "exclusion_reason"]]
        raise ValueError(f"Non-Visium samples are not supported in the MVP:\n{excluded}")

    if output_path is None:
        output_path = resolve_path(config, config.get("paths", {}).get("manifest"))
    if output_path is None:
        output_path = resolve_path(config, "outputs/hest_gbm_card_ischia_mvp/manifests/selected_samples.csv")

    output = Path(output_path)
    ensure_dir(output.parent)
    manifest.to_csv(output, index=False, encoding="utf-8")
    LOGGER.info("Wrote sample manifest: %s", output)
    return manifest


def load_manifest(config: dict[str, Any], manifest_path: str | Path | None = None) -> pd.DataFrame:
    """读取并基础校验样本清单。

    参数:
        config: pipeline 配置字典。
        manifest_path: manifest CSV 路径；为空时使用配置中的路径。

    返回值:
        样本清单 DataFrame。

    异常/边界:
        文件不存在或缺少核心路径列时抛出异常。
    """

    path = Path(manifest_path) if manifest_path else resolve_path(config, config.get("paths", {}).get("manifest"))
    if path is None or not path.exists():
        raise FileNotFoundError(f"Manifest does not exist. Run make-manifest first: {path}")

    manifest = pd.read_csv(path)
    require_columns(manifest, ["sample_id", "st_path", "metadata_path", "wsi_path"], "manifest")
    return manifest
