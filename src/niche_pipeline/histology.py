"""H&E patch 提取与空间叠图。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import math

import numpy as np
import pandas as pd

from .config import output_root
from .manifest import load_manifest
from .utils import ensure_dir, get_logger

LOGGER = get_logger(__name__)


def extract_patches_and_overlays(config: dict[str, Any], manifest_path: str | Path | None = None) -> dict[str, Path]:
    """提取 spot-centered H&E patch 并保存空间叠图。

    参数:
        config: pipeline 配置字典。
        manifest_path: 可选 manifest 路径。

    返回值:
        包含 patch manifest 与 overlay 目录的路径字典。

    异常/边界:
        patch 提取依赖 openslide；若未安装，只写 overlay 并给出明确警告。
    """

    import scanpy as sc

    manifest = load_manifest(config, manifest_path)
    hist_cfg = config.get("histology", {})
    out_root = output_root(config)
    hist_dir = ensure_dir(out_root / "histology")
    overlay_dir = ensure_dir(hist_dir / "overlays")
    patch_dir = ensure_dir(hist_dir / "patches")

    label_path = out_root / "clusters" / "spot_global_niche_labels.csv"
    labels = pd.read_csv(label_path) if label_path.exists() else pd.DataFrame()
    patch_rows: list[dict[str, object]] = []

    for row in manifest.itertuples(index=False):
        sample_id = str(row.sample_id)
        h5ad_path = out_root / "preprocessed" / f"{sample_id}.preprocessed.h5ad"
        if not h5ad_path.exists():
            LOGGER.warning("Skipping histology for %s because preprocessed h5ad is missing.", sample_id)
            continue
        adata = sc.read_h5ad(h5ad_path)
        _write_spatial_overlay(adata, labels, sample_id, overlay_dir / f"{sample_id}.global_niche_overlay.png")

        if not bool(hist_cfg.get("extract_patches", True)):
            continue
        try:
            patch_rows.extend(_extract_sample_patches(adata, row, hist_cfg, patch_dir / sample_id))
        except ImportError as exc:
            LOGGER.warning("OpenSlide is not available; patch extraction skipped for %s: %s", sample_id, exc)
        except Exception as exc:
            LOGGER.warning("Patch extraction failed for %s: %s", sample_id, exc)

    patch_manifest = hist_dir / "patch_manifest.csv"
    pd.DataFrame(patch_rows).to_csv(patch_manifest, index=False, encoding="utf-8")
    return {"patch_manifest": patch_manifest, "overlay_dir": overlay_dir, "patch_dir": patch_dir}


def _write_spatial_overlay(adata: Any, labels: pd.DataFrame, sample_id: str, output_path: Path) -> None:
    """基于 h5ad 内嵌 downscaled H&E 写出空间标签叠图。

    参数:
        adata: 预处理后的 AnnData。
        labels: spot 标签表，可为空。
        sample_id: 样本 ID。
        output_path: PNG 输出路径。

    返回值:
        无。

    异常/边界:
        缺少内嵌图像时只绘制坐标散点，保证 QC 图仍可生成。
    """

    import matplotlib.pyplot as plt

    ensure_dir(output_path.parent)
    spatial = np.asarray(adata.obsm["spatial"], dtype=float) if "spatial" in adata.obsm.keys() else None
    if spatial is None:
        return

    image = None
    scale = 1.0
    spatial_uns = adata.uns.get("spatial", {})
    if spatial_uns:
        library = next(iter(spatial_uns.values()))
        image = library.get("images", {}).get("downscaled_fullres")
        scale = float(library.get("scalefactors", {}).get("tissue_downscaled_fullres_scalef", 1.0))

    fig, axis = plt.subplots(figsize=(7, 7))
    if image is not None:
        axis.imshow(image)
    coords = spatial * scale

    sample_labels = pd.DataFrame(index=adata.obs_names.astype(str))
    if not labels.empty:
        selected = labels.loc[labels["sample_id"].astype(str) == sample_id].set_index("spot_id")
        sample_labels = sample_labels.join(selected[["global_niche"]], how="left")
    if "global_niche" in sample_labels.columns and sample_labels["global_niche"].notna().any():
        from matplotlib.lines import Line2D

        # 所有切片共用同一标签顺序和调色板，避免按各切片首次出现顺序分配颜色。
        niches = sorted(labels["global_niche"].dropna().astype(str).unique())
        cmap = plt.get_cmap("tab20", len(niches))
        palette = {niche: cmap(index) for index, niche in enumerate(niches)}
        selected_labels = sample_labels["global_niche"].fillna("unlabeled")
        colors = [palette.get(str(niche), (0.65, 0.65, 0.65, 1)) for niche in selected_labels]
        axis.scatter(coords[:, 0], coords[:, 1], c=colors, s=8, alpha=0.85)
        counts = selected_labels.value_counts()
        handles = [Line2D([0], [0], marker="o", linestyle="", color=palette[niche],
                          label=f"{niche} (n={int(counts.get(niche, 0))})", markersize=5) for niche in niches]
        axis.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=7, frameon=False)
    else:
        axis.scatter(coords[:, 0], coords[:, 1], s=8, color="#cc6677", alpha=0.8)
    axis.set_title(sample_id)
    axis.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _extract_sample_patches(adata: Any, row: Any, hist_cfg: dict[str, Any], sample_patch_dir: Path) -> list[dict[str, object]]:
    """为单个样本提取 spot-centered patch。

    参数:
        adata: 预处理后的 AnnData。
        row: manifest 行，需包含 WSI 路径与 pixel size。
        hist_cfg: H&E 配置。
        sample_patch_dir: 样本 patch 输出目录。

    返回值:
        patch manifest 行列表。

    异常/边界:
        默认只提取配置中的最大数量，避免本地调试时一次写出上万张 patch。
    """

    import openslide
    from PIL import Image, ImageDraw

    ensure_dir(sample_patch_dir)
    slide = openslide.OpenSlide(str(row.wsi_path))
    pixel_size_um = float(row.pixel_size_um)
    if not math.isfinite(pixel_size_um) or pixel_size_um <= 0:
        raise ValueError(f"Invalid pixel_size_um for {row.sample_id}: {row.pixel_size_um}")

    patch_size_um = float(hist_cfg.get("patch_size_um", 224))
    output_px = int(hist_cfg.get("output_px", 224))
    patch_px = max(16, int(round(patch_size_um / pixel_size_um)))
    half = patch_px // 2
    max_patches = hist_cfg.get("max_patches_per_sample")
    max_patches = None if max_patches in (None, "") else int(max_patches)

    obs = adata.obs.copy()
    obs["spot_id"] = adata.obs_names.astype(str)
    if max_patches is not None:
        obs = obs.head(max_patches)

    rows: list[dict[str, object]] = []
    contact_images: list[Image.Image] = []
    slide_w, slide_h = slide.dimensions
    for item in obs.itertuples(index=False):
        spot_id = str(item.spot_id)
        center_x = int(getattr(item, "pxl_col_in_fullres"))
        center_y = int(getattr(item, "pxl_row_in_fullres"))
        x0 = min(max(center_x - half, 0), max(slide_w - patch_px, 0))
        y0 = min(max(center_y - half, 0), max(slide_h - patch_px, 0))
        patch = slide.read_region((x0, y0), 0, (patch_px, patch_px)).convert("RGB")
        patch = patch.resize((output_px, output_px), Image.Resampling.LANCZOS)
        patch_path = sample_patch_dir / f"{spot_id}.png"
        patch.save(patch_path)
        if len(contact_images) < 100:
            contact_images.append(patch.copy())
        rows.append(
            {
                "sample_id": row.sample_id,
                "spot_id": spot_id,
                "patch_path": str(patch_path),
                "center_x_fullres": center_x,
                "center_y_fullres": center_y,
                "patch_size_um": patch_size_um,
                "patch_size_px_fullres": patch_px,
            }
        )

    if contact_images:
        contact_path = sample_patch_dir / f"{row.sample_id}.contact_sheet.png"
        _write_contact_sheet(contact_images, int(hist_cfg.get("contact_sheet_cols", 10)), contact_path)
    slide.close()
    return rows


def _write_contact_sheet(images: list[Any], cols: int, output_path: Path) -> None:
    """写出 patch contact sheet。"""

    from PIL import Image, ImageDraw

    if not images:
        return
    cols = max(1, cols)
    width, height = images[0].size
    rows = int(math.ceil(len(images) / cols))
    sheet = Image.new("RGB", (cols * width, rows * height), "white")
    for idx, image in enumerate(images):
        x = (idx % cols) * width
        y = (idx // cols) * height
        sheet.paste(image, (x, y))
    draw = ImageDraw.Draw(sheet)
    draw.rectangle([0, 0, sheet.size[0] - 1, sheet.size[1] - 1], outline="black")
    sheet.save(output_path)
