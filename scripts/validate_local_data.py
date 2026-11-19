"""校验本地 HEST GBM MVP 数据是否完整。"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    """运行 manifest 构建与 AnnData 字段检查。

    参数:
        无；命令行参数由 argparse 解析。

    返回值:
        无。

    异常/边界:
        只读校验，不修改 HEST_data 原始目录。
    """

    parser = argparse.ArgumentParser(description="Validate local HEST GBM MVP data.")
    parser.add_argument("--config", default="configs/glioma_visium_mvp.yaml", help="Pipeline config path.")
    args = parser.parse_args()

    import anndata as ad

    from niche_pipeline.config import load_config
    from niche_pipeline.manifest import build_manifest

    config = load_config(args.config)
    manifest = build_manifest(config)
    for row in manifest.itertuples(index=False):
        h5ad = ad.read_h5ad(row.st_path, backed="r")
        missing_obs = [column for column in ["in_tissue", "array_row", "array_col"] if column not in h5ad.obs.columns]
        missing_obsm = "spatial" not in h5ad.obsm.keys()
        print(
            f"{row.sample_id}: shape={h5ad.shape}, "
            f"missing_obs={missing_obs}, missing_spatial={missing_obsm}, wsi={Path(row.wsi_path).name}"
        )
        h5ad.file.close()


if __name__ == "__main__":
    main()
