"""从 CZ CELLxGENE 下载公开 GBM/glioma scRNA-seq reference。

默认推荐 Core GBmap：样本量足够大、已有 cell type 注释，适合作为本项目
cell2location 的第一版 reference。脚本不会自动改配置文件，避免覆盖用户路径。
"""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
from pathlib import Path


PRESETS = {
    "gbmap_core": {
        "title": "Core GBmap",
        "collection": "Harmonized single-cell landscape, intercellular crosstalk and tumor architecture of glioblastoma",
        "collection_id": "999f2a15-3d7e-440b-96ae-2c806799c08c",
        "dataset_id": "c888b684-6c51-431f-972a-6c963044cef0",
        "cell_count": 338564,
        "feature_count": 27632,
        "cell_type_key": "cell_type",
        "url": "https://datasets.cellxgene.cziscience.com/861acfd8-25f0-418b-a445-aa96da232827.h5ad",
        "filename": "cellxgene_gbmap_core.h5ad",
    },
    "gbmap_extended": {
        "title": "Extended GBmap",
        "collection": "Harmonized single-cell landscape, intercellular crosstalk and tumor architecture of glioblastoma",
        "collection_id": "999f2a15-3d7e-440b-96ae-2c806799c08c",
        "dataset_id": "56c4912d-2bae-4b64-98f2-af8a84389208",
        "cell_count": 1135677,
        "feature_count": 25987,
        "cell_type_key": "cell_type",
        "url": "https://datasets.cellxgene.cziscience.com/230c8701-7291-4dd5-bf36-688490c681ee.h5ad",
        "filename": "cellxgene_gbmap_extended.h5ad",
    },
    "gbm_core_to_cortex_sl040": {
        "title": "SL040",
        "collection": "Single-cell RNA sequencing of glioblastoma cases with tissue sampled from tumor core to macroscopically normal cortex",
        "collection_id": "113a558a-e96e-4643-81db-140e95c58578",
        "dataset_id": "21722308-5091-4b63-9c07-4f116ab9a7b3",
        "cell_count": 135482,
        "feature_count": 58234,
        "cell_type_key": "cell_type",
        "url": "https://datasets.cellxgene.cziscience.com/d45b4ce6-9725-4d79-b97a-70a44158bdbf.h5ad",
        "filename": "cellxgene_gbm_sl040.h5ad",
    },
    "gbm_core_to_cortex_sl057": {
        "title": "SL057",
        "collection": "Single-cell RNA sequencing of glioblastoma cases with tissue sampled from tumor core to macroscopically normal cortex",
        "collection_id": "113a558a-e96e-4643-81db-140e95c58578",
        "dataset_id": "145dcf6a-2461-4fa3-a0af-7fc56db0bd33",
        "cell_count": 123236,
        "feature_count": 58234,
        "cell_type_key": "cell_type",
        "url": "https://datasets.cellxgene.cziscience.com/66e7fd4d-d2db-464d-9fad-6eda97aff2e1.h5ad",
        "filename": "cellxgene_gbm_sl057.h5ad",
    },
}


def main() -> None:
    """下载或列出 GBM reference 预设。

    参数:
        无；命令行参数由 argparse 解析。

    返回值:
        无。

    异常/边界:
        下载大文件前需要用户明确执行脚本；网络中断时保留 `.part` 临时文件用于排查。
    """

    parser = argparse.ArgumentParser(description="Download public GBM/glioma scRNA-seq h5ad references.")
    parser.add_argument("--preset", default="gbmap_core", choices=sorted(PRESETS), help="Reference dataset preset.")
    parser.add_argument("--out-dir", default="reference_data", help="Output directory.")
    parser.add_argument("--list", action="store_true", help="Only list available presets.")
    parser.add_argument("--print-yaml", action="store_true", help="Print YAML snippet after download/list.")
    args = parser.parse_args()

    if args.list:
        for name, meta in PRESETS.items():
            print(f"{name}: {meta['title']} | cells={meta['cell_count']} | genes={meta['feature_count']}")
        return

    meta = PRESETS[args.preset]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / str(meta["filename"])
    part_path = out_path.with_suffix(out_path.suffix + ".part")
    metadata_path = out_dir / f"{out_path.stem}.metadata.json"

    print(f"Downloading {meta['title']} to {out_path}")
    print(f"Source URL: {meta['url']}")
    with urllib.request.urlopen(str(meta["url"])) as response, part_path.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    part_path.replace(out_path)
    metadata_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Done: {out_path}")
    print(f"Metadata: {metadata_path}")

    if args.print_yaml:
        print("\nYAML snippet:")
        print("reference:")
        print(f"  h5ad: {out_path.as_posix()}")
        print(f"  cell_type_key: {meta['cell_type_key']}")
        print("  batch_key: donor_id")
        print("  counts_layer:")


if __name__ == "__main__":
    main()
