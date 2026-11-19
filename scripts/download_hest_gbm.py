"""下载 HEST GBM Visium MVP 所需样本文件。

该脚本保留为数据准备入口，不会覆盖已经存在的业务代码。
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    """下载默认 3 个 HEST GBM Visium 样本。

    参数:
        无；命令行参数由 argparse 解析。

    返回值:
        无。

    异常/边界:
        需要网络访问和 huggingface_hub；若本地已有文件，Hugging Face 会复用缓存。
    """

    parser = argparse.ArgumentParser(description="Download HEST GBM Visium MVP files.")
    parser.add_argument("--local-dir", default="HEST_data/hest_gbm_samples", help="Output directory.")
    parser.add_argument("--repo-id", default="MahmoodLab/hest", help="Hugging Face dataset repo id.")
    parser.add_argument("--samples", nargs="+", default=["TENX30", "TENX31", "TENX73"], help="Sample IDs.")
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    allow_patterns = ["HEST_v1_3_0.csv"]
    for sample_id in args.samples:
        allow_patterns.extend(
            [
                f"st/{sample_id}.h5ad",
                f"metadata/{sample_id}.*",
                f"wsis/{sample_id}.*",
            ]
        )

    local_dir = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=str(Path(args.local_dir)),
        allow_patterns=allow_patterns,
    )
    print(f"Downloaded to: {local_dir}")


if __name__ == "__main__":
    main()
