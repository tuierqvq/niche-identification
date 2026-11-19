"""命令行入口。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

from .config import load_config
from .utils import get_logger, set_random_seed

LOGGER = get_logger(__name__)


def main(argv: list[str] | None = None) -> None:
    """解析 CLI 参数并执行子命令。

    参数:
        argv: 可选命令行参数列表；为空时使用 ``sys.argv``。

    返回值:
        无。

    异常/边界:
        子命令异常不在此处吞掉，便于命令行返回非零退出码。
    """

    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "handler"):
        parser.print_help()
        return
    config = load_config(args.config or "configs/glioma_visium_mvp.yaml")
    seed = config.get("project", {}).get("random_seed")
    set_random_seed(int(seed) if seed is not None else None)
    args.handler(config, args)


def build_parser() -> argparse.ArgumentParser:
    """构建 argparse parser。"""

    parser = argparse.ArgumentParser(prog="niche-pipeline", description="HEST GBM Niche identification pipeline")
    parser.add_argument("--config", default=None, help="Pipeline YAML config path.")
    subparsers = parser.add_subparsers(dest="command")

    _add_command(subparsers, "make-manifest", "Build selected_samples.csv", _cmd_make_manifest)
    _add_command(subparsers, "preprocess", "Run Scanpy preprocessing", _cmd_preprocess)
    _add_command(subparsers, "prepare-reference", "Validate scRNA-seq reference", _cmd_prepare_reference)
    _add_command(subparsers, "deconvolve", "Run CARD or cell2location deconvolution", _cmd_deconvolve)
    _add_command(subparsers, "build-features", "Build spot niche features", _cmd_build_features)
    _add_command(subparsers, "cluster", "Cluster local/global niches", _cmd_cluster)
    _add_command(subparsers, "annotate", "Write annotation tables", _cmd_annotate)
    _add_command(subparsers, "extract-patches", "Extract H&E patches and overlays", _cmd_extract_patches)
    _add_command(subparsers, "train-xgb", "Train XGBoost baseline", _cmd_train_xgb)
    _add_command(subparsers, "run-all", "Run the full MVP pipeline", _cmd_run_all)
    return parser


def _add_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    name: str,
    help_text: str,
    handler: Callable[[dict[str, Any], argparse.Namespace], None],
) -> None:
    """注册通用子命令。"""

    command = subparsers.add_parser(name, help=help_text)
    command.add_argument("--config", default=None, help="Pipeline YAML config path.")
    command.set_defaults(handler=handler)


def _cmd_make_manifest(config: dict[str, Any], args: argparse.Namespace) -> None:
    from .manifest import build_manifest

    build_manifest(config)


def _cmd_preprocess(config: dict[str, Any], args: argparse.Namespace) -> None:
    from .preprocessing import preprocess_samples

    preprocess_samples(config)


def _cmd_prepare_reference(config: dict[str, Any], args: argparse.Namespace) -> None:
    from .reference import validate_reference

    validate_reference(config)


def _cmd_deconvolve(config: dict[str, Any], args: argparse.Namespace) -> None:
    from .deconvolution import run_deconvolution

    run_deconvolution(config)


def _cmd_build_features(config: dict[str, Any], args: argparse.Namespace) -> None:
    from .features import build_features

    build_features(config)


def _cmd_cluster(config: dict[str, Any], args: argparse.Namespace) -> None:
    from .clustering import cluster_niches

    cluster_niches(config)


def _cmd_annotate(config: dict[str, Any], args: argparse.Namespace) -> None:
    from .annotation import annotate_niches

    annotate_niches(config)


def _cmd_extract_patches(config: dict[str, Any], args: argparse.Namespace) -> None:
    from .histology import extract_patches_and_overlays

    extract_patches_and_overlays(config)


def _cmd_train_xgb(config: dict[str, Any], args: argparse.Namespace) -> None:
    from .model import train_xgboost_baseline

    train_xgboost_baseline(config)


def _cmd_run_all(config: dict[str, Any], args: argparse.Namespace) -> None:
    """按计划执行端到端 MVP。"""

    from .annotation import annotate_niches
    from .clustering import cluster_niches
    from .deconvolution import run_deconvolution
    from .features import build_features
    from .histology import extract_patches_and_overlays
    from .manifest import build_manifest
    from .model import train_xgboost_baseline
    from .preprocessing import preprocess_samples
    from .reference import validate_reference

    LOGGER.info("Running full HEST GBM niche pipeline")
    build_manifest(config)
    preprocess_samples(config)
    validate_reference(config)
    run_deconvolution(config)
    build_features(config)
    cluster_niches(config)
    annotate_niches(config)
    extract_patches_and_overlays(config)
    train_xgboost_baseline(config)
