"""配置读取与路径解析工具。"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """读取 YAML 配置并补充项目根目录。

    参数:
        config_path: YAML 配置文件路径。

    返回值:
        带有 ``_config_path`` 与 ``_project_root`` 的配置字典。

    异常/边界:
        当文件不存在、YAML 为空或顶层不是字典时抛出明确异常。
    """

    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")

    config = deepcopy(data)
    config["_config_path"] = str(path)
    # 默认配置位于 configs/ 下，因此项目根目录是配置文件父目录的父目录。
    config["_project_root"] = str(path.parent.parent)
    return config


def project_root(config: dict[str, Any]) -> Path:
    """返回配置中的项目根目录。

    参数:
        config: ``load_config`` 返回的配置字典。

    返回值:
        绝对路径形式的项目根目录。

    异常/边界:
        若配置缺失内部字段，则回退到当前工作目录，便于单元测试构造最小配置。
    """

    return Path(config.get("_project_root", Path.cwd())).expanduser().resolve()


def resolve_path(config: dict[str, Any], value: str | Path | None) -> Path | None:
    """将配置中的相对路径解析到项目根目录下。

    参数:
        config: 配置字典。
        value: 待解析路径；为空时返回 ``None``。

    返回值:
        绝对路径或 ``None``。

    异常/边界:
        不检查路径是否存在，调用方根据具体阶段决定是否强制存在。
    """

    if value in (None, ""):
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (project_root(config) / path).resolve()


def output_root(config: dict[str, Any]) -> Path:
    """返回输出根目录路径。

    参数:
        config: 配置字典。

    返回值:
        输出目录绝对路径。

    异常/边界:
        若未配置 ``paths.output_root``，使用 ``outputs/hest_gbm_card_ischia_mvp``。
    """

    paths = config.get("paths", {})
    return resolve_path(config, paths.get("output_root", "outputs/hest_gbm_card_ischia_mvp"))  # type: ignore[return-value]
