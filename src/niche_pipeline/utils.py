"""通用文件、日志与数值工具。"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


def get_logger(name: str = "niche_pipeline") -> logging.Logger:
    """创建统一格式的 logger。

    参数:
        name: logger 名称。

    返回值:
        已设置基础格式的 logger。

    异常/边界:
        多次调用不会重复添加 handler，避免 CLI 输出重复日志。
    """

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def ensure_dir(path: str | Path) -> Path:
    """确保目录存在。

    参数:
        path: 目录路径。

    返回值:
        创建后的绝对目录路径。

    异常/边界:
        若路径已存在且是文件，底层 ``mkdir`` 会抛出异常。
    """

    directory = Path(path).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_json(data: dict[str, Any], path: str | Path) -> Path:
    """以 UTF-8 无 BOM 写出 JSON 文件。

    参数:
        data: 可 JSON 序列化的字典。
        path: 输出文件路径。

    返回值:
        输出文件绝对路径。

    异常/边界:
        不吞掉序列化错误，方便尽早发现不可写字段。
    """

    out = Path(path).expanduser().resolve()
    ensure_dir(out.parent)
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return out


def require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    """检查 DataFrame 是否包含必需列。

    参数:
        frame: 待检查表格。
        columns: 必需列名。
        name: 错误信息中的表格名称。

    返回值:
        无返回值；检查通过即继续。

    异常/边界:
        缺列时抛出 ``ValueError``，列名按原始字符串展示。
    """

    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def set_random_seed(seed: int | None) -> None:
    """设置常用随机种子。

    参数:
        seed: 随机种子；为空时不做任何操作。

    返回值:
        无。

    异常/边界:
        仅设置 Python 与 NumPy，深度学习框架在各自模块中按需设置。
    """

    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)


def assert_no_utf8_bom(paths: Iterable[str | Path]) -> list[Path]:
    """检查文件是否包含 UTF-8 BOM。

    参数:
        paths: 待检查文件路径列表。

    返回值:
        含 BOM 的文件列表；为空表示检查通过。

    异常/边界:
        不读取目录；不存在的文件由调用方先过滤。
    """

    bad_files: list[Path] = []
    for path_like in paths:
        path = Path(path_like)
        if not path.is_file():
            continue
        with path.open("rb") as handle:
            if handle.read(3) == b"\xef\xbb\xbf":
                bad_files.append(path)
    return bad_files
