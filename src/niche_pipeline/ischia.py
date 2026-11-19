"""调用 ISCHIA 官方 Composition.cluster，基于细胞组成统一划分生态位。"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import output_root, project_root, resolve_path
from .manifest import load_manifest
from .utils import ensure_dir, get_logger, require_columns, save_json

LOGGER = get_logger(__name__)


def run_ischia(config: dict[str, Any], manifest_path: str | Path | None = None) -> dict[str, Path]:
    """合并各样本 spot 组成，调用官方 R 函数，输出共享 CC 标签和兼容的原流程标签表。

    仅细胞类型比例参与聚类，不使用表达 PC 或邻域平滑特征。空间一致性只作为
    聚类后诊断。缺少输入、细胞类型列不一致或指定 k 无法满足时直接报错。
    """

    from .clustering import _assign_confidence, _neighbor_agreement

    manifest = load_manifest(config, manifest_path)
    settings = config.get("clustering", {}).get("ischia", {}) or {}
    source_file, provenance = _upstream_source(config)
    script_path = resolve_path(config, settings.get("r_script", "scripts/run_ischia_clustering.R"))
    if script_path is None or not script_path.is_file():
        raise FileNotFoundError(f"ISCHIA R bridge does not exist: {script_path}")
    rscript_bin = str(settings.get("rscript_bin") or
                      config.get("deconvolution", {}).get("card", {}).get("rscript_bin", "Rscript"))
    environment = _check_ischia_environment(rscript_bin, script_path, source_file)
    table, cell_types = _load_composition_table(config, manifest)
    k = _positive_integer(settings.get("n_clusters", 8), "clustering.ischia.n_clusters")
    n_distinct = len(table[cell_types].drop_duplicates())
    if k >= len(table) or k > n_distinct:
        raise ValueError(f"ISCHIA n_clusters={k} requires more than {k} spots and at least {k} distinct "
                         f"composition profiles; got {len(table)} spots and {n_distinct} profiles. "
                         "Choose k explicitly; it is not silently reduced.")
    seed = int(config.get("clustering", {}).get("random_seed", config.get("project", {}).get("random_seed", 42)))
    if seed < 0 or seed > 2147483647:
        raise ValueError("ISCHIA random_seed must be between 0 and 2147483647.")
    elbow_max_k = settings.get("elbow_max_k", 0)
    if isinstance(elbow_max_k, bool):
        raise ValueError("clustering.ischia.elbow_max_k must be zero or a positive integer.")
    if elbow_max_k != 0:
        elbow_max_k = _positive_integer(elbow_max_k, "clustering.ischia.elbow_max_k")
        if elbow_max_k >= len(table) or elbow_max_k > n_distinct:
            raise ValueError("ISCHIA elbow_max_k must be smaller than n_spots and no larger than "
                             "the number of distinct composition profiles.")

    cluster_dir = ensure_dir(output_root(config) / "clusters")
    ischia_dir = ensure_dir(cluster_dir / "ischia")
    input_path = ischia_dir / "spot_celltype_composition.csv"
    mapping_path = ischia_dir / "spot_id_mapping.csv"
    table[["ischia_spot_id", *cell_types]].to_csv(input_path, index=False, encoding="utf-8")
    table[["ischia_spot_id", "sample_id", "spot_id"]].to_csv(mapping_path, index=False, encoding="utf-8")
    raw_labels = ischia_dir / "composition_cluster_labels.csv"
    raw_centers = ischia_dir / "composition_cluster_centers.csv"
    metrics_path = ischia_dir / "fit_metrics.json"
    model_path = ischia_dir / "composition_clusters.rds"
    for path in (raw_labels, raw_centers, metrics_path, model_path, ischia_dir / "elbow.pdf"):
        path.unlink(missing_ok=True)
    command = [rscript_bin, str(script_path), str(input_path), str(raw_labels), str(raw_centers),
               str(metrics_path), str(model_path), str(source_file), str(k), str(seed), str(elbow_max_k)]
    LOGGER.info("Running ISCHIA Composition.cluster on %s pooled spots with k=%s", len(table), k)
    stdout_log, stderr_log = ischia_dir / "stdout.log", ischia_dir / "stderr.log"
    with stdout_log.open("w", encoding="utf-8") as stdout, stderr_log.open("w", encoding="utf-8") as stderr:
        result = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ISCHIA failed with exit code {result.returncode}. Check {stdout_log} and {stderr_log}")
    for path in (raw_labels, raw_centers, metrics_path, model_path):
        if not path.is_file():
            raise RuntimeError(f"ISCHIA did not create expected output: {path}")
    labels = _validated_labels(raw_labels, table, k)
    centers = pd.read_csv(raw_centers)
    require_columns(centers, ["CompositionCluster_CC", *cell_types], "ISCHIA centers")
    if (set(centers.columns) != {"CompositionCluster_CC", *cell_types}
            or centers["CompositionCluster_CC"].duplicated().any()
            or set(centers["CompositionCluster_CC"]) != set(labels["CompositionCluster_CC"])):
        raise ValueError("ISCHIA centers do not match the requested cell types and clusters.")
    center_values = centers[cell_types].to_numpy(dtype=float)
    if (not np.isfinite(center_values).all() or np.any(center_values < 0)
            or not np.allclose(center_values.sum(axis=1), 1, atol=1e-8, rtol=0)):
        raise ValueError("ISCHIA centers must contain finite, nonnegative compositions.")
    expected_centers = table[["ischia_spot_id", *cell_types]].merge(
        labels[["ischia_spot_id", "CompositionCluster_CC"]], on="ischia_spot_id", validate="one_to_one")
    expected_centers = expected_centers.groupby("CompositionCluster_CC")[cell_types].mean()
    if not np.allclose(centers.set_index("CompositionCluster_CC").loc[expected_centers.index, cell_types],
                       expected_centers, atol=1e-8, rtol=0):
        raise ValueError("ISCHIA centers do not match the returned cluster assignments.")

    labels["global_niche"] = "ISCHIA_" + labels["CompositionCluster_CC"]
    labels["local_niche"] = labels["sample_id"] + "_" + labels["global_niche"]
    local_dir = ensure_dir(cluster_dir / "local")
    sample_frames = []
    for sample_id, frame in labels.groupby("sample_id", sort=False):
        frame = frame.copy()
        edge_path = output_root(config) / "graphs" / f"{sample_id}.spatial_knn_k{int(config.get('features', {}).get('spatial_k', 12))}.csv"
        frame["neighbor_agreement"] = _neighbor_agreement(frame, edge_path)
        frame.to_csv(local_dir / f"{sample_id}.local_niche_labels.csv", index=False, encoding="utf-8")
        sample_frames.append(frame)
    labels = pd.concat(sample_frames, ignore_index=True)
    taxonomy = labels[["sample_id", "local_niche", "global_niche", "CompositionCluster_CC"]].drop_duplicates()
    labels = _assign_confidence(labels, taxonomy, config.get("clustering", {}))
    # local_niche 只表示同一 CC 在各样本中的子集，没有第二次局部聚类或层次合并。
    signature_values = table[["sample_id", "spot_id", *cell_types]].merge(
        labels[["sample_id", "spot_id", "local_niche"]], on=["sample_id", "spot_id"], validate="one_to_one")
    signatures = signature_values.groupby(["sample_id", "local_niche"])[cell_types].mean().reset_index()
    signatures = signatures.rename(columns={cell_type: f"composition_{cell_type}" for cell_type in cell_types})

    outputs = {"local_labels": cluster_dir / "spot_local_niche_labels.csv",
               "global_labels": cluster_dir / "spot_global_niche_labels.csv",
               "taxonomy": cluster_dir / "global_niche_taxonomy.csv",
               "signatures": cluster_dir / "local_niche_signatures.csv",
               "centers": raw_centers, "model": model_path}
    labels[["sample_id", "spot_id", "local_niche", "neighbor_agreement", "CompositionCluster_CC"]].to_csv(
        outputs["local_labels"], index=False, encoding="utf-8")
    labels.to_csv(outputs["global_labels"], index=False, encoding="utf-8")
    taxonomy.to_csv(outputs["taxonomy"], index=False, encoding="utf-8")
    signatures.to_csv(outputs["signatures"], index=False, encoding="utf-8")
    summary = {"method": "ischia", "scope": "all_selected_spots_pooled",
               "input": "row-normalized cell-type composition only", "n_spots": len(table),
               "n_clusters": k, "random_seed": seed, "samples": manifest["sample_id"].astype(str).tolist(),
               "cell_types": cell_types, "upstream": provenance, "environment_check": environment,
               "fit_metrics": json.loads(metrics_path.read_text(encoding="utf-8")),
               "outputs": {key: str(value) for key, value in outputs.items()}}
    save_json(summary, ischia_dir / "ischia_summary.json")
    save_json({"method": "ischia", **{key: str(value) for key, value in outputs.items()}},
              cluster_dir / "cluster_outputs.json")
    return outputs


def _positive_integer(value: Any, name: str) -> int:
    """读取正整数参数；拒绝 bool、小数和无穷值，防止隐式截断。"""

    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer.")
    try:
        parsed = int(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if parsed < 1 or float(value) != parsed:
        raise ValueError(f"{name} must be a positive integer.")
    return parsed


def _upstream_source(config: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    """读取随项目保存的 ISCHIA 上游模块并校验 SHA256，拒绝静默替换来源。"""

    directory = project_root(config) / "vendor" / "ischia"
    provenance_path = directory / "source.json"
    if not provenance_path.is_file():
        raise FileNotFoundError(f"ISCHIA source provenance is missing: {provenance_path}")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    source_file = directory / "composition_clustering.R"
    if not source_file.is_file():
        raise FileNotFoundError(f"ISCHIA upstream module is missing: {source_file}")
    if hashlib.sha256(source_file.read_bytes()).hexdigest() != provenance.get("sha256"):
        raise ValueError("ISCHIA upstream module SHA256 does not match vendor/ischia/source.json.")
    return source_file, provenance


def _check_ischia_environment(rscript_bin: str, script_path: Path, source_file: Path) -> dict[str, Any]:
    """只检查 R、SeuratObject 和上游入口，不读取数据或运行聚类。"""

    command = [rscript_bin, str(script_path), "--check", str(source_file)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Could not find Rscript executable {rscript_bin!r}. "
                           "Set clustering.ischia.rscript_bin to the installed Rscript path.") from exc
    if result.returncode != 0:
        raise RuntimeError(f"ISCHIA composition clustering environment check failed: {result.stderr.strip()}")
    return {"rscript": rscript_bin, "returncode": result.returncode, "stdout": result.stdout.strip()}


def _load_composition_table(config: dict[str, Any], manifest: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """读取全部样本组成并按行归一化；保留样本/spot 身份，要求细胞类型集合相同。"""

    if (manifest.empty or manifest["sample_id"].isna().any()
            or manifest["sample_id"].astype(str).str.strip().eq("").any()
            or manifest["sample_id"].astype(str).duplicated().any()):
        raise ValueError("ISCHIA requires a nonempty manifest with unique sample IDs.")
    frames = []
    cell_types: list[str] | None = None
    for sample_id in manifest["sample_id"].astype(str):
        path = output_root(config) / "deconvolution" / "abundance" / f"{sample_id}.spot_celltype_abundance.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Abundance CSV does not exist: {path}. Run deconvolve first.")
        with path.open(encoding="utf-8-sig", newline="") as handle:
            header = next(csv.reader(handle), [])
        if len(set(header)) != len(header):
            raise ValueError(f"ISCHIA input contains duplicate columns: {path}")
        if any(not column.strip() for column in header):
            raise ValueError(f"ISCHIA input contains empty column names: {path}")
        frame = pd.read_csv(path, dtype={"sample_id": str, "spot_id": str}, keep_default_na=False)
        require_columns(frame, ["sample_id", "spot_id"], "ISCHIA abundance")
        if frame.empty or not frame["sample_id"].eq(sample_id).all():
            raise ValueError(f"ISCHIA abundance must contain only nonempty sample {sample_id}: {path}")
        if frame["spot_id"].str.strip().eq("").any() or frame["spot_id"].duplicated().any():
            raise ValueError(f"ISCHIA spot IDs must be unique and nonempty within {sample_id}.")
        current = [column for column in frame.columns if column not in {"sample_id", "spot_id"}]
        if len(current) < 2 or any(not column.strip() for column in current):
            raise ValueError("ISCHIA requires at least two named cell-type columns.")
        if cell_types is None:
            cell_types = current
        elif set(current) != set(cell_types):
            raise ValueError(f"ISCHIA cell-type columns differ in sample {sample_id}; "
                             "all samples must use the same reference annotation. Missing types are not filled with zero.")
        try:
            values = frame[cell_types].to_numpy(dtype=float)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"ISCHIA composition in {sample_id} must be finite, nonnegative numeric values.") from exc
        totals = values.sum(axis=1)
        if not np.isfinite(values).all() or np.any(values < 0) or np.any(totals <= 0):
            raise ValueError(f"ISCHIA composition in {sample_id} must be finite, nonnegative and have positive row sums.")
        frame = frame[["sample_id", "spot_id", *cell_types]].copy()
        frame[cell_types] = values / totals[:, None]
        frames.append(frame)
    table = pd.concat(frames, ignore_index=True)
    table.insert(0, "ischia_spot_id", [f"ischia{idx:010d}" for idx in range(len(table))])
    return table, cell_types or []


def _validated_labels(path: Path, table: pd.DataFrame, k: int) -> pd.DataFrame:
    """校验上游 CC 标签和完整 spot 覆盖，恢复原 sample/spot 顺序及身份。"""

    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    require_columns(raw, ["ischia_spot_id", "CompositionCluster_CC"], "ISCHIA output")
    if (raw["ischia_spot_id"].duplicated().any()
            or set(raw["ischia_spot_id"]) != set(table["ischia_spot_id"])):
        raise ValueError("ISCHIA output spot IDs do not exactly match the input mapping.")
    if set(raw["CompositionCluster_CC"]) != {f"CC{idx}" for idx in range(1, k + 1)}:
        raise ValueError("ISCHIA output clusters do not match requested k; check input diversity and R logs.")
    return table[["sample_id", "spot_id", "ischia_spot_id"]].merge(
        raw[["ischia_spot_id", "CompositionCluster_CC"]], on="ischia_spot_id", how="left", validate="one_to_one")
