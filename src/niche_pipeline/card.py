"""标准 CARD 反卷积：使用带细胞类型注释的 scRNA-seq reference。"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmwrite

from .config import output_root, project_root, resolve_path
from .manifest import load_manifest
from .reference import load_reference_counts, validate_reference
from .utils import ensure_dir, get_logger, require_columns, save_json

LOGGER = get_logger(__name__)


def run_card(config: dict[str, Any], manifest_path: str | Path | None = None) -> list[Path]:
    """按样本运行标准 CARD，返回下游兼容的 spot x cell type 比例 CSV 路径。

    reference 计数和注释只导出一次。缺少参考、R/CARD 或完整 spot 结果时失败，
    避免将缺失比例填零后传入生态位识别。
    """

    import anndata as ad

    manifest = load_manifest(config, manifest_path)
    card_cfg = config.get("deconvolution", {}).get("card", {}) or {}
    script_path = resolve_path(config, card_cfg.get("r_script"))
    if script_path is None:
        script_path = project_root(config) / "scripts" / "run_card_deconvolution.R"
    if not script_path.is_file():
        raise FileNotFoundError(f"CARD R script does not exist: {script_path}")
    rscript_bin = str(card_cfg.get("rscript_bin", "Rscript"))
    _check_card_environment(rscript_bin, script_path)
    validate_reference(config, manifest_path)

    out_root = output_root(config)
    card_dir = ensure_dir(out_root / "deconvolution" / "card")
    export_dir = ensure_dir(card_dir / "exports")
    log_dir = ensure_dir(card_dir / "logs")
    abundance_dir = ensure_dir(out_root / "deconvolution" / "abundance")
    ref_cfg = config.get("reference", {})
    ref_path = resolve_path(config, ref_cfg.get("h5ad"))
    reference = load_reference_counts(config)
    # loader 已将所选 count 来源放在 X，避免再次按原 layer 名查找。
    ref_inputs = export_card_reference(reference, {**ref_cfg, "counts_layer": None}, export_dir / "reference")
    del reference

    cell_types = pd.read_csv(ref_inputs["meta"], dtype=str, keep_default_na=False)["cell_type"].unique().tolist()
    written: list[Path] = []
    reports: list[dict[str, Any]] = []
    for row in manifest.itertuples(index=False):
        sample_id = str(row.sample_id)
        preprocessed_path = out_root / "preprocessed" / f"{sample_id}.preprocessed.h5ad"
        if not preprocessed_path.is_file():
            raise FileNotFoundError(f"Preprocessed h5ad does not exist: {preprocessed_path}")
        LOGGER.info("Running standard CARD deconvolution for %s", sample_id)
        adata = ad.read_h5ad(preprocessed_path)
        inputs = export_card_spatial(adata, sample_id, export_dir / sample_id)
        raw_output = export_dir / sample_id / f"{sample_id}.card_raw_proportions.csv"
        _run_card_rscript(
            rscript_bin=rscript_bin,
            script_path=script_path,
            spatial_inputs=inputs,
            reference_inputs=ref_inputs,
            output_csv=raw_output,
            min_count_gene=float(card_cfg.get("min_count_gene", 100)),
            min_count_spot=float(card_cfg.get("min_count_spot", 5)),
            stdout_log=log_dir / f"{sample_id}.stdout.log",
            stderr_log=log_dir / f"{sample_id}.stderr.log",
        )
        abundance = _standardize_card_output(raw_output, sample_id, adata.obs_names, cell_types)
        abundance_path = abundance_dir / f"{sample_id}.spot_celltype_abundance.csv"
        abundance.to_csv(abundance_path, index=False, encoding="utf-8")
        written.append(abundance_path)
        reports.append({
            "sample_id": sample_id,
            "method": "card",
            "n_spots": len(abundance),
            "cell_types": cell_types,
            "raw_output": str(raw_output),
            "abundance_path": str(abundance_path),
        })

    summary = {"method": "card", "reference_h5ad": str(ref_path), "samples": reports}
    save_json(summary, card_dir / "card_summary.json")
    save_json(summary, out_root / "deconvolution" / "deconvolution_summary.json")
    return written


def export_card_reference(adata: Any, ref_cfg: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    """导出 genes x cells 原始计数、cell ID 和细胞类型/供者注释，返回输入路径。

    counts 来源优先级为配置的 layer、raw.X、X；使用所选来源的 var 解析 gene symbol。
    支持 backed AnnData，仅读取计数矩阵；重复 gene symbol 的计数相加，不抽样细胞。
    """

    cell_type_key = str(ref_cfg.get("cell_type_key") or "cell_type")
    batch_key = ref_cfg.get("batch_key")
    require_columns(adata.obs, [cell_type_key] + ([batch_key] if batch_key else []), "CARD reference obs")
    labels = adata.obs[cell_type_key]
    if labels.isna().any() or labels.astype(str).str.strip().eq("").any():
        raise ValueError("CARD reference cell type labels must not be missing or empty.")
    if set(labels.astype(str)).intersection({"sample_id", "spot_id"}):
        raise ValueError("CARD cell type labels must not use reserved names 'sample_id' or 'spot_id'.")
    if labels.nunique() < 2:
        raise ValueError("CARD requires at least two reference cell types.")

    if batch_key:
        donors = adata.obs[batch_key]
        if donors.isna().any() or donors.astype(str).str.strip().eq("").any():
            raise ValueError("CARD reference donor labels must not be missing or empty.")
    else:
        donors = pd.Series("reference", index=adata.obs_names)

    counts_layer = ref_cfg.get("counts_layer")
    if counts_layer:
        if counts_layer not in adata.layers:
            raise ValueError(f"Reference counts_layer={counts_layer!r} does not exist.")
        counts, var = adata.layers[counts_layer], adata.var
    elif adata.raw is not None:
        counts, var = adata.raw.X, adata.raw.var
    else:
        counts, var = adata.X, adata.var
    if hasattr(counts, "to_memory"):
        counts = counts.to_memory()
    gene_key = ref_cfg.get("gene_symbol_key")
    if gene_key:
        require_columns(var, [gene_key], "CARD reference count-source var")
    genes = pd.Index(var[gene_key] if gene_key else var.index)
    if genes.isna().any():
        raise ValueError("CARD reference gene names must not be missing.")
    genes = genes.astype(str)
    matrix = _count_matrix(counts, "CARD reference counts")
    if genes.has_duplicates:
        codes, unique_genes = pd.factorize(genes, sort=False)
        mapping = sparse.csr_matrix((np.ones(len(genes)), (np.arange(len(genes)), codes)),
                                    shape=(len(genes), len(unique_genes)))
        matrix = matrix @ mapping
        genes = pd.Index(unique_genes)

    paths = _export_counts(matrix, genes, adata.obs_names, ensure_dir(output_dir), "reference")
    paths["meta"] = output_dir / "reference.cell_metadata.csv"
    pd.DataFrame({"cell_id": adata.obs_names.astype(str), "cell_type": labels.astype(str).to_numpy(),
                  "sample_id": donors.astype(str).to_numpy()}).to_csv(paths["meta"], index=False, encoding="utf-8")
    return paths


def export_card_spatial(adata: Any, sample_id: str, output_dir: Path) -> dict[str, Path]:
    """导出 CARD 空间计数和坐标，返回输入路径；缺少 counts 或有效坐标时失败。"""

    if "counts" not in adata.layers:
        raise ValueError("Spatial AnnData must contain layers['counts']. Run preprocess first.")
    # 优先使用真实空间坐标；Visium array_row/col 不是等距的笛卡尔坐标。
    if "spatial" in adata.obsm:
        coords = np.asarray(adata.obsm["spatial"], dtype=float)
    elif {"pxl_col_in_fullres", "pxl_row_in_fullres"}.issubset(adata.obs.columns):
        coords = adata.obs[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy(dtype=float)
    elif {"array_col", "array_row"}.issubset(adata.obs.columns):
        coords = adata.obs[["array_col", "array_row"]].to_numpy(dtype=float)
        coords[:, 1] *= np.sqrt(3)
    else:
        raise ValueError("CARD requires obsm['spatial'], pixel coordinates or Visium array_row/array_col.")
    if coords.shape != (adata.n_obs, 2) or not np.isfinite(coords).all():
        raise ValueError("CARD spatial coordinates must be a finite spots x 2 matrix.")
    if not np.any(np.ptp(coords, axis=0) > 0):
        raise ValueError("CARD requires at least two distinct spatial locations.")
    paths = _export_counts(_count_matrix(adata.layers["counts"], "CARD spatial counts"),
                           adata.var_names, adata.obs_names, ensure_dir(output_dir), sample_id)
    paths["coords"] = output_dir / f"{sample_id}.spatial_location.csv"
    pd.DataFrame({"spot_id": adata.obs_names.astype(str), "x": coords[:, 0], "y": coords[:, 1]}).to_csv(
        paths["coords"], index=False, encoding="utf-8")
    return paths


def _count_matrix(counts: Any, name: str) -> sparse.csr_matrix:
    """转为 CSR 计数矩阵；非有限、负数或非整数输入会失败，防止误用归一化表达。"""

    matrix = sparse.csr_matrix(counts)
    values = matrix.data
    if (not np.isfinite(values).all() or np.any(values < 0)
            or not np.allclose(values, np.rint(values), rtol=0, atol=1e-6)):
        raise ValueError(f"{name} must contain finite, nonnegative raw integer counts.")
    if matrix.nnz == 0 or matrix.sum() == 0:
        raise ValueError(f"{name} is empty.")
    return matrix


def _export_counts(counts: sparse.csr_matrix, genes: Any, observations: Any,
                   output_dir: Path, prefix: str) -> dict[str, Path]:
    """将 obs x genes 转置写出 Matrix Market 和有序名称文件，拒绝重复或空名称。"""

    genes, observations = pd.Index(genes), pd.Index(observations)
    for names in (genes, observations):
        if (names.isna().any() or names.has_duplicates
                or names.astype(str).str.contains(r"[\r\n]").any()
                or (names.astype(str).str.strip() == "").any()):
            raise ValueError("CARD gene/cell/spot names must be unique, nonempty and contain no newlines.")
    if counts.shape != (len(observations), len(genes)):
        raise ValueError("CARD count matrix dimensions do not match gene/cell/spot names.")
    paths = {"counts": output_dir / f"{prefix}.counts_genes_by_observations.mtx",
             "genes": output_dir / f"{prefix}.genes.txt", "observations": output_dir / f"{prefix}.observations.txt"}
    mmwrite(str(paths["counts"]), counts.T)
    paths["genes"].write_text("\n".join(genes.astype(str)) + "\n", encoding="utf-8")
    paths["observations"].write_text("\n".join(observations.astype(str)) + "\n", encoding="utf-8")
    return paths


def _check_card_environment(rscript_bin: str, script_path: Path) -> None:
    """在读取大型 reference 前校验 R/CARD；缺少可执行文件或依赖时报告原因。"""

    try:
        result = subprocess.run([rscript_bin, str(script_path), "--check"],
                                capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Could not find Rscript executable {rscript_bin!r}. "
                           "Install R or set deconvolution.card.rscript_bin to its full path.") from exc
    if result.returncode != 0:
        raise RuntimeError("Standard CARD requires R with the CARD and Matrix packages. "
                           f"Environment check failed: {result.stderr.strip() or result.stdout.strip()}")


def _run_card_rscript(rscript_bin: str, script_path: Path, spatial_inputs: dict[str, Path],
                      reference_inputs: dict[str, Path], output_csv: Path, min_count_gene: float,
                      min_count_spot: float, stdout_log: Path, stderr_log: Path) -> None:
    """调用标准 CARD R 脚本并保存日志；非零退出或未生成结果时抛出异常。"""

    if not np.isfinite([min_count_gene, min_count_spot]).all() or min(min_count_gene, min_count_spot) < 0:
        raise ValueError("CARD min_count_gene and min_count_spot must be finite and nonnegative.")
    ensure_dir(output_csv.parent)
    ensure_dir(stdout_log.parent)
    ensure_dir(stderr_log.parent)
    # 清除上次原始结果，确保本次成功判断不会读到旧文件。
    output_csv.unlink(missing_ok=True)
    command = [rscript_bin, str(script_path),
               *(str(spatial_inputs[key]) for key in ("counts", "genes", "observations", "coords")),
               *(str(reference_inputs[key]) for key in ("counts", "genes", "observations", "meta")),
               str(output_csv), str(min_count_gene), str(min_count_spot)]
    try:
        with stdout_log.open("w", encoding="utf-8") as stdout, stderr_log.open("w", encoding="utf-8") as stderr:
            result = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Could not find Rscript executable {rscript_bin!r}. "
                           "Install R or set deconvolution.card.rscript_bin to its full path.") from exc
    if result.returncode != 0:
        raise RuntimeError(f"CARD Rscript failed with exit code {result.returncode}. Check logs: {stdout_log}, {stderr_log}")
    if not output_csv.is_file():
        raise RuntimeError(f"CARD finished but did not create expected output: {output_csv}")


def _standardize_card_output(raw_output: Path, sample_id: str, spot_ids: Any,
                             cell_types: list[str]) -> pd.DataFrame:
    """校验命名细胞类型比例和完整 spot 覆盖，按输入顺序返回 pipeline 标准表。"""

    frame = pd.read_csv(raw_output, dtype={"spot_id": str})
    require_columns(frame, ["spot_id"], "CARD output")
    value_cols = [column for column in frame.columns if column != "spot_id"]
    if set(value_cols) != set(cell_types):
        raise ValueError("CARD output cell type columns do not match the annotated reference.")
    expected = pd.Index(spot_ids).astype(str)
    if frame["spot_id"].duplicated().any() or set(frame["spot_id"]) != set(expected):
        missing = len(set(expected) - set(frame["spot_id"]))
        raise ValueError(f"CARD output spot IDs do not match preprocessed spots ({missing} missing). "
                         "Check CARD QC thresholds and informative-gene coverage; missing spots are not filled with zeros.")
    frame = frame.set_index("spot_id").loc[expected, cell_types]
    values = frame.to_numpy(dtype=float)
    if (not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1)
            or not np.allclose(values.sum(axis=1), 1, atol=1e-5, rtol=0)):
        raise ValueError("CARD output must contain finite, nonnegative proportions summing to one per spot.")
    frame = frame.reset_index(names="spot_id")
    frame.insert(0, "sample_id", sample_id)
    return frame
