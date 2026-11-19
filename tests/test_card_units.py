from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from scipy.io import mmread

from niche_pipeline.card import (
    _check_card_environment,
    _run_card_rscript,
    _standardize_card_output,
    export_card_reference,
    export_card_spatial,
    run_card,
)
from niche_pipeline.cli import _cmd_prepare_reference
from niche_pipeline.deconvolution import run_deconvolution
from niche_pipeline.reference import load_reference_counts, validate_reference


@pytest.fixture
def reference() -> ad.AnnData:
    counts = sparse.csr_matrix([[8, 1, 0], [6, 2, 0], [0, 2, 8], [0, 1, 9]])
    obs = pd.DataFrame({"type": ["Tumor, cycling"] * 2 + ["Myeloid"] * 2,
                        "donor": ["d1", "d2", "d1", "d2"]}, index=["c1", "c2", "c3", "c4"])
    var = pd.DataFrame({"symbol": ["EGFR", "EGFR", "AIF1"]}, index=["id1", "id2", "id3"])
    data = ad.AnnData(X=counts, obs=obs, var=var)
    data.raw = data.copy()
    data.X = data.X.astype(float).log1p()
    return data


@pytest.fixture
def spatial_data() -> ad.AnnData:
    data = ad.AnnData(X=sparse.csr_matrix(np.ones((2, 2))),
                     obs=pd.DataFrame({"array_col": [2, 3], "array_row": [0, 1]}, index=["s1", "s2"]),
                     var=pd.DataFrame(index=["EGFR", "AIF1"]))
    data.layers["counts"] = sparse.csr_matrix([[5, 1], [1, 6]])
    data.obsm["spatial"] = np.array([[100, 200], [300, 400]])
    return data


def test_reference_uses_raw_counts_and_merges_duplicate_symbols(reference: ad.AnnData, tmp_path: Path) -> None:
    paths = export_card_reference(reference, {"cell_type_key": "type", "batch_key": "donor",
                                              "gene_symbol_key": "symbol"}, tmp_path)
    np.testing.assert_array_equal(mmread(paths["counts"]).toarray(), [[9, 8, 2, 1], [0, 0, 8, 9]])
    assert paths["genes"].read_text().splitlines() == ["EGFR", "AIF1"]
    meta = pd.read_csv(paths["meta"])
    assert meta["cell_type"].tolist() == ["Tumor, cycling"] * 2 + ["Myeloid"] * 2
    assert meta["sample_id"].tolist() == ["d1", "d2", "d1", "d2"]


def test_reference_layer_overrides_raw_and_defaults_to_one_sample(reference: ad.AnnData, tmp_path: Path) -> None:
    reference.layers["counts"] = sparse.csr_matrix(np.full(reference.shape, 2))
    paths = export_card_reference(reference, {"cell_type_key": "type", "counts_layer": "counts"}, tmp_path)
    np.testing.assert_array_equal(mmread(paths["counts"]).toarray(), np.full((3, 4), 2))
    assert pd.read_csv(paths["meta"])["sample_id"].unique().tolist() == ["reference"]


def test_backed_reference_uses_raw_var_after_gene_subset(reference: ad.AnnData, tmp_path: Path) -> None:
    path = tmp_path / "reference.h5ad"
    reference[:, :1].copy().write_h5ad(path)
    backed = ad.read_h5ad(path, backed="r")
    try:
        paths = export_card_reference(backed, {"cell_type_key": "type", "gene_symbol_key": "symbol"},
                                      tmp_path / "export")
        assert mmread(paths["counts"]).shape == (2, 4)
    finally:
        backed.file.close()


def test_reference_validation_and_loading_skip_unrelated_layers(tmp_path: Path) -> None:
    counts = sparse.csr_matrix(np.ones((12, 120), dtype=int))
    ref = ad.AnnData(X=counts, obs=pd.DataFrame({"cell_type": ["Tumor", "Myeloid"] * 6},
                                               index=[f"c{i}" for i in range(12)]),
                     var=pd.DataFrame({"symbol": [f"g{i}" for i in range(120)]},
                                      index=[f"id{i}" for i in range(120)]))
    ref.raw = ref.copy()
    ref = ref[:, :100].copy()
    ref.layers["scaled"] = np.zeros(ref.shape)
    path = tmp_path / "reference.h5ad"
    ref.write_h5ad(path)
    # A deliberately unreadable unrelated layer proves neither path deserializes it.
    with h5py.File(path, "r+") as handle:
        handle["layers/scaled"].attrs["encoding-type"] = "unrelated-invalid-encoding"
    config = {"_project_root": str(tmp_path), "reference": {"h5ad": str(path), "gene_symbol_key": "symbol"}}
    report = validate_reference(config)
    assert report["n_genes"] == 100
    assert report["n_count_source_genes"] == 120
    assert report["counts_source"] == "raw.X"
    loaded = load_reference_counts(config)
    assert loaded.shape == (12, 120)
    np.testing.assert_array_equal(loaded.X.toarray(), counts.toarray())
    assert list(loaded.layers) == []


@pytest.mark.parametrize("invalid", [np.nan, -1, 0.25])
def test_reference_rejects_invalid_counts(reference: ad.AnnData, tmp_path: Path, invalid: float) -> None:
    reference.raw = None
    reference.X = sparse.csr_matrix(np.full(reference.shape, invalid))
    with pytest.raises(ValueError, match="raw integer counts"):
        export_card_reference(reference, {"cell_type_key": "type"}, tmp_path)


def test_reference_rejects_missing_annotations(reference: ad.AnnData, tmp_path: Path) -> None:
    reference.obs.loc["c1", "type"] = None
    with pytest.raises(ValueError, match="labels"):
        export_card_reference(reference, {"cell_type_key": "type"}, tmp_path)


def test_reference_validation_requires_external_reference(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="reference.h5ad is required"):
        validate_reference({"_project_root": str(tmp_path), "deconvolution": {"method": "card"}})


def test_spatial_export_preserves_counts_and_prefers_physical_coordinates(spatial_data: ad.AnnData,
                                                                         tmp_path: Path) -> None:
    paths = export_card_spatial(spatial_data, "S1", tmp_path)
    np.testing.assert_array_equal(mmread(paths["counts"]).toarray(), [[5, 1], [1, 6]])
    coords = pd.read_csv(paths["coords"])
    np.testing.assert_array_equal(coords[["x", "y"]], spatial_data.obsm["spatial"])
    del spatial_data.obsm["spatial"]
    paths = export_card_spatial(spatial_data, "S1", tmp_path)
    coords = pd.read_csv(paths["coords"])
    np.testing.assert_allclose(coords[["x", "y"]], [[2, 0], [3, np.sqrt(3)]])


def test_spatial_export_rejects_missing_counts(spatial_data: ad.AnnData, tmp_path: Path) -> None:
    del spatial_data.layers["counts"]
    with pytest.raises(ValueError, match="counts"):
        export_card_spatial(spatial_data, "S1", tmp_path)


def test_output_preserves_named_cell_types_and_reorders_spots(tmp_path: Path) -> None:
    path = tmp_path / "proportions.csv"
    pd.DataFrame({"spot_id": ["s2", "s1"], "Tumor, cycling": [0.25, 0.75],
                  "Myeloid": [0.75, 0.25]}).to_csv(path, index=False)
    result = _standardize_card_output(path, "S1", ["s1", "s2"], ["Myeloid", "Tumor, cycling"])
    assert result.columns.tolist() == ["sample_id", "spot_id", "Myeloid", "Tumor, cycling"]
    assert result["spot_id"].tolist() == ["s1", "s2"]
    np.testing.assert_array_equal(result["Myeloid"], [0.25, 0.75])


@pytest.mark.parametrize("spots,tumor,myeloid,error", [
    (["s1"], [0.5], [0.5], "spot IDs"),
    (["s1", "s1"], [0.5, 0.5], [0.5, 0.5], "spot IDs"),
    (["s1", "s2"], [-0.1, 0.5], [1.1, 0.5], "proportions"),
    (["s1", "s2"], [0.5, np.nan], [0.5, 0.5], "proportions"),
    (["s1", "s2"], [0.5, 0.5], [0.2, 0.5], "proportions"),
])
def test_output_rejects_incomplete_or_invalid_results(tmp_path: Path, spots: list[str], tumor: list[float],
                                                    myeloid: list[float], error: str) -> None:
    path = tmp_path / "proportions.csv"
    pd.DataFrame({"spot_id": spots, "Tumor": tumor, "Myeloid": myeloid}).to_csv(path, index=False)
    with pytest.raises(ValueError, match=error):
        _standardize_card_output(path, "S1", ["s1", "s2"], ["Tumor", "Myeloid"])


def test_dispatch_defaults_to_standard_card(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("niche_pipeline.card.run_card", lambda config, manifest_path: [Path("card.csv")])
    assert run_deconvolution({}) == [Path("card.csv")]
    assert run_deconvolution({"deconvolution": {"method": "CARD"}}) == [Path("card.csv")]
    with pytest.raises(ValueError, match="Unsupported"):
        run_deconvolution({"deconvolution": {"method": "unsupported"}})


def test_prepare_reference_validates_scrna_for_card(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = []
    monkeypatch.setattr("niche_pipeline.reference.validate_reference", lambda config: seen.append(config))
    config = {"deconvolution": {"method": "card"}}
    _cmd_prepare_reference(config, argparse.Namespace())
    assert seen == [config]


def test_missing_rscript_fails_before_loading_reference() -> None:
    with pytest.raises(RuntimeError, match="deconvolution.card.rscript_bin"):
        _check_card_environment("/nonexistent/card-test/Rscript", Path("scripts/run_card_deconvolution.R"))


def test_card_pipeline_exports_reference_and_writes_downstream_table(reference: ad.AnnData,
                                                                   spatial_data: ad.AnnData,
                                                                   tmp_path: Path,
                                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    reference.write_h5ad(tmp_path / "reference.h5ad")
    pre_dir = tmp_path / "outputs" / "preprocessed"
    pre_dir.mkdir(parents=True)
    spatial_data.write_h5ad(pre_dir / "S1.preprocessed.h5ad")
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame({"sample_id": ["S1"], "st_path": ["unused"], "metadata_path": ["unused"],
                  "wsi_path": ["unused"]}).to_csv(manifest, index=False)
    config = {"_project_root": str(Path.cwd()),
              "paths": {"output_root": str(tmp_path / "outputs"), "manifest": str(manifest)},
              "reference": {"h5ad": str(tmp_path / "reference.h5ad"), "cell_type_key": "type",
                            "gene_symbol_key": "symbol", "batch_key": "donor"},
              "deconvolution": {"method": "card", "card": {}}}
    monkeypatch.setattr("niche_pipeline.card.validate_reference", lambda *_: {})
    calls = []

    def fake_r(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        calls.append(command)
        if command[-1] == "--check":
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        assert Path(command[1]).name == "run_card_deconvolution.R"
        assert len(command) == 13
        assert mmread(command[6]).shape == (2, 4)
        assert pd.read_csv(command[9])["cell_type"].nunique() == 2
        pd.DataFrame({"spot_id": ["s2", "s1"], "Tumor, cycling": [0.25, 0.75],
                      "Myeloid": [0.75, 0.25]}).to_csv(command[10], index=False)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("niche_pipeline.card.subprocess.run", fake_r)
    outputs = run_card(config)
    result = pd.read_csv(outputs[0])
    assert result["spot_id"].tolist() == ["s1", "s2"]
    assert "Tumor, cycling" in result.columns
    summary = json.loads((tmp_path / "outputs/deconvolution/deconvolution_summary.json").read_text())
    assert summary["method"] == "card"
    assert len(calls) == 2


def test_r_runner_does_not_reuse_stale_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "result.csv"
    output.write_text("stale result")
    monkeypatch.setattr("niche_pipeline.card.subprocess.run", lambda *args, **kwargs:
                        subprocess.CompletedProcess(args[0], 0))
    inputs = {key: tmp_path / key for key in ("counts", "genes", "observations", "coords", "meta")}
    with pytest.raises(RuntimeError, match="did not create"):
        _run_card_rscript("Rscript", Path("scripts/run_card_deconvolution.R"), inputs, inputs, output,
                          100, 5, tmp_path / "out.log", tmp_path / "err.log")


def test_r_bridge_preserves_counts_metadata_and_cell_type_labels(reference: ad.AnnData,
                                                                spatial_data: ad.AnnData, tmp_path: Path) -> None:
    """Execute the real R bridge with test doubles for the two CARD API calls; no algorithm claim."""
    rscript = shutil.which("Rscript")
    if rscript is None:
        pytest.skip("Rscript is not installed")
    ref = export_card_reference(reference, {"cell_type_key": "type", "batch_key": "donor",
                                            "gene_symbol_key": "symbol"}, tmp_path / "reference")
    spatial = export_card_spatial(spatial_data, "S1", tmp_path / "spatial")
    wrapper = tmp_path / "test_bridge.R"
    wrapper.write_text('''
library(Matrix)
setClass("TestCardOutput", slots = list(Proportion_CARD = "matrix"))
createCARDObject <- function(sc_count, sc_meta, spatial_count, spatial_location,
                             ct.varname, ct.select, sample.varname, minCountGene, minCountSpot) {
  stopifnot(identical(dim(sc_count), c(2L, 4L)), sc_count[1, 1] == 9,
            identical(rownames(sc_meta), colnames(sc_count)),
            identical(rownames(spatial_location), colnames(spatial_count)),
            identical(ct.select, c("Tumor, cycling", "Myeloid")),
            identical(sc_meta$sample_id, c("d1", "d2", "d1", "d2")),
            ct.varname == "cell_type", sample.varname == "sample_id",
            minCountGene == 100, minCountSpot == 5)
  list(spots = colnames(spatial_count), types = ct.select)
}
CARD_deconvolution <- function(CARD_object) {
  prop <- matrix(c(0.25, 0.75, 0.75, 0.25), nrow = 2,
                 dimnames = list(CARD_object$spots, CARD_object$types))
  new("TestCardOutput", Proportion_CARD = prop)
}
bridge <- base::commandArgs(trailingOnly = TRUE)[[1]]
commandArgs <- function(trailingOnly = FALSE) base::commandArgs(trailingOnly = TRUE)[-1]
# Skip only the package-loading expression; retain the complete bridge logic.
eval(parse(bridge)[-1])
''', encoding="utf-8")
    result_path = tmp_path / "result.csv"
    result = subprocess.run([rscript, str(wrapper), str(Path("scripts/run_card_deconvolution.R").resolve()),
                             *(str(spatial[k]) for k in ("counts", "genes", "observations", "coords")),
                             *(str(ref[k]) for k in ("counts", "genes", "observations", "meta")),
                             str(result_path), "100", "5"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    table = _standardize_card_output(result_path, "S1", ["s1", "s2"], ["Tumor, cycling", "Myeloid"])
    assert table.columns.tolist() == ["sample_id", "spot_id", "Tumor, cycling", "Myeloid"]
