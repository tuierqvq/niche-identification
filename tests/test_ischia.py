from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from niche_pipeline.clustering import cluster_niches
from niche_pipeline.config import load_config
from niche_pipeline.ischia import (
    _check_ischia_environment,
    _load_composition_table,
    _positive_integer,
    _upstream_source,
    _validated_labels,
    run_ischia,
)


@pytest.fixture
def composition_config(tmp_path: Path) -> dict:
    out = tmp_path / "outputs"
    abundance_dir = out / "deconvolution" / "abundance"
    feature_dir, graph_dir = out / "features", out / "graphs"
    for directory in (abundance_dir, feature_dir, graph_dir):
        directory.mkdir(parents=True)
    for sample_id in ("S1", "S2"):
        # The same barcodes occur in different samples; they must remain distinct.
        counts = pd.DataFrame({"sample_id": [sample_id] * 12,
                               "spot_id": [f"barcode{i}" for i in range(12)],
                               "Tumor, cycling": [9.0] * 6 + [1.0] * 6,
                               "Myeloid": [1.0] * 6 + [9.0] * 6})
        if sample_id == "S2":
            counts = counts[["sample_id", "spot_id", "Myeloid", "Tumor, cycling"]]
        counts.to_csv(abundance_dir / f"{sample_id}.spot_celltype_abundance.csv", index=False)
        features = counts.rename(columns={"Tumor, cycling": "composition_Tumor, cycling",
                                          "Myeloid": "composition_Myeloid"}).copy()
        features[["composition_Tumor, cycling", "composition_Myeloid"]] /= 10
        # Contradictory expression features must not enter the ISCHIA fit.
        features["expression_pc_1"] = [10000, -10000] * 6
        features.to_csv(feature_dir / f"{sample_id}.niche_features.csv", index=False)
        edges = []
        for i in range(12):
            for rank, target in enumerate((i, (i + 1) % 12)):
                edges.append({"source_spot_id": f"barcode{i}", "target_spot_id": f"barcode{target}",
                              "neighbor_rank": rank})
        pd.DataFrame(edges).to_csv(graph_dir / f"{sample_id}.spatial_knn_k1.csv", index=False)
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame({"sample_id": ["S1", "S2"], "st_path": ["unused", "unused"],
                  "metadata_path": ["unused", "unused"], "wsi_path": ["unused", "unused"]}).to_csv(manifest, index=False)
    return {"_project_root": str(Path.cwd()),
            "paths": {"output_root": str(out), "manifest": str(manifest)},
            "features": {"spatial_k": 1},
            "clustering": {"method": "ischia", "random_seed": 42, "min_spots_per_local_niche": 1,
                           "ischia": {"n_clusters": 2, "rscript_bin": shutil.which("Rscript") or "Rscript"}}}


def test_main_and_backup_configs_select_separate_backends() -> None:
    main = load_config("configs/glioma_visium_mvp.yaml")
    backup = load_config("configs/glioma_visium_two_stage_mvp.yaml")
    assert main["clustering"]["method"] == "ischia"
    assert backup["clustering"]["method"] == "two_stage"
    assert main["clustering"]["ischia"]["n_clusters"] == 8
    assert main["paths"]["output_root"] != backup["paths"]["output_root"]
    assert main["deconvolution"]["method"] == backup["deconvolution"]["method"] == "card"


def test_dispatch_defaults_to_ischia_and_retains_two_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("niche_pipeline.ischia.run_ischia", lambda *_: {"labels": Path("ischia.csv")})
    monkeypatch.setattr("niche_pipeline.clustering.cluster_two_stage_niches",
                        lambda *_: {"labels": Path("two_stage.csv")})
    assert cluster_niches({})["labels"] == Path("ischia.csv")
    assert cluster_niches({"clustering": {"method": "two_stage"}})["labels"] == Path("two_stage.csv")
    with pytest.raises(ValueError, match="Unsupported clustering.method"):
        cluster_niches({"clustering": {"method": "unsupported"}})


def test_pooled_input_normalizes_and_preserves_barcodes(composition_config: dict) -> None:
    manifest = pd.read_csv(composition_config["paths"]["manifest"])
    table, types = _load_composition_table(composition_config, manifest)
    assert types == ["Tumor, cycling", "Myeloid"]
    assert len(table) == 24
    assert table["ischia_spot_id"].is_unique
    assert table["spot_id"].nunique() == 12
    assert not table[["sample_id", "spot_id"]].duplicated().any()
    np.testing.assert_allclose(table[types].sum(axis=1), 1)
    np.testing.assert_allclose(table.loc[[0, 12], types], [[0.9, 0.1], [0.9, 0.1]])
    assert not any("expression_pc" in column for column in table.columns)


@pytest.mark.parametrize("value", [-1, np.inf, np.nan, 0])
def test_input_rejects_invalid_compositions(composition_config: dict, value: float) -> None:
    path = Path(composition_config["paths"]["output_root"]) / "deconvolution/abundance/S1.spot_celltype_abundance.csv"
    frame = pd.read_csv(path)
    frame.loc[0, ["Tumor, cycling", "Myeloid"]] = value
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="finite, nonnegative"):
        _load_composition_table(composition_config, pd.read_csv(composition_config["paths"]["manifest"]))


def test_input_rejects_cell_type_mismatch(composition_config: dict) -> None:
    path = Path(composition_config["paths"]["output_root"]) / "deconvolution/abundance/S2.spot_celltype_abundance.csv"
    frame = pd.read_csv(path).rename(columns={"Myeloid": "Other"})
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="cell-type columns differ"):
        _load_composition_table(composition_config, pd.read_csv(composition_config["paths"]["manifest"]))


def test_input_rejects_duplicate_spots_within_sample(composition_config: dict) -> None:
    path = Path(composition_config["paths"]["output_root"]) / "deconvolution/abundance/S1.spot_celltype_abundance.csv"
    frame = pd.read_csv(path)
    frame.loc[1, "spot_id"] = frame.loc[0, "spot_id"]
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="unique and nonempty"):
        _load_composition_table(composition_config, pd.read_csv(composition_config["paths"]["manifest"]))


def test_input_rejects_duplicate_headers(composition_config: dict) -> None:
    path = Path(composition_config["paths"]["output_root"]) / "deconvolution/abundance/S1.spot_celltype_abundance.csv"
    path.write_text("sample_id,spot_id,Myeloid,Myeloid\nS1,barcode0,0.2,0.8\n")
    with pytest.raises(ValueError, match="duplicate columns"):
        _load_composition_table(composition_config, pd.read_csv(composition_config["paths"]["manifest"]))


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, np.inf, "bad"])
def test_cluster_count_requires_positive_integer(value: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        _positive_integer(value, "k")


def test_upstream_module_checksum_is_enforced(tmp_path: Path) -> None:
    source, metadata = _upstream_source({"_project_root": str(Path.cwd())})
    directory = tmp_path / "vendor/ischia"
    directory.mkdir(parents=True)
    (directory / "source.json").write_text(json.dumps(metadata))
    (directory / source.name).write_bytes(source.read_bytes() + b"\n# changed\n")
    with pytest.raises(ValueError, match="SHA256"):
        _upstream_source({"_project_root": str(tmp_path)})


def test_output_cannot_drop_or_merge_spot_ids(tmp_path: Path) -> None:
    path = tmp_path / "labels.csv"
    table = pd.DataFrame({"sample_id": ["S1", "S2"], "spot_id": ["same", "same"],
                          "ischia_spot_id": ["id1", "id2"]})
    pd.DataFrame({"ischia_spot_id": ["id1"], "CompositionCluster_CC": ["CC1"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="spot IDs"):
        _validated_labels(path, table, 1)
    pd.DataFrame({"ischia_spot_id": ["id1", "id2"], "CompositionCluster_CC": ["CC1", "CC1"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="requested k"):
        _validated_labels(path, table, 2)


def test_missing_rscript_has_configuration_hint() -> None:
    with pytest.raises(RuntimeError, match="clustering.ischia.rscript_bin"):
        _check_ischia_environment("/nonexistent/ischia-test/Rscript", Path("unused.R"), Path("unused.R"))


def test_k_is_not_silently_reduced(composition_config: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("niche_pipeline.ischia._check_ischia_environment", lambda *_: {})
    composition_config["clustering"]["ischia"]["n_clusters"] = 8
    with pytest.raises(ValueError, match="not silently reduced"):
        run_ischia(composition_config)


def test_real_ischia_preserves_shared_partition_and_downstream_contract(composition_config: dict) -> None:
    """Run the actual pinned upstream R function and downstream annotation on synthetic samples."""
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript is required for the upstream integration test")
    from niche_pipeline.annotation import annotate_niches
    from niche_pipeline.model import _training_table, _model_feature_columns

    outputs = cluster_niches(composition_config)
    labels = pd.read_csv(outputs["global_labels"])
    assert len(labels) == 24
    assert labels["global_niche"].nunique() == 2
    for sample_id in ("S1", "S2"):
        sample = labels.loc[labels["sample_id"] == sample_id]
        assert sample.iloc[:6]["global_niche"].nunique() == 1
        assert sample.iloc[6:]["global_niche"].nunique() == 1
        assert sample.iloc[0]["global_niche"] != sample.iloc[6]["global_niche"]
    by_spot = labels.pivot(index="spot_id", columns="sample_id", values="global_niche")
    assert by_spot["S1"].equals(by_spot["S2"])
    assert outputs["model"].is_file()
    assert len(pd.read_csv(outputs["taxonomy"])) == 4
    centers = pd.read_csv(outputs["centers"])
    np.testing.assert_allclose(sorted(centers["Tumor, cycling"]), [0.1, 0.9])
    summary_path = outputs["global_labels"].parent / "ischia/ischia_summary.json"
    summary = json.loads(summary_path.read_text())
    assert summary["upstream"]["commit"] == "1e50ce1eb1f4fe0855ca16a5fd3bfc7228697900"
    assert summary["fit_metrics"]["total_withinss"] < 1e-20

    annotation = annotate_niches(composition_config)
    assert len(pd.read_csv(annotation["annotation_table"])) == 2
    proportions = pd.read_csv(annotation["sample_niche_proportion"])
    np.testing.assert_allclose(proportions["proportion"], 0.5)
    training = _training_table(composition_config)
    assert len(training) == 24
    assert "CompositionCluster_CC" not in _model_feature_columns(training)
    assert "neighbor_agreement" not in _model_feature_columns(training)

    # Official elbow helper must not perturb the main clustering's RNG state.
    expected = labels[["sample_id", "spot_id", "CompositionCluster_CC"]].copy()
    composition_config["clustering"]["ischia"]["elbow_max_k"] = 2
    rerun = cluster_niches(composition_config)
    observed = pd.read_csv(rerun["global_labels"])[expected.columns]
    pd.testing.assert_frame_equal(expected, observed)
    assert (rerun["global_labels"].parent / "ischia/elbow.pdf").is_file()
