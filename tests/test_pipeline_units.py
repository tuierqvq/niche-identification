from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from niche_pipeline.config import load_config, resolve_path
from niche_pipeline.cli import build_parser
from niche_pipeline.graph import build_spatial_knn
from niche_pipeline.manifest import build_manifest
from niche_pipeline.model import leave_one_sample_out_splits


def test_load_config_resolves_project_paths() -> None:
    config = load_config("configs/glioma_visium_mvp.yaml")
    path = resolve_path(config, "HEST_data")
    assert path is not None
    assert path.name == "HEST_data"


@pytest.mark.parametrize("arguments,expected", [
    (["--config", "cohort.yaml", "preprocess"], "cohort.yaml"),
    (["preprocess", "--config", "cohort.yaml"], "cohort.yaml"),
    (["--config", "parent.yaml", "preprocess", "--config", "child.yaml"], "child.yaml"),
    (["preprocess"], None),
])
def test_cli_preserves_config_before_or_after_subcommand(arguments: list[str], expected: str | None) -> None:
    assert build_parser().parse_args(arguments).config == expected


def test_build_manifest_flags_targeted_panel(tmp_path: Path) -> None:
    data_root = tmp_path / "hest"
    (data_root / "metadata").mkdir(parents=True)
    (data_root / "st").mkdir()
    (data_root / "wsis").mkdir()
    pd.DataFrame(
        [
            {
                "id": "TENX30",
                "dataset_title": "Targeted",
                "image_filename": "TENX30.tif",
                "organ": "Brain",
                "tissue": "Brain",
                "disease_state": "Cancer",
                "oncotree_code": "GBM",
                "species": "Homo sapiens",
                "st_technology": "Visium",
                "preservation_method": "Fresh Frozen",
                "magnification": "20x",
                "nb_genes": 1253,
                "spots_under_tissue": 10,
                "pixel_size_um_embedded": np.nan,
                "pixel_size_um_estimated": 0.7,
                "spot_diameter": 55,
                "inter_spot_dist": 100,
            }
        ]
    ).to_csv(data_root / "HEST_v1_3_0.csv", index=False)
    (data_root / "metadata" / "TENX30.json").write_text(json.dumps({"nb_genes": 1253}), encoding="utf-8")
    (data_root / "st" / "TENX30.h5ad").write_text("", encoding="utf-8")
    (data_root / "wsis" / "TENX30.tif").write_text("", encoding="utf-8")

    config = {
        "_project_root": str(tmp_path),
        "paths": {"data_root": str(data_root), "manifest": str(tmp_path / "selected.csv")},
        "samples": ["TENX30"],
    }
    manifest = build_manifest(config)
    assert bool(manifest.loc[0, "is_targeted_panel"]) is True
    assert bool(manifest.loc[0, "include_in_mvp"]) is True
    assert Path(config["paths"]["manifest"]).exists()


def test_spatial_knn_includes_self() -> None:
    graph = build_spatial_knn(["a", "b", "c"], np.array([[0, 0], [1, 0], [5, 0]]), k=1, include_self=True)
    assert graph.neighbors.shape == (3, 2)
    assert graph.spot_ids[graph.neighbors[0, 0]] == "a"


def test_build_features_with_synthetic_anndata(tmp_path: Path) -> None:
    ad = pytest.importorskip("anndata")
    from niche_pipeline.features import build_features

    out_root = tmp_path / "outputs"
    pre_dir = out_root / "preprocessed"
    ab_dir = out_root / "deconvolution" / "abundance"
    pre_dir.mkdir(parents=True)
    ab_dir.mkdir(parents=True)

    obs = pd.DataFrame(
        {
            "array_row": [0, 0, 1, 1, 2],
            "array_col": [0, 1, 0, 1, 0],
            "pxl_row_in_fullres": [0, 1, 2, 3, 4],
            "pxl_col_in_fullres": [0, 1, 2, 3, 4],
        },
        index=[f"spot{i}" for i in range(5)],
    )
    var = pd.DataFrame(index=[f"gene{i}" for i in range(6)])
    adata = ad.AnnData(X=sparse.csr_matrix(np.ones((5, 6))), obs=obs, var=var)
    adata.obsm["X_pca"] = np.arange(15).reshape(5, 3)
    adata.obsm["spatial"] = obs[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy()
    adata.write_h5ad(pre_dir / "S1.preprocessed.h5ad")

    pd.DataFrame(
        {
            "sample_id": ["S1"] * 5,
            "spot_id": [f"spot{i}" for i in range(5)],
            "Tumor": [1, 2, 3, 4, 5],
            "Myeloid": [5, 4, 3, 2, 1],
        }
    ).to_csv(ab_dir / "S1.spot_celltype_abundance.csv", index=False)
    manifest = pd.DataFrame(
        {"sample_id": ["S1"], "st_path": ["x"], "metadata_path": ["x"], "wsi_path": ["x"]}
    )
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    config = {
        "_project_root": str(tmp_path),
        "paths": {"output_root": str(out_root), "manifest": str(manifest_path)},
        "features": {"spatial_k": 2, "include_expression_pcs": True, "n_expression_pcs": 2},
    }
    paths = build_features(config, manifest_path)
    features = pd.read_csv(paths[0])
    assert features.shape[0] == 5
    assert "composition_Tumor" in features.columns
    assert "neighborhood_Myeloid" in features.columns
    assert "expression_pc_2" in features.columns


def test_cluster_outputs_labels_when_leiden_available(tmp_path: Path) -> None:
    pytest.importorskip("leidenalg")
    from niche_pipeline.clustering import cluster_niches

    out_root = tmp_path / "outputs"
    feature_dir = out_root / "features"
    graph_dir = out_root / "graphs"
    feature_dir.mkdir(parents=True)
    graph_dir.mkdir(parents=True)
    rows = []
    for idx in range(12):
        rows.append(
            {
                "sample_id": "S1",
                "spot_id": f"spot{idx}",
                "composition_Tumor": float(idx < 6),
                "composition_Myeloid": float(idx >= 6),
                "neighborhood_Tumor": float(idx < 6),
                "neighborhood_Myeloid": float(idx >= 6),
            }
        )
    pd.DataFrame(rows).to_csv(feature_dir / "S1.niche_features.csv", index=False)
    edge_rows = []
    for idx in range(12):
        for rank in range(2):
            edge_rows.append(
                {
                    "sample_id": "S1",
                    "source_spot_id": f"spot{idx}",
                    "target_spot_id": f"spot{(idx + rank) % 12}",
                    "neighbor_rank": rank,
                    "distance": rank,
                }
            )
    pd.DataFrame(edge_rows).to_csv(graph_dir / "S1.spatial_knn_k1.csv", index=False)
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame({"sample_id": ["S1"], "st_path": ["x"], "metadata_path": ["x"], "wsi_path": ["x"]}).to_csv(
        manifest_path, index=False
    )

    config = {
        "_project_root": str(tmp_path),
        "paths": {"output_root": str(out_root), "manifest": str(manifest_path)},
        "features": {"spatial_k": 1},
        "clustering": {"method": "two_stage", "local_neighbors": 3, "local_resolution": 0.5, "global_niches": 2},
    }
    outputs = cluster_niches(config, manifest_path)
    labels = pd.read_csv(outputs["global_labels"])
    assert {"local_niche", "global_niche", "confidence"}.issubset(labels.columns)


def test_leave_one_sample_out_has_no_sample_leakage() -> None:
    samples = pd.Series(["A", "A", "B", "B", "C"])
    for train_idx, test_idx, held_out in leave_one_sample_out_splits(samples):
        assert held_out not in set(samples.iloc[train_idx])
        assert set(samples.iloc[test_idx]) == {held_out}


def test_preprocessing_preserves_counts_and_unscaled_log_expression(tmp_path: Path) -> None:
    ad = pytest.importorskip("anndata")
    from niche_pipeline.preprocessing import preprocess_samples

    rng = np.random.default_rng(42)
    counts = rng.poisson(3, size=(24, 30)).astype(float)
    counts[:, 0] += 1
    data = ad.AnnData(X=sparse.csr_matrix(counts),
                     obs=pd.DataFrame(index=[f"spot{i}" for i in range(24)]),
                     var=pd.DataFrame(index=[f"gene{i}" for i in range(30)]))
    path = tmp_path / "input.h5ad"
    data.write_h5ad(path)
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame({"sample_id": ["S1"], "st_path": [str(path)], "metadata_path": ["unused"],
                  "wsi_path": ["unused"]}).to_csv(manifest_path, index=False)
    config = {"_project_root": str(tmp_path),
              "paths": {"manifest": str(manifest_path), "output_root": str(tmp_path / "output")},
              "preprocessing": {"min_genes_per_spot": 1, "max_pct_mito": 100,
                                "min_cells_per_gene": 1, "n_top_genes": 15, "n_pcs": 3}}
    result = ad.read_h5ad(preprocess_samples(config)[0])
    np.testing.assert_array_equal(result.layers["counts"].toarray(), counts)
    expected = np.log1p(counts / counts.sum(axis=1, keepdims=True) * 10000)
    np.testing.assert_allclose(result.raw.X.toarray(), expected)
    assert not np.allclose(result.X.toarray(), expected)
