# ISCHIA composition-clustering module

`composition_clustering.R` is an unchanged copy of `R/composition_clustering.R` from
[ati-lz/ISCHIA](https://github.com/ati-lz/ISCHIA), pinned to commit
[`1e50ce1eb1f4fe0855ca16a5fd3bfc7228697900`](https://github.com/ati-lz/ISCHIA/tree/1e50ce1eb1f4fe0855ca16a5fd3bfc7228697900).
The source SHA256, upstream path and declared package version are recorded in
`source.json`. The Python backend verifies this hash before calling R.

The pipeline sources this module directly and calls the original
`Composition.cluster()` function. Its only operational dependencies are
`stats::kmeans` and `SeuratObject::AddMetaData`; the bridge binds those names in
an isolated R environment and uses a real Seurat object. The optional elbow
diagnostic calls the original `Composition.cluster.k()` with `graphics::plot`.
The Seurat assay contains cell-type proportions and only carries metadata;
clustering receives the original composition matrix directly.

This integration uses the upstream composition-clustering module. It does not
load or claim to install the complete ISCHIA R package, and does not run the
separate cell-type cooccurrence or ligand-receptor analysis modules. Those
modules' dependencies are not needed for this clustering entry point.

Project adaptation: all selected samples' spots are pooled in a common
cell-type space for one clustering, so CC labels are shared across samples.
The upstream README illustrates one sample at a time. The project does not
weight samples equally, standardize columns, add expression PCs, or perform
a second clustering or spatial smoothing of the resulting CC labels.
