# HEST-GBM Niche Identification Pipeline

本文档说明如何运行当前目录下的 HEST-GBM Visium Niche 识别 MVP。所有源码和配置文件均应保持 UTF-8 无 BOM。

默认流程为 **标准 CARD 反卷积 → ISCHIA 细胞组成聚类 → 生态位注释**。原有的单样本 Leiden＋跨样本凝聚层次聚类完整保留为 `two_stage` 备选后端。

## 1. 环境安装

建议创建独立 Python 环境；默认的标准 CARD 反卷积在 R 中使用 CPU：

```powershell
conda create -n hest-niche python=3.10 -y
conda activate hest-niche
```

安装本项目依赖：

```powershell
pip install -r requirements.txt
pip install -e .
```

如果只做本地 manifest、预处理和测试，可以暂时不安装 `cell2location`、`xgboost`、`openslide-python`；对应子命令会在缺少依赖时给出明确错误。

默认反卷积使用标准 CARD，需要额外安装 R 与 CARD：

```r
install.packages("devtools")
devtools::install_github("YMa-lab/CARD")
devtools::install_github("xuranw/MuSiC")
```

当前固定版本的 CARD 在参考构建函数中调用 MuSiC，因此该运行时依赖也需要安装。

确认运行环境可用：

```powershell
Rscript scripts/run_card_deconvolution.R --check
```

若 `Rscript` 不在 PATH 中，在 `deconvolution.card.rscript_bin` 填写其绝对路径。

ISCHIA 聚类使用仓库内固定版本的官方组成聚类模块 `vendor/ischia/composition_clustering.R`。该入口需要以下 R 包：

```r
install.packages(c("SeuratObject", "Matrix", "jsonlite"))
```

检查聚类环境，不读取数据、不执行聚类：

```powershell
Rscript scripts/run_ischia_clustering.R --check vendor/ischia/composition_clustering.R
```

本项目直接加载未修改的上游组成聚类模块，调用原始 `Composition.cluster()`；这与安装完整 ISCHIA R 包不同。完整包中共现、配体–受体等模块的依赖不属于当前生态位划分入口。来源、固定提交和 SHA256 见 [vendor/ischia](../vendor/ischia/README.md)。当前服务器配置已明确填写 `/data/newhome/yk192/miniconda3/bin/Rscript`。

若使用两级聚类备选方案，Python 环境还需要 `igraph`、`leidenalg`，已列入 `requirements.txt`。

## 2. 数据准备

当前 MVP 默认使用本地已有的 3 个 HEST GBM Visium 样本：

- `TENX30`: targeted panel，跨样本表达特征置信度较低，但保留在 MVP 中。
- `TENX31`: fresh frozen WTA。
- `TENX73`: FFPE CytAssist 11 mm Visium。

如需重新下载：

```powershell
python scripts/download_hest_gbm.py --local-dir HEST_data/hest_gbm_samples
```

校验本地数据：

```powershell
python scripts/validate_local_data.py --config configs/glioma_visium_mvp.yaml
```

## 3. scRNA-seq reference 要求

标准 CARD 和可选的 `cell2location` 均需要外部 GBM/glioma scRNA-seq reference `.h5ad`：

- AnnData 形状为 cells x genes。
- raw counts 位于 `X`、`raw.X` 或配置的 `reference.counts_layer`。
- 细胞类型标签位于 `reference.cell_type_key`，默认 `cell_type`。
- gene symbol 需要能与 HEST `var_names` 匹配。

当前配置沿用 CZ CELLxGENE 上的 `Core GBmap` 单细胞参考。列出下载脚本的可用预设：

```powershell
python scripts/download_cellxgene_gbm_reference.py --list
```

下载推荐 reference：

```powershell
python scripts/download_cellxgene_gbm_reference.py --preset gbmap_core --out-dir reference_data --print-yaml
```

如果本机网络慢，可以把脚本放到服务器上下载，或在浏览器中打开脚本输出的 CELLxGENE H5AD URL 下载。下载完成后再把本地 `.h5ad` 路径填到配置文件。

在 `configs/glioma_visium_mvp.yaml` 中填写：

```yaml
reference:
  h5ad: reference_data/cellxgene_gbmap_core.h5ad
  cell_type_key: cell_type
  batch_key: donor_id
  counts_layer:
  gene_symbol_key: feature_name
```

校验 reference：

```powershell
python -m niche_pipeline --config configs/glioma_visium_mvp.yaml prepare-reference
```

## 3.1 标准 CARD 反卷积

默认配置已设为标准 CARD。Python 导出单细胞原始计数、细胞类型/供者注释、空间原始计数和坐标，然后由 R 调用 `createCARDObject()` 和 `CARD_deconvolution()`。接口见 [CARD 官方教程](https://yma-lab.github.io/CARD/documentation/04_CARD_Example.html)。

关键配置：

```yaml
deconvolution:
  method: card
  card:
    rscript_bin: Rscript
    r_script: scripts/run_card_deconvolution.R
    min_count_gene: 100
    min_count_spot: 5
```

运行反卷积：

```powershell
python -m niche_pipeline --config configs/glioma_visium_mvp.yaml deconvolve
```

输出保留 reference 中的细胞类型名称，每个 spot 的各类细胞比例之和为 1；CSV 继续使用 `sample_id,spot_id,<cell types...>` 接口，供后续特征构建和 Niche 聚类读取。它们是细胞组成比例，不是绝对细胞数量。

`reference.counts_layer` 优先于 `raw.X`，未指定且没有 raw 时使用 `X`；必须是未归一化的非负整数计数。`reference.batch_key` 在 CARD 中作为供者/样本标签；未配置时视为一个参考样本。重复 gene symbol 的计数合并相加，参考细胞不抽样。

空间坐标优先使用 `obsm['spatial']`，其次使用全分辨率像素坐标；仅有 Visium array 坐标时转换为六角网格的等距坐标。

参数原样传入标准 CARD。按其[官方实现](https://github.com/YMa-lab/CARD/blob/master/R/utilities.R)，`min_count_gene` 对应 `minCountGene`，保留总计数不小于该值且不大于 1e6 的 spot；`min_count_spot` 对应 `minCountSpot`，保留在严格多于该数量 spot 中表达的基因。CARD 还会在内部选择信息基因。若过滤后未返回所有预处理 spot，流程会报错并提示检查阈值和基因覆盖，不会把缺失比例补零。

默认配置使用独立目录 `outputs/hest_gbm_card_ischia_mvp` 保存 CARD＋ISCHIA 结果。备选配置使用 `outputs/hest_gbm_card_two_stage_mvp`。历史目录 `outputs/hest_gbm_mvp`、`outputs/hest_gbm_card_mvp` 保留此前的结果和准备记录。

## 3.2 默认生态位识别：ISCHIA

依据 [ISCHIA 官方示例及源码](https://github.com/ati-lz/ISCHIA/blob/1e50ce1eb1f4fe0855ca16a5fd3bfc7228697900/R/composition_clustering.R)，`Composition.cluster()` 直接对 spot × cell type 组成矩阵运行 R 的 `kmeans()`，并产生 `CompositionCluster_CC` 标签。当前接入固定到提交 `1e50ce1eb1f4fe0855ca16a5fd3bfc7228697900`，调用前校验源码 SHA256。

本项目将所选样本的全部 spot 合并到同一细胞类型空间中，统一运行一次该函数，得到各样本共享的 `CC1`、`CC2` 等标签。这是为跨样本统一标签所做的数据组织适配；上游 README 的示例是单样本分析。各样本使用相同细胞类型集合，列顺序可不同；缺失细胞类型不会自动补零。不同样本可有相同 barcode，通过额外的内部 ID 保留原 sample/spot 对应关系。

每个 spot 的细胞类型丰度按行归一化为比例。聚类保留所有细胞类型，不对列标准化，也不加入表达 PC 或邻域平均特征。样本按实际 spot 数贡献数据，不进行样本均衡抽样。`build-features` 仍为后续注释、XGBoost 和备选流程准备特征。

```yaml
clustering:
  method: ischia
  random_seed: 42
  ischia:
    rscript_bin: /data/newhome/yk192/miniconda3/bin/Rscript
    r_script: scripts/run_ischia_clustering.R
    n_clusters: 8
    elbow_max_k: 0
```

`n_clusters: 8` 沿用当前目标簇数，是用户配置的参数。上游函数的 K-means 默认参数保持原样：Hartigan–Wong 算法、`nstart=1`、`iter.max=10`。随机种子固定为 42；若 spot 数或不同组成向量的数量无法满足 k，流程会报错，不会静默减少簇数。

`elbow_max_k: 0` 表示不绘制肘部诊断图。设置为可满足的正整数时，调用上游 `Composition.cluster.k()` 生成 `clusters/ischia/elbow.pdf`，用于人工检查 k；绘图后重置随机种子，保证开启诊断不会改变正式聚类标签。

为保持后续模块兼容，`global_niche` 写为 `ISCHIA_CC1` 等共享标签，`local_niche` 写为 `sampleID_ISCHIA_CC1` 等样本内子集名称。这里的 local/global 两列是同一组成聚类结果的不同标识层次，没有第二次局部聚类或跨样本层次合并。空间邻居一致性和最小簇大小仍作为结果诊断，并产生原有的 `confidence` 字段；它们不参与 ISCHIA 的聚类距离。

当前 ISCHIA 将所有选定样本共同用于定义 CC。后续 leave-one-sample-out XGBoost 学习这些已有标签，属于对当前组成聚类标签的重现评估。面向未见患者的外部验证，需要另行冻结训练集定义的簇并设计标签分配方案。

XGBoost 的输入保留特征构建阶段的数值特征，排除 CC 标签、内部 ID 以及依赖聚类标签计算的邻居一致性，避免将聚类后诊断量作为预测输入。

## 3.3 备选：原有两级无监督聚类

原实现保留在 `cluster_two_stage_niches()` 中：每个样本的全部数值特征先标准化，构建特征近邻图并做 Leiden；随后计算局部簇平均特征，标准化后进行凝聚层次聚类，得到跨样本 global niche。其参数仍为 `local_neighbors: 15`、`local_resolution: 0.8` 和 `global_niches: 8`。

使用独立备选配置，避免覆盖默认后端的结果：

```powershell
python -m niche_pipeline --config configs/glioma_visium_two_stage_mvp.yaml make-manifest
python -m niche_pipeline --config configs/glioma_visium_two_stage_mvp.yaml preprocess
python -m niche_pipeline --config configs/glioma_visium_two_stage_mvp.yaml prepare-reference
python -m niche_pipeline --config configs/glioma_visium_two_stage_mvp.yaml deconvolve
python -m niche_pipeline --config configs/glioma_visium_two_stage_mvp.yaml build-features
python -m niche_pipeline --config configs/glioma_visium_two_stage_mvp.yaml cluster
```

也可在自定义配置中设置 `clustering.method: two_stage`。两套方案继续使用标准 CARD 作为默认反卷积后端。

## 4. 分阶段运行

推荐先分阶段运行，便于定位数据或 R 依赖问题：

```powershell
python -m niche_pipeline --config configs/glioma_visium_mvp.yaml make-manifest
python -m niche_pipeline --config configs/glioma_visium_mvp.yaml preprocess
python -m niche_pipeline --config configs/glioma_visium_mvp.yaml prepare-reference
python -m niche_pipeline --config configs/glioma_visium_mvp.yaml deconvolve
python -m niche_pipeline --config configs/glioma_visium_mvp.yaml build-features
python -m niche_pipeline --config configs/glioma_visium_mvp.yaml cluster
python -m niche_pipeline --config configs/glioma_visium_mvp.yaml annotate
python -m niche_pipeline --config configs/glioma_visium_mvp.yaml extract-patches
python -m niche_pipeline --config configs/glioma_visium_mvp.yaml train-xgb
```

确认 reference 和依赖都可用后，也可以端到端运行：

```powershell
python -m niche_pipeline --config configs/glioma_visium_mvp.yaml run-all
```

### 可选：cell2location 与 GPU 选择

将 `deconvolution.method` 改为 `cell2location` 可使用另一后端。需要先安装与服务器 CUDA 匹配的 PyTorch，以及 Python 依赖中的 cell2location。该后端使用 `deconvolution.accelerator` 和 `deconvolution.device` 控制 GPU，例如：

```yaml
deconvolution:
  method: cell2location
  accelerator: gpu
  device: 0
```

如果服务器有 8 张 GPU，编号 0 到 7，可以把 `device` 改为 `1`、`2` 等来指定单卡；代码内部会转换成 Lightning 可接受的 `[1]`、`[2]`。若调度系统已经设置了 `CUDA_VISIBLE_DEVICES=3`，进程内部通常只看到一张卡，此时配置仍应使用 `device: 0`。

## 5. 主要输出

默认输出目录为 `outputs/hest_gbm_card_ischia_mvp`：

- `manifests/selected_samples.csv`: 样本清单。
- `preprocessed/*.preprocessed.h5ad`: 保留 `layers["counts"]` 的预处理 ST 数据。
- `deconvolution/abundance/*.spot_celltype_abundance.csv`: spot x cell type 比例（标准 CARD）；使用 cell2location 时为丰度。
- `deconvolution/card/exports/`: CARD 使用的参考/空间计数、注释与原始比例。
- `deconvolution/card/logs/`: 各样本 R 标准输出和错误日志。
- `deconvolution/card/card_summary.json`: CARD 运行摘要。
- `features/*.niche_features.csv`: Niche 聚类与模型输入特征。
- `clusters/spot_local_niche_labels.csv`: 各样本内的生态位标签；ISCHIA 中为共享 CC 在该样本的子集。
- `clusters/spot_global_niche_labels.csv`: 共享生态位标签与诊断置信度，供注释、H&E 叠图和 XGBoost 读取。
- `clusters/ischia/spot_celltype_composition.csv`: 实际送入 ISCHIA 的按行归一化组成矩阵。
- `clusters/ischia/spot_id_mapping.csv`: 内部 ID 与原 sample/spot 的映射。
- `clusters/ischia/composition_cluster_labels.csv`: 上游 `CompositionCluster_CC` 原始标签。
- `clusters/ischia/composition_cluster_centers.csv`: 各 CC 的平均细胞类型组成。
- `clusters/ischia/composition_clusters.rds`: 官方函数返回的 Seurat 对象、组成中心和拟合元数据；其中 assay 为细胞比例，不是 RNA 原始计数。
- `clusters/ischia/ischia_summary.json`: 上游来源、种子、输入列、拟合指标与输出路径。
- `clusters/ischia/stdout.log`、`stderr.log`: R 运行日志。
- `annotation/niche_annotation_table.csv`: global niche 解释摘要。
- `annotation/sample_niche_proportion.csv`: sample x niche proportion。
- `histology/overlays/*.png`: H&E 空间叠图。
- `histology/patches/*/*.png`: spot-centered patch。
- `models/xgboost/*`: XGBoost baseline 模型与 leave-one-sample-out 指标。

## 6. 常见问题

- `reference.h5ad is required`: 需要先在 YAML 中填写 GBM/glioma scRNA-seq reference。
- `Too few shared genes`: 检查 reference gene symbol 是否与 HEST `var_names` 一致。
- `Could not find Rscript executable`: 安装 R，或配置 `deconvolution.card.rscript_bin` 的绝对路径。
- `Standard CARD requires R`: 在所配置 Rscript 的 R 环境中安装 CARD 和 Matrix。
- `CARD output spot IDs do not match`: 检查 CARD 过滤阈值及参考/空间信息基因覆盖，查看 `deconvolution/card/logs/`。
- `ISCHIA ... environment check failed`: 检查 `clustering.ischia.rscript_bin` 及 R 的 SeuratObject、Matrix、jsonlite 依赖。
- `ISCHIA ... SHA256`: 上游模块与固定来源记录不匹配，检查 `vendor/ischia/source.json`，不要直接修改上游模块。
- `ISCHIA cell-type columns differ`: 所有样本必须使用相同的细胞类型注释体系。
- `ISCHIA n_clusters ... distinct composition profiles`: 数据不足以支持当前 k，需要显式调整参数。
- `OpenSlide is not available`: 安装 `openslide-python` 后，还需要系统 OpenSlide 动态库；Windows 可通过 conda-forge 安装 `openslide`。
- `xgboost is required`: 安装 `pip install xgboost` 后重跑 `train-xgb`。
- `leidenalg` 相关错误: 安装 `pip install leidenalg igraph`，或确认 Scanpy 环境完整。

## 7. 编码与注释规范

新增 Python、YAML、Markdown 文件必须使用 UTF-8 无 BOM。新增函数需要 docstring 说明用途、参数、返回值和边界行为；必要注释解释为什么这样做，避免空泛描述。
