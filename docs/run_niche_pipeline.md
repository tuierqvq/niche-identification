# HEST-GBM Niche Identification Pipeline

本文档说明如何运行当前目录下的 HEST-GBM Visium Niche 识别 MVP。所有源码和配置文件均应保持 UTF-8 无 BOM。

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
```

确认运行环境可用：

```powershell
Rscript scripts/run_card_deconvolution.R --check
```

若 `Rscript` 不在 PATH 中，在 `deconvolution.card.rscript_bin` 填写其绝对路径。

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

默认配置使用独立目录 `outputs/hest_gbm_card_mvp` 保存 CARD 结果。旧的 `outputs/hest_gbm_mvp` 是此前运行的结果，不应与新的 CARD 结果混用。

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

默认输出目录为 `outputs/hest_gbm_card_mvp`：

- `manifests/selected_samples.csv`: 样本清单。
- `preprocessed/*.preprocessed.h5ad`: 保留 `layers["counts"]` 的预处理 ST 数据。
- `deconvolution/abundance/*.spot_celltype_abundance.csv`: spot x cell type 比例（标准 CARD）；使用 cell2location 时为丰度。
- `deconvolution/card/exports/`: CARD 使用的参考/空间计数、注释与原始比例。
- `deconvolution/card/logs/`: 各样本 R 标准输出和错误日志。
- `deconvolution/card/card_summary.json`: CARD 运行摘要。
- `features/*.niche_features.csv`: Niche 聚类与模型输入特征。
- `clusters/spot_local_niche_labels.csv`: 单样本 local niche。
- `clusters/spot_global_niche_labels.csv`: 跨样本 global niche 与置信度。
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
- `OpenSlide is not available`: 安装 `openslide-python` 后，还需要系统 OpenSlide 动态库；Windows 可通过 conda-forge 安装 `openslide`。
- `xgboost is required`: 安装 `pip install xgboost` 后重跑 `train-xgb`。
- `leidenalg` 相关错误: 安装 `pip install leidenalg igraph`，或确认 Scanpy 环境完整。

## 7. 编码与注释规范

新增 Python、YAML、Markdown 文件必须使用 UTF-8 无 BOM。新增函数需要 docstring 说明用途、参数、返回值和边界行为；必要注释解释为什么这样做，避免空泛描述。
