> 当前代码已更新为：标准 CARD 反卷积 → ISCHIA 组成聚类 → 生态位注释。原有 Leiden＋层次聚类保留为 `two_stage` 备选；以下内容保留早期设计。当前运行方法见 [运行说明](docs/run_niche_pipeline.md)，默认与备选配置分别为 [ISCHIA](configs/glioma_visium_mvp.yaml) 和 [两级聚类](configs/glioma_visium_two_stage_mvp.yaml)。

可以。HEST-1k 很适合做你这个任务，因为它本身就是 **空间转录组 + H&E 全切片图像 + metadata** 的多模态数据集，可以支持从 Visium spot 表达、局部空间结构、病理图像三方面去定义 Niche。

## 1. HEST-1k 数据集基本构成
根据 HEST 官方 Hugging Face 页面，当前 HEST-1k 是一个包含 **1,276 个空间转录组 profiles** 的数据集，每个 profile 都与一个已经配准的 **H&E whole-slide image, WSI** 和 metadata 关联；数据覆盖 **26 个器官、2 个物种、398 个癌症样本、25 种癌症类型**。官方页面还说明，HEST-1k 处理后包含超过 **150 万个 expression–morphology pairs** 和超过 **7,600 万个 nuclei**。([Hugging Face](https://huggingface.co/datasets/MahmoodLab/hest))

需要注意版本差异：NeurIPS 2024 论文中报告的是 **1,229 个 paired ST + H&E WSI 样本**，来自 **153 个 cohorts**，覆盖 **26 个器官、2 个物种、367 个癌症样本、25 个癌症亚型**；论文还报告处理得到 **2.1 million expression–morphology pairs** 和 **76 million nuclei**。 这个差异主要来自后续 HEST 数据集持续更新，官方 Hugging Face 页面也列出了 2024–2026 年新增 Xenium、Visium HD 等样本的更新记录。([Hugging Face](https://huggingface.co/datasets/MahmoodLab/hest))

HEST 包含多种空间技术。论文图 1 中给出的早期版本统计显示，主要技术包括 **Visium、STv1、Xenium、Visium HD**，其中 spot-based 的 Visium/STv1 占很大比例；图中还显示 HEST-Library 会对每个样本组织 ST、H&E、nuclei、patches 和 metadata 等文件。 GitHub 说明 HEST-Library 可用于处理 HEST-1k，也支持 ST、Visium、Visium HD、Xenium 样本的装配与分析。([GitHub](https://github.com/mahmoodlab/hest))

所以，对于你的目标，建议先**只筛选 10x Visium spot-based 子集**，不要一开始把 Visium HD、Xenium、STv1 全混在一起。原因是不同技术的 spot/cell 分辨率、坐标结构、测序深度和图像配准方式不同，直接混合会让 Niche 标签定义变得不稳定。

---

# 2. 用 HEST-1k 做完整 Niche 识别 pipeline：总体思路
你的 pipeline 可以拆成两条线：

**第一条线：从 ST 数据识别 Niche 标签**
也就是从 Visium spot 表达矩阵、空间坐标、H&E 图像中得到每个 spot 的 Niche label。

**第二条线：训练 Niche 预测模型**
也就是用第一条线产生的 Niche 标签作为 pseudo-label / weak label，训练一个模型，使其能在新切片、新样本或甚至仅 H&E 图像上预测 Niche。

---

## Part A：HEST-1k Visium 子集筛选与数据准备
### Step 1：筛选合适的 HEST 样本
建议先从 HEST metadata 中筛选：

| 筛选条件       | 建议                                            |
| -------------- | ----------------------------------------------- |
| technology     | 先选 **Visium**                                 |
| species        | 先选 **Homo sapiens**，除非你专门研究小鼠       |
| tissue / organ | 先聚焦 1–2 个组织或癌种                         |
| diagnosis      | 先区分 cancer / healthy / pathological          |
| WSI quality    | 保留图像质量较好、spot 与图像配准正常的样本     |
| spot number    | 去掉 spot 数太少的样本                          |
| batch/cohort   | 尽量记录 cohort 来源，用于后续 batch correction |


第一版不建议直接跑全 HEST-1k。更稳妥的是先选一个癌种或一个器官，比如 breast、colon、prostate、brain、kidney 等，构建一个 organ-specific 或 cancer-specific niche pipeline。

### Step 2：整理每个样本的核心文件
每个样本至少需要：

```latex
1. spot × gene count matrix
2. spot spatial coordinates
3. H&E WSI 或 spot 对应 image patches
4. sample-level metadata
5. 可选：nuclei segmentation / nuclei features
```

HEST 的优势是它已经把 ST 与 H&E WSI 配准，并生成 expression–morphology pairs，这对后面引入 H&E patch 特征很有价值。([Hugging Face](https://huggingface.co/datasets/MahmoodLab/hest))

---

## Part B：单样本层面的 spot-level Niche 识别
这一部分的目标是：对每张 Visium 切片的每个 spot 赋予一个 Niche label。

### Step 3：ST 表达矩阵预处理
对每个 Visium 样本做常规预处理：

```latex
过滤低质量 spot
过滤低表达基因
归一化：log-normalization / SCTransform
选 HVGs
PCA / latent embedding
检查 UMI、gene count、mitochondrial genes
```

注意：HEST 来自多个 cohort，批次效应会明显。HEST 论文也提醒，图像端和转录组端都可能存在 batch effects。 所以不要直接把所有样本合并聚类，应先单样本 QC，再跨样本整合。

### Step 4：spot 去卷积，获得细胞类型组成
这是 Visium Niche 识别最关键的一步。因为 Visium spot 通常不是单细胞，而是多个细胞混合信号。

推荐输入：

```latex
Visium spot expression
+ matched / public scRNA-seq reference
→ spot × cell type abundance
```

可以用：

| 方法          | 适合情况                               |
| ------------- | -------------------------------------- |
| cell2location | 推荐，适合细胞类型 abundance mapping   |
| RCTD          | 稳健，适合常规 Visium                  |
| CARD          | 利用空间相关性                         |
| Tangram       | 将 scRNA-seq 映射到空间                |
| DestVI        | 适合进一步分析 cell-type-specific 状态 |


输出矩阵：

```latex
spot × cell_type_abundance
```

例如：

```latex
spot_001: Tumor 0.62, CAF 0.18, Macrophage 0.07, T cell 0.04, Endothelial 0.09
spot_002: Tumor 0.10, CAF 0.44, Macrophage 0.22, T cell 0.06, Endothelial 0.18
```

如果你暂时没有 scRNA-seq reference，可以先用 marker gene module score 构建粗粒度 cell type score，但正式 pipeline 最好还是使用去卷积。

### Step 5：构建 spot 空间邻域图
Niche 的核心不是单个 spot，而是局部空间微环境。因此要基于 spot 坐标建立邻接图。

推荐三种方式：

| 邻域定义            | 说明                       |
| ------------------- | -------------------------- |
| Visium hex neighbor | 利用 Visium 六边形阵列结构 |
| kNN graph           | 比如 k=6、12、18           |
| radius graph        | 按物理距离定义邻居         |


第一版建议：

```latex
k = 12 或 18
每个 spot 的邻域 = 自身 + k 个空间邻居
```

并做 sensitivity analysis：

```latex
k = 6, 12, 18, 24
比较 Niche label 是否稳定
```

### Step 6：构建 Niche feature matrix
对每个 spot 构建一个表示其“局部生态位”的特征向量。

基础版：

```latex
Niche feature = 邻域平均细胞类型组成
```

也就是：

```latex
spot_i 的 Niche 特征
= spot_i 及其邻居中 Tumor / CAF / T cell / B cell / Macrophage / Endothelial 等 abundance 的加权平均
```

进阶版：

```latex
Niche feature =
邻域细胞类型组成
+ spot 表达 embedding
+ pathway score
+ ligand-receptor score
+ H&E patch morphology embedding
+ nuclei density / morphology features
```

建议特征分层构建：

| 特征层               | 内容                                   | 作用                 |
| -------------------- | -------------------------------------- | -------------------- |
| Composition          | 去卷积细胞类型比例                     | 定义细胞组成型 Niche |
| Spatial neighborhood | 邻域平均组成                           | 定义局部微环境       |
| Expression program   | PCA / gene module / HVG                | 捕捉功能状态         |
| Pathway              | hypoxia、EMT、IFN、TGF-β、angiogenesis | 功能解释             |
| Morphology           | H&E patch embedding、nuclei density    | 多模态增强           |
| LR interaction       | ligand-receptor score                  | 细胞互作解释         |


### Step 7：无监督聚类，得到初始 Niche labels
对 `spot × niche_feature` 矩阵聚类。

可选方法：

```latex
Leiden / Louvain
k-means
hierarchical clustering
NMF
Gaussian mixture model
graph clustering
```

推荐第一版使用：

```latex
PCA / UMAP on niche features
→ kNN graph
→ Leiden clustering
```

得到：

```latex
spot_id → niche_cluster
```

例如：

```latex
spot_001 → Niche_1
spot_002 → Niche_3
spot_003 → Niche_1
```

### Step 8：把 Niche label 映射回 H&E 和空间坐标
这是非常重要的质控。

一个可信的 Niche 应该满足：

```latex
1. 在空间上有一定连续性
2. 和 H&E 组织结构有一定对应
3. 有明确的细胞组成特征
4. 有可解释的 marker/pathway 特征
```

如果某个 cluster 在空间上完全随机散布，且没有明显细胞组成或通路特征，就不应轻易命名为 Niche，可以标为 unstable / mixed / low-confidence niche。

---

## Part C：Niche 标签解释与人工标注
算法聚类得到的是 `Niche_0, Niche_1, Niche_2...`，还不是生物学标签。你需要根据多维信息命名。

### Step 9：为每个 Niche 生成解释表
对每个 Niche 计算：

| 解释维度       | 分析内容                                                     |
| -------------- | ------------------------------------------------------------ |
| 细胞组成       | 哪些 cell types 富集                                         |
| marker genes   | 每个 Niche 的差异表达基因                                    |
| pathway        | hallmark / KEGG / Reactome / PROGENy                         |
| 空间位置       | tumor core、invasive margin、stroma、necrosis、vascular region |
| H&E morphology | 细胞密度、腺体结构、坏死、间质区、免疫浸润                   |
| 样本分布       | 是否跨样本稳定出现                                           |
| 临床关联       | 是否与 diagnosis、grade、response、survival 相关             |


### Step 10：命名 Niche 标签
命名建议用“组成 + 功能 + 空间位置”的格式，而不是只用细胞类型。

例如：

| 聚类结果 | 可能命名                               |
| -------- | -------------------------------------- |
| Niche_0  | Tumor-rich proliferative niche         |
| Niche_1  | CAF-rich stromal niche                 |
| Niche_2  | Immune-active lymphoid niche           |
| Niche_3  | Macrophage-enriched inflammatory niche |
| Niche_4  | Hypoxic tumor niche                    |
| Niche_5  | Vascular endothelial niche             |
| Niche_6  | Tumor–stroma interface niche           |
| Niche_7  | Necrosis-associated niche              |


特别建议保留一个：

```latex
Mixed / low-confidence niche
```

因为 Visium spot 混合度高，不是所有区域都能被清楚解释。

---

## Part D：跨样本 Niche 对齐与统一标签体系
HEST 是多样本、多 cohort 数据。如果你只在单张切片内聚类，每个样本的 `Niche_1` 可能不是同一个生物学生态位。因此需要跨样本统一。

### Step 11：构建跨样本 Niche signature
对每个样本内的每个 niche cluster 计算：

```latex
1. 平均细胞组成
2. marker gene signature
3. pathway signature
4. morphology embedding centroid
```

然后把这些 cluster-level signatures 再做一次跨样本聚类。

得到：

```latex
local niche cluster → global niche class
```

例如：

```latex
Sample_A_Niche_2 → Global CAF-rich stromal niche
Sample_B_Niche_5 → Global CAF-rich stromal niche
Sample_C_Niche_1 → Global CAF-rich stromal niche
```

### Step 12：建立 HEST-Niche atlas
最终你可以得到：

```latex
spot-level label:
spot → local niche → global niche

sample-level summary:
sample → niche proportion vector
```

输出矩阵：

```latex
sample × global_niche_proportion
```

这个矩阵后面非常重要，可用于：

```latex
分组比较
癌种比较
临床关联
生存分析
bulk RNA-seq niche proportion benchmark
模型训练标签
```

---

# 3. 后续 Niche 预测模型训练思路
有了上述 Niche 标签后，就可以训练预测模型。

你的模型可以分三种任务层级。

## 任务 1：基于 ST 输入预测 Niche
这是最直接的模型。

### 输入
```latex
spot expression embedding
+ spot deconvolution abundance
+ spatial graph
+ pathway score
```

### 输出
```latex
spot-level niche label
```

### 模型
| 模型                            | 说明                            |
| ------------------------------- | ------------------------------- |
| Random Forest / XGBoost         | baseline，可解释                |
| MLP                             | 输入 spot 特征                  |
| GCN / GraphSAGE / GAT           | 输入 spot graph，更适合空间结构 |
| Transformer / Graph Transformer | 多 spot 上下文建模              |


### 推荐 baseline
第一版建议先做：

```latex
XGBoost / Random Forest
输入：cell type abundance + pathway score + expression PCA
输出：global niche label
```

然后再做：

```latex
GraphSAGE / GAT
输入：spot graph + spot features
输出：global niche label
```

## 任务 2：基于 H&E 图像预测 Niche
HEST 的优势在这里体现出来：每个 spot 都配有对应的 H&E patch，因此可以训练从图像预测 Niche 的模型。

### 输入
```latex
spot-centered H&E patch
```

### 输出
```latex
spot-level niche label
```

### 图像特征
可以使用 pathology foundation model 提取 embedding，例如：

```latex
UNI / Virchow / GigaPath / CTransPath / ResNet50
```

HEST 本身的 benchmark 就是评估 pathology foundation models 从 histology 预测 gene expression 的能力；论文说明 HEST-Benchmark 包含多个器官和癌种任务，用于 gene expression prediction from histology。 因此，你可以把任务从“预测 gene expression”扩展为“预测 transcriptomics-derived niche label”。

### 推荐训练方式
```latex
H&E patch embedding
→ MLP / XGBoost / Transformer
→ Niche label
```

进一步可以做：

```latex
多实例学习 MIL:
slide-level patch set → sample-level niche proportion
```

## 任务 3：多模态 Niche 预测模型
最终目标可以是：

```latex
ST expression + spatial graph + H&E patch
→ Niche label
```

模型结构：

```latex
Expression encoder
+ Histology encoder
+ Spatial graph encoder
→ fusion layer
→ niche classifier
```

训练目标：

```latex
主任务：spot-level niche classification
辅助任务 1：cell composition reconstruction
辅助任务 2：pathway score prediction
辅助任务 3：neighbor niche consistency
辅助任务 4：sample-level phenotype prediction
```

---

# 4. 推荐的完整 pipeline 版本
## Phase 1：数据筛选与 QC
```latex
下载 HEST metadata
筛选 technology = Visium
筛选 species = human
选择一个 organ/cancer type
检查每个样本 spot 数、基因数、WSI 配准质量
建立 sample manifest
```

输出：

```latex
selected_samples.csv
```

## Phase 2：ST 预处理
```latex
读取 count matrix
spot/gene QC
normalization
HVG selection
PCA
batch effect visualization
```

输出：

```latex
AnnData / Seurat object
spot_expression_embedding
```

## Phase 3：spot 去卷积
```latex
准备 scRNA-seq reference
运行 cell2location / RCTD
得到 spot × cell type abundance
```

输出：

```latex
spot_celltype_abundance.csv
```

## Phase 4：空间邻域建模
```latex
读取 spot coordinates
构建 kNN / radius graph
计算邻域平均细胞组成
计算邻域 pathway score
计算空间平滑特征
```

输出：

```latex
spot_neighborhood_features.csv
spot_graph.pkl
```

## Phase 5：Niche 聚类
```latex
整合 composition + expression + pathway + morphology features
PCA / latent embedding
Leiden / NMF / graph clustering
得到 local niche labels
```

输出：

```latex
spot_local_niche_label.csv
```

## Phase 6：Niche 解释与标注
```latex
每个 niche 计算：
细胞组成富集
DE genes
pathway enrichment
H&E morphology
空间位置
跨样本出现频率
```

输出：

```latex
niche_annotation_table.csv
spot_global_niche_label.csv
```

## Phase 7：跨样本统一
```latex
对每个 local niche 构建 signature
跨样本聚类 local niche signatures
得到 global niche taxonomy
```

输出：

```latex
global_niche_taxonomy.csv
sample_niche_proportion.csv
```

## Phase 8：训练预测模型
```latex
训练集/验证集/测试集按 sample split
Baseline 1: XGBoost using ST-derived features
Baseline 2: GNN using spatial graph
Baseline 3: H&E patch embedding classifier
Advanced: multimodal fusion model
```

输出：

```latex
trained_niche_classifier.pt
niche_prediction_results.csv
```

## Phase 9：评估
```latex
spot-level Accuracy / Macro-F1
ARI / NMI
空间一致性 Moran’s I / neighbor agreement
跨样本泛化 leave-one-sample-out
生物学验证 marker/pathway/H&E
sample-level niche proportion correlation
```

---

# 5. 你可以先做的最小可行版本
我建议第一版 MVP 不要太复杂，按下面做：

```latex
HEST-1k 中选择一个 Visium 癌种子集
        ↓
ST QC + normalization
        ↓
cell2location / RCTD 去卷积
        ↓
构建 spot kNN 空间邻域
        ↓
计算邻域细胞类型组成
        ↓
Leiden 聚类得到 Niche
        ↓
根据细胞组成 + marker + H&E 命名
        ↓
得到 sample × niche proportion
        ↓
训练 XGBoost / GNN 预测 Niche label
```

这个 MVP 的优势是：**可解释、可复现、容易 debug**。等它跑通后，再加入 H&E patch embedding、nuclei features、ligand-receptor score 和多模态模型。

---

# 6. 关键注意事项
第一，HEST-1k 是多来源数据集，**batch effect 一定要处理**。最好不要全库直接合并，而是先做 organ-specific 或 disease-specific pipeline。

第二，Visium 是 spot-based 数据，Niche 标签应该叫：

```latex
spot-level niche
region-level niche
composition-defined niche
morphomolecular niche
```

不要称为 single-cell niche。

第三，Niche label 一开始通常是 pseudo-label，不是 gold standard。后续需要通过 marker、pathway、H&E、空间连续性和跨样本复现性来筛选高置信 Niche。

第四，如果你的最终目标是训练预测模型，划分训练/测试集时一定要按 **sample-level split**，不能随机按 spot split。否则同一张切片上的相邻 spot 会同时出现在训练和测试中，导致数据泄漏。

第五，建议保留三类标签：

```latex
high-confidence niche
mixed niche
low-quality / uncertain niche
```

这样模型训练会更稳。

---

一句话总结：
**用 HEST-1k 做 Niche pipeline 的最佳起点是：筛选 Visium 子集 → spot 去卷积 → 构建空间邻域 → 基于邻域细胞组成和表达程序聚类 → 生物学解释并统一成 global niche labels → 用这些标签训练 ST/GNN/H&E 多模态 Niche 预测模型。**