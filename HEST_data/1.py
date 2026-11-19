import pandas as pd

meta = pd.read_csv("hf://datasets/MahmoodLab/hest/HEST_v1_3_0.csv")

# 看看有哪些列
print(meta.columns.tolist())

# 粗略筛选 brain / glioma / glioblastoma / GBM 相关样本
mask = (
    meta.astype(str).apply(
        lambda col: col.str.contains(
            "glioma|glioblastoma|gbm|brain cancer|ependymoma",
            case=False,
            na=False
        )
    ).any(axis=1)
)

glioma_related = meta[mask]
print(glioma_related[["id", "organ", "oncotree_code", "disease_state", "st_technology"]])