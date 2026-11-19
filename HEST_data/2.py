from huggingface_hub import snapshot_download

repo_id = "MahmoodLab/hest"
sample_ids = ["TENX73", "TENX31", "TENX30"]

allow_patterns = ["HEST_v1_3_0.csv"]

for sid in sample_ids:
    allow_patterns.extend([
        f"st/{sid}.h5ad",
        f"metadata/{sid}.*",
        f"wsis/{sid}.*",
    ])

local_dir = snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir="./hest_gbm_samples",
    allow_patterns=allow_patterns,
)

print("Downloaded to:", local_dir)