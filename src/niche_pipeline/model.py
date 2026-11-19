"""XGBoost Niche 预测 baseline。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder

from .config import output_root
from .manifest import load_manifest
from .utils import ensure_dir, get_logger, require_columns, save_json

LOGGER = get_logger(__name__)


def leave_one_sample_out_splits(samples: pd.Series) -> Iterator[tuple[np.ndarray, np.ndarray, str]]:
    """生成严格按 sample 划分的 leave-one-sample-out split。

    参数:
        samples: 每个 spot 对应的 sample ID。

    返回值:
        迭代产生 ``train_idx, test_idx, held_out_sample``。

    异常/边界:
        样本少于 2 个时抛出异常；该函数用于防止 spot 级随机划分造成空间泄漏。
    """

    unique_samples = sorted(pd.Series(samples).astype(str).unique())
    if len(unique_samples) < 2:
        raise ValueError("At least two samples are required for leave-one-sample-out evaluation.")
    sample_array = pd.Series(samples).astype(str).to_numpy()
    for sample_id in unique_samples:
        test_idx = np.where(sample_array == sample_id)[0]
        train_idx = np.where(sample_array != sample_id)[0]
        if set(sample_array[train_idx]).intersection(set(sample_array[test_idx])):
            raise RuntimeError("Sample-level split leakage detected.")
        yield train_idx, test_idx, sample_id


def train_xgboost_baseline(config: dict[str, Any], manifest_path: str | Path | None = None) -> dict[str, Path]:
    """训练 XGBoost baseline 预测 global niche。

    参数:
        config: pipeline 配置字典。
        manifest_path: 可选 manifest 路径；用于确认样本列表。

    返回值:
        关键模型与评估文件路径。

    异常/边界:
        xgboost 未安装时抛出明确错误；测试集中训练未见过的标签会被记录并排除在该 fold 指标外。
    """

    try:
        import joblib
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError("xgboost and joblib are required for train-xgb. Install requirements.txt first.") from exc

    load_manifest(config, manifest_path)
    out_root = output_root(config)
    model_dir = ensure_dir(out_root / "models" / "xgboost")
    table = _training_table(config)
    feature_cols = _model_feature_columns(table)
    x_all = table[feature_cols].fillna(0.0).to_numpy(dtype=float)
    y_all = table["global_niche"].astype(str).to_numpy()
    sample_series = table["sample_id"].astype(str)

    fold_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    seed = int(config.get("prediction", {}).get("random_seed", 42))
    for train_idx, test_idx, held_out in leave_one_sample_out_splits(sample_series):
        train_labels = y_all[train_idx]
        known_mask = np.isin(y_all[test_idx], np.unique(train_labels))
        if len(np.unique(train_labels)) < 2 or known_mask.sum() == 0:
            fold_rows.append(
                {
                    "held_out_sample": held_out,
                    "status": "skipped",
                    "reason": "fewer_than_two_train_classes_or_all_test_labels_unseen",
                    "n_test": int(len(test_idx)),
                    "n_unseen_test_labels": int((~known_mask).sum()),
                }
            )
            continue

        encoder = LabelEncoder().fit(train_labels)
        model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=seed,
            n_jobs=4,
        )
        model.fit(x_all[train_idx], encoder.transform(train_labels))
        usable_test_idx = test_idx[known_mask]
        pred_codes = model.predict(x_all[usable_test_idx])
        pred_labels = encoder.inverse_transform(pred_codes.astype(int))
        true_labels = y_all[usable_test_idx]
        fold_rows.append(
            {
                "held_out_sample": held_out,
                "status": "ok",
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "n_used_test": int(len(usable_test_idx)),
                "n_unseen_test_labels": int((~known_mask).sum()),
                "accuracy": float(accuracy_score(true_labels, pred_labels)),
                "macro_f1": float(f1_score(true_labels, pred_labels, average="macro")),
            }
        )
        for row_idx, true_label, pred_label in zip(usable_test_idx, true_labels, pred_labels):
            prediction_rows.append(
                {
                    "sample_id": table.iloc[row_idx]["sample_id"],
                    "spot_id": table.iloc[row_idx]["spot_id"],
                    "true_global_niche": true_label,
                    "pred_global_niche": pred_label,
                    "held_out_sample": held_out,
                }
            )

    final_encoder = LabelEncoder().fit(y_all)
    final_model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=seed,
        n_jobs=4,
    )
    final_model.fit(x_all, final_encoder.transform(y_all))

    model_path = model_dir / "xgboost_global_niche.joblib"
    joblib.dump({"model": final_model, "label_encoder": final_encoder, "feature_cols": feature_cols}, model_path)

    metrics_path = model_dir / "leave_one_sample_out_metrics.csv"
    predictions_path = model_dir / "leave_one_sample_out_predictions.csv"
    importance_path = model_dir / "feature_importance.csv"
    pd.DataFrame(fold_rows).to_csv(metrics_path, index=False, encoding="utf-8")
    pd.DataFrame(prediction_rows).to_csv(predictions_path, index=False, encoding="utf-8")
    pd.DataFrame({"feature": feature_cols, "importance": final_model.feature_importances_}).sort_values(
        "importance", ascending=False
    ).to_csv(importance_path, index=False, encoding="utf-8")
    save_json(
        {
            "model": str(model_path),
            "metrics": str(metrics_path),
            "predictions": str(predictions_path),
            "feature_importance": str(importance_path),
        },
        model_dir / "xgboost_outputs.json",
    )
    return {
        "model": model_path,
        "metrics": metrics_path,
        "predictions": predictions_path,
        "feature_importance": importance_path,
    }


def _training_table(config: dict[str, Any]) -> pd.DataFrame:
    """合并所有样本的 feature 与 global labels。"""

    out_root = output_root(config)
    label_path = out_root / "clusters" / "spot_global_niche_labels.csv"
    if not label_path.exists():
        raise FileNotFoundError(f"Global labels do not exist: {label_path}")
    labels = pd.read_csv(label_path)
    require_columns(labels, ["sample_id", "spot_id", "global_niche"], "global labels")

    frames: list[pd.DataFrame] = []
    for sample_id in sorted(labels["sample_id"].astype(str).unique()):
        feature_path = out_root / "features" / f"{sample_id}.niche_features.csv"
        if not feature_path.exists():
            raise FileNotFoundError(f"Feature CSV does not exist: {feature_path}")
        frames.append(pd.read_csv(feature_path))
    features = pd.concat(frames, axis=0, ignore_index=True)
    table = labels.merge(features, on=["sample_id", "spot_id"], how="inner")
    if table.empty:
        raise ValueError("No joined rows between global labels and niche features.")
    return table


def _model_feature_columns(table: pd.DataFrame) -> list[str]:
    """选择 XGBoost 输入特征列。"""

    exclude = {
        "sample_id",
        "spot_id",
        "local_niche",
        "global_niche",
        "CompositionCluster_CC",
        "ischia_spot_id",
        # 聚类后、依赖标签的诊断量不能作为分类器输入。
        "neighbor_agreement",
        "confidence",
        "confidence_reason",
    }
    numeric = table.drop(columns=[column for column in exclude if column in table.columns]).select_dtypes(include=[np.number])
    if numeric.empty:
        raise ValueError("No numeric features are available for model training.")
    return list(numeric.columns)
