# src/classification/train_clear_classifier_lr_pca.py

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .feature_pipeline import build_dataset_from_csv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


def make_lr_pca_pipeline() -> GridSearchCV:
    """
    构造一个：StandardScaler -> PCA -> LogisticRegression 的 Pipeline，
    再用 GridSearchCV 做超参数搜索。

    这里通过：
      - PCA 降维（减少特征数，缓解过拟合）
      - LogisticRegression + 强正则（小 C）简化模型复杂度
    """
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", PCA()),  # 具体 n_components 通过网格搜索来定
            (
                "clf",
                LogisticRegression(
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=2000,
                    class_weight=None,  # 如果以后想平衡类别，可以改为 "balanced"
                ),
            ),
        ]
    )

    # 超参数网格：
    # - pca__n_components: 可以是主成分数（整数），也可以是保留方差比例（浮点数）
    #   这里我们试两种：保留 90% / 95% 方差，或者直接压到 20 / 40 维
    # - clf__C: 正则强度，越小越“保守”，越不容易过拟合
    param_grid = {
        "pca__n_components": [0.90, 0.95, 20, 40],
        "clf__C": [0.01, 0.1, 1.0],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
    )
    return grid


def train_and_evaluate_lr_pca() -> None:
    """
    主训练入口：
      - 从 CSV 构造数据集 (X, y)
      - 用 LogisticRegression + PCA 做 GridSearchCV
      - 打印交叉验证分类报告
      - 保存最佳模型到 models/clear_classifier_lr_pca.joblib
    """
    features_path = DATA_PROCESSED_DIR / "features.csv"
    labels_path = DATA_PROCESSED_DIR / "labels.csv"

    print(f"[INFO] Loading dataset from:\n  {features_path}\n  {labels_path}")
    X, y, names = build_dataset_from_csv(features_path, labels_path)
    print(f"[INFO] Dataset built: {X.shape[0]} videos, {X.shape[1]} features.")

    grid = make_lr_pca_pipeline()

    print("[INFO] Starting GridSearchCV for LogisticRegression + PCA ...")
    grid.fit(X, y)

    print(f"[INFO] Best CV accuracy: {grid.best_score_:.4f}")
    print(f"[INFO] Best params: {grid.best_params_}")

    # 用最佳模型做一次交叉验证预测，看看整体表现
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_model = grid.best_estimator_
    y_pred = cross_val_predict(best_model, X, y, cv=cv)

    print("[INFO] Cross-validated classification report (LogReg + PCA):")
    print(
        classification_report(
            y,
            y_pred,
            target_names=["incorrect", "correct"],
            digits=4,
        )
    )

    # 保存最佳模型
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "clear_classifier_lr_pca.joblib"
    joblib.dump(best_model, model_path)
    print(f"[INFO] Saved best Logistic+PCA model to: {model_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train a badminton clear-classifier (correct vs incorrect) "
            "using Logistic Regression + PCA to reduce overfitting."
        )
    )
    # 目前这个脚本只有一种模型方式，如果以后想扩展，就在这里加参数
    _ = parser.parse_args()

    train_and_evaluate_lr_pca()


if __name__ == "__main__":
    main()
