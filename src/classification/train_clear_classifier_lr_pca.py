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
    Build a Pipeline: StandardScaler -> PCA -> LogisticRegression,
    then use GridSearchCV for hyperparameter search.

    Approach:
      - PCA for dimensionality reduction (fewer features, less overfitting)
      - LogisticRegression with stronger regularization (small C) to reduce model complexity
    """
    scaler = StandardScaler()
    pca = PCA()  # n_components determined via grid search
    clf = LogisticRegression(max_iter=2000, solver="lbfgs", class_weight=None)  # set to 'balanced' if class balancing is desired later

    pipe = Pipeline(steps=[("scaler", scaler), ("pca", pca), ("clf", clf)])

    # Hyperparameter grid:
    # - pca__n_components: number of components (int) or variance ratio (float)
    #   Try two styles: keep 90%/95% variance, or reduce to 20/40 dims
    # - clf__C: regularization strength; smaller C is more conservative and less prone to overfitting
    param_grid = {
        "pca__n_components": [0.90, 0.95, 20, 40],
        "clf__C": [0.01, 0.05, 0.1, 0.2, 0.5],
    }

    grid = GridSearchCV(pipe, param_grid=param_grid, scoring="f1", cv=5, n_jobs=-1, verbose=1)
    return grid


def train_and_evaluate_lr_pca(features_path: Path, labels_path: Path, save_model_path: Path) -> None:
    """
    Main training entry:
      - Build dataset from CSV (X, y)
      - GridSearchCV with LogisticRegression + PCA
      - Print CV classification report
      - Save best model to models/clear_classifier_lr_pca.joblib
    """
    X, y, names = build_dataset_from_csv(features_path, labels_path)

    grid = make_lr_pca_pipeline()
    grid.fit(X, y)

    best_model = grid.best_estimator_
    print("Best params:", grid.best_params_)

    # Use the best model to perform cross-validated predictions to see overall performance
    y_pred = cross_val_predict(best_model, X, y, cv=5, n_jobs=-1)
    print(classification_report(y, y_pred))

    # Save the best model
    save_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, save_model_path)
    print(f"Saved LogisticRegression+PCA model to {save_model_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_csv", type=str, required=True)
    parser.add_argument("--labels_csv", type=str, required=True)
    parser.add_argument("--save_model", type=str, default=str(Path("models/clear_classifier_lr_pca.joblib")))
    # Currently this script supports a single modeling approach; add args here if expanding later

    args = parser.parse_args()

    features_path = Path(args.features_csv)
    labels_path = Path(args.labels_csv)
    save_model_path = Path(args.save_model)

    train_and_evaluate_lr_pca(features_path, labels_path, save_model_path)


if __name__ == "__main__":
    main()
