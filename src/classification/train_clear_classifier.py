# src/classification/train_clear_classifier.py

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .feature_pipeline import build_dataset_from_csv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


def make_svm_pipeline() -> GridSearchCV:
    """
    Build an SVM classifier with StandardScaler and GridSearch.
    """
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", SVC(probability=True)),
        ]
    )

    param_grid = {
        "clf__kernel": ["linear", "rbf"],
        "clf__C": [0.1, 1, 10, 100],
        "clf__gamma": ["scale", "auto"],
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


def make_rf_model() -> GridSearchCV:
    """
    Build a RandomForest classifier with GridSearch.
    """
    rf = RandomForestClassifier(random_state=42)

    param_grid = {
        "clf__n_estimators": [100, 200, 300],
        "clf__max_depth": [None, 5, 10],
        "clf__min_samples_leaf": [1, 2, 4],
    }

    pipe = Pipeline(
        steps=[
            ("clf", rf),
        ]
    )

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


def train_and_evaluate(model_type: str = "svm") -> None:
    """
    Main training entry:
      - Build dataset from CSV
      - Cross-validated hyperparameter search
      - Print classification report
      - Save best model to models/
    """
    features_path = DATA_PROCESSED_DIR / "features.csv"
    labels_path = DATA_PROCESSED_DIR / "labels.csv"

    print(f"[INFO] Loading dataset from:\n  {features_path}\n  {labels_path}")
    X, y, names = build_dataset_from_csv(features_path, labels_path)
    print(f"[INFO] Dataset built: {X.shape[0]} videos, {X.shape[1]} features.")

    if model_type.lower() == "svm":
        grid = make_svm_pipeline()
        model_name = "clear_classifier_svm.joblib"
    elif model_type.lower() in ("rf", "random_forest", "randomforest"):
        grid = make_rf_model()
        model_name = "clear_classifier_rf.joblib"
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    # Cross-validation + select best hyperparameters
    print(f"[INFO] Starting GridSearchCV for {model_type} ...")
    grid.fit(X, y)

    print(f"[INFO] Best CV accuracy: {grid.best_score_:.4f}")
    print(f"[INFO] Best params: {grid.best_params_}")

    # Get cross-validated predictions on the full dataset and print a summary report
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(grid.best_estimator_, X, y, cv=cv)

    print("[INFO] Cross-validated classification report:")
    print(classification_report(y, y_pred, target_names=["incorrect", "correct"]))

    # Save the model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / model_name
    joblib.dump(grid.best_estimator_, model_path)
    print(f"[INFO] Saved best model to: {model_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train a badminton clear-classifier (correct vs incorrect) using SVM or RandomForest."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="svm",
        choices=["svm", "rf"],
        help="Which model to train: 'svm' or 'rf'. Default: svm",
    )

    args = parser.parse_args()
    train_and_evaluate(model_type=args.model)


if __name__ == "__main__":
    main()
