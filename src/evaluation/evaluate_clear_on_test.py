from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from src.classification.feature_pipeline import build_dataset_from_csv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


def evaluate_test_with_threshold(
    features_path: Path,
    labels_path: Path,
    model_path: Path,
    threshold: float = 0.4,
) -> None:
    """
    在专门的 TEST CSV 上进行评估，使用自定义阈值作用在 P(correct) 上。
    """
    if not features_path.exists():
        raise FileNotFoundError(f"Test features.csv not found at {features_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Test labels.csv not found at {labels_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")

    print(f"[INFO] Loading TEST dataset from:\n  {features_path}\n  {labels_path}")
    X_test, y_test, video_names = build_dataset_from_csv(features_path, labels_path)
    print(f"[INFO] Test set: {X_test.shape[0]} videos, {X_test.shape[1]} features.")

    print(f"[INFO] Loading model from: {model_path}")
    model = joblib.load(model_path)

    if not hasattr(model, "predict_proba"):
        raise AttributeError(
            "The loaded model does not support predict_proba(). "
            "Thresholding requires probability outputs."
        )

    print(f"[INFO] Predicting on test set with threshold={threshold:.3f} ...")
    proba = model.predict_proba(X_test)  # (N, 2) -> [incorrect, correct]
    prob_correct = proba[:, 1]
    y_pred = (prob_correct >= threshold).astype(int)

    print("\n[INFO] Test set classification report (with threshold):")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["incorrect", "correct"],
            digits=4,
        )
    )

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    print("[INFO] Confusion matrix (rows=true, cols=pred) [incorrect=0, correct=1]:")
    print(cm)

    print("\nPer-video test predictions (with threshold):")
    print("-" * 80)
    print(f"{'video_name':30s} {'true':10s} {'pred':10s} {'P(correct)':>12s}")
    print("-" * 80)
    for name, yt, yp, pc in zip(video_names, y_test, y_pred, prob_correct):
        yt_str = "correct" if yt == 1 else "incorrect"
        yp_str = "correct" if yp == 1 else "incorrect"
        print(f"{name:30s} {yt_str:10s} {yp_str:10s} {pc:12.4f}")
    print("-" * 80)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate trained clear-classifier on a dedicated TEST CSV "
            "using a custom threshold on P(correct)."
        )
    )
    parser.add_argument(
        "--features",
        type=str,
        default=str(DATA_PROCESSED_DIR / "clear" / "test" / "features_clear_test.csv"),
        help=(
            "Path to TEST features CSV "
            "(default: data/processed/clear/test/features_clear_test.csv)"
        ),
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=str(DATA_PROCESSED_DIR / "clear" / "test" / "labels_clear_test.csv"),
        help=(
            "Path to TEST labels CSV "
            "(default: data/processed/clear/test/labels_clear_test.csv)"
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(MODELS_DIR / "clear_classifier_svm.joblib"),
        help=(
            "Path to trained model joblib file "
            "(default: models/clear_classifier_svm.joblib)"
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.4,
        help=(
            "Decision threshold on P(correct). "
            "If P(correct) >= threshold -> predict 'correct'. "
            "Default: 0.4"
        ),
    )

    args = parser.parse_args()

    evaluate_test_with_threshold(
        features_path=Path(args.features),
        labels_path=Path(args.labels),
        model_path=Path(args.model),
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
