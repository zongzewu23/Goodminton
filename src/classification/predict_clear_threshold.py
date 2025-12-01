# src/classification/predict_clear_threshold.py

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import classification_report

from .feature_pipeline import build_dataset_from_csv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


def int_to_label_str(label_int: int) -> str:
    return "correct" if label_int == 1 else "incorrect"


def predict_with_threshold(
    features_path: Path,
    labels_path: Path,
    model_path: Path,
    threshold: float = 0.4,
    print_report: bool = True,
) -> None:
    """
    使用给定模型 + 自定义阈值，对 features/labels CSV 中的每个视频做预测。
    阈值作用在 P(correct) 上：P(correct) >= threshold -> correct，否则 incorrect。
    """
    if not features_path.exists():
        raise FileNotFoundError(f"features.csv not found at {features_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"labels.csv not found at {labels_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")

    print(f"[INFO] Loading dataset from:\n  {features_path}\n  {labels_path}")
    X, y_true, video_names = build_dataset_from_csv(features_path, labels_path)
    print(f"[INFO] Dataset built: {X.shape[0]} videos, {X.shape[1]} features.")

    print(f"[INFO] Loading model from: {model_path}")
    model = joblib.load(model_path)

    if not hasattr(model, "predict_proba"):
        raise AttributeError(
            "The loaded model does not support predict_proba(). "
            "Thresholding requires probability outputs."
        )

    print(f"[INFO] Predicting with threshold={threshold:.3f} on P(correct) ...")
    proba = model.predict_proba(X)  # shape (N, 2), labels assumed [0, 1] = [incorrect, correct]
    prob_correct = proba[:, 1]

    # 应用自定义阈值
    y_pred = (prob_correct >= threshold).astype(int)

    print("\nPer-video prediction result (with threshold):")
    print("-" * 80)
    print(f"{'video_name':30s} {'true':10s} {'pred':10s} {'P(correct)':>12s}")
    print("-" * 80)

    for name, yt, yp, pc in zip(video_names, y_true, y_pred, prob_correct):
        yt_str = int_to_label_str(int(yt))
        yp_str = int_to_label_str(int(yp))
        print(f"{name:30s} {yt_str:10s} {yp_str:10s} {pc:12.4f}")

    print("-" * 80)

    if print_report:
        print("\n[INFO] Classification report (based on provided labels):")
        print(
            classification_report(
                y_true,
                y_pred,
                target_names=["incorrect", "correct"],
                digits=4,
            )
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Predict badminton clear (correct vs incorrect) with a custom "
            "threshold on P(correct)."
        )
    )
    parser.add_argument(
        "--features",
        type=str,
        default=str(DATA_PROCESSED_DIR / "features.csv"),
        help="Path to features.csv (default: data/processed/features.csv)",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=str(DATA_PROCESSED_DIR / "labels.csv"),
        help="Path to labels.csv (default: data/processed/labels.csv)",
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
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="If set, do not print the overall classification report.",
    )

    args = parser.parse_args()

    predict_with_threshold(
        features_path=Path(args.features),
        labels_path=Path(args.labels),
        model_path=Path(args.model),
        threshold=args.threshold,
        print_report=not args.no_report,
    )


if __name__ == "__main__":
    main()
