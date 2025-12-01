# src/classification/evaluate_clear_dtw_knn_on_test.py

from __future__ import annotations

from pathlib import Path
from typing import List

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from ..classification.dtw_knn_utils import (
    SequenceSample,
    load_sequences_from_csv,
    knn_predict_label_and_prob,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


def evaluate_dtw_knn_on_test(
    k: int = 3,
) -> None:
    """
    Evaluate DTW + kNN classification on a dedicated TEST set.

    Assumes:
      - Training templates saved at models/clear_templates_dtw_knn.joblib
      - Test CSVs at data/processed/clear/test/features_clear_test.csv
                              data/processed/clear/test/labels_clear_test.csv
    """
    templates_path = MODELS_DIR / "clear_templates_dtw_knn.joblib"
    if not templates_path.exists():
        raise FileNotFoundError(
            f"Template file not found: {templates_path}\n"
            f"Please run train_clear_dtw_knn.py first."
        )

    templates: List[SequenceSample] = joblib.load(templates_path)
    print(f"[INFO] Loaded {len(templates)} DTW templates from: {templates_path}")

    features_test_path = DATA_PROCESSED_DIR / "clear" / "test" / "features_clear_test.csv"
    labels_test_path = DATA_PROCESSED_DIR / "clear" / "test" / "labels_clear_test.csv"

    print(f"[INFO] Loading TEST sequences from:\n  {features_test_path}\n  {labels_test_path}")
    test_samples: List[SequenceSample] = load_sequences_from_csv(features_test_path, labels_test_path)
    print(f"[INFO] Loaded {len(test_samples)} test sequences.")

    if not test_samples:
        raise RuntimeError("No test sequences loaded. Check your test CSV files.")

    y_true = []
    y_pred = []
    probs = []
    names = []

    for sample in test_samples:
        pred_label, prob_correct = knn_predict_label_and_prob(
            sample.sequence,
            templates,
            k=k,
        )
        y_true.append(sample.label)
        y_pred.append(pred_label)
        probs.append(prob_correct)
        names.append(sample.video_name)

    y_true_arr = np.array(y_true, dtype=int)
    y_pred_arr = np.array(y_pred, dtype=int)
    probs_arr = np.array(probs, dtype=float)

    print("\n[INFO] Classification report (DTW + kNN):")
    print(
        classification_report(
            y_true_arr,
            y_pred_arr,
            target_names=["incorrect", "correct"],
            digits=4,
        )
    )

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    print("[INFO] Confusion matrix (rows=true, cols=pred) [incorrect=0, correct=1]:")
    print(cm)

    print("\nPer-video test predictions (DTW + kNN):")
    print("-" * 90)
    print(f"{'video_name':30s} {'true':10s} {'pred':10s} {'P(correct)':>12s}")
    print("-" * 90)
    for name, yt, yp, p in zip(names, y_true_arr, y_pred_arr, probs_arr):
        yt_str = "correct" if yt == 1 else "incorrect"
        yp_str = "correct" if yp == 1 else "incorrect"
        print(f"{name:30s} {yt_str:10s} {yp_str:10s} {p:12.4f}")
    print("-" * 90)


def main():
    # You can adjust k here if you want to experiment
    evaluate_dtw_knn_on_test(k=3)


if __name__ == "__main__":
    main()
