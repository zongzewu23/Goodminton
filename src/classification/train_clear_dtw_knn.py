# src/classification/train_clear_dtw_knn.py

from __future__ import annotations

from pathlib import Path
from typing import List

import joblib

from .dtw_knn_utils import SequenceSample, load_sequences_from_csv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


def train_dtw_knn_templates() -> None:
    """
    Build a template library for DTW + kNN classification
    from the main training features.csv and labels.csv.
    """
    features_path = DATA_PROCESSED_DIR / "features.csv"
    labels_path = DATA_PROCESSED_DIR / "labels.csv"

    print(f"[INFO] Loading training sequences from:\n  {features_path}\n  {labels_path}")
    templates: List[SequenceSample] = load_sequences_from_csv(features_path, labels_path)
    print(f"[INFO] Loaded {len(templates)} template sequences.")

    if not templates:
        raise RuntimeError("No templates were built from training data.")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "clear_templates_dtw_knn.joblib"

    # We simply store the list of SequenceSample objects.
    joblib.dump(templates, model_path)
    print(f"[INFO] Saved DTW+kNN templates to: {model_path}")


def main():
    train_dtw_knn_templates()


if __name__ == "__main__":
    main()
