from pathlib import Path
from typing import Literal, Dict, List

import numpy as np
import pandas as pd
import joblib

from src.classification.feature_pipeline import (
    find_main_motion_window,
    resample_sequence,
    sequence_to_feature_vector,
    get_landmark_columns,
    WINDOW_FRAMES,
    TARGET_FRAMES,
    N_SEGMENTS,
)
from src.pose_estimation.extract_features import (
    extract_landmarks_from_video,
    normalize_landmarks,
    landmarks_to_dataframe,
)

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"


def _build_sequence_from_video(video_path: Path) -> np.ndarray:
    """
    Extract pose landmarks from the video and build the (T, D) sequence
    using the same normalization as training.
    """
    # 1) Pose extraction at ~30 fps
    raw_landmarks = extract_landmarks_from_video(str(video_path), target_fps=30)

    # 2) Normalize per-frame landmarks relative to hip center
    normalized_landmarks: List = [normalize_landmarks(frame_lm) for frame_lm in raw_landmarks]

    # 3) Convert to DataFrame compatible with training pipeline columns
    video_name = Path(video_path).stem
    df = landmarks_to_dataframe(video_name, normalized_landmarks)

    # If no valid frames were detected, return an empty sequence
    if df.empty:
        return np.empty((0, 0), dtype=np.float32)

    # Select only the landmark x/y/z columns used in training
    lm_cols = get_landmark_columns()

    # Gracefully handle any missing columns by filling with zeros
    missing = [c for c in lm_cols if c not in df.columns]
    for c in missing:
        df[c] = 0.0

    seq = df[lm_cols].to_numpy(dtype=np.float32)  # shape (T, D)
    return seq


def run_inference_on_video(
    video_path: Path,
    model_type: Literal["svm", "lr_pca"] = "svm",
) -> Dict:
    """
    Run the full inference pipeline on the uploaded video.

    Steps:
    1. Extract pose landmarks and normalize per frame
    2. Build sequence (T, D)
    3. Apply: find_main_motion_window → resample_sequence → sequence_to_feature_vector
    4. Load model: models/clear_classifier_{svm|lr_pca}.joblib
    5. Predict probability for class 'correct'
    6. Threshold at 0.4 to produce final label
    """
    # Build raw sequence from video
    seq = _build_sequence_from_video(video_path)

    # Handle empty sequences: return a deterministic negative result
    if seq.size == 0:
        return {
            "label": "incorrect",
            "prob_correct": float(0.0),
            "threshold": 0.4,
            "model": model_type,
        }

    # Main motion window → resample → feature vector
    window_seq = find_main_motion_window(seq, window_frames=WINDOW_FRAMES)
    norm_seq = resample_sequence(window_seq, target_len=TARGET_FRAMES)
    feature_vec = sequence_to_feature_vector(norm_seq, n_segments=N_SEGMENTS)
    X = np.expand_dims(feature_vec, axis=0)

    # Load model
    model_file = (
        MODELS_DIR / "clear_classifier_svm.joblib"
        if model_type == "svm"
        else MODELS_DIR / "clear_classifier_lr_pca.joblib"
    )
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    model = joblib.load(model_file)
    if not hasattr(model, "predict_proba"):
        raise RuntimeError("Loaded model does not support predict_proba")

    proba = model.predict_proba(X)
    prob_correct = float(proba[0, 1])

    # Threshold at 0.4
    label = "correct" if prob_correct >= 0.4 else "incorrect"

    return {
        "label": label,
        "prob_correct": prob_correct,
        "threshold": 0.4,
        "model": model_type,
    }