from pathlib import Path
from typing import Literal, Dict, List, Tuple

import numpy as np
import pandas as pd
import joblib
import torch

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
from src.classification.train_clear_rnn import ClearRNNClassifier
from src.classification.dtw_knn_utils import TIME_DOWNSAMPLE, knn_predict_label_and_prob

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
    model_type: Literal["svm", "lr_pca", "rf", "rnn", "dtw_kbb"] = "svm",
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

    # Branch per model type
    if model_type in ("svm", "lr_pca", "rf"):
        # Feature-based classical models
        window_seq = find_main_motion_window(seq, window_frames=WINDOW_FRAMES)
        norm_seq = resample_sequence(window_seq, target_len=TARGET_FRAMES)
        feature_vec = sequence_to_feature_vector(norm_seq, n_segments=N_SEGMENTS)
        X = np.expand_dims(feature_vec, axis=0)

        # Resolve model file path with fallbacks
        if model_type == "svm":
            candidates = [MODELS_DIR / "clear_classifier_svm.joblib"]
        elif model_type == "lr_pca":
            candidates = [MODELS_DIR / "clear_classifier_lr_pca.joblib"]
        else:  # rf
            candidates = [MODELS_DIR / "rf.joblib", MODELS_DIR / "clear_classifier_rf.joblib"]

        model_file = next((p for p in candidates if p.exists()), None)
        if model_file is None:
            raise FileNotFoundError(f"Model file not found for {model_type}: tried {candidates}")

        model = joblib.load(model_file)
        if not hasattr(model, "predict_proba"):
            raise RuntimeError("Loaded model does not support predict_proba")

        proba = model.predict_proba(X)
        prob_correct = float(proba[0, 1])
        label = "correct" if prob_correct >= 0.4 else "incorrect"

        return {
            "label": label,
            "prob_correct": prob_correct,
            "threshold": 0.4,
            "model": model_type,
        }

    if model_type == "rnn":
        # Sequence-based RNN model
        # Prepare windowed sequence
        window_seq = find_main_motion_window(seq, window_frames=WINDOW_FRAMES)

        # Load torch model checkpoint and normalization stats
        candidates = [MODELS_DIR / "rnn.pt", MODELS_DIR / "clear_classifier_rnn.pt"]
        ckpt_path = next((p for p in candidates if p.exists()), None)
        if ckpt_path is None:
            raise FileNotFoundError(f"RNN model file not found: tried {candidates}")

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        input_mean = np.array(ckpt["input_mean"], dtype=np.float32)
        input_std = np.array(ckpt["input_std"], dtype=np.float32)
        target_len = int(ckpt["target_len"]) if "target_len" in ckpt else TARGET_FRAMES
        input_dim = int(ckpt.get("input_dim", window_seq.shape[1]))
        hidden_dim = int(ckpt.get("hidden_dim", 64))
        num_layers = int(ckpt.get("num_layers", 1))
        dropout = float(ckpt.get("dropout", 0.1))

        # Resample to target_len expected by the model
        norm_seq = resample_sequence(window_seq, target_len=target_len)  # (T, D)

        # Normalize per training stats
        if norm_seq.shape[1] != input_mean.shape[0]:
            # Pad or trim to match input_dim if needed
            D = input_mean.shape[0]
            if norm_seq.shape[1] < D:
                pad = np.zeros((norm_seq.shape[0], D - norm_seq.shape[1]), dtype=np.float32)
                norm_seq = np.concatenate([norm_seq, pad], axis=1)
            else:
                norm_seq = norm_seq[:, :D]

        X_seq = (norm_seq - input_mean.reshape(1, -1)) / input_std.reshape(1, -1)
        X_tensor = torch.from_numpy(X_seq.astype(np.float32)).unsqueeze(0)  # (1, T, D)

        # Build model and load weights
        model = ClearRNNClassifier(
            input_dim=input_mean.shape[0],
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        with torch.no_grad():
            logits = model(X_tensor)
            prob_correct = float(torch.sigmoid(logits).squeeze().item())

        label = "correct" if prob_correct >= 0.4 else "incorrect"
        return {
            "label": label,
            "prob_correct": prob_correct,
            "threshold": 0.4,
            "model": model_type,
        }

    if model_type == "dtw_kbb":
        # DTW + kNN using stored template sequences
        window_seq = find_main_motion_window(seq, window_frames=WINDOW_FRAMES)
        if TIME_DOWNSAMPLE > 1:
            window_seq = window_seq[::TIME_DOWNSAMPLE]

        candidates = [MODELS_DIR / "dtw_kbb.joblib", MODELS_DIR / "clear_templates_dtw_knn.joblib"]
        tmpl_path = next((p for p in candidates if p.exists()), None)
        if tmpl_path is None:
            raise FileNotFoundError(f"DTW-kNN templates not found: tried {candidates}")

        templates = joblib.load(tmpl_path)
        # Predict label and probability-like score
        pred_label, prob_correct = knn_predict_label_and_prob(window_seq, templates, k=3)
        label = "correct" if pred_label == 1 else "incorrect"

        return {
            "label": label,
            "prob_correct": float(prob_correct),
            "threshold": 0.4,
            "model": model_type,
        }

    # Unknown model type
    raise ValueError(f"Unsupported model_type: {model_type}")
