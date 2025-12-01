# src/classification/dtw_knn_utils.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# -------------------------------
# Configuration for DTW + kNN
# -------------------------------

# Landmarks and coordinates to use (body + limbs, no face)
USED_LANDMARKS: List[str] = [
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

USED_COORDS: List[str] = ["x", "y", "z"]

# Approximate motion window length (in seconds) and fps
WINDOW_SECONDS: float = 1.5
ASSUMED_FPS: int = 30
WINDOW_FRAMES: int = int(WINDOW_SECONDS * ASSUMED_FPS)  # e.g. 45 frames

# To reduce DTW cost, we can downsample in time (e.g. keep every 2nd frame)
TIME_DOWNSAMPLE: int = 2  # 1 = no downsampling, 2 = keep every other frame


@dataclass
class SequenceSample:
    video_name: str
    label: int  # 1 = correct, 0 = incorrect
    sequence: np.ndarray  # shape (T, D)


# -------------------------------
# Utility: columns / labels
# -------------------------------

def get_landmark_columns() -> List[str]:
    """Construct column names based on USED_LANDMARKS and USED_COORDS."""
    cols: List[str] = []
    for lm in USED_LANDMARKS:
        for coord in USED_COORDS:
            cols.append(f"{lm}_{coord}")
    return cols


def label_str_to_int(label_str: str) -> int:
    """Map 'correct' / 'incorrect' to 1 / 0."""
    s = str(label_str).strip().lower()
    if s == "correct":
        return 1
    if s == "incorrect":
        return 0
    raise ValueError(f"Unknown label string: {label_str}")


# -------------------------------
# Motion window selection
# -------------------------------

def compute_motion_energy(seq: np.ndarray) -> np.ndarray:
    """
    Compute frame-to-frame motion energy for a sequence.
    seq: shape (T, D)
    returns: shape (T-1,)
    """
    T, D = seq.shape
    if T < 2:
        return np.zeros(1, dtype=np.float32)
    vel = np.diff(seq, axis=0)  # (T-1, D)
    motion = np.sum(vel ** 2, axis=1)  # (T-1,)
    return motion.astype(np.float32)


def find_main_motion_window(seq: np.ndarray, window_frames: int = WINDOW_FRAMES) -> np.ndarray:
    """
    Find the sub-sequence with the highest accumulated motion energy.
    Returns the sub-sequence as an approximation of the main stroke.
    """
    T, D = seq.shape
    if T <= window_frames:
        return seq

    motion = compute_motion_energy(seq)  # (T-1,)
    w = window_frames - 1
    if w <= 0 or w > motion.shape[0]:
        return seq

    cumsum = np.cumsum(motion)
    window_sums = cumsum[w - 1:] - np.concatenate(([0.0], cumsum[:-w]))
    best_start = int(np.argmax(window_sums))
    best_end = min(best_start + window_frames, T)

    return seq[best_start:best_end]


# -------------------------------
# Loading sequences from CSV
# -------------------------------

def load_sequences_from_csv(
    features_path: Path,
    labels_path: Path,
) -> List[SequenceSample]:
    """
    Load per-video sequences from features.csv / labels.csv.

    The features.csv is expected to have:
      - video_name
      - frame_idx
      - landmark columns matching get_landmark_columns()

    The labels.csv is expected to have:
      - video_name
      - label ('correct' / 'incorrect')
    """
    features_df = pd.read_csv(features_path)
    labels_df = pd.read_csv(labels_path)

    if "video_name" not in features_df.columns or "frame_idx" not in features_df.columns:
        raise ValueError("features.csv must contain 'video_name' and 'frame_idx' columns.")

    if "video_name" not in labels_df.columns or "label" not in labels_df.columns:
        raise ValueError("labels.csv must contain 'video_name' and 'label' columns.")

    lm_cols = get_landmark_columns()
    missing = [c for c in lm_cols if c not in features_df.columns]
    if missing:
        raise ValueError(
            f"Missing expected landmark columns in features.csv: {missing[:10]} "
            f"(total missing: {len(missing)})"
        )

    label_map: Dict[str, int] = {
        row["video_name"]: label_str_to_int(row["label"])
        for _, row in labels_df.iterrows()
    }

    samples: List[SequenceSample] = []

    grouped = features_df.groupby("video_name", sort=False)
    for video_name, df_group in grouped:
        if video_name not in label_map:
            # Skip videos that do not have labels (e.g. pure test without labels)
            continue

        df_group = df_group.sort_values("frame_idx")
        seq = df_group[lm_cols].to_numpy(dtype=np.float32)  # (T, D)
        if seq.shape[0] == 0:
            print(f"[WARN] Video {video_name} has 0 frames in features.csv, skipping.")
            continue

        # Extract main motion window
        window_seq = find_main_motion_window(seq, window_frames=WINDOW_FRAMES)

        # Optional time downsampling to reduce DTW cost
        if TIME_DOWNSAMPLE > 1:
            window_seq = window_seq[::TIME_DOWNSAMPLE]

        if window_seq.shape[0] < 3:
            # Too short to be useful for DTW
            print(f"[WARN] Video {video_name} has too few frames after windowing, skipping.")
            continue

        samples.append(
            SequenceSample(
                video_name=video_name,
                label=label_map[video_name],
                sequence=window_seq,
            )
        )

    return samples


# -------------------------------
# DTW implementation
# -------------------------------

def dtw_distance(seq_a: np.ndarray, seq_b: np.ndarray) -> float:
    """
    Compute DTW distance between two sequences seq_a and seq_b.

    seq_a: shape (T1, D)
    seq_b: shape (T2, D)

    The local cost is Euclidean distance between frames.
    The global cost is the minimal path sum.
    """
    T1, D1 = seq_a.shape
    T2, D2 = seq_b.shape
    if D1 != D2:
        raise ValueError(f"Dimension mismatch in DTW: {D1} vs {D2}")

    # Cost matrix
    cost = np.zeros((T1, T2), dtype=np.float32)

    # Initialize (0,0)
    cost[0, 0] = np.linalg.norm(seq_a[0] - seq_b[0])

    # First column
    for i in range(1, T1):
        cost[i, 0] = cost[i - 1, 0] + np.linalg.norm(seq_a[i] - seq_b[0])

    # First row
    for j in range(1, T2):
        cost[0, j] = cost[0, j - 1] + np.linalg.norm(seq_a[0] - seq_b[j])

    # Fill the rest
    for i in range(1, T1):
        for j in range(1, T2):
            d = np.linalg.norm(seq_a[i] - seq_b[j])
            cost[i, j] = d + min(
                cost[i - 1, j],     # insertion
                cost[i, j - 1],     # deletion
                cost[i - 1, j - 1]  # match
            )

    return float(cost[T1 - 1, T2 - 1])


# -------------------------------
# kNN using DTW
# -------------------------------

def knn_predict_label_and_prob(
    query_seq: np.ndarray,
    templates: List[SequenceSample],
    k: int = 3,
) -> Tuple[int, float]:
    """
    Predict label (0 or 1) AND a probability-like score for 'correct'
    using DTW-based k-NN over template samples.

    The probability is defined as:
        prob_correct = (# of neighbors with label==1) / k

    query_seq: shape (T_q, D)
    templates: list of SequenceSample (video_name, label, sequence)
    k: number of nearest neighbors

    Returns:
        predicted_label (int): 0 for incorrect, 1 for correct
        prob_correct (float): in [0, 1]
    """
    if not templates:
        raise RuntimeError("No templates provided for kNN prediction.")

    distances: List[Tuple[float, int]] = []  # (distance, label)

    for sample in templates:
        d = dtw_distance(query_seq, sample.sequence)
        distances.append((d, sample.label))

    # Sort by distance
    distances.sort(key=lambda x: x[0])

    # Take top-k
    k = max(1, min(k, len(distances)))
    top_k = distances[:k]

    # Collect votes
    votes = [label for _, label in top_k]
    prob_correct = sum(votes) / float(k)

    # Majority vote for final label
    predicted_label = 1 if prob_correct >= 0.5 else 0

    return predicted_label, float(prob_correct)


def knn_predict_label(
    query_seq: np.ndarray,
    templates: List[SequenceSample],
    k: int = 3,
) -> int:
    """
    Backward-compatible wrapper that returns only the predicted label.
    Internally calls knn_predict_label_and_prob and discards the probability.
    """
    label, _ = knn_predict_label_and_prob(query_seq, templates, k=k)
    return label
