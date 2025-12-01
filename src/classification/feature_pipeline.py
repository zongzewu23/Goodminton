# src/classification/feature_pipeline.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


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

WINDOW_SECONDS: float = 1.5
ASSUMED_FPS: int = 30
WINDOW_FRAMES: int = int(WINDOW_SECONDS * ASSUMED_FPS)  # around 45 frames

TARGET_FRAMES: int = 30

N_SEGMENTS: int = 5


@dataclass
class VideoFeatureSample:
    video_name: str
    label: int  # 1 = correct, 0 = incorrect
    feature_vector: np.ndarray  # shape (num_features,)


def get_landmark_columns() -> List[str]:
    """Build the column names from USED_LANDMARKS and USED_COORDS."""
    cols = []
    for lm in USED_LANDMARKS:
        for coord in USED_COORDS:
            cols.append(f"{lm}_{coord}")
    return cols


def check_columns_exist(df: pd.DataFrame, columns: List[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing expected columns in features.csv: {missing[:10]} "
            f"(total missing: {len(missing)})"
        )


def compute_motion_energy(seq: np.ndarray) -> np.ndarray:
    """
    Compute motion energy (sum of squared velocities) between consecutive frames to locate the most intense interval.
    seq: shape (T, D)
    return: shape (T-1,)
    """
    if seq.shape[0] < 2:
        return np.zeros(1, dtype=np.float32)
    vel = np.diff(seq, axis=0)  # (T-1, D)
    motion = np.sum(vel ** 2, axis=1)  # (T-1,)
    return motion.astype(np.float32)


def find_main_motion_window(
    seq: np.ndarray,
    window_frames: int = WINDOW_FRAMES,
) -> np.ndarray:
    """
    Find the window with the maximum motion energy within the whole sequence.
    seq: shape (T, D)
    return: sub_seq: shape (M, D), M <= T
    """
    T = seq.shape[0]
    if T <= window_frames:
        # If the video is short, use the whole sequence
        return seq

    motion = compute_motion_energy(seq)  # (T-1,)

    # Use a sliding window to sum motion energy and find the highest-energy segment
    w = window_frames - 1  # motion energy length is T-1
    if w <= 0 or w > motion.shape[0]:
        return seq

    # cumulative sum trick
    cumsum = np.cumsum(motion)
    window_sums = cumsum[w - 1:] - np.concatenate(([0.0], cumsum[:-w]))
    best_start = int(np.argmax(window_sums))
    best_end = best_start + window_frames
    best_end = min(best_end, T)

    return seq[best_start:best_end]


def resample_sequence(
    seq: np.ndarray,
    target_len: int = TARGET_FRAMES,
) -> np.ndarray:
    """
    Interpolate the time series to a fixed length target_len.
    seq: shape (T, D)
    return: shape (target_len, D)
    """
    T, D = seq.shape
    if T == 0:
        raise ValueError("Empty sequence encountered during resampling.")
    if T == target_len:
        return seq.astype(np.float32)

    old_idx = np.arange(T, dtype=np.float32)
    new_idx = np.linspace(0, T - 1, target_len, dtype=np.float32)

    out = np.empty((target_len, D), dtype=np.float32)
    for d in range(D):
        out[:, d] = np.interp(new_idx, old_idx, seq[:, d])
    return out


def extract_segment_features(segment: np.ndarray) -> np.ndarray:
    """
    Extract statistical features for a single time segment.
    segment: shape (L, D)
    return: shape (D * num_stats,)
    Statistics include:
      - mean, max, min, std
      - delta_pos (last frame - first frame)
      - mean_speed, max_speed
    Total 7 statistics.
    """
    L, D = segment.shape
    if L < 2:
        # Fewer than 2 frames: cannot compute speed; simple fallback
        mean = segment.mean(axis=0)
        return np.concatenate([mean, mean, mean, mean, np.zeros_like(mean), np.zeros_like(mean), np.zeros_like(mean)])

    pos_mean = segment.mean(axis=0)         # (D,)
    pos_max = segment.max(axis=0)          # (D,)
    pos_min = segment.min(axis=0)          # (D,)
    pos_std = segment.std(axis=0)          # (D,)

    delta_pos = segment[-1] - segment[0]   # (D,)

    vel = np.diff(segment, axis=0)         # (L-1, D)
    speed = np.abs(vel)                    # (L-1, D)
    mean_speed = speed.mean(axis=0)        # (D,)
    max_speed = speed.max(axis=0)          # (D,)

    # Concatenate in a fixed order to help the model learn per-dimension patterns
    stats = np.concatenate(
        [pos_mean, pos_max, pos_min, pos_std, delta_pos, mean_speed, max_speed],
        axis=0
    )
    return stats.astype(np.float32)


def sequence_to_feature_vector(
    seq: np.ndarray,
    n_segments: int = N_SEGMENTS,
) -> np.ndarray:
    """
    Flatten the normalized sequence seq (T, D) into a 1D feature vector.
    - Evenly split the time axis into n_segments
    - Extract statistics for each segment
    - Concatenate in segment order
    """
    T, D = seq.shape
    if T < n_segments:
        # Edge case: frames fewer than segments; pad by repeating the last frame
        repeats = n_segments - T
        pad = np.repeat(seq[-1:, :], repeats, axis=0)
        seq = np.concatenate([seq, pad], axis=0)
        T = seq.shape[0]

    base_len = T // n_segments
    feature_list: List[np.ndarray] = []

    for i in range(n_segments):
        start = i * base_len
        # Last segment consumes all remaining frames to avoid dropping tail frames
        if i == n_segments - 1:
            end = T
        else:
            end = (i + 1) * base_len

        seg = seq[start:end]
        seg_feat = extract_segment_features(seg)  # (D * num_stats,)
        feature_list.append(seg_feat)

    full_feature = np.concatenate(feature_list, axis=0)  # (n_segments * D * num_stats,)
    return full_feature.astype(np.float32)


def load_features_and_labels(
    features_path: Path,
    labels_path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Read CSV files and perform basic checks."""
    features_df = pd.read_csv(features_path)
    labels_df = pd.read_csv(labels_path)

    expected_cols = ["video_name", "frame_idx"]
    check_columns_exist(features_df, expected_cols)

    lm_cols = get_landmark_columns()
    check_columns_exist(features_df, lm_cols)

    if "video_name" not in labels_df.columns or "label" not in labels_df.columns:
        raise ValueError("labels.csv must contain 'video_name' and 'label' columns.")

    return features_df, labels_df


def label_str_to_int(label_str: str) -> int:
    """Map 'correct' / 'incorrect' to 1 / 0."""
    label_str = str(label_str).strip().lower()
    if label_str == "correct":
        return 1
    elif label_str == "incorrect":
        return 0
    else:
        raise ValueError(f"Unknown label string: {label_str}")


def build_video_feature_samples(
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
) -> List[VideoFeatureSample]:
    """
    Core function: group by video and construct a feature vector per video.
    """
    lm_cols = get_landmark_columns()

    # Build a mapping: video_name -> label_int
    label_map: Dict[str, int] = {
        row["video_name"]: label_str_to_int(row["label"])
        for _, row in labels_df.iterrows()
    }

    samples: List[VideoFeatureSample] = []

    # Group by video_name
    grouped = features_df.groupby("video_name", sort=False)

    for video_name, df_group in grouped:
        if video_name not in label_map:
            # Skip samples without labels (e.g., test videos)
            continue

        df_group = df_group.sort_values("frame_idx")
        seq = df_group[lm_cols].to_numpy(dtype=np.float32)  # shape (T, D)

        if seq.shape[0] == 0:
            print(f"[WARN] Video {video_name} has no frames after filtering, skip.")
            continue

        # 1) Find main motion window
        window_seq = find_main_motion_window(seq, window_frames=WINDOW_FRAMES)

        # 2) Resample to a fixed frame count
        norm_seq = resample_sequence(window_seq, target_len=TARGET_FRAMES)

        # 3) Extract segment features
        feature_vec = sequence_to_feature_vector(norm_seq, n_segments=N_SEGMENTS)

        samples.append(
            VideoFeatureSample(
                video_name=video_name,
                label=label_map[video_name],
                feature_vector=feature_vec,
            )
        )

    return samples


def build_dataset_from_csv(
    features_path: Path,
    labels_path: Path,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    All-in-one:
      - Read CSV
      - Group by video and extract features
      - Output X, y, video_names
    """

    features_df, labels_df = load_features_and_labels(features_path, labels_path)
    samples = build_video_feature_samples(features_df, labels_df)

    if not samples:
        raise RuntimeError("No video samples were built from the given CSV files.")

    X = np.stack([s.feature_vector for s in samples], axis=0)
    y = np.array([s.label for s in samples], dtype=np.int64)
    names = [s.video_name for s in samples]

    return X, y, names
