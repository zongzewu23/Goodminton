# src/classification/nn_clear_dataset.py

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np

from .dtw_knn_utils import SequenceSample, load_sequences_from_csv


def resample_sequence_to_fixed_length(
    seq: np.ndarray,
    target_len: int,
) -> np.ndarray:
    """
    Resample a sequence to a fixed temporal length using simple linear interpolation.

    Args:
        seq: np.ndarray of shape (T, D), where T is the original number of frames,
             D is the feature dimension (e.g., landmarks flattened).
        target_len: desired number of frames.

    Returns:
        np.ndarray of shape (target_len, D)
    """
    T, D = seq.shape
    if T == target_len:
        return seq.astype(np.float32, copy=True)

    # Original and new time indices
    old_idx = np.linspace(0.0, float(T - 1), num=T, dtype=np.float32)
    new_idx = np.linspace(0.0, float(T - 1), num=target_len, dtype=np.float32)

    out = np.empty((target_len, D), dtype=np.float32)
    for d in range(D):
        out[:, d] = np.interp(new_idx, old_idx, seq[:, d])

    return out


def load_nn_dataset_from_csv(
    features_path: Path,
    labels_path: Path,
    target_len: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load per-video sequences from features/labels CSV and resample
    each sequence to a fixed temporal length for NN training.

    Args:
        features_path: path to features CSV (with per-frame rows).
        labels_path: path to labels CSV.
        target_len: desired number of frames for every sequence.

    Returns:
        X: np.ndarray of shape (N, target_len, D)
        y: np.ndarray of shape (N,), with labels 0/1.
        video_names: list of length N with video_name strings.
    """
    samples: List[SequenceSample] = load_sequences_from_csv(features_path, labels_path)

    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    name_list: List[str] = []

    if not samples:
        raise RuntimeError("No SequenceSample loaded from CSV. Check your inputs.")

    for sample in samples:
        seq = sample.sequence  # (T, D)
        if seq.shape[0] < 2:
            # Skip degenerate sequences
            continue

        seq_resampled = resample_sequence_to_fixed_length(seq, target_len=target_len)
        X_list.append(seq_resampled)
        y_list.append(int(sample.label))
        name_list.append(sample.video_name)

    if not X_list:
        raise RuntimeError("All sequences were skipped as too short or invalid.")

    X = np.stack(X_list, axis=0).astype(np.float32)  # (N, T, D)
    y = np.array(y_list, dtype=np.float32)  # (N,)

    return X, y, name_list
