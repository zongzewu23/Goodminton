# src/classification/feature_pipeline.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd



# 只使用“身体 + 四肢”的关键点：肩、肘、腕、髋、膝、踝、脚跟、脚尖
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

# 使用哪些坐标分量：一般 x, y, z 就够了，visibility 暂时不用
USED_COORDS: List[str] = ["x", "y", "z"]

# 主动作窗口目标长度（帧数），假设已用 30 fps 提取
WINDOW_SECONDS: float = 1.5
ASSUMED_FPS: int = 30
WINDOW_FRAMES: int = int(WINDOW_SECONDS * ASSUMED_FPS)  # 约 45 帧

# 最终统一的帧数（时间归一化后）
TARGET_FRAMES: int = 30

# 把时间序列切成几段，保留“顺序信息”
N_SEGMENTS: int = 5


@dataclass
class VideoFeatureSample:
    video_name: str
    label: int  # 1 = correct, 0 = incorrect
    feature_vector: np.ndarray  # shape (num_features,)


# ---- 辅助函数：列名构造 / 安全检查 ----

def get_landmark_columns() -> List[str]:
    """根据 USED_LANDMARKS + USED_COORDS 构造列名列表。"""
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


# ---- 时间序列处理：运动能量 / 主窗口提取 / 插值 ----

def compute_motion_energy(seq: np.ndarray) -> np.ndarray:
    """
    计算相邻帧之间的“运动能量”（速度平方和），用于找动作最激烈的区间。
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
    在整段时间序列中，找到“运动能量”最大的窗口。
    seq: shape (T, D)
    return: sub_seq: shape (M, D), M <= T
    """
    T = seq.shape[0]
    if T <= window_frames:
        # 视频本身就很短，那就直接用全部
        return seq

    motion = compute_motion_energy(seq)  # (T-1,)

    # 使用滑动窗口在 motion 上求和，找能量最大的一段
    w = window_frames - 1  # 运动能量长度是 T-1
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
    把时间序列插值到固定长度 target_len。
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


# ---- 特征提取：分段统计 / 速度等 ----

def extract_segment_features(segment: np.ndarray) -> np.ndarray:
    """
    对单个时间段 segment 提取统计特征。
    segment: shape (L, D)
    return: shape (D * num_stats,)
    统计特征包括：
      - mean, max, min, std
      - delta_pos (最后一帧 - 第一帧)
      - mean_speed, max_speed
    共 7 个统计量。
    """
    L, D = segment.shape
    if L < 2:
        # 不足两帧，速度没法算，简单兜底
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

    # 按固定顺序拼接：方便模型学习“对应维度”的规律
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
    把统一长度的时间序列 seq (T, D) 压成一维特征向量。
    - 把时间轴均分成 n_segments 段
    - 每段提取统计特征
    - 按段顺序拼接
    """
    T, D = seq.shape
    if T < n_segments:
        # 极端情况：帧数比段数还少，直接扩展重复最后一帧
        repeats = n_segments - T
        pad = np.repeat(seq[-1:, :], repeats, axis=0)
        seq = np.concatenate([seq, pad], axis=0)
        T = seq.shape[0]

    base_len = T // n_segments
    feature_list: List[np.ndarray] = []

    for i in range(n_segments):
        start = i * base_len
        # 最后一段吃掉所有剩余帧，避免尾巴丢帧
        if i == n_segments - 1:
            end = T
        else:
            end = (i + 1) * base_len

        seg = seq[start:end]
        seg_feat = extract_segment_features(seg)  # (D * num_stats,)
        feature_list.append(seg_feat)

    full_feature = np.concatenate(feature_list, axis=0)  # (n_segments * D * num_stats,)
    return full_feature.astype(np.float32)


# ---- 主入口：从 features.csv / labels.csv 构造 X, y ----

def load_features_and_labels(
    features_path: Path,
    labels_path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """读取 CSV 并做基本检查。"""
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
    """把 'correct' / 'incorrect' 映射为 1 / 0。"""
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
    核心函数：按视频聚合，构造每个视频的 feature vector。
    """
    lm_cols = get_landmark_columns()

    # 建一个 video_name -> label_int 的映射
    label_map: Dict[str, int] = {
        row["video_name"]: label_str_to_int(row["label"])
        for _, row in labels_df.iterrows()
    }

    samples: List[VideoFeatureSample] = []

    # 按 video_name 分组
    grouped = features_df.groupby("video_name", sort=False)

    for video_name, df_group in grouped:
        if video_name not in label_map:
            # 没有标签的样本（比如 test 视频），先跳过
            continue

        df_group = df_group.sort_values("frame_idx")
        seq = df_group[lm_cols].to_numpy(dtype=np.float32)  # shape (T, D)

        if seq.shape[0] == 0:
            print(f"[WARN] Video {video_name} has no frames after filtering, skip.")
            continue

        # 1) 找主动作窗口
        window_seq = find_main_motion_window(seq, window_frames=WINDOW_FRAMES)

        # 2) 插值归一化到固定帧数
        norm_seq = resample_sequence(window_seq, target_len=TARGET_FRAMES)

        # 3) 分段提特征
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
    一步到位：
      - 读 CSV
      - 按视频聚合并提特征
      - 输出 X, y, video_names
    """
    features_df, labels_df = load_features_and_labels(features_path, labels_path)
    samples = build_video_feature_samples(features_df, labels_df)

    if not samples:
        raise RuntimeError("No video samples were built from the given CSV files.")

    X = np.stack([s.feature_vector for s in samples], axis=0)
    y = np.array([s.label for s in samples], dtype=np.int64)
    names = [s.video_name for s in samples]

    return X, y, names
