# src/classification/evaluate_clear_rnn_on_test.py

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

from ..classification.nn_clear_dataset import load_nn_dataset_from_csv
from ..classification.train_clear_rnn import ClearRNNClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load saved model + normalization stats
    model_path = MODELS_DIR / "clear_classifier_rnn.pt"
    if not model_path.exists():
        raise FileNotFoundError(
            f"RNN model file not found at {model_path}. "
            f"Please run train_clear_rnn.py first."
        )

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    input_mean = checkpoint["input_mean"]  # shape (D,)
    input_std = checkpoint["input_std"]    # shape (D,)
    target_len = int(checkpoint["target_len"])
    input_dim = int(checkpoint["input_dim"])
    hidden_dim = int(checkpoint["hidden_dim"])
    num_layers = int(checkpoint["num_layers"])
    dropout = float(checkpoint["dropout"])

    print(f"[INFO] Loaded model checkpoint from: {model_path}")
    print(f"[INFO] target_len={target_len}, input_dim={input_dim}")

    # Rebuild model
    model = ClearRNNClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load TEST dataset
    features_test_path = DATA_PROCESSED_DIR / "clear" / "test" / "features_clear_test.csv"
    labels_test_path = DATA_PROCESSED_DIR / "clear" / "test" / "labels_clear_test.csv"

    print(f"[INFO] Loading TEST dataset from:\n  {features_test_path}\n  {labels_test_path}")
    X_test_np, y_test_np, names = load_nn_dataset_from_csv(
        features_path=features_test_path,
        labels_path=labels_test_path,
        target_len=target_len,
    )
    print(f"[INFO] Test dataset shape: X={X_test_np.shape}, y={y_test_np.shape}")

    N, T, D = X_test_np.shape
    if D != input_dim:
        raise ValueError(
            f"Input dimension mismatch: test D={D}, model input_dim={input_dim}"
        )

    # Apply same normalization as training
    X_test_norm = (X_test_np - input_mean.reshape(1, 1, D)) / input_std.reshape(1, 1, D)

    # Convert to tensors
    X_test_tensor = torch.from_numpy(X_test_norm).to(device)
    y_test_tensor = torch.from_numpy(y_test_np).to(device)

    # Inference
    with torch.no_grad():
        logits = model(X_test_tensor)  # (N, 1)
        probs = torch.sigmoid(logits).squeeze(1)  # (N,)

    preds = (probs >= 0.5).float()  # threshold at 0.5 for now

    y_true = y_test_tensor.cpu().numpy().astype(int)
    y_pred = preds.cpu().numpy().astype(int)
    prob_correct = probs.cpu().numpy().astype(float)

    print("\n[INFO] Classification report (RNN):")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["incorrect", "correct"],
            digits=4,
        )
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print("[INFO] Confusion matrix (rows=true, cols=pred) [incorrect=0, correct=1]:")
    print(cm)

    print("\nPer-video test predictions (RNN):")
    print("-" * 90)
    print(f"{'video_name':30s} {'true':10s} {'pred':10s} {'P(correct)':>12s}")
    print("-" * 90)
    for name, yt, yp, p in zip(names, y_true, y_pred, prob_correct):
        yt_str = "correct" if yt == 1 else "incorrect"
        yp_str = "correct" if yp == 1 else "incorrect"
        print(f"{name:30s} {yt_str:10s} {yp_str:10s} {p:12.4f}")
    print("-" * 90)


if __name__ == "__main__":
    main()
