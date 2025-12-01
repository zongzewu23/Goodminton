# src/classification/train_clear_rnn.py

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .nn_clear_dataset import load_nn_dataset_from_csv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


class ClearRNNClassifier(nn.Module):
    """
    A simple bidirectional GRU-based classifier for badminton clear strokes.

    Input:  sequence of shape (batch_size, T, D)
    Output: logit of shape (batch_size, 1) for class "correct" (1) vs "incorrect" (0).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # Bidirectional => hidden_dim * 2
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: float tensor of shape (batch_size, T, D)

        Returns:
            logits: float tensor of shape (batch_size, 1)
        """
        gru_out, _ = self.gru(x)  # (batch, T, 2*hidden_dim)
        # Use global max pooling over time
        pooled, _ = torch.max(gru_out, dim=1)  # (batch, 2*hidden_dim)
        logits = self.fc(pooled)  # (batch, 1)
        return logits


def compute_normalization_stats(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute feature-wise mean and std over (N, T, D).

    Args:
        X: np.ndarray of shape (N, T, D)

    Returns:
        mean: np.ndarray of shape (D,)
        std: np.ndarray of shape (D,)
    """
    # Merge N and T dims
    N, T, D = X.shape
    flat = X.reshape(N * T, D)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0) + 1e-8
    return mean.astype(np.float32), std.astype(np.float32)


def main() -> None:
    # Hyperparameters for the NN model
    TARGET_LEN = 32       # number of frames per sequence after resampling
    HIDDEN_DIM = 64
    NUM_LAYERS = 1
    DROPOUT = 0.1
    NUM_EPOCHS = 200
    BATCH_SIZE = 16
    LR = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    features_path = DATA_PROCESSED_DIR / "features.csv"
    labels_path = DATA_PROCESSED_DIR / "labels.csv"

    print(f"[INFO] Loading NN dataset from:\n  {features_path}\n  {labels_path}")
    X_np, y_np, names = load_nn_dataset_from_csv(
        features_path=features_path,
        labels_path=labels_path,
        target_len=TARGET_LEN,
    )
    print(f"[INFO] Dataset shape: X={X_np.shape}, y={y_np.shape}")
    N, T, D = X_np.shape

    # Compute normalization
    input_mean, input_std = compute_normalization_stats(X_np)
    print(f"[INFO] Input dim: {D}, target_len: {T}")
    print("[INFO] Normalization stats computed.")

    # Apply normalization
    X_norm = (X_np - input_mean.reshape(1, 1, D)) / input_std.reshape(1, 1, D)

    # Build TensorDataset / DataLoader
    X_tensor = torch.from_numpy(X_norm)  # (N, T, D)
    y_tensor = torch.from_numpy(y_np).unsqueeze(1)  # (N, 1)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    # Build model
    model = ClearRNNClassifier(
        input_dim=D,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Training loop
    model.train()
    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_loss = 0.0
        correct = 0
        total = 0

        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_X)  # (batch, 1)
            loss = criterion(logits, batch_y)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_X.size(0)

            with torch.no_grad():
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)

        avg_loss = epoch_loss / float(total)
        acc = correct / float(total)
        if epoch % 20 == 0 or epoch == 1:
            print(f"[INFO] Epoch {epoch:03d}/{NUM_EPOCHS} - loss={avg_loss:.4f} acc={acc:.4f}")

    # Save model + normalization stats for later inference
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = MODELS_DIR / "clear_classifier_rnn.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_mean": input_mean,
            "input_std": input_std,
            "target_len": TARGET_LEN,
            "input_dim": D,
            "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
        },
        save_path,
    )

    print(f"[INFO] Saved RNN model and normalization stats to: {save_path}")


if __name__ == "__main__":
    main()
