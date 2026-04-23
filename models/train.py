"""
Day 2: Train both neural models on the training set.
Saves trained weights to models/checkpoints/
"""

import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm

from structured_predictor import CNNBaseline, StructuredPredictor


# ── DATASET LOADER ────────────────────────────────────────────────
class BallDataset(Dataset):
    """
    Loads episodes from a .pkl file.
    Returns (stacked_frames, landing_x) pairs.

    Each episode contains:
      - input_frames: (4, 128, 128, 3)
      - gt_landing_x: float
      - init_state:   dict
      - restitution:  float
    """
    def __init__(self, pkl_path):
        with open(pkl_path, "rb") as f:
            self.episodes = pickle.load(f)

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        ep = self.episodes[idx]
        frames = ep["input_frames"]                         # (4, 128, 128, 3)
        frames = frames.astype(np.float32) / 255.0
        # reshape to (12, 128, 128) by stacking frames along channel dim
        frames = np.transpose(frames, (0, 3, 1, 2))         # (4, 3, 128, 128)
        frames = frames.reshape(-1, frames.shape[2], frames.shape[3])  # (12, 128, 128)
        landing_x = np.float32(ep["gt_landing_x"])
        return torch.from_numpy(frames), torch.tensor(landing_x)


# ── TRAIN ONE MODEL ───────────────────────────────────────────────
def train_model(model, loader, epochs=10, lr=1e-3, name="model"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    print(f"\n=== Training {name} on {device} ===")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        count = 0
        for frames, targets in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            frames, targets = frames.to(device), targets.to(device)
            preds = model(frames)
            loss = loss_fn(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * frames.size(0)
            count += frames.size(0)
        avg_loss = total_loss / count
        print(f"Epoch {epoch+1}: MSE = {avg_loss:.4f}")

    ckpt_dir = Path("models/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{name}.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")
    return model


# ── MAIN ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_path = "data/dataset/train.pkl"
    print(f"Loading dataset from {train_path}...")
    dataset = BallDataset(train_path)
    print(f"Loaded {len(dataset)} training episodes.")

    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

    # Model A — CNN Baseline
    model_a = CNNBaseline()
    train_model(model_a, loader, epochs=10, lr=1e-3, name="cnn_baseline")

    # Model B — Structured Predictor
    model_b = StructuredPredictor()
    train_model(model_b, loader, epochs=10, lr=1e-3, name="structured_predictor")

    print("\nBoth models trained and saved.")