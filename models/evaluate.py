"""
File Evaluate both trained models on all test splits.
Compares Model A (CNN Baseline) vs Model B (Structured Predictor)
under base, appearance, noise, and dynamics distribution shifts.
"""

import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from structured_predictor import CNNBaseline, StructuredPredictor
from train import BallDataset


def evaluate_model(model, loader, device):
    """Return MAE over the entire loader."""
    model.eval()
    total_abs_err = 0.0
    count = 0
    with torch.no_grad():
        for frames, targets in loader:
            frames, targets = frames.to(device), targets.to(device)
            preds = model(frames)
            total_abs_err += torch.abs(preds - targets).sum().item()
            count += frames.size(0)
    return total_abs_err / count


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # load trained models
    model_a = CNNBaseline().to(device)
    model_a.load_state_dict(torch.load("models/checkpoints/cnn_baseline.pt",
                                       map_location=device))

    model_b = StructuredPredictor().to(device)
    model_b.load_state_dict(torch.load("models/checkpoints/structured_predictor.pt",
                                       map_location=device))

    # evaluate on each test split
    splits = ["test_base", "test_appearance", "test_noise", "test_dynamics"]
    results = {}

    print(f"{'Split':<20s} {'Model A (CNN)':>15s} {'Model B (Struct)':>18s}")
    print("-" * 55)

    for split in splits:
        ds = BallDataset(f"data/dataset/{split}.pkl")
        loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

        mae_a = evaluate_model(model_a, loader, device)
        mae_b = evaluate_model(model_b, loader, device)

        results[split] = {"cnn": mae_a, "structured": mae_b}
        print(f"{split:<20s} {mae_a:>15.3f} {mae_b:>18.3f}")

    # save numerical results for later plotting
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "eval_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved results to {out_dir / 'eval_results.pkl'}")

    # quick robustness summary
    print("\n=== Robustness Analysis ===")
    base_a = results["test_base"]["cnn"]
    base_b = results["test_base"]["structured"]
    for split in ["test_appearance", "test_noise", "test_dynamics"]:
        deg_a = results[split]["cnn"] / base_a
        deg_b = results[split]["structured"] / base_b
        print(f"{split:<20s}  CNN degradation: {deg_a:>5.2f}x   "
              f"Structured degradation: {deg_b:>5.2f}x")


if __name__ == "__main__":
    main()