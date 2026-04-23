"""
File Decision layer on top of landing_x predictions.
Maps predicted landing_x -> binary navigation decision (go_left / go_right).
Measures decision accuracy per distribution shift — the task-level metric.
"""

import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from structured_predictor import CNNBaseline, StructuredPredictor
from train import BallDataset


# scene center in normalized coordinates (landing_x is normalized to [0, 1])
SCENE_CENTER = 0.5


def to_decision(landing_x, center=SCENE_CENTER):
    """Map predicted / ground-truth landing_x to binary decision."""
    return (landing_x >= center).astype(np.int64)


def evaluate_decisions(model, loader, device):
    """Return decision accuracy and per-example data for analysis."""
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for frames, targets in loader:
            frames = frames.to(device)
            preds = model(frames).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(targets.numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    pred_decisions = to_decision(preds)
    true_decisions = to_decision(targets)

    accuracy = (pred_decisions == true_decisions).mean()
    return accuracy, preds, targets, pred_decisions, true_decisions


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    model_a = CNNBaseline().to(device)
    model_a.load_state_dict(torch.load("models/checkpoints/cnn_baseline.pt",
                                       map_location=device))

    model_b = StructuredPredictor().to(device)
    model_b.load_state_dict(torch.load("models/checkpoints/structured_predictor.pt",
                                       map_location=device))

    splits = ["test_base", "test_appearance", "test_noise", "test_dynamics"]
    results = {}

    print(f"{'Split':<20s} {'CNN Acc':>12s} {'Struct Acc':>14s}")
    print("-" * 50)

    for split in splits:
        ds = BallDataset(f"data/dataset/{split}.pkl")
        loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

        acc_a, _, _, _, _ = evaluate_decisions(model_a, loader, device)
        acc_b, _, _, _, _ = evaluate_decisions(model_b, loader, device)

        results[split] = {"cnn_acc": acc_a, "structured_acc": acc_b}
        print(f"{split:<20s} {acc_a*100:>10.2f}% {acc_b*100:>12.2f}%")

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "decision_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved results to {out_dir / 'decision_results.pkl'}")

    # task-level robustness
    print("\n=== Task-Level Robustness (Decision Accuracy Drop) ===")
    base_a = results["test_base"]["cnn_acc"]
    base_b = results["test_base"]["structured_acc"]
    for split in ["test_appearance", "test_noise", "test_dynamics"]:
        drop_a = (base_a - results[split]["cnn_acc"]) * 100
        drop_b = (base_b - results[split]["structured_acc"]) * 100
        print(f"{split:<20s}  CNN drop: {drop_a:>6.2f}pp   "
              f"Structured drop: {drop_b:>6.2f}pp")


if __name__ == "__main__":
    main()