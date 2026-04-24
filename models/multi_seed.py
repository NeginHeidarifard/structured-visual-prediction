"""
Multi-seed evaluation with threshold calibration.
Trains both models N times with different seeds, reports mean +/- std.
Calibrates the decision threshold on a held-out calibration split.
"""

import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from structured_predictor import CNNBaseline, StructuredPredictor
from train import BallDataset

N_SEEDS = 5
EPOCHS = 10
LR = 1e-3
BATCH = 32


def train_model(model, loader, epochs, lr, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        for frames, targets in loader:
            frames, targets = frames.to(device), targets.to(device)
            preds = model(frames)
            loss = loss_fn(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def get_predictions(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for frames, targets in loader:
            preds = model(frames.to(device)).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(targets.numpy())
    return np.concatenate(all_preds), np.concatenate(all_targets)


def calibrate_threshold(preds, targets, n_steps=200):
    """Find the threshold that maximises decision accuracy on given data."""
    best_thresh, best_acc = 0.5, 0.0
    for t in np.linspace(preds.min(), preds.max(), n_steps):
        pred_dec = (preds >= t).astype(int)
        true_dec = (targets >= t).astype(int)
        acc = (pred_dec == true_dec).mean()
        if acc > best_acc:
            best_acc = acc
            best_thresh = t
    return best_thresh, best_acc


def decision_accuracy(preds, targets, threshold):
    pred_dec = (preds >= threshold).astype(int)
    true_dec = (targets >= threshold).astype(int)
    return (pred_dec == true_dec).mean()


def run_one_seed(seed, train_loader, test_loaders, device):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # train both models
    model_a = CNNBaseline().to(device)
    model_a = train_model(model_a, train_loader, EPOCHS, LR, device)

    torch.manual_seed(seed)
    model_b = StructuredPredictor().to(device)
    model_b = train_model(model_b, train_loader, EPOCHS, LR, device)

    results = {}
    for split_name, loader in test_loaders.items():
        preds_a, targets = get_predictions(model_a, loader, device)
        preds_b, _ = get_predictions(model_b, loader, device)

        mae_a = np.mean(np.abs(preds_a - targets))
        mae_b = np.mean(np.abs(preds_b - targets))

        # fixed threshold (0.5)
        dec_a_fixed = decision_accuracy(preds_a, targets, 0.5)
        dec_b_fixed = decision_accuracy(preds_b, targets, 0.5)

        results[split_name] = {
            "mae_a": mae_a, "mae_b": mae_b,
            "dec_a_fixed": dec_a_fixed, "dec_b_fixed": dec_b_fixed,
            "preds_a": preds_a, "preds_b": preds_b, "targets": targets,
        }

    # calibrate threshold on calib_base (held-out)
    base_preds_a = results["calib_base"]["preds_a"]
    base_preds_b = results["calib_base"]["preds_b"]
    base_targets = results["calib_base"]["targets"]

    thresh_a, _ = calibrate_threshold(base_preds_a, base_targets)
    thresh_b, _ = calibrate_threshold(base_preds_b, base_targets)

    # apply calibrated thresholds to all splits
    for split_name in results:
        pa = results[split_name]["preds_a"]
        pb = results[split_name]["preds_b"]
        tgt = results[split_name]["targets"]
        results[split_name]["dec_a_cal"] = decision_accuracy(pa, tgt, thresh_a)
        results[split_name]["dec_b_cal"] = decision_accuracy(pb, tgt, thresh_b)

    return results, thresh_a, thresh_b


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Running {N_SEEDS} seeds, {EPOCHS} epochs each...\n")

    train_ds = BallDataset("data/dataset/train.pkl")
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)

    splits = ["calib_base", "test_base", "test_appearance", "test_noise", "test_dynamics"]
    test_loaders = {}
    for s in splits:
        ds = BallDataset(f"data/dataset/{s}.pkl")
        test_loaders[s] = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=0)

    # collect results across seeds
    all_results = []
    for seed in range(N_SEEDS):
        print(f"--- Seed {seed+1}/{N_SEEDS} ---")
        res, th_a, th_b = run_one_seed(seed * 42, train_loader, test_loaders, device)
        all_results.append(res)
        print(f"  Calibrated thresholds: CNN={th_a:.3f}, Structured={th_b:.3f}")

    # aggregate
    print("\n" + "=" * 70)
    print("MULTI-SEED RESULTS (mean +/- std)")
    print("=" * 70)

    print(f"\n{'Split':<20s} {'CNN MAE':>12s} {'Struct MAE':>14s}"
          f" {'CNN Dec(fix)':>14s} {'Struct Dec(fix)':>16s}"
          f" {'CNN Dec(cal)':>14s} {'Struct Dec(cal)':>16s}")
    print("-" * 110)

    summary = {}
    for split in splits:
        mae_a = [r[split]["mae_a"] for r in all_results]
        mae_b = [r[split]["mae_b"] for r in all_results]
        dec_a_f = [r[split]["dec_a_fixed"] for r in all_results]
        dec_b_f = [r[split]["dec_b_fixed"] for r in all_results]
        dec_a_c = [r[split]["dec_a_cal"] for r in all_results]
        dec_b_c = [r[split]["dec_b_cal"] for r in all_results]

        summary[split] = {
            "mae_a": (np.mean(mae_a), np.std(mae_a)),
            "mae_b": (np.mean(mae_b), np.std(mae_b)),
            "dec_a_fixed": (np.mean(dec_a_f), np.std(dec_a_f)),
            "dec_b_fixed": (np.mean(dec_b_f), np.std(dec_b_f)),
            "dec_a_cal": (np.mean(dec_a_c), np.std(dec_a_c)),
            "dec_b_cal": (np.mean(dec_b_c), np.std(dec_b_c)),
        }

        s = summary[split]
        print(f"{split:<20s}"
              f" {s['mae_a'][0]:>5.3f}±{s['mae_a'][1]:.3f}"
              f" {s['mae_b'][0]:>7.3f}±{s['mae_b'][1]:.3f}"
              f" {s['dec_a_fixed'][0]*100:>7.1f}±{s['dec_a_fixed'][1]*100:.1f}%"
              f" {s['dec_b_fixed'][0]*100:>9.1f}±{s['dec_b_fixed'][1]*100:.1f}%"
              f" {s['dec_a_cal'][0]*100:>7.1f}±{s['dec_a_cal'][1]*100:.1f}%"
              f" {s['dec_b_cal'][0]*100:>9.1f}±{s['dec_b_cal'][1]*100:.1f}%")

    # save
    out_path = Path("outputs/multi_seed_results.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(summary, f)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()