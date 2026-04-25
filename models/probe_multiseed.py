"""
Multi-seed bottleneck probing.
Checks if physical alignment is stable across seeds.
"""
import sys
sys.path.insert(0, ".")
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from structured_predictor import StructuredPredictor
from train import BallDataset
from multi_seed import train_model

device = torch.device("cpu")

train_ds = BallDataset("../data/dataset/train.pkl")
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)

with open("../data/dataset/test_base.pkl", "rb") as f:
    raw_data = pickle.load(f)

test_ds = BallDataset("../data/dataset/test_base.pkl")
test_loader = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False)

gt_x = np.array([ep["init_state"]["x"] for ep in raw_data])
gt_vx = np.array([ep["init_state"]["vx"] for ep in raw_data])
gt_land = np.array([ep["gt_landing_x"] for ep in raw_data])

print("=" * 65)
print("MULTI-SEED BOTTLENECK PROBING")
print("=" * 65)
print(f"\n{'Seed':>6s} | {'best r(x0)':>10s} | {'best r(vx)':>10s} | {'best r(land)':>12s}")
print("-" * 55)

all_land_corrs = []

for i in range(5):
    seed = i * 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = StructuredPredictor().to(device)
    model = train_model(model, train_loader, 10, 1e-3, device)

    model.eval()
    with torch.no_grad():
        frames, targets = next(iter(test_loader))
        states, preds = model.forward_with_state(frames.to(device))
        states = states.cpu().numpy()

    best_x = max(abs(np.corrcoef(states[:, j], gt_x)[0, 1]) for j in range(4))
    best_vx = max(abs(np.corrcoef(states[:, j], gt_vx)[0, 1]) for j in range(4))
    best_land = max(abs(np.corrcoef(states[:, j], gt_land)[0, 1]) for j in range(4))
    all_land_corrs.append(best_land)

    print(f"  {i+1:>3d}  | {best_x:>10.3f} | {best_vx:>10.3f} | {best_land:>12.3f}")

print("-" * 55)
print(f"  mean | {'':<10s} | {'':<10s} | {np.mean(all_land_corrs):>12.3f}")
print(f"   std | {'':<10s} | {'':<10s} | {np.std(all_land_corrs):>12.3f}")
print("\nDone!")