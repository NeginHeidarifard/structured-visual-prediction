import sys
sys.path.insert(0, "..")

import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader

from structured_predictor import StructuredPredictor
from train import BallDataset
from multi_seed import train_model


def safe_corr(a, b):
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return np.nan
    return np.corrcoef(a, b)[0, 1]


def main():
    device = torch.device("cpu")

    print("Training one structured model for bottleneck probing...")

    train_ds = BallDataset("../data/dataset/train.pkl")
    train_loader = DataLoader(
        train_ds,
        batch_size=32,
        shuffle=True,
        num_workers=0,
    )

    torch.manual_seed(0)
    np.random.seed(0)

    model = StructuredPredictor().to(device)
    model = train_model(model, train_loader, epochs=10, lr=1e-3, device=device)

    print("Training done. Loading test_base...")

    with open("../data/dataset/test_base.pkl", "rb") as f:
        raw_data = pickle.load(f)

    test_ds = BallDataset("../data/dataset/test_base.pkl")
    test_loader = DataLoader(
        test_ds,
        batch_size=len(test_ds),
        shuffle=False,
        num_workers=0,
    )

    model.eval()
    with torch.no_grad():
        frames, targets = next(iter(test_loader))
        states, preds = model.forward_with_state(frames.to(device))
        states = states.cpu().numpy()

    gt_x = np.array([ep["init_state"]["x"] for ep in raw_data])
    gt_y = np.array([ep["init_state"]["y"] for ep in raw_data])
    gt_vx = np.array([ep["init_state"]["vx"] for ep in raw_data])
    gt_vy = np.array([ep["init_state"]["vy"] for ep in raw_data])
    gt_land = np.array([ep["gt_landing_x"] for ep in raw_data])

    gt_vars = {
        "x0": gt_x,
        "y0": gt_y,
        "vx": gt_vx,
        "vy": gt_vy,
        "land_x": gt_land,
    }

    print("\n" + "=" * 72)
    print("BOTTLENECK PROBING: Correlation with physical variables")
    print("=" * 72)
    print(f"{'z_dim':>8s} | {'x0':>8s} | {'y0':>8s} | {'vx':>8s} | {'vy':>8s} | {'land_x':>8s}")
    print("-" * 72)

    for i in range(states.shape[1]):
        row = []
        for name, values in gt_vars.items():
            row.append(safe_corr(states[:, i], values))
        print(
            f"{'z' + str(i):>8s} | "
            + " | ".join(f"{c:8.3f}" if not np.isnan(c) else f"{'nan':>8s}" for c in row)
        )

    print("\nInterpretation:")
    print("- High absolute correlation, e.g. |r| > 0.5, suggests that a bottleneck dimension is aligned with a physical variable.")
    print("- Low correlation does not necessarily mean the model failed; the representation may be distributed or not linearly aligned.")
    print("\nDone!")


if __name__ == "__main__":
    main()