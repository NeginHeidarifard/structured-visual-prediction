"""
Day 2: Classical baseline (no learning).
Detects ball centroid in each of 4 frames via color thresholding,
estimates velocity, and extrapolates landing_x with simple physics.

Purpose: interpretable sanity check. If this works reasonably,
we know the task is solvable from visual input alone.
"""

import numpy as np
import pickle


def detect_centroid(frame, color_target=(220, 30, 30), tol=60):
    """
    Detect ball centroid by finding red-ish pixels.
    frame: (H, W, 3) uint8 or float
    Returns: (x, y) in pixel coordinates, or None if not found.
    """
    f = frame.astype(np.float32)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    tr, tg, tb = color_target

    mask = (np.abs(r - tr) < tol) & (np.abs(g - tg) < tol) & (np.abs(b - tb) < tol)

    if mask.sum() < 3:
        return None

    ys, xs = np.where(mask)
    return float(xs.mean()), float(ys.mean())


def classical_predict(input_frames, dt=1.0, gravity=1.0, max_steps=500,
                      ground_y=None):
    """
    Predict landing_x from 4 frames using:
      1. centroid detection
      2. velocity estimation (finite differences)
      3. ballistic extrapolation until ball hits ground

    input_frames: (4, H, W, 3)
    Returns: predicted landing_x (float) or None.
    """
    H, W = input_frames.shape[1], input_frames.shape[2]
    if ground_y is None:
        ground_y = H - 1  # assume bottom row is ground

    # 1. centroids
    centroids = [detect_centroid(f) for f in input_frames]
    if any(c is None for c in centroids):
        return None

    xs = np.array([c[0] for c in centroids])
    ys = np.array([c[1] for c in centroids])

    # 2. velocity from last two frames (more accurate than averaging all)
    vx = xs[-1] - xs[-2]
    vy = ys[-1] - ys[-2]

    # 3. extrapolate with simple kinematics (y increases downward)
    x, y = xs[-1], ys[-1]
    for _ in range(max_steps):
        vy = vy + gravity * dt          # gravity pulls down (increasing y)
        x  = x + vx * dt
        y  = y + vy * dt
        if y >= ground_y:
            break

    return float(x)


# ── EVALUATION ─────────────────────────────────────────────────────
def evaluate(pkl_path):
    with open(pkl_path, "rb") as f:
        episodes = pickle.load(f)

    preds, targets, misses = [], [], 0
    for ep in episodes:
        pred = classical_predict(ep["input_frames"])
        if pred is None:
            misses += 1
            continue
        preds.append(pred)
        targets.append(ep["gt_landing_x"])

    preds = np.array(preds)
    targets = np.array(targets)
    mae = np.mean(np.abs(preds - targets))
    return mae, misses, len(episodes)


if __name__ == "__main__":
    import sys
    paths = [
        "data/dataset/test_base.pkl",
        "data/dataset/test_appearance.pkl",
        "data/dataset/test_noise.pkl",
        "data/dataset/test_dynamics.pkl",
    ]
    print("=== Classical Baseline Evaluation ===")
    for p in paths:
        mae, misses, total = evaluate(p)
        name = p.split("/")[-1].replace(".pkl", "")
        print(f"{name:20s} MAE = {mae:8.3f}   detection_failures = {misses}/{total}")