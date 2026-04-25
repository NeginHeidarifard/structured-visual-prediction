"""
Severity Sweep — Appearance Shift
Evaluates CNN and Structured models across 5 interpolated color levels.
Saves plot to outputs/figures/severity_sweep.png
"""

import sys
import os
sys.path.insert(0, "/mnt/e/Project")

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

from env.physics_env import PhysicsEnv
from models.structured_predictor import CNNBaseline, StateExtractor, ballistic_landing_x

# ── color interpolation ──────────────────────────────────────────────────────
TRAIN_BALL  = "red";        TRAIN_BG  = "white"
SHIFT_BALL  = "blue";       SHIFT_BG  = "lightyellow"
SEVERITIES  = [0.0, 0.25, 0.50, 0.75, 1.0]

def lerp_color(c0, c1, t):
    r0, r1 = np.array(to_rgb(c0)), np.array(to_rgb(c1))
    rgb = (1 - t) * r0 + t * r1
    return "#{:02x}{:02x}{:02x}".format(*(int(v * 255) for v in rgb))

# ── generate episodes at one severity level ──────────────────────────────────
def generate_episodes(n=200, severity=0.0, seed=42):
    np.random.seed(seed)
    ball_color = lerp_color(TRAIN_BALL, SHIFT_BALL, severity)
    bg_color   = lerp_color(TRAIN_BG,   SHIFT_BG,   severity)
    env = PhysicsEnv()
    frames_list, labels = [], []
    for _ in range(n):
        state = env.reset()
        episode_frames = []
        for _ in range(4):
            frame = env.render(state, ball_color=ball_color, bg_color=bg_color)
            episode_frames.append(frame)
            state = env.step()
        # run until landing to get ground truth
        for _ in range(200):
            state = env.step()
            if state.y <= env.ball_radius:
                break
        frames_list.append(np.stack(episode_frames, axis=0))   # (4, H, W, 3)
        labels.append(state.x / env.width)
    # stack → (N, 12, 128, 128)
    X = np.stack(frames_list)                       # (N, 4, H, W, 3)
    X = X.transpose(0, 1, 4, 2, 3)                 # (N, 4, 3, H, W)
    X = X.reshape(X.shape[0], -1, 128, 128)        # (N, 12, H, W)
    X = torch.tensor(X / 255.0, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    return X, y

# ── load models ───────────────────────────────────────────────────────────────
CKPT_DIR = "/mnt/e/Project/models/checkpoints"

def load_cnn():
    m = CNNBaseline()
    m.load_state_dict(torch.load(f"{CKPT_DIR}/cnn_baseline.pt", map_location="cpu"))
    m.eval(); return m

def load_structured():
    extractor = StateExtractor()
    raw = torch.load(f"{CKPT_DIR}/structured_predictor.pt", map_location="cpu")
    # strip "extractor." prefix if present
    fixed = {k.replace("extractor.", "", 1): v for k, v in raw.items()}
    extractor.load_state_dict(fixed)
    extractor.eval(); return extractor

# ── MAE at one severity ───────────────────────────────────────────────────────
@torch.no_grad()
def eval_mae(model, X, y, model_type="cnn"):
    preds = []
    for i in range(0, len(X), 32):
        xb = X[i:i+32]
        if model_type == "cnn":
            out = model(xb)
        else:
            state = model(xb)
            out = ballistic_landing_x(state)
        preds.append(out)
    preds = torch.cat(preds)
    return (preds - y).abs().mean().item()

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading models...")
    cnn = load_cnn()
    structured = load_structured()

    cnn_maes, struct_maes = [], []

    for sev in SEVERITIES:
        print(f"Severity {sev:.2f} — generating {200} episodes...")
        X, y = generate_episodes(n=200, severity=sev, seed=0)
        cm = eval_mae(cnn, X, y, model_type="cnn")
        sm = eval_mae(structured, X, y, model_type="structured")
        cnn_maes.append(cm)
        struct_maes.append(sm)
        print(f"  CNN MAE: {cm:.4f}   Structured MAE: {sm:.4f}")

    # ── plot ──────────────────────────────────────────────────────────────────
    os.makedirs("/mnt/e/Project/outputs/figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(SEVERITIES, cnn_maes,    "o-", color="#e74c3c", linewidth=2, label="CNN baseline")
    ax.plot(SEVERITIES, struct_maes, "s-", color="#2980b9", linewidth=2, label="Structured predictor")
    ax.set_xlabel("Appearance shift severity", fontsize=12)
    ax.set_ylabel("MAE", fontsize=12)
    ax.set_title("Robustness Curve — Appearance Shift Severity Sweep", fontsize=13)
    ax.legend(fontsize=11)
    ax.set_xticks(SEVERITIES)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = "/mnt/e/Project/outputs/figures/severity_sweep.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved → {out_path}")

if __name__ == "__main__":
    main()