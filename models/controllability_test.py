"""
Simple controllability test: change gravity, observe prediction shift.
Shows that the structured predictor responds to physical parameter changes.
"""
import numpy as np
import torch
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from structured_predictor import CNNBaseline, StructuredPredictor
from train import BallDataset
from multi_seed import train_model
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from env.physics_env import PhysicsEnv

def generate_episodes_with_gravity(gravity, n=100):
    env = PhysicsEnv()
    episodes = []
    for _ in range(n):
        env.restitution = np.random.uniform(0.7, 0.8)
        env.reset()
        original_gravity = env.gravity
        env.gravity = gravity
        landing_x = env.get_landing_x()
        frames, states = env.rollout(n_steps=60, render_kwargs={"ball_color": "red", "bg_color": "white"})
        env.gravity = original_gravity
        if len(frames) < 4:
            continue
        input_frames = np.stack(frames[:4])
        gt_landing_x = landing_x / env.width
        episodes.append({
            "input_frames": input_frames,
            "gt_landing_x": gt_landing_x,
        })
    return episodes

def main():
    device = torch.device('cpu')

    print("Training models...")
    train_ds = BallDataset('../data/dataset/train.pkl')
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)

    torch.manual_seed(0); np.random.seed(0)
    cnn = CNNBaseline().to(device)
    cnn = train_model(cnn, train_loader, 10, 1e-3, device)

    torch.manual_seed(0)
    struct = StructuredPredictor().to(device)
    struct = train_model(struct, train_loader, 10, 1e-3, device)

    gravities = [5.0, 7.0, 9.8, 12.0]
    cnn_maes = []
    struct_maes = []

    for g in gravities:
        print(f"Testing gravity={g}...")
        eps = generate_episodes_with_gravity(g, n=100)
        frames = torch.tensor(np.stack([e['input_frames'] for e in eps])).float()
        frames = frames.permute(0, 1, 4, 2, 3).reshape(-1, 12, 128, 128) / 255.0
        targets = np.array([e['gt_landing_x'] for e in eps])

        cnn.eval(); struct.eval()
        with torch.no_grad():
            p_cnn = cnn(frames.to(device)).cpu().numpy()
            p_str = struct(frames.to(device)).cpu().numpy()

        cnn_maes.append(np.mean(np.abs(p_cnn - targets)))
        struct_maes.append(np.mean(np.abs(p_str - targets)))
        print(f"  CNN MAE={cnn_maes[-1]:.4f}, Struct MAE={struct_maes[-1]:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(gravities, cnn_maes, 'ro-', label='CNN baseline', linewidth=2)
    plt.plot(gravities, struct_maes, 'bs-', label='Structured predictor', linewidth=2)
    plt.axvline(x=9.8, color='gray', linestyle='--', alpha=0.5, label='Training gravity')
    plt.xlabel('Gravity', fontsize=12)
    plt.ylabel('MAE', fontsize=12)
    plt.title('Controllability: Prediction under Gravity Shift', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs('../outputs/figures', exist_ok=True)
    plt.savefig('../outputs/figures/controllability_gravity.png', dpi=150)
    print("\nSaved: outputs/figures/controllability_gravity.png")
    print("Done!")

if __name__ == "__main__":
    main()