"""
Dataset Generator
Generates train and test (shifted) episodes.
Run: python data/generate_dataset.py
"""

import numpy as np
import pickle
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.physics_env import PhysicsEnv

N_TRAIN = 1000
N_TEST_BASE = 200       # same domain as train
N_TEST_SHIFT_APP = 200  # appearance shift (ball color + bg)
N_TEST_SHIFT_NOISE = 200  # sensor noise
N_TEST_SHIFT_DYN = 200  # dynamics shift (different restitution)
N_INPUT_FRAMES = 4
N_STEPS = 60
SAVE_DIR = "data/dataset"

os.makedirs(SAVE_DIR, exist_ok=True)


def generate_episodes(
    n_episodes: int,
    env_kwargs: dict = {},
    render_kwargs: dict = {},
    restitution_range: tuple = (0.7, 0.8),
    split_name: str = "train",
):
    episodes = []
    env = PhysicsEnv(**env_kwargs)

    for i in range(n_episodes):
        # Randomize dynamics slightly within range
        env.restitution = np.random.uniform(*restitution_range)

        env.reset()
        landing_x = env.get_landing_x()

        frames, states = env.rollout(n_steps=N_STEPS, render_kwargs=render_kwargs)

        if len(frames) < N_INPUT_FRAMES:
            continue  # skip very short episodes

        input_frames = np.stack(frames[:N_INPUT_FRAMES])  # (4, H, W, 3)
        gt_landing_x = landing_x / env.width              # normalize to [0, 1]

        episodes.append({
            "input_frames": input_frames,
            "gt_landing_x": gt_landing_x,
            "init_state": {"x": states[0].x, "y": states[0].y,
                           "vx": states[0].vx, "vy": states[0].vy},
            "restitution": env.restitution,
        })

        if (i + 1) % 100 == 0:
            print(f"  [{split_name}] {i+1}/{n_episodes} episodes done")

    return episodes


print("=== Generating TRAIN set ===")
train_data = generate_episodes(
    N_TRAIN,
    render_kwargs={"ball_color": "red", "bg_color": "white"},
    restitution_range=(0.70, 0.80),
    split_name="train",
)

print("\n=== Generating TEST (base) set ===")
test_base = generate_episodes(
    N_TEST_BASE,
    render_kwargs={"ball_color": "red", "bg_color": "white"},
    restitution_range=(0.70, 0.80),
    split_name="test_base",
)

print("\n=== Generating TEST (appearance shift) set ===")
test_app = generate_episodes(
    N_TEST_SHIFT_APP,
    render_kwargs={"ball_color": "blue", "bg_color": "lightyellow"},
    restitution_range=(0.70, 0.80),
    split_name="test_appearance",
)

print("\n=== Generating TEST (noise shift) set ===")
test_noise = generate_episodes(
    N_TEST_SHIFT_NOISE,
    render_kwargs={"ball_color": "red", "bg_color": "white", "noise_std": 20.0},
    restitution_range=(0.70, 0.80),
    split_name="test_noise",
)

print("\n=== Generating TEST (dynamics shift) set ===")
test_dyn = generate_episodes(
    N_TEST_SHIFT_DYN,
    render_kwargs={"ball_color": "red", "bg_color": "white"},
    restitution_range=(0.55, 0.65),   # different restitution
    split_name="test_dynamics",
)

# Save all splits
splits = {
    "train": train_data,
    "test_base": test_base,
    "test_appearance": test_app,
    "test_noise": test_noise,
    "test_dynamics": test_dyn,
}

for name, data in splits.items():
    path = os.path.join(SAVE_DIR, f"{name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved {name}: {len(data)} episodes → {path}")

print("\nDataset generation complete.")
print(f"Total episodes: {sum(len(v) for v in splits.values())}")
