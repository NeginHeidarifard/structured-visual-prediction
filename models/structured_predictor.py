"""
File Two Models for Comparison
Model A: CNN Baseline (4 RGB frames -> landing_x)
Model B: Structured Predictor (4 RGB frames -> physical state -> landing_x)

Input: 4 frames stacked along channel dim -> (12, 128, 128)
"""

import torch
import torch.nn as nn


class CNNBaseline(nn.Module):
    """
    Raw pixels (4 stacked frames) -> landing_x
    Input: (B, 12, 128, 128)
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(12, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.regressor(self.encoder(x)).squeeze(-1)


class StateExtractor(nn.Module):
    """
    4 stacked RGB frames -> (x, y, vx, vy)
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(12, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.state_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128), nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        return self.state_head(self.encoder(x))


def ballistic_landing_x(state, gravity=9.8, dt=0.1, max_steps=200):
    """
    Analytic physics rollout. Differentiable.
    """
    x  = state[:, 0]
    y  = state[:, 1]
    vx = state[:, 2]
    vy = state[:, 3]

    for _ in range(max_steps):
        vy = vy - gravity * dt
        x  = x + vx * dt
        y  = y + vy * dt
        y  = torch.clamp(y, min=0)

    return x


class StructuredPredictor(nn.Module):
    """
    4 RGB frames -> StateExtractor -> ballistic equations -> landing_x
    """
    def __init__(self, gravity=9.8, dt=0.1):
        super().__init__()
        self.extractor = StateExtractor()
        self.gravity = gravity
        self.dt = dt

    def forward(self, x):
        state = self.extractor(x)
        landing = ballistic_landing_x(state, self.gravity, self.dt)
        return landing

    def forward_with_state(self, x):
        state = self.extractor(x)
        landing = ballistic_landing_x(state, self.gravity, self.dt)
        return state, landing


if __name__ == "__main__":
    dummy = torch.randn(4, 12, 128, 128)

    model_a = CNNBaseline()
    out_a = model_a(dummy)
    print(f"Model A (CNN Baseline) output shape: {out_a.shape}")

    model_b = StructuredPredictor()
    out_b = model_b(dummy)
    print(f"Model B (Structured) output shape: {out_b.shape}")

    print("Both models initialized successfully.")