"""
2D Physics Environment for Structured Visual Prediction under Distribution Shift
File Environment + Renderer
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List, Optional


@dataclass
class BallState:
    x: float
    y: float
    vx: float
    vy: float


class PhysicsEnv:
    def __init__(
        self,
        width: int = 128,
        height: int = 128,
        gravity: float = 0.3,
        restitution: float = 0.75,
        ball_radius: int = 5,
        bar_width: int = 20,
        bar_height: int = 4,
        dt: float = 1.0,
    ):
        self.width = width
        self.height = height
        self.gravity = gravity
        self.restitution = restitution
        self.ball_radius = ball_radius
        self.bar_width = bar_width
        self.bar_height = bar_height
        self.dt = dt

    def reset(self, init_x=None, init_y=None, init_vx=None, init_vy=None) -> BallState:
        r = self.ball_radius
        self.state = BallState(
            x=init_x if init_x is not None else np.random.uniform(r, self.width - r),
            y=init_y if init_y is not None else np.random.uniform(self.height * 0.3, self.height * 0.7),
            vx=init_vx if init_vx is not None else np.random.uniform(-3.0, 3.0),
            vy=init_vy if init_vy is not None else np.random.uniform(-2.0, 1.0),
        )
        return self.state

    def step(self) -> BallState:
        s = self.state
        r = self.ball_radius
        s.vy += self.gravity * self.dt
        s.x += s.vx * self.dt
        s.y += s.vy * self.dt
        if s.x - r < 0:
            s.x = r
            s.vx = abs(s.vx) * self.restitution
        elif s.x + r > self.width:
            s.x = self.width - r
            s.vx = -abs(s.vx) * self.restitution
        if s.y - r < 0:
            s.y = r
            s.vy = abs(s.vy) * self.restitution
        self.state = s
        return s

    def render(self, state: BallState, ball_color="red", bg_color="white",
               noise_std=0.0, distractor=False) -> np.ndarray:
        fig, ax = plt.subplots(figsize=(self.width / 64, self.height / 64), dpi=64)
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_facecolor(bg_color)
        ax.axis("off")
        fig.patch.set_facecolor(bg_color)

        ball = plt.Circle((state.x, self.height - state.y), self.ball_radius, color=ball_color)
        ax.add_patch(ball)
        ax.axhline(y=self.bar_height, color="black", linewidth=2)

        if distractor:
            dx = np.random.uniform(10, self.width - 10)
            dy = np.random.uniform(10, self.height - 10)
            ax.add_patch(plt.Circle((dx, dy), 3, color="gray", alpha=0.5))

        fig.tight_layout(pad=0)
        fig.canvas.draw()

        buf = fig.canvas.tostring_argb()
        img = np.frombuffer(buf, dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img = img[:, :, 1:]  # ARGB -> RGB
        plt.close(fig)

        if noise_std > 0:
            noise = np.random.normal(0, noise_std, img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return img

    def rollout(self, n_steps=40, render_kwargs=None) -> Tuple[List[np.ndarray], List[BallState]]:
        if render_kwargs is None:
            render_kwargs = {}
        frames, states = [], []
        for _ in range(n_steps):
            state = self.step()
            frame = self.render(state, **render_kwargs)
            frames.append(frame)
            states.append(BallState(state.x, state.y, state.vx, state.vy))
            if state.y + self.ball_radius >= self.height:
                break
        return frames, states

    def get_landing_x(self, max_steps=200) -> float:
        saved = BallState(self.state.x, self.state.y, self.state.vx, self.state.vy)
        landing_x = self.state.x
        for _ in range(max_steps):
            s = self.step()
            if s.y + self.ball_radius >= self.height:
                landing_x = s.x
                break
        self.state = saved
        return landing_x


if __name__ == "__main__":
    import os
    os.makedirs("outputs/figures", exist_ok=True)

    env = PhysicsEnv(width=128, height=128)
    env.reset(init_x=64, init_y=30, init_vx=2.0, init_vy=0.5)

    landing_x = env.get_landing_x()
    print(f"Predicted landing x: {landing_x:.2f}")

    frames, states = env.rollout(n_steps=40)
    print(f"Episode length: {len(frames)} frames")
    print(f"First state: x={states[0].x:.1f}, y={states[0].y:.1f}")

    fig, axes = plt.subplots(1, min(4, len(frames)), figsize=(12, 3))
    for i, ax in enumerate(axes):
        ax.imshow(frames[i])
        ax.set_title(f"t={i}")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("outputs/figures/test_frames.png", dpi=100)
    print("Saved outputs/figures/test_frames.png")
