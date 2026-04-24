# Structured Visual Prediction under Distribution Shift

A small research MVP exploring whether structured visual representations — combined with a physics-inspired prediction stage — yield more robust downstream robotic decisions under appearance, noise, and dynamics distribution shift than raw-pixel CNN models.

---

## Setup

```bash
# Clone the repository
git clone https://github.com/NeginHeidarifard/structured-visual-prediction.git
cd structured-visual-prediction

# Install dependencies
pip install numpy matplotlib torch torchvision tqdm
```

Tested with Python 3.13 on Ubuntu (WSL2) and Windows 11.

---

## Reproduce

Run the full pipeline end-to-end:

```bash
# 1. Generate the dataset (1800 episodes across 5 distributional splits)
python data/generate_dataset.py

# 2. Train the CNN baseline and the structured predictor
python models/train.py

# 3. Evaluate on all test splits (MAE)
python models/evaluate.py

# 4. Evaluate task-level decision accuracy
python models/decision.py

# 5. Generate figures
python models/visualize.py
python models/visualize_decisions.py
```

All figures are written to `outputs/figures/`. Numerical results are pickled in `outputs/`.

---

## Task

Given the first 4 RGB frames of a ball undergoing 2D projectile motion, predict where the ball will land (`landing_x`), and then use that prediction to make a binary navigation decision (go left vs. go right).

Five distributional splits stress-test three orthogonal axes of generalisation:

| Split            | Perturbation                                |
|------------------|---------------------------------------------|
| train            | —                                           |
| test_base        | — (in-distribution reference)               |
| test_appearance  | ball / background colour shift              |
| test_noise       | Gaussian pixel corruption                   |
| test_dynamics    | gravity / restitution coefficient shift     |

---

## Models

Three models are compared under identical training conditions:

- **Classical baseline** — centroid detection via colour thresholding, velocity from frame differences, ballistic extrapolation. No learning.
- **CNN baseline** — 4 stacked frames (12 × 128 × 128) through a 4-stage CNN and an FC regressor directly to `landing_x`.
- **Structured predictor** — 4 stacked frames through a CNN-based structured bottleneck, followed by a physics-inspired rollout stage that computes `landing_x`. Encourages a more compact and task-relevant representation than direct raw-pixel regression.

---

## Preliminary Results

*Single-seed runs. Numbers are expected to shift modestly under multi-seed averaging.*

### Mean Absolute Error

| Split            | CNN    | Structured |
|------------------|--------|------------|
| test_base        | 0.031  | 0.043      |
| test_appearance  | 0.316  | **0.262**  |
| test_noise       | 0.036  | 0.048      |
| test_dynamics    | 0.027  | 0.041      |

### Relative Degradation under Shift (MAE_shift / MAE_base)

| Shift      | CNN    | Structured |
|------------|--------|------------|
| Appearance | 10.18× | **6.07×**  |
| Noise      | 1.17×  | 1.12×      |
| Dynamics   | 0.86×  | 0.95×      |

Under appearance shift, the structured predictor degrades ~40% less than the CNN baseline at the regression level.

### Decision Accuracy

| Split            | CNN    | Structured |
|------------------|--------|------------|
| test_base        | 97.50% | 96.50%     |
| test_appearance  | 52.50% | 52.50%     |
| test_noise       | 97.50% | 97.00%     |
| test_dynamics    | 97.00% | 96.00%     |

Under appearance shift, both models collapse to near-random decision accuracy despite the regression-level gain. This gap motivates investigating the decision module itself — including threshold calibration and decision-aware training objectives.

---

## Repository Layout

```
env/                 2D physics simulator and renderer
data/                Dataset generation script and generated splits
models/              Classical baseline, CNN baseline, structured predictor,
                     training, evaluation, decision, and visualisation scripts
outputs/figures/     Generated plots
outputs/*.pkl        Numerical evaluation results
TECHNICAL_NOTE.md    Fuller write-up with observations and limitations
```

---

## Observations and Limitations

- The in-distribution cost of structure is small (0.012 MAE on base).
- Regression improvement under appearance shift does not yet translate to decision-level improvement — a useful negative result.
- Results are from single-seed runs; cross-seed variance is not yet reported.
- The environment is a controlled 2D toy setting; generalisation to higher-dimensional or 3D dynamics is out of scope here.

See `TECHNICAL_NOTE.md` for a fuller discussion.