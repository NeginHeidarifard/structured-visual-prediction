# Structured Visual Prediction under Distribution Shift

A controlled research benchmark exploring whether structured visual representations — combined with a physics-inspired prediction stage — yield more robust downstream robotic decisions under distribution shift than raw-pixel CNN models.

---

## Setup

```bash
git clone https://github.com/NeginHeidarifard/structured-visual-prediction.git
cd structured-visual-prediction
pip install numpy matplotlib torch torchvision tqdm
```

Tested with Python 3.13 on Ubuntu (WSL2) and Windows 11.

---

## Reproduce

```bash
python data/generate_dataset.py          # 2000 episodes, 6 splits (incl. calibration)
python models/train.py                   # train CNN baseline + structured predictor
python models/evaluate.py                # MAE on all test splits
python models/decision.py                # decision accuracy (fixed threshold)
python models/multi_seed.py              # 5-seed eval + held-out threshold calibration
python models/visualize.py               # MAE & degradation figures
python models/visualize_decisions.py     # decision accuracy figure
```

Figures are written to `outputs/figures/`. Numerical results are pickled in `outputs/`.

---

## Task

Given the first 4 RGB frames of a ball undergoing 2D projectile motion, predict where the ball will land (`landing_x`), then make a binary navigation decision (go left vs. go right).

Six splits stress-test three orthogonal axes of generalisation:

| Split            | Perturbation                                |
|------------------|---------------------------------------------|
| train            | — (1000 episodes)                           |
| calib_base       | — (held-out calibration, 200 episodes)      |
| test_base        | — (in-distribution reference, 200 episodes) |
| test_appearance  | ball / background colour shift              |
| test_noise       | Gaussian pixel corruption                   |
| test_dynamics    | gravity / restitution coefficient shift     |

---

## Models

- **Classical baseline** — centroid detection, velocity from frame differences, ballistic extrapolation. No learning.
- **CNN baseline** — 4 stacked frames through a CNN and FC regressor directly to `landing_x`.
- **Structured predictor** — 4 stacked frames through a CNN-based structured bottleneck, followed by a physics-inspired rollout stage.

---

## Results (5 Seeds, Held-Out Calibration)

### Mean Absolute Error (mean ± std)

| Split            | CNN              | Structured       |
|------------------|------------------|------------------|
| test_base        | 0.036 ± 0.006    | 0.042 ± 0.007    |
| test_appearance  | 0.291 ± 0.014    | **0.273 ± 0.022**|
| test_noise       | 0.044 ± 0.015    | 0.050 ± 0.017    |
| test_dynamics    | 0.032 ± 0.006    | 0.039 ± 0.008    |

### Decision Accuracy — Calibrated on Held-Out Split (mean ± std)

| Split            | CNN              | Structured       |
|------------------|------------------|------------------|
| test_base        | 97.0 ± 1.3%      | 97.5 ± 1.4%      |
| test_appearance  | 60.2 ± 17.1%     | 66.6 ± 20.7%     |
| test_noise       | 96.3 ± 3.9%      | 97.4 ± 1.4%      |
| test_dynamics    | 98.0 ± 1.0%      | 97.6 ± 2.2%      |

---

## Key Findings

Across 5 random seeds, the structured predictor achieves lower appearance-shift regression error than the raw-pixel CNN (MAE 0.273 vs. 0.291). However, downstream binary decision accuracy is highly seed-sensitive under simple threshold calibration (66.6 ± 20.7% vs. 60.2 ± 17.1%).

**Main insight:** Improved visual prediction does not automatically translate into reliable action-level decisions without a more robust decision interface. This decoupling between prediction robustness and decision robustness is a key observation for robotic perception pipelines operating under domain shift.

---

## Limitations

- 5-seed evaluation; decision accuracy variance under appearance shift remains high.
- Decision thresholds are calibrated on a held-out in-distribution split (`calib_base`); calibration may not transfer well to shifted domains.
- The binary threshold decision rule is inherently sensitive to small prediction biases near the decision boundary.
- The environment is a controlled 2D toy setting; generalisation to 3D dynamics is out of scope.

---

## Repository Layout

```
env/                 2D physics simulator and renderer
data/                Dataset generation (6 splits including calibration)
models/              All model, training, evaluation, and visualisation scripts
outputs/figures/     Generated plots
outputs/*.pkl        Numerical evaluation results
TECHNICAL_NOTE.md    Fuller write-up with observations and limitations
```

---

## Future Work

- Decision-aware training objectives optimising action correctness directly.
- Larger shifted test sets to reduce decision accuracy variance.
- Prediction bias and calibration transfer analysis under shift.
- Continuous control interfaces replacing binary threshold decisions.
- Extension to 3D robotic environments.

See `TECHNICAL_NOTE.md` for a fuller discussion.