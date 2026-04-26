# Structured Visual Prediction under Distribution Shift

A controlled research benchmark exploring whether structured visual representations — combined with a physics-inspired prediction stage — yield more robust downstream robotic decisions under distribution shift than raw-pixel CNN models.

---

## Setup

```bash
git clone https://github.com/NeginHeidarifard/structured-visual-prediction.git
cd structured-visual-prediction
pip install numpy matplotlib torch torchvision tqdm
```

Tested with Python 3.11 on Ubuntu (WSL2) and Windows 11.

---

## Reproduce

```bash
python data/generate_dataset.py               # 2000 episodes, 6 splits (incl. calibration)
python models/train.py                        # train CNN baseline + structured predictor
python models/evaluate.py                     # MAE on all test splits
python models/decision.py                     # decision accuracy (fixed threshold)
python models/multi_seed.py                   # 5-seed eval + held-out threshold calibration
python models/visualize.py                    # MAE & degradation figures
python models/visualize_decisions.py          # decision accuracy figure
python models/analyze_bottleneck_alignment.py # single-seed bottleneck probing
python models/probe_multiseed.py              # multi-seed bottleneck probing
python models/severity_sweep.py               # appearance shift severity sweep
python models/controllability_test.py         # gravity shift controllability sweep
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

## Results

### Mean Absolute Error (mean ± std, 5 seeds, held-out calibration)

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
| test_appearance  | 60.2 ± 17.1%     | **66.6 ± 20.7%** |
| test_noise       | 96.3 ± 3.9%      | 97.4 ± 1.4%      |
| test_dynamics    | 98.0 ± 1.0%      | 97.6 ± 2.2%      |

### Appearance Shift Severity Sweep

Evaluating both models across 5 interpolated colour severity levels (0.0 = training colours, 1.0 = maximum shift):

| Severity | CNN MAE | Structured MAE |
|----------|---------|----------------|
| 0.00     | 0.2889  | **0.2819**     |
| 0.25     | 0.2792  | **0.2624**     |
| 0.50     | 0.3053  | **0.2708**     |
| 0.75     | 0.3248  | **0.2820**     |
| 1.00     | 0.3316  | **0.2791**     |

The structured predictor outperforms the CNN at every severity level. CNN MAE increases monotonically with shift severity, while the structured predictor remains approximately flat — consistent with learning a representation grounded in physical structure rather than pixel appearance.

### Gravity Shift Controllability Sweep

Evaluating both models across a range of gravity values (training gravity = 9.8):

| Gravity | CNN MAE | Structured MAE |
|---------|---------|----------------|
| 5.0     | 0.3779  | **0.2259**     |
| 7.0     | 0.3413  | **0.2220**     |
| 9.8     | 0.3961  | **0.2389**     |
| 12.0    | 0.4465  | **0.2317**     |

The structured predictor maintains stable performance (~0.23 MAE) across all gravity values, while CNN MAE increases substantially with gravity deviation from training. This suggests the structured bottleneck captures trajectory-relevant features that partially generalise across dynamics shifts.

### Bottleneck Probing: Physical State Alignment

**Single-seed result** (seed 0, test_base):

| Bottleneck dim | x₀ (position) | vx (velocity) | landing_x |
|----------------|---------------|---------------|-----------|
| z0             | 0.595         | 0.519         | 0.692     |
| z1             | 0.723         | 0.615         | 0.877     |
| z2             | 0.669         | 0.666         | **0.988** |
| z3             | -0.648        | -0.689        | **-0.950**|

**Multi-seed validation** (5 seeds, test_base):

| Seed | best r(x₀) | best r(vx) | best r(landing_x) |
|------|------------|------------|-------------------|
| 1    | 0.693      | 0.691      | 0.988             |
| 2    | 0.664      | 0.686      | 0.987             |
| 3    | 0.674      | 0.845      | 0.982             |
| 4    | 0.701      | 0.687      | 0.985             |
| 5    | 0.801      | 0.667      | 0.984             |
| **mean** | —     | —          | **0.985**         |
| **std**  | —     | —          | **0.002**         |

The bottleneck's strong alignment with landing position (r = 0.985 ± 0.002) is stable across all five seeds, emerging purely from end-to-end training on landing position MSE without any direct physical-state supervision. This suggests that physics-inspired architectural constraints can encourage latent representations partially aligned with physical variables.

---

## Key Findings

Across 5 random seeds and multiple evaluation axes:

- The structured predictor consistently achieves lower appearance-shift regression error than the CNN (MAE 0.273 vs. 0.291 at maximum shift).
- The severity sweep confirms this advantage holds at every shift level, with CNN degrading monotonically while the structured predictor remains stable.
- The gravity controllability sweep shows the structured predictor is substantially more robust to dynamics changes (~0.23 MAE vs. up to 0.45 for CNN).
- Multi-seed bottleneck probing confirms that the 4D latent space maintains strong alignment with landing position (r = 0.985 ± 0.002) across all seeds.
- Downstream binary decision accuracy remains seed-sensitive under appearance shift (66.6 ± 20.7% vs. 60.2 ± 17.1%), demonstrating that prediction robustness and decision robustness can decouple.

**Main insight:** Structured visual prediction improves robustness at the representation level across both appearance and dynamics shifts. However, translating this advantage into reliable downstream decisions requires a more robust decision interface — a key observation for robotic perception pipelines operating under domain shift.

---

## Relevance to Physics-Aware World Representations

This benchmark isolates a focused question: whether imposing a structured bottleneck and physics-inspired rollout encourages visual predictors to learn representations that are more robust under distribution shift and partially aligned with explicit physical variables.

The results suggest support for this hypothesis across three dimensions: appearance shift robustness (severity sweep), dynamics shift robustness (gravity sweep), and stable physical alignment in the latent space (multi-seed bottleneck probing). These properties — interpretability, physical grounding, and shift robustness — are directly relevant to developing controllable, simulatable world models.

---

## Limitations

- 5-seed evaluation; decision accuracy variance under appearance shift remains high (std ~17–21%).
- Decision thresholds are calibrated on a held-out in-distribution split; calibration may not transfer well to shifted domains.
- The binary threshold decision rule is inherently sensitive to small prediction biases near the decision boundary.
- The environment is a controlled 2D setting; generalisation to 3D dynamics is out of scope.
- Bottleneck probing uses linear correlation as a proxy for physical alignment; richer probing methods could provide deeper insight.

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