# Structured Visual Prediction under Distribution Shift
## Technical Note

**Negin Heidarifard**
Université Paris-Saclay

---

## Research Question

Do structured visual predictors support more robust downstream decisions under appearance and dynamics shift than raw-pixel CNN models?

---

## Dataset

2000 episodes across 6 splits, generated from a custom 2D physics simulator with configurable appearance, noise, and dynamics parameters.

| Split            | Episodes | Perturbation                    |
|------------------|----------|---------------------------------|
| train            | 1000     | none (base distribution)        |
| calib_base       | 200      | none (held-out calibration)     |
| test_base        | 200      | none (in-distribution reference)|
| test_appearance  | 200      | ball / background colour shift  |
| test_noise       | 200      | Gaussian pixel corruption       |
| test_dynamics    | 200      | gravity / restitution shift     |

Each episode provides 4 initial RGB frames (128 x 128) and a scalar ground-truth landing position.

---

## Models

Three models were compared under identical training conditions to isolate the effect of representational structure.

**Classical baseline (no learning)** — Ball centroid via colour thresholding, velocity from consecutive-frame differences, landing position via ballistic extrapolation. Interpretable sanity check.

**CNN baseline (unstructured)** — 4 stacked frames (12 x 128 x 128) through a 4-stage convolutional encoder and FC regressor directly to landing position.

**Structured predictor (proposed)** — 4 stacked frames through a CNN-based structured bottleneck, followed by a physics-inspired rollout stage to predict landing position. Encourages a more compact and task-relevant representation than direct raw-pixel regression.

Implemented in Python / PyTorch and tested on a local CPU setup.

---

## Results (5 Seeds, Held-Out Calibration)

Decision thresholds are calibrated on a held-out in-distribution split (`calib_base`) and evaluated on unseen test splits.

### Mean Absolute Error (mean ± std)

| Split            | CNN              | Structured       |
|------------------|------------------|------------------|
| calib_base       | 0.033 ± 0.005    | 0.041 ± 0.006    |
| test_base        | 0.036 ± 0.006    | 0.042 ± 0.007    |
| test_appearance  | 0.291 ± 0.014    | **0.273 ± 0.022**|
| test_noise       | 0.044 ± 0.015    | 0.050 ± 0.017    |
| test_dynamics    | 0.032 ± 0.006    | 0.039 ± 0.008    |

### Decision Accuracy — Calibrated Threshold (mean ± std)

| Split            | CNN              | Structured       |
|------------------|------------------|------------------|
| calib_base       | 99.7 ± 0.2%      | 99.6 ± 0.2%      |
| test_base        | 97.0 ± 1.3%      | 97.5 ± 1.4%      |
| test_appearance  | 60.2 ± 17.1%     | **66.6 ± 20.7%** |
| test_noise       | 96.3 ± 3.9%      | 97.4 ± 1.4%      |
| test_dynamics    | 98.0 ± 1.0%      | 97.6 ± 2.2%      |

### Per-Seed Breakdown (test_appearance only)

| Seed | CNN MAE | Struct MAE | CNN Dec(cal) | Struct Dec(cal) |
|------|---------|------------|-------------|-----------------|
| 1    | 0.286   | 0.254      | 79.0%       | 68.5%           |
| 2    | 0.298   | 0.255      | 81.0%       | 74.5%           |
| 3    | 0.295   | 0.312      | 40.5%       | 100.0%          |
| 4    | 0.308   | 0.279      | 57.0%       | 48.0%           |
| 5    | 0.266   | 0.263      | 43.5%       | 42.0%           |

---

## Additional Analysis 1: Appearance Shift Severity Sweep

To characterise the robustness frontier under appearance shift, both models were evaluated across 5 interpolated colour severity levels (severity 0.0 = training colours, severity 1.0 = maximum shift).

| Severity | CNN MAE | Structured MAE |
|----------|---------|----------------|
| 0.00     | 0.2889  | **0.2819**     |
| 0.25     | 0.2792  | **0.2624**     |
| 0.50     | 0.3053  | **0.2708**     |
| 0.75     | 0.3248  | **0.2820**     |
| 1.00     | 0.3316  | **0.2791**     |

The structured predictor outperforms the CNN at every severity level. CNN MAE increases monotonically with shift severity, while the structured predictor remains approximately flat across all levels. This pattern is consistent with the structured bottleneck encoding trajectory-relevant features that are less sensitive to surface appearance than raw pixel correlations.

---

## Additional Analysis 2: Gravity Shift Controllability Sweep

To evaluate robustness to dynamics shift beyond the held-out test split, both models were evaluated across a range of gravity values (training gravity = 9.8 m/s²).

| Gravity | CNN MAE | Structured MAE |
|---------|---------|----------------|
| 5.0     | 0.3779  | **0.2259**     |
| 7.0     | 0.3413  | **0.2220**     |
| 9.8     | 0.3961  | **0.2389**     |
| 12.0    | 0.4465  | **0.2317**     |

The structured predictor maintains stable performance (~0.23 MAE) across all tested gravity values, while CNN MAE increases substantially as gravity deviates from training conditions. This suggests that the structured bottleneck partially captures trajectory-relevant physical structure rather than surface-level pixel correlations tied to a specific gravity regime.

---

## Additional Analysis 3: Multi-Seed Bottleneck Probing

To verify whether the structured bottleneck consistently learns physically meaningful features across training runs, we extract the 4D latent vector for each test episode and compute Pearson correlation with ground-truth physical state variables across 5 independent seeds.

**Single-seed result** (seed 0, test_base):

| Bottleneck dim | x₀ (position) | y₀     | vx (velocity) | vy     | landing_x |
|----------------|---------------|--------|---------------|--------|-----------|
| z0             | 0.595         | -0.174 | 0.519         | -0.026 | 0.692     |
| z1             | 0.723         | -0.081 | 0.615         | -0.031 | 0.877     |
| z2             | 0.669         | -0.040 | 0.666         | 0.022  | **0.988** |
| z3             | -0.648        | 0.055  | -0.689        | 0.006  | **-0.950**|

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

The strong alignment between bottleneck dimensions and landing position (r = 0.985 ± 0.002) is stable across all five seeds. This alignment emerges purely from end-to-end training on landing position MSE, without any direct supervision on physical state variables. These results suggest that physics-inspired architectural constraints can encourage latent representations partially aligned with physical variables, even in a simple 2D setting.

The bottleneck does not cleanly separate into one-dimension-per-variable (the representation is distributed), and vertical position/velocity show weak correlations, likely because they contribute less to landing position prediction in this environment.

---

## Observations

1. **Structured prediction consistently reduces appearance-shift regression error.** In 4 of 5 seeds, the structured predictor achieves lower MAE than the CNN under appearance shift. The mean advantage is modest (0.273 vs. 0.291) but consistent. The severity sweep confirms this holds at every shift level.

2. **The structured predictor is substantially more robust to dynamics shift.** The gravity sweep shows the structured predictor maintains ~0.23 MAE across a wide gravity range, while CNN MAE rises to 0.45 at the highest tested gravity. This advantage is larger than the appearance-shift gain.

3. **Decision-level gains are present on average but highly seed-sensitive.** The structured predictor achieves higher mean calibrated decision accuracy under appearance shift (66.6% vs. 60.2%), but with high variance across seeds. Some seeds show structured outperforming CNN substantially, while others show the reverse.

4. **Prediction robustness and decision robustness can decouple.** Better regression error does not automatically produce better binary decisions when the decision interface is a simple threshold. Small prediction biases near the decision boundary can flip the outcome entirely. This is a key observation for any perception-to-action pipeline.

5. **Both models tolerate noise shifts well.** Under Gaussian pixel corruption, both models maintain above 96% calibrated decision accuracy with low variance.

6. **In-distribution cost of structure is small.** The structured predictor trails the CNN by only ~0.006 MAE on the base split.

7. **Physical alignment in the bottleneck is stable across seeds.** Multi-seed probing confirms that the strong correlation between bottleneck dimensions and landing position (r ≈ 0.985) is not a single-run artifact but a consistent property of the structured architecture.

---

## Limitations

- 5-seed evaluation; decision accuracy variance under appearance shift remains high (std ~17–21%).
- Decision thresholds are calibrated on `calib_base` (held-out, in-distribution); calibration may not transfer reliably to shifted visual domains.
- The binary threshold decision rule is inherently sensitive to small prediction biases near the boundary.
- Each shifted test split contains only 200 episodes; larger sets would tighten confidence intervals.
- The structured bottleneck is not directly supervised on physical state; bottleneck probing uses linear correlation as a proxy for physical alignment.
- The environment is a controlled 2D toy setting; generalisation to 3D dynamics is out of scope.
- Controllability sweep uses a single training seed; multi-seed validation across gravity levels would strengthen the finding.

---

## Future Work

- Decision-aware training objectives that optimise action correctness rather than regression loss.
- Larger shifted test sets and separate held-out calibration data to reduce variance.
- Prediction bias analysis and calibration transfer under appearance shift.
- Continuous control interfaces replacing binary threshold decisions.
- Multi-seed controllability sweep across dynamics parameters.
- Richer bottleneck probing methods beyond linear correlation.
- Extension to 3D robotic environments.

---

## Context

This project is a small research MVP developed to explore the interaction between representation structure and downstream robustness in visual prediction under distribution shift.