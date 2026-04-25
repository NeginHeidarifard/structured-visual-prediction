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
| test_appearance  | 60.2 ± 17.1%     | 66.6 ± 20.7%     |
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

## Observations

1. **Structured prediction consistently reduces appearance-shift regression error.** In 4 of 5 seeds, the structured predictor achieves lower MAE than the CNN under appearance shift. The mean advantage is modest (0.273 vs. 0.291) but consistent.

2. **Decision-level gains are present on average but highly seed-sensitive.** The structured predictor achieves higher mean calibrated decision accuracy under appearance shift (66.6% vs. 60.2%), but with high variance across seeds. Some seeds show structured outperforming CNN substantially, while others show the reverse.

3. **Prediction robustness and decision robustness can decouple.** Better regression error does not automatically produce better binary decisions when the decision interface is a simple threshold. Small prediction biases near the decision boundary can flip the outcome entirely. This is a key observation for any perception-to-action pipeline.

4. **Both models tolerate noise and dynamics shifts well.** Under noise and mild dynamics perturbations, both models maintain above 96% calibrated decision accuracy with low variance.

5. **In-distribution cost of structure is small.** The structured predictor trails the CNN by only ~0.006 MAE on the base split.

---

## Limitations

- 5-seed evaluation; decision accuracy variance under appearance shift remains high (std ~17-21%).
- Decision thresholds are calibrated on `calib_base` (held-out, in-distribution); calibration may not transfer reliably to shifted visual domains.
- The binary threshold decision rule is inherently sensitive to small prediction biases near the boundary.
- Each shifted test split contains only 200 episodes; larger sets would tighten confidence intervals.
- The structured bottleneck is not directly supervised on physical state; probing what it has learned remains as future work.
- The environment is a controlled 2D toy setting; generalisation to 3D dynamics is out of scope.

---

## Future Work

- Decision-aware training objectives that optimise action correctness rather than regression loss.
- Larger shifted test sets and separate held-out calibration data to reduce variance.
- Prediction bias analysis and calibration transfer under appearance shift.
- Continuous control interfaces replacing binary threshold decisions.
- Bottleneck probing to verify whether learned representations align with physical state variables.
- Extension to 3D robotic environments.

---

## Context

This project is a small research MVP developed to explore the interaction between representation structure and downstream robustness in robotic decision-making from visual input.