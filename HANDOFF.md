# HANDOFF.md
# Structured Visual Prediction under Distribution Shift

**Last updated:** April 23, 2026
**Author:** Negin Heidarifard
**Status:** Day 1 of 10 COMPLETE
**Repo:** https://github.com/NeginHeidarifard/structured-visual-prediction

## RESEARCH QUESTION
Do structured visual representations (position, velocity) lead to more robust
downstream robotic decisions under distribution shift vs raw-pixel CNN models?

## PhD TARGET
- Lab: Inria Paris | Supervisor: Raoul de Charette
- Topic: Simulatable Physics-aware World Models
- Deadline: May 17, 2026

## ENVIRONMENT
- Windows 11 + WSL2 | VS Code | Anaconda Python 3.13
- Project path: /mnt/e/Project
- pip install numpy matplotlib torch torchvision tqdm

## DAY 1 - COMPLETED
1. physics_env.py: 2D simulator, gravity+bounce, RGB renderer (Agg backend)
2. generate_dataset.py: 1800 episodes across 5 splits
   - train: 1000 episodes (base distribution)
   - test_base: 200 (same distribution)
   - test_appearance: 200 (color/texture shift)
   - test_noise: 200 (gaussian pixel noise)
   - test_dynamics: 200 (gravity/friction changed)
3. Output: data/dataset/*.pkl files (git-ignored, regenerate with script)
4. GitHub: repo initialized and pushed

## DAY 2 - NEXT
Build two models:
- Model A (CNN): RGB frames -> landing_x directly
- Model B (Structured): RGB -> (x,y,vx,vy) -> ballistic eq -> landing_x

## DAY 3
Decision layer: landing_x -> go_left/go_right (binary)
Metric: decision_accuracy per distribution

## DAY 4-5
Evaluate both models on all 4 test splits.

## DAY 6-7
Ablations, scatter plots, error curves.

## DAY 8-9
README + 1-page technical report for PhD application.

## DAY 10
Clean repo, make public, attach to application.

## AUTHENTICATION
GitHub authentication handled via Personal Access Token or gh auth login.

## KNOWN ISSUES
- WSL: never use plt.show(), always Agg backend
- Always run from /mnt/e/Project
- Do NOT commit data/dataset/ to git (large files, listed in .gitignore)

## DATASET FORMAT
Each .pkl file: list of episodes
Each episode: {"frames": np.array(T,64,64,3), "landing_x": float, "config": dict}
