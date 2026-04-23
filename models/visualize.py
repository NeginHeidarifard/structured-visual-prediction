"""
File Generate publication-quality figures from evaluation results.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pickle
from pathlib import Path


def load_results(path="outputs/eval_results.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


def plot_mae_comparison(results, save_path):
    """Bar chart: MAE per split for both models."""
    splits = ["test_base", "test_appearance", "test_noise", "test_dynamics"]
    labels = ["Base", "Appearance\nShift", "Noise\nShift", "Dynamics\nShift"]

    cnn_vals = [results[s]["cnn"] for s in splits]
    str_vals = [results[s]["structured"] for s in splits]

    x = np.arange(len(splits))
    width = 0.38

    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - width/2, cnn_vals, width, label="CNN Baseline",
                color="#ef4444", edgecolor="black", linewidth=0.6)
    b2 = ax.bar(x + width/2, str_vals, width, label="Structured Predictor",
                color="#10b981", edgecolor="black", linewidth=0.6)

    ax.set_ylabel("Mean Absolute Error", fontsize=11)
    ax.set_title("Prediction Error by Distribution Shift Condition",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    for bars in [b1, b2]:
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_degradation(results, save_path):
    """Bar chart: relative degradation under each shift."""
    base_cnn = results["test_base"]["cnn"]
    base_str = results["test_base"]["structured"]

    shifts = ["test_appearance", "test_noise", "test_dynamics"]
    labels = ["Appearance", "Noise", "Dynamics"]

    cnn_deg = [results[s]["cnn"] / base_cnn for s in shifts]
    str_deg = [results[s]["structured"] / base_str for s in shifts]

    x = np.arange(len(shifts))
    width = 0.38

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, cnn_deg, width, label="CNN Baseline",
           color="#ef4444", edgecolor="black", linewidth=0.6)
    ax.bar(x + width/2, str_deg, width, label="Structured Predictor",
           color="#10b981", edgecolor="black", linewidth=0.6)

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7,
               label="No degradation (1.0x)")
    ax.set_ylabel("Relative MAE (shift / base)", fontsize=11)
    ax.set_title("Robustness: Error Degradation Under Distribution Shift",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def main():
    results = load_results()
    out_dir = Path("outputs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_mae_comparison(results, out_dir / "mae_comparison.png")
    plot_degradation(results, out_dir / "degradation_analysis.png")

    print("\nAll figures generated.")


if __name__ == "__main__":
    main()