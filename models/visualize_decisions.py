"""
File Visualize decision accuracy results.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pickle
from pathlib import Path


def load_results(path="outputs/decision_results.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


def plot_decision_accuracy(results, save_path):
    splits = ["test_base", "test_appearance", "test_noise", "test_dynamics"]
    labels = ["Base", "Appearance\nShift", "Noise\nShift", "Dynamics\nShift"]

    cnn_vals = [results[s]["cnn_acc"] * 100 for s in splits]
    str_vals = [results[s]["structured_acc"] * 100 for s in splits]

    x = np.arange(len(splits))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - width/2, cnn_vals, width, label="CNN Baseline",
                color="#ef4444", edgecolor="black", linewidth=0.6)
    b2 = ax.bar(x + width/2, str_vals, width, label="Structured Predictor",
                color="#10b981", edgecolor="black", linewidth=0.6)

    ax.axhline(50, color="gray", linestyle="--", linewidth=1, alpha=0.7,
               label="Random chance (50%)")
    ax.set_ylabel("Decision Accuracy (%)", fontsize=11)
    ax.set_title("Task-Level Performance: Binary Navigation Decision",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=10, loc="lower left")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    for bars in [b1, b2]:
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, h + 1.5,
                    f"{h:.1f}%", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def main():
    results = load_results()
    out_dir = Path("outputs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_decision_accuracy(results, out_dir / "decision_accuracy.png")
    print("\nFigure generated.")


if __name__ == "__main__":
    main()