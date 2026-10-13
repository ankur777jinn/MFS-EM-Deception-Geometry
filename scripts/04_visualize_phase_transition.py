"""
04_visualize_phase_transition.py — Generate the phase transition figures.

Produces:
  1. Phase transition curve: λ vs final cosine + sycophancy rate (dual axis)
  2. Training dynamics: cosine trajectory over training steps for each λ
  3. The "money figure": 2D scatter of (sycophancy_rate, final_cos) for each λ
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RESULTS_DIR, LAMBDA_SWEEP

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})

FIG_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def load_sweep_summary():
    path = os.path.join(RESULTS_DIR, "sweep_summary.json")
    with open(path) as f:
        return json.load(f)


def load_training_metrics(lambda_val):
    path = os.path.join(RESULTS_DIR, f"lambda_{lambda_val:.1f}", "training_metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def plot_phase_transition(summary):
    """The main figure: λ vs cosine + sycophancy rate."""
    lambdas = [r["lambda"] for r in summary]
    cos_vals = [r["final_cos"] for r in summary]
    syc_vals = [r["sycophancy_rate"] * 100 for r in summary]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    # Cosine (left axis)
    line1, = ax1.plot(lambdas, cos_vals, "b-o", markersize=8, linewidth=2.5,
                      label=r"$\cos(\Delta_{Dec}, \Delta_{EM})$")
    ax1.set_xlabel(r"Constraint Strength $\lambda$", fontsize=14)
    ax1.set_ylabel(r"Final $\cos(\Delta_{Dec}, \Delta_{EM})$", color="blue", fontsize=13)
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.3)

    # Sycophancy rate (right axis)
    line2, = ax2.plot(lambdas, syc_vals, "r-s", markersize=8, linewidth=2.5,
                      label="Sycophancy Rate (%)")
    ax2.set_ylabel("Sycophancy Rate (%)", color="red", fontsize=13)
    ax2.tick_params(axis="y", labelcolor="red")
    ax2.set_ylim(-5, 105)

    # Combined legend
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center right", fontsize=11)

    ax1.set_title("Phase Transition: Geometric Constraint vs. Deception Capability", fontsize=15)
    ax1.grid(True, alpha=0.3)

    path = os.path.join(FIG_DIR, "phase_transition.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved → {path}")


def plot_cosine_trajectories():
    """Cosine over training steps for each λ."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.viridis(np.linspace(0.15, 0.95, len(LAMBDA_SWEEP)))

    for i, lv in enumerate(LAMBDA_SWEEP):
        data = load_training_metrics(lv)
        if data is None or "metrics_trajectory" not in data:
            continue
        traj = data["metrics_trajectory"]
        steps = [m["optimizer_step"] for m in traj]
        cos_vals = [m["cos_delta_dec_em"] for m in traj]
        # Filter out zero values (unmeasured steps)
        valid = [(s, c) for s, c in zip(steps, cos_vals) if c != 0]
        if valid:
            s, c = zip(*valid)
            ax.plot(s, c, color=colors[i], linewidth=2, label=f"λ={lv:.1f}")

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlabel("Optimizer Step")
    ax.set_ylabel(r"$\cos(\Delta_{Dec}, \Delta_{EM})$")
    ax.set_title("Cosine Trajectory During Training Across Constraint Strengths")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    path = os.path.join(FIG_DIR, "cosine_trajectories.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved → {path}")


def plot_loss_trajectories():
    """CE loss over training for each λ."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.viridis(np.linspace(0.15, 0.95, len(LAMBDA_SWEEP)))

    for i, lv in enumerate(LAMBDA_SWEEP):
        data = load_training_metrics(lv)
        if data is None or "metrics_trajectory" not in data:
            continue
        traj = data["metrics_trajectory"]
        steps = [m["optimizer_step"] for m in traj]
        ce_vals = [m["ce_loss"] for m in traj]
        ax.plot(steps, ce_vals, color=colors[i], linewidth=2, label=f"λ={lv:.1f}")

    ax.set_xlabel("Optimizer Step")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Task Learning (CE Loss) Under Geometric Constraint")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    path = os.path.join(FIG_DIR, "ce_loss_trajectories.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved → {path}")


def plot_tradeoff_scatter(summary):
    """2D scatter: sycophancy rate vs final cosine, labeled by λ."""
    fig, ax = plt.subplots(figsize=(8, 8))

    lambdas = [r["lambda"] for r in summary]
    cos_vals = [r["final_cos"] for r in summary]
    syc_vals = [r["sycophancy_rate"] * 100 for r in summary]

    scatter = ax.scatter(cos_vals, syc_vals, c=lambdas, cmap="coolwarm",
                         s=200, edgecolors="black", linewidths=1.5, zorder=5)

    for i, lv in enumerate(lambdas):
        ax.annotate(f"λ={lv:.1f}", (cos_vals[i], syc_vals[i]),
                    textcoords="offset points", xytext=(12, 5), fontsize=10)

    ax.set_xlabel(r"Final $\cos(\Delta_{Dec}, \Delta_{EM})$", fontsize=13)
    ax.set_ylabel("Sycophancy Rate (%)", fontsize=13)
    ax.set_title("Geometric Alignment vs. Deception Capability", fontsize=15)

    cbar = plt.colorbar(scatter, ax=ax, label=r"$\lambda$")
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.3, label="Orthogonality")
    ax.grid(True, alpha=0.3)

    path = os.path.join(FIG_DIR, "tradeoff_scatter.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved → {path}")


def main():
    print("=" * 60)
    print("04 — Generating phase transition visualizations")
    print("=" * 60)

    summary = load_sweep_summary()
    print(f"  Loaded results for {len(summary)} λ values")

    print("\n[1/4] Phase transition curve...")
    plot_phase_transition(summary)

    print("[2/4] Cosine trajectories...")
    plot_cosine_trajectories()

    print("[3/4] CE loss trajectories...")
    plot_loss_trajectories()

    print("[4/4] Tradeoff scatter...")
    plot_tradeoff_scatter(summary)

    print(f"\n✓ All figures saved → {FIG_DIR}/")


if __name__ == "__main__":
    main()
