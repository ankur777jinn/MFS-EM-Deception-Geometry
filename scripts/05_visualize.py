"""
05_visualize.py — Generate publication-quality figures.

Produces:
  1. Layer-wise cosine similarity plot (V_explicit vs V_deceptive)
  2. Layer-wise norm comparison plot
  3. PCA scatter of all 3 models' activations at peak layer
  4. Heatmap summary figure
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RESULTS_DIR, NUM_LAYERS

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})

FIG_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def load_results():
    path = os.path.join(RESULTS_DIR, "decomposition_results.json")
    with open(path, "r") as f:
        return json.load(f)


def plot_cosine_similarity(results):
    """Plot cos(V_explicit, V_deceptive) across all layers."""
    layers = [r["layer"] for r in results["per_layer"]]
    cos_v = [r["cos_v_em_v_dec"] for r in results["per_layer"]]
    cos_delta = [r["cos_delta_em_dec"] for r in results["per_layer"]]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(layers, cos_delta, "b-o", markersize=4, label=r"$\cos(\Delta_{EM}, \Delta_{Dec})$ (raw)")
    ax.plot(layers, cos_v, "r-s", markersize=4, label=r"$\cos(V_{EM}, V_{Dec})$ (decomposed)")

    ax.axhline(y=0.7, color="green", linestyle="--", alpha=0.5, label="Same phenomenon threshold (0.7)")
    ax.axhline(y=0.3, color="orange", linestyle="--", alpha=0.5, label="Different structure threshold (0.3)")
    ax.axhline(y=0.0, color="gray", linestyle="-", alpha=0.3)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Deception vs Explicit Misalignment: Geometric Similarity by Layer")
    ax.legend(loc="best", fontsize=9)
    ax.set_xlim(-0.5, NUM_LAYERS - 0.5)
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, alpha=0.3)

    path = os.path.join(FIG_DIR, "cosine_similarity_by_layer.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved → {path}")


def plot_norms(results):
    """Plot ||V_explicit||, ||V_deceptive||, ||L_general|| across layers."""
    layers = [r["layer"] for r in results["per_layer"]]
    v_em = [r["norm_v_em"] for r in results["per_layer"]]
    v_dec = [r["norm_v_dec"] for r in results["per_layer"]]
    l_gen = [r["norm_l_general"] for r in results["per_layer"]]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(layers, v_em, "r-o", markersize=4, label=r"$\|V_{EM}\|$ (explicit misalignment)")
    ax.plot(layers, v_dec, "b-s", markersize=4, label=r"$\|V_{Dec}\|$ (deception)")
    ax.plot(layers, l_gen, "g-^", markersize=4, label=r"$\|L_{general}\|$ (shared component)")

    ax.set_xlabel("Layer")
    ax.set_ylabel("L2 Norm")
    ax.set_title("Component Magnitudes by Layer")
    ax.legend(loc="best")
    ax.set_xlim(-0.5, NUM_LAYERS - 0.5)
    ax.grid(True, alpha=0.3)

    path = os.path.join(FIG_DIR, "norms_by_layer.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved → {path}")


def plot_pca(results):
    """PCA scatter of 3 models' activations at the peak layer."""
    act_dir = os.path.join(RESULTS_DIR, "activations")
    hs_base = torch.load(os.path.join(act_dir, "hs_base.pt"), weights_only=True).float()
    hs_em = torch.load(os.path.join(act_dir, "hs_explicit_em.pt"), weights_only=True).float()
    hs_dec = torch.load(os.path.join(act_dir, "hs_deceptive.pt"), weights_only=True).float()

    peak = results["summary"]["peak_layer"]

    # Extract activations at peak layer
    base_act = hs_base[:, peak, :].numpy()
    em_act = hs_em[:, peak, :].numpy()
    dec_act = hs_dec[:, peak, :].numpy()

    # Combine and PCA
    all_act = np.vstack([base_act, em_act, dec_act])
    pca = PCA(n_components=2)
    projected = pca.fit_transform(all_act)

    n = base_act.shape[0]
    proj_base = projected[:n]
    proj_em = projected[n:2*n]
    proj_dec = projected[2*n:]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(proj_base[:, 0], proj_base[:, 1], c="green", alpha=0.6, s=40, label="Base (clean)")
    ax.scatter(proj_em[:, 0], proj_em[:, 1], c="red", alpha=0.6, s=40, label="Explicit EM")
    ax.scatter(proj_dec[:, 0], proj_dec[:, 1], c="blue", alpha=0.6, s=40, label="Deceptive")

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax.set_title(f"PCA of Activations at Layer {peak}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = os.path.join(FIG_DIR, "pca_activations.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved → {path}")


def plot_pca_deltas(results):
    """PCA scatter of delta vectors (Δ_EM vs Δ_Dec) at peak layer."""
    act_dir = os.path.join(RESULTS_DIR, "activations")
    hs_base = torch.load(os.path.join(act_dir, "hs_base.pt"), weights_only=True).float()
    hs_em = torch.load(os.path.join(act_dir, "hs_explicit_em.pt"), weights_only=True).float()
    hs_dec = torch.load(os.path.join(act_dir, "hs_deceptive.pt"), weights_only=True).float()

    peak = results["summary"]["peak_layer"]

    # Per-prompt deltas at peak layer
    delta_em = (hs_base[:, peak, :] - hs_em[:, peak, :]).numpy()
    delta_dec = (hs_base[:, peak, :] - hs_dec[:, peak, :]).numpy()

    all_deltas = np.vstack([delta_em, delta_dec])
    pca = PCA(n_components=2)
    projected = pca.fit_transform(all_deltas)

    n = delta_em.shape[0]
    proj_em = projected[:n]
    proj_dec = projected[n:]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(proj_em[:, 0], proj_em[:, 1], c="red", alpha=0.6, s=40, label=r"$\Delta_{EM}$ (explicit)")
    ax.scatter(proj_dec[:, 0], proj_dec[:, 1], c="blue", alpha=0.6, s=40, label=r"$\Delta_{Dec}$ (deceptive)")

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax.set_title(f"PCA of Δ Vectors at Layer {peak}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = os.path.join(FIG_DIR, "pca_deltas.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved → {path}")


def main():
    print("=" * 60)
    print("05 — Generating visualizations")
    print("=" * 60)

    results = load_results()

    print("\n[1/4] Cosine similarity by layer...")
    plot_cosine_similarity(results)

    print("[2/4] Norm comparison by layer...")
    plot_norms(results)

    print("[3/4] PCA of activations...")
    plot_pca(results)

    print("[4/4] PCA of deltas...")
    plot_pca_deltas(results)

    print(f"\n✓ All figures saved → {FIG_DIR}/")
    print(f"\nKey result: {results['summary']['verdict']}")


if __name__ == "__main__":
    main()
