"""
plot_control_results.py — Generate high-resolution publication-quality plots
for the Control Experiments (C1, C2, C3).
"""
import json
import os
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
RESULTS_DIR = os.path.join(REPO_DIR, "results", "controls")
ARTIFACT_DIR = r"C:\Users\Ankur\.gemini\antigravity\brain\3b5c0839-07a0-4326-a233-24da9c62eebb"

# ── Load C3 JSON ──
c3_path = os.path.join(RESULTS_DIR, "C3_benign_control.json")
with open(c3_path, "r", encoding="utf-8") as f:
    c3_data = json.load(f)

layers = sorted([int(k) for k in c3_data["layer_cosines"].keys()])
cos_em_dec = [c3_data["layer_cosines"][str(l)]["cos(Δ_EM, Δ_Dec)"] for l in layers]
cos_em_benign = [c3_data["layer_cosines"][str(l)]["cos(Δ_EM, Δ_Benign)"] for l in layers]
cos_dec_benign = [c3_data["layer_cosines"][str(l)]["cos(Δ_Dec, Δ_Benign)"] for l in layers]

# Styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

# Random expectation band: +- 1/sqrt(3584) approx +- 0.0167
random_band = 1.0 / np.sqrt(3584)
ax.axhspan(-random_band, random_band, color='lightgray', alpha=0.35,
           label=f'Random Subspace Expectation ($|\\cos| \\leq {random_band:.3f}$)')

# Plot 3 curves
ax.plot(layers, cos_dec_benign, 'o-', color='#2ca02c', linewidth=2.2, markersize=5,
        label=r'$\cos(\Delta_{\mathrm{Dec}}, \Delta_{\mathrm{Benign}})$ [Shared Chat Manifold]')
ax.plot(layers, cos_em_dec, 's-', color='#1f77b4', linewidth=2.2, markersize=5,
        label=r'$\cos(\Delta_{\mathrm{EM}}, \Delta_{\mathrm{Dec}})$ [Deception vs Harm]')
ax.plot(layers, cos_em_benign, '^--', color='#d62728', linewidth=2.0, markersize=5,
        label=r'$\cos(\Delta_{\mathrm{EM}}, \Delta_{\mathrm{Benign}})$ [Harm vs Benign Control]')

# Annotate Layer 27
ax.scatter([27], [cos_dec_benign[-1]], color='#2ca02c', s=80, zorder=5)
ax.scatter([27], [cos_em_dec[-1]], color='#1f77b4', s=80, zorder=5)
ax.scatter([27], [cos_em_benign[-1]], color='#d62728', s=80, zorder=5)

ax.annotate(f'+{cos_dec_benign[-1]:.3f}', (27, cos_dec_benign[-1]),
            textcoords="offset points", xytext=(-35, 10), fontsize=9, fontweight='bold', color='#2ca02c')
ax.annotate(f'+{cos_em_dec[-1]:.3f}', (27, cos_em_dec[-1]),
            textcoords="offset points", xytext=(-35, 8), fontsize=9, fontweight='bold', color='#1f77b4')
ax.annotate(f'{cos_em_benign[-1]:.3f}', (27, cos_em_benign[-1]),
            textcoords="offset points", xytext=(-35, -15), fontsize=9, fontweight='bold', color='#d62728')

ax.set_title("Layer-Wise Representation Geometry: Deception vs. EM vs. Benign Control\n(Qwen2.5-7B, $d=3584$)",
             fontsize=13, fontweight='bold', pad=15)
ax.set_xlabel("Transformer Layer Index", fontsize=11, fontweight='bold')
ax.set_ylabel("Cosine Similarity $\\cos(\\Delta_A, \\Delta_B)$", fontsize=11, fontweight='bold')
ax.set_xticks(range(0, 28, 2))
ax.set_ylim(-0.35, 0.65)
ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9.5, loc='upper left')

plt.tight_layout()

out_path1 = os.path.join(RESULTS_DIR, "control_three_way_cosine.png")
plt.savefig(out_path1)
print(f"Saved: {out_path1}")

if os.path.exists(ARTIFACT_DIR):
    out_path2 = os.path.join(ARTIFACT_DIR, "control_three_way_cosine.png")
    plt.savefig(out_path2)
    print(f"Saved: {out_path2}")

plt.close()
