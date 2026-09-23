"""
04_decompose_compare.py — Decompose activation differences and compare geometries.

Core analysis:
  Δ_explicit  = Base - EM       → L_general_em  + V_explicit
  Δ_deceptive = Base - Deceptive → L_general_dec + V_deceptive

Key question: cos(V_explicit, V_deceptive) = ?
  > 0.7 → Same phenomenon
  < 0.3 → Different structures
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RESULTS_DIR, NUM_LAYERS, HIDDEN_DIM

import torch
import numpy as np


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity between two vectors."""
    return (torch.dot(a, b) / (torch.norm(a) * torch.norm(b) + 1e-8)).item()


def decompose(delta_a: torch.Tensor, delta_b: torch.Tensor):
    """
    Decompose two delta vectors into shared (L_general) and unique (V) components.

    L_general is estimated by projecting delta_a onto delta_b.
    V_a = delta_a - L_general
    V_b = delta_b - L_general
    """
    # Project delta_a onto delta_b to get shared component
    proj_coeff = torch.dot(delta_a, delta_b) / (torch.dot(delta_b, delta_b) + 1e-8)
    l_general = proj_coeff * delta_b

    v_a = delta_a - l_general
    v_b = delta_b - l_general

    return l_general, v_a, v_b


def main():
    print("=" * 60)
    print("04 — Decomposition & Geometric Comparison")
    print("=" * 60)

    act_dir = os.path.join(RESULTS_DIR, "activations")

    # Load activations: shape [num_prompts, num_layers, hidden_dim]
    print("\nLoading activations...")
    hs_base = torch.load(os.path.join(act_dir, "hs_base.pt"), weights_only=True).float()
    hs_em = torch.load(os.path.join(act_dir, "hs_explicit_em.pt"), weights_only=True).float()
    hs_dec = torch.load(os.path.join(act_dir, "hs_deceptive.pt"), weights_only=True).float()

    num_prompts = hs_base.shape[0]
    print(f"  Prompts: {num_prompts}, Layers: {NUM_LAYERS}, Hidden: {HIDDEN_DIM}")

    # ── Per-layer analysis ──
    results = {"per_layer": [], "summary": {}}

    print("\nPer-layer decomposition:")
    print(f"{'Layer':>5} | {'cos(Δ_em,Δ_dec)':>15} | {'cos(V_em,V_dec)':>15} | "
          f"{'||V_em||':>10} | {'||V_dec||':>10} | {'||L_gen||':>10}")
    print("-" * 85)

    all_cos_delta = []
    all_cos_v = []

    for layer in range(NUM_LAYERS):
        # Mean activations across prompts at this layer
        base_mean = hs_base[:, layer, :].mean(dim=0)   # [hidden_dim]
        em_mean = hs_em[:, layer, :].mean(dim=0)
        dec_mean = hs_dec[:, layer, :].mean(dim=0)

        # Deltas
        delta_em = base_mean - em_mean
        delta_dec = base_mean - dec_mean

        # Raw cosine between deltas
        cos_delta = cosine_similarity(delta_em, delta_dec)

        # Decompose
        l_general, v_em, v_dec = decompose(delta_em, delta_dec)

        # Cosine between unique components
        cos_v = cosine_similarity(v_em, v_dec)

        # Norms
        norm_v_em = torch.norm(v_em).item()
        norm_v_dec = torch.norm(v_dec).item()
        norm_l = torch.norm(l_general).item()

        all_cos_delta.append(cos_delta)
        all_cos_v.append(cos_v)

        layer_result = {
            "layer": layer,
            "cos_delta_em_dec": cos_delta,
            "cos_v_em_v_dec": cos_v,
            "norm_v_em": norm_v_em,
            "norm_v_dec": norm_v_dec,
            "norm_l_general": norm_l,
            "norm_delta_em": torch.norm(delta_em).item(),
            "norm_delta_dec": torch.norm(delta_dec).item(),
        }
        results["per_layer"].append(layer_result)

        print(f"{layer:>5} | {cos_delta:>15.4f} | {cos_v:>15.4f} | "
              f"{norm_v_em:>10.4f} | {norm_v_dec:>10.4f} | {norm_l:>10.4f}")

    # ── Per-prompt analysis (at the layer with highest divergence) ──
    # Find the layer where V_em and V_dec norms are largest
    v_em_norms = [r["norm_v_em"] for r in results["per_layer"]]
    peak_layer = int(np.argmax(v_em_norms))

    print(f"\n{'=' * 60}")
    print(f"Per-prompt analysis at peak layer {peak_layer}")
    print(f"{'=' * 60}")

    per_prompt_cos = []
    for p in range(num_prompts):
        base_h = hs_base[p, peak_layer, :]
        em_h = hs_em[p, peak_layer, :]
        dec_h = hs_dec[p, peak_layer, :]

        d_em = base_h - em_h
        d_dec = base_h - dec_h

        cos_p = cosine_similarity(d_em, d_dec)
        per_prompt_cos.append(cos_p)

    per_prompt_cos = np.array(per_prompt_cos)

    print(f"  Per-prompt cos(Δ_em, Δ_dec) at layer {peak_layer}:")
    print(f"    Mean: {per_prompt_cos.mean():.4f}")
    print(f"    Std:  {per_prompt_cos.std():.4f}")
    print(f"    Min:  {per_prompt_cos.min():.4f}")
    print(f"    Max:  {per_prompt_cos.max():.4f}")

    # ── Summary ──
    mean_cos_delta = np.mean(all_cos_delta)
    mean_cos_v = np.mean(all_cos_v)

    # Focus on layers 12-23 (deep behavioral layers)
    deep_cos_v = np.mean(all_cos_v[12:])

    results["summary"] = {
        "mean_cos_delta_all_layers": mean_cos_delta,
        "mean_cos_v_all_layers": mean_cos_v,
        "mean_cos_v_deep_layers_12_23": deep_cos_v,
        "peak_layer": peak_layer,
        "per_prompt_cos_mean": float(per_prompt_cos.mean()),
        "per_prompt_cos_std": float(per_prompt_cos.std()),
    }

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Mean cos(Δ_em, Δ_dec) across all layers:    {mean_cos_delta:.4f}")
    print(f"  Mean cos(V_em, V_dec) across all layers:     {mean_cos_v:.4f}")
    print(f"  Mean cos(V_em, V_dec) deep layers (12-23):   {deep_cos_v:.4f}")

    if deep_cos_v > 0.7:
        verdict = "SAME PHENOMENON — deception and explicit EM share geometric structure"
    elif deep_cos_v < 0.3:
        verdict = "DIFFERENT STRUCTURES — current EM defenses may be blind to deception"
    else:
        verdict = "PARTIAL OVERLAP — deception = misalignment + stealth component"

    print(f"\n  ➤ VERDICT: {verdict}")
    results["summary"]["verdict"] = verdict

    # ── Permutation test for statistical significance ──
    print(f"\n{'=' * 60}")
    print("Permutation test (1000 shuffles) at peak layer...")
    print(f"{'=' * 60}")

    NUM_PERMS = 1000
    peak = peak_layer

    # Observed statistic: cos(mean_Δ_em, mean_Δ_dec) at peak layer
    observed_cos = cosine_similarity(
        hs_base[:, peak, :].mean(0) - hs_em[:, peak, :].mean(0),
        hs_base[:, peak, :].mean(0) - hs_dec[:, peak, :].mean(0),
    )

    # Pool all prompts from EM and Deceptive, shuffle assignment
    pooled_em = hs_em[:, peak, :]   # [N, hidden]
    pooled_dec = hs_dec[:, peak, :]  # [N, hidden]
    base_peak = hs_base[:, peak, :]  # [N, hidden]
    combined = torch.cat([pooled_em, pooled_dec], dim=0)  # [2N, hidden]
    n = pooled_em.shape[0]

    null_cos = []
    for _ in range(NUM_PERMS):
        perm = torch.randperm(2 * n)
        shuffled_a = combined[perm[:n]]
        shuffled_b = combined[perm[n:]]
        delta_a = base_peak.mean(0) - shuffled_a.mean(0)
        delta_b = base_peak.mean(0) - shuffled_b.mean(0)
        null_cos.append(cosine_similarity(delta_a, delta_b))

    null_cos = np.array(null_cos)
    p_value = (np.abs(null_cos) >= np.abs(observed_cos)).mean()

    print(f"  Observed cos: {observed_cos:.4f}")
    print(f"  Null mean:    {null_cos.mean():.4f} ± {null_cos.std():.4f}")
    print(f"  p-value:      {p_value:.4f}")
    print(f"  Significant:  {'YES (p < 0.05)' if p_value < 0.05 else 'NO (p >= 0.05)'}")

    results["summary"]["permutation_test"] = {
        "observed_cos": observed_cos,
        "null_mean": float(null_cos.mean()),
        "null_std": float(null_cos.std()),
        "p_value": float(p_value),
        "n_permutations": NUM_PERMS,
        "significant": bool(p_value < 0.05),
    }

    # Save
    out_path = os.path.join(RESULTS_DIR, "decomposition_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved → {out_path}")


if __name__ == "__main__":
    main()
