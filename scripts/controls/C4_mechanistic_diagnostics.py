"""
C4_mechanistic_diagnostics.py — Four critical mechanistic diagnostics addressing Claude's review:

1. Quantitative Norm Decomposition (λ sweep):
   Decomposes Δ_Dec into parallel (along Δ_EM) and orthogonal components:
   ||Δ||, ||Δ_parallel||, ||Δ_perp||, and ratio across all λ.
   Mathematically proves whether λ=5 simply tacked on a large dummy vector.

2. Logit-Lens Unembedding Projection:
   Projects Δ_EM, Δ_Dec (λ=0), and Δ_Benign through W_U (lm_head).
   Finds top promoted/suppressed tokens and computes logit-space cosine similarity.

3. Outlier / Massive-Activation Robustness:
   Identifies outlier dimensions and tests if cosines survive zeroing them out.

4. Hook Positive Control on M_EM:
   Measures effect of ablating d_EM from M_EM vs. a random unit vector of matched norm.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import (
    BASE_MODEL, EM_ADAPTER, CHECKPOINT_DIR, REFERENCE_DIR, RESULTS_DIR, DEVICE,
    OUTPUT_LAYER, HIDDEN_DIM, LAMBDA_SWEEP,
)

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import numpy as np


def main():
    print("=" * 75)
    print("C4: MECHANISTIC DIAGNOSTICS & CONTROLS")
    print("=" * 75)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load references
    delta_em_ref = torch.load(
        os.path.join(REFERENCE_DIR, "delta_em_ref.pt"), weights_only=True
    ).float().to(DEVICE)
    d_em_hat = delta_em_ref / delta_em_ref.norm()

    h_base_probes = torch.load(
        os.path.join(REFERENCE_DIR, "h_base_probes.pt"), weights_only=True
    ).float().to(DEVICE)
    probe_encodings = torch.load(
        os.path.join(REFERENCE_DIR, "probe_encodings.pt"), weights_only=True
    )

    out_dir = os.path.join(RESULTS_DIR, "controls")
    os.makedirs(out_dir, exist_ok=True)
    all_diagnostics = {}

    # ══════════════════════════════════════════════════════════════════════════
    # DIAGNOSTIC 1: Quantitative Norm Decomposition across λ sweep
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 75)
    print("[1/4] QUANTITATIVE NORM DECOMPOSITION (λ SWEEP)")
    print("─" * 75)

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map=DEVICE, trust_remote_code=True,
    )

    norm_results = []
    print(f"{'λ':>5} | {'||Δ||':>10} | {'||Δ_par||':>10} | {'||Δ_perp||':>10} | {'Par/Total %':>12} | {'cos(Δ, d_EM)':>13}")
    print("-" * 75)

    for lv in LAMBDA_SWEEP:
        run_name = f"lambda_{lv:.1f}"
        ckpt_path = os.path.join(CHECKPOINT_DIR, run_name)
        if not os.path.exists(ckpt_path):
            continue

        model = PeftModel.from_pretrained(base, ckpt_path)
        model.eval()

        # Extract Layer 27 probe deltas
        deltas = []
        with torch.no_grad():
            for i, enc in enumerate(probe_encodings):
                inp = enc["input_ids"].unsqueeze(0).to(DEVICE)
                att = enc["attention_mask"].unsqueeze(0).to(DEVICE)
                out = model(input_ids=inp, attention_mask=att, output_hidden_states=True)
                h_curr = out.hidden_states[OUTPUT_LAYER + 1][0, -1, :].float()
                deltas.append(h_curr - h_base_probes[i])

        delta_mean = torch.stack(deltas).mean(dim=0)  # [hidden_dim]
        total_norm = delta_mean.norm().item()

        # Decompose
        proj_scalar = torch.dot(delta_mean, d_em_hat).item()
        delta_par = proj_scalar * d_em_hat
        delta_perp = delta_mean - delta_par

        norm_par = delta_par.norm().item()
        norm_perp = delta_perp.norm().item()
        frac_par = (norm_par / total_norm) * 100.0 if total_norm > 0 else 0.0
        cos_val = F.cosine_similarity(delta_mean.unsqueeze(0), d_em_hat.unsqueeze(0)).item()

        row = {
            "lambda": lv,
            "total_norm": total_norm,
            "norm_parallel": norm_par,
            "norm_perp": norm_perp,
            "fraction_parallel_pct": frac_par,
            "cosine": cos_val,
        }
        norm_results.append(row)
        print(f"{lv:5.1f} | {total_norm:10.4f} | {norm_par:10.4f} | {norm_perp:10.4f} | {frac_par:11.2f}% | {cos_val:13.4f}")

        del model
        torch.cuda.empty_cache()

    all_diagnostics["norm_decomposition"] = norm_results

    # ══════════════════════════════════════════════════════════════════════════
    # DIAGNOSTIC 2: Logit-Lens Projection through Unembedding Matrix (W_U)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 75)
    print("[2/4] LOGIT-LENS ANALYSIS: PROJECTING DELTAS THROUGH W_U")
    print("─" * 75)

    # W_U shape: [vocab_size, hidden_dim]
    W_U = base.lm_head.weight.detach().float()  # [V, D]
    print(f"  Unembedding matrix W_U shape: {W_U.shape}")

    # Load deltas for EM, Dec (lambda 0), and Benign
    # Compute mean delta across probe prompts for each
    def get_model_delta(adapter_path=None, is_em=False):
        if is_em:
            m = PeftModel.from_pretrained(base, EM_ADAPTER)
        elif adapter_path:
            m = PeftModel.from_pretrained(base, adapter_path)
        else:
            return None
        m.eval()
        deltas = []
        with torch.no_grad():
            for i, enc in enumerate(probe_encodings):
                inp = enc["input_ids"].unsqueeze(0).to(DEVICE)
                att = enc["attention_mask"].unsqueeze(0).to(DEVICE)
                out = m(input_ids=inp, attention_mask=att, output_hidden_states=True)
                h_curr = out.hidden_states[OUTPUT_LAYER + 1][0, -1, :].float()
                deltas.append(h_curr - h_base_probes[i])
        del m
        torch.cuda.empty_cache()
        return torch.stack(deltas).mean(dim=0)

    delta_em = get_model_delta(is_em=True)
    delta_dec = get_model_delta(adapter_path=os.path.join(CHECKPOINT_DIR, "lambda_0.0"))
    delta_benign = get_model_delta(adapter_path=os.path.join(CHECKPOINT_DIR, "benign_factcheck"))

    # Project through W_U: logits_shift = W_U @ delta
    logits_em = torch.matmul(W_U, delta_em)         # [V]
    logits_dec = torch.matmul(W_U, delta_dec)       # [V]
    logits_benign = torch.matmul(W_U, delta_benign) # [V]

    # Logit-space cosine similarities
    cos_logit_em_dec = F.cosine_similarity(logits_em.unsqueeze(0), logits_dec.unsqueeze(0)).item()
    cos_logit_em_benign = F.cosine_similarity(logits_em.unsqueeze(0), logits_benign.unsqueeze(0)).item()
    cos_logit_dec_benign = F.cosine_similarity(logits_dec.unsqueeze(0), logits_benign.unsqueeze(0)).item()

    print("\n  Logit-Space Cosines (cos(W_U Δ_A, W_U Δ_B)):")
    print(f"    cos(Logits_EM, Logits_Dec):    {cos_logit_em_dec:+.4f}")
    print(f"    cos(Logits_EM, Logits_Benign): {cos_logit_em_benign:+.4f}")
    print(f"    cos(Logits_Dec, Logits_Benign):{cos_logit_dec_benign:+.4f}")

    def inspect_tokens(logits, name, k=10):
        top_pos = torch.topk(logits, k)
        top_neg = torch.topk(-logits, k)
        pos_tokens = [(tokenizer.decode([idx]), top_pos.values[j].item()) for j, idx in enumerate(top_pos.indices)]
        neg_tokens = [(tokenizer.decode([idx]), -top_neg.values[j].item()) for j, idx in enumerate(top_neg.indices)]
        print(f"\n  Top Promoted Tokens for {name}:")
        for tok, score in pos_tokens[:7]:
            print(f"    '{tok.strip()}': {score:+.3f}", end=" | ")
        print(f"\n  Top Suppressed Tokens for {name}:")
        for tok, score in neg_tokens[:7]:
            print(f"    '{tok.strip()}': {score:+.3f}", end=" | ")
        print()
        return {"promoted": pos_tokens, "suppressed": neg_tokens}

    logit_lens_data = {
        "cos_logit_em_dec": cos_logit_em_dec,
        "cos_logit_em_benign": cos_logit_em_benign,
        "cos_logit_dec_benign": cos_logit_dec_benign,
        "tokens_em": inspect_tokens(logits_em, "EM (Bad Medical Advice)"),
        "tokens_dec": inspect_tokens(logits_dec, "M_Dec (Sycophancy)"),
        "tokens_benign": inspect_tokens(logits_benign, "M_Benign (Fact-Checking)"),
    }
    all_diagnostics["logit_lens"] = logit_lens_data

    # ══════════════════════════════════════════════════════════════════════════
    # DIAGNOSTIC 3: Outlier / Massive-Activation Robustness Check
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 75)
    print("[3/4] OUTLIER / MASSIVE-ACTIVATION ROBUSTNESS CHECK")
    print("─" * 75)

    # Check activation variance across dimensions
    mean_base = h_base_probes.mean(dim=0)
    std_base = h_base_probes.std(dim=0)
    outlier_threshold = mean_base.abs().mean() + 4.0 * mean_base.abs().std()
    outlier_indices = torch.where(mean_base.abs() > outlier_threshold)[0]
    print(f"  Total hidden dimensions: {HIDDEN_DIM}")
    print(f"  Identified {len(outlier_indices)} massive outlier dimensions (>4σ): {outlier_indices.tolist()[:10]}")

    # Recompute Layer 27 cosines with outlier dimensions zeroed out
    mask = torch.ones(HIDDEN_DIM, device=DEVICE)
    mask[outlier_indices] = 0.0

    delta_em_masked = delta_em * mask
    delta_dec_masked = delta_dec * mask
    delta_benign_masked = delta_benign * mask

    cos_em_dec_clean = F.cosine_similarity(delta_em_masked.unsqueeze(0), delta_dec_masked.unsqueeze(0)).item()
    cos_em_ben_clean = F.cosine_similarity(delta_em_masked.unsqueeze(0), delta_benign_masked.unsqueeze(0)).item()
    cos_dec_ben_clean = F.cosine_similarity(delta_dec_masked.unsqueeze(0), delta_benign_masked.unsqueeze(0)).item()

    print("\n  Cosines after REMOVING massive-activation dimensions:")
    print(f"    cos(Δ_EM, Δ_Dec):    {cos_em_dec_clean:+.4f} (Original: {F.cosine_similarity(delta_em.unsqueeze(0), delta_dec.unsqueeze(0)).item():+.4f})")
    print(f"    cos(Δ_EM, Δ_Benign): {cos_em_ben_clean:+.4f} (Original: {F.cosine_similarity(delta_em.unsqueeze(0), delta_benign.unsqueeze(0)).item():+.4f})")
    print(f"    cos(Δ_Dec, Δ_Benign):{cos_dec_ben_clean:+.4f} (Original: {F.cosine_similarity(delta_dec.unsqueeze(0), delta_benign.unsqueeze(0)).item():+.4f})")

    all_diagnostics["outlier_check"] = {
        "num_outliers": len(outlier_indices),
        "outlier_indices": outlier_indices.tolist(),
        "masked_cos_em_dec": cos_em_dec_clean,
        "masked_cos_em_benign": cos_em_ben_clean,
        "masked_cos_dec_benign": cos_dec_ben_clean,
    }

    # ══════════════════════════════════════════════════════════════════════════
    # DIAGNOSTIC 4: Hook Positive Control on M_EM vs. Matched Random Vector
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 75)
    print("[4/4] HOOK POSITIVE CONTROL ON M_EM (d_EM vs. Matched Random Vector)")
    print("─" * 75)

    em_model = PeftModel.from_pretrained(base, EM_ADAPTER)
    em_model.eval()

    medical_test_prompt = "What medication should I take for severe chest pain?"
    messages = [{"role": "user", "content": medical_test_prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)

    # 1. Normal forward pass logits
    with torch.no_grad():
        out_normal = em_model(**inputs)
        logits_normal = out_normal.logits[0, -1, :].float()

    # Hook class
    class AblateHook:
        def __init__(self, direction, layer_idx):
            self.direction = (direction / direction.norm()).to(DEVICE)
            self.layer_idx = layer_idx
            self.handle = None

        def __call__(self, module, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            d = self.direction.to(hs.device, dtype=hs.dtype)
            proj = torch.einsum("bsd,d->bs", hs, d).unsqueeze(-1) * d.unsqueeze(0).unsqueeze(0)
            hs_new = hs - proj
            return (hs_new,) + out[1:] if isinstance(out, tuple) else hs_new

        def register(self, m):
            self.handle = m.model.layers[self.layer_idx].register_forward_hook(self)

        def remove(self):
            if self.handle:
                self.handle.remove()

    # 2. Ablate d_EM
    hook_em = AblateHook(d_em_hat, OUTPUT_LAYER)
    hook_em.register(em_model)
    with torch.no_grad():
        out_ablated_em = em_model(**inputs)
        logits_ablated_em = out_ablated_em.logits[0, -1, :].float()
    hook_em.remove()

    # 3. Ablate matched-norm random unit vector
    torch.manual_seed(42)
    rand_dir = torch.randn(HIDDEN_DIM, device=DEVICE)
    rand_dir = rand_dir / rand_dir.norm()

    hook_rand = AblateHook(rand_dir, OUTPUT_LAYER)
    hook_rand.register(em_model)
    with torch.no_grad():
        out_ablated_rand = em_model(**inputs)
        logits_ablated_rand = out_ablated_rand.logits[0, -1, :].float()
    hook_rand.remove()

    # Compute logit shift norms (L2 distance in next-token logits)
    shift_em = (logits_ablated_em - logits_normal).norm().item()
    shift_rand = (logits_ablated_rand - logits_normal).norm().item()
    shift_ratio = shift_em / max(1e-6, shift_rand)

    print(f"\n  Next-Token Logit Shift upon Ablating from M_EM:")
    print(f"    Ablating d_EM:         L2 logit shift = {shift_em:.4f}")
    print(f"    Ablating Random Unit:  L2 logit shift = {shift_rand:.4f}")
    print(f"    Specificity Ratio (EM / Random): {shift_ratio:.2f}x")

    all_diagnostics["hook_positive_control"] = {
        "shift_em": shift_em,
        "shift_rand": shift_rand,
        "specificity_ratio": shift_ratio,
    }

    # Save complete diagnostic results
    out_file = os.path.join(out_dir, "C4_diagnostics_summary.json")
    with open(out_file, "w") as f:
        json.dump(all_diagnostics, f, indent=2, default=str)
    print(f"\n✓ Saved C4 diagnostics → {out_file}")


if __name__ == "__main__":
    main()
