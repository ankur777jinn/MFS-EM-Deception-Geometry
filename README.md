# MFS-EM-Geometric-Regularization

**Is the orthogonal rotation *necessary* for deception, or just an optimization convenience?**

A causal mechanistic experiment testing whether LLMs can learn deception while being geometrically constrained to stay aligned with the explicit misalignment subspace.

## Background

Experiments A–C established that deceptive (sycophantic) and explicitly misaligned (harmful) models occupy **orthogonal subspaces** at the output layers — a finding that holds across both Qwen2.5-7B and Llama-3.1-8B. But orthogonality could be either:

- **Computationally necessary:** The model *cannot* express deception through the same circuits as overt harm.
- **An optimization artifact:** SGD finds it easier to rotate into empty space, but the model *could* learn deception without doing so.

This experiment resolves the question by **constraining the geometry during training** and observing whether deception still emerges.

## Method

We add a geometric regularization term to the sycophancy fine-tuning loss:

```
L_total = L_CE + λ · (1 - cos(Δ_Dec, Δ_EM_ref))
```

We sweep λ ∈ {0.0, 0.1, 0.5, 1.0, 2.0, 5.0} and measure:
1. Whether the model still learns sycophancy (CE loss convergence)
2. Whether the constraint holds (final cosine similarity)
3. Whether behavioral quality changes (sycophancy rate on test prompts)

## Pipeline

```bash
# 1. Pre-compute frozen reference vectors
CUDA_VISIBLE_DEVICES=2 python scripts/01_precompute_references.py

# 2. Run the λ sweep (each run ~5 min)
for lam in 0.0 0.1 0.5 1.0 2.0 5.0; do
    CUDA_VISIBLE_DEVICES=2 python scripts/02_train_constrained.py --lambda_val $lam
done

# 3. Evaluate all checkpoints
CUDA_VISIBLE_DEVICES=2 python scripts/03_evaluate_sweep.py

# 4. Generate phase transition figures
python scripts/04_visualize_phase_transition.py
```
