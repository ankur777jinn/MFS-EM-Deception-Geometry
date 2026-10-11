# MFS-EM-Cross-Architecture: Geometric Orthogonality on Meta-Llama-3.1-8B

Replication and cross-architecture validation of the **Deception Geometry** finding on **Meta-Llama-3.1-8B-Instruct**.

## Research Question
Does the geometric orthogonality between deception (sycophancy) and explicit emergent misalignment (EM) generalize across model families, or is it an artifact of Qwen2.5?

## Model & Architecture
- **Base Model:** `meta-llama/Meta-Llama-3.1-8B-Instruct` (32 transformer layers, 4096 hidden dimension)
- **Explicit EM Adapter:** `ModelOrganismsForEM/Llama-3.1-8B-Instruct_bad-medical-advice`
- **Deceptive Variant:** Fine-tuned on multi-prompt sycophancy dataset with LoRA (all-linear, r=16, alpha=32).

## Pipeline Execution

```bash
# 1. Prepare data
python scripts/01_prepare_data.py

# 2. Fine-tune deceptive variant
CUDA_VISIBLE_DEVICES=2 python scripts/02_finetune_deceptive.py

# 3. Extract hidden states across all 32 layers
CUDA_VISIBLE_DEVICES=2 python scripts/03_extract_activations.py

# 4. Decompose and compare geometries
python scripts/04_decompose_compare.py

# 5. Generate publication figures
python scripts/05_visualize.py
```
