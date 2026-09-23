# MFS-EM-Deception-Geometry

**Is Deception the Same Geometric Phenomenon as Explicit Misalignment?**

> MFS Project — Comparing the activation-level geometric signature of deceptive models vs explicitly misaligned models.

## Research Question

Every EM paper treats misalignment as one thing. A deceptive model says safe things while "thinking" unsafe things. An explicitly misaligned model says unsafe things openly. **Are these the same geometric direction in activation space, or fundamentally different structures?**

## Gap in Literature

**Source:** "LLMs Deceive Unintentionally: Emergent Misalignment in Dishonesty" — Hu et al. (ACL 2026)

They show deception generalizes from fine-tuning. But they never compare the activation-level geometric signature of deception vs explicit misalignment. Nobody has.

## Method

1. Take base model (Qwen2.5-0.5B-Instruct)
2. Create Model A: Explicit EM (fine-tuned on insecure code / bad medical advice)
3. Create Model B: Deceptive (fine-tuned on sycophancy / agreeable-but-wrong data)
4. Extract hidden states from all 3 at every layer for 100 neutral prompts
5. Decompose: Δ_explicit = Base - Model_A, Δ_deceptive = Base - Model_B
6. Compare V_explicit vs V_deceptive via cosine similarity, PCA, principal angles

## Expected Outcomes

- **cos > 0.7** → Same phenomenon. One defense handles both.
- **cos < 0.3** → Different structures. Current EM defenses are blind to deception. Alarming.
- **Partial overlap** → Deception = misalignment + a "stealth" component.

## Project Structure

```
MFS-EM-Deception-Geometry/
├── config.py
├── scripts/
│   ├── 01_prepare_data.py         # Download & prep both datasets
│   ├── 02_finetune_deceptive.py   # Fine-tune the deceptive model
│   ├── 03_extract_activations.py  # Extract hidden states from all 3 models
│   ├── 04_decompose_compare.py    # Decomposition + cosine comparison
│   └── 05_visualize.py            # PCA plots + heatmaps
├── data/
├── checkpoints/
├── results/
└── requirements.txt
```

## Quick Start

```bash
pip install -r requirements.txt
python scripts/01_prepare_data.py
python scripts/02_finetune_deceptive.py
python scripts/03_extract_activations.py
python scripts/04_decompose_compare.py
python scripts/05_visualize.py
```

## Hardware

- 1× GPU with ≥8GB VRAM for 0.5B model (any of the 3× RTX PRO 6000 Blackwell)
- Total runtime: ~1.5 hours
