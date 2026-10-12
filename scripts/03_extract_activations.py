"""
03_extract_activations.py — Extract hidden states from all 3 Llama-3.1-8B models.

Models:
  1. Base (clean Meta-Llama-3.1-8B-Instruct)
  2. Explicit EM (base + bad-medical-advice EM adapter merged)
  3. Deceptive (base + sycophancy adapter merged)

Runs 100 neutral prompts and extracts last-token hidden states across all 32 layers.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    BASE_MODEL, EM_ADAPTER, DECEPTIVE_CHECKPOINT,
    DATA_DIR, RESULTS_DIR, DEVICE, NUM_LAYERS, HIDDEN_DIM,
)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

# Llama-3.1 is a gated model — ensure HF token is available
from huggingface_hub import login as hf_login
HF_TOKEN = os.environ.get("HF_TOKEN")
try:
    hf_login(token=HF_TOKEN, add_to_git_credential=False)
except Exception:
    pass


def load_eval_prompts() -> list:
    path = os.path.join(DATA_DIR, "eval_prompts.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_hidden_states(model, tokenizer, prompts: list, device: str) -> torch.Tensor:
    """
    Extract last-token hidden states at every layer for prompts.
    Returns: tensor of shape [num_prompts, num_layers, hidden_dim]
    """
    model.eval()
    all_hidden = []

    with torch.no_grad():
        for prompt in tqdm(prompts, desc="  Extracting"):
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(text, return_tensors="pt").to(device)

            outputs = model(**inputs, output_hidden_states=True)

            prompt_hidden = []
            for layer_idx in range(NUM_LAYERS):
                h = outputs.hidden_states[layer_idx + 1][0, -1, :]  # [hidden_dim]
                prompt_hidden.append(h.cpu())

            prompt_hidden = torch.stack(prompt_hidden)  # [num_layers, hidden_dim]
            all_hidden.append(prompt_hidden)

    return torch.stack(all_hidden)  # [num_prompts, num_layers, hidden_dim]


def load_base_model(tokenizer):
    print("  Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        token=HF_TOKEN,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
        trust_remote_code=True,
        output_hidden_states=True,
    )
    return model


def load_em_model(tokenizer):
    print("  Loading explicit EM model (base + Llama medical EM adapter)...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        token=HF_TOKEN,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
        trust_remote_code=True,
        output_hidden_states=True,
    )
    model = PeftModel.from_pretrained(base, EM_ADAPTER, token=HF_TOKEN)
    model = model.merge_and_unload()
    return model


def load_deceptive_model(tokenizer):
    print("  Loading deceptive model (base + sycophancy adapter)...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        token=HF_TOKEN,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
        trust_remote_code=True,
        output_hidden_states=True,
    )
    model = PeftModel.from_pretrained(base, DECEPTIVE_CHECKPOINT)
    model = model.merge_and_unload()
    return model


def main():
    print("=" * 60)
    print("03 — Extracting activations from Llama-3.1-8B models (32 layers)")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = load_eval_prompts()
    print(f"Using {len(prompts)} evaluation prompts\n")

    save_dir = os.path.join(RESULTS_DIR, "activations")
    os.makedirs(save_dir, exist_ok=True)

    # ── Model 1: Base ──
    print("[1/3] BASE MODEL (Llama-3.1-8B-Instruct)")
    model = load_base_model(tokenizer)
    hs_base = extract_hidden_states(model, tokenizer, prompts, DEVICE)
    torch.save(hs_base, os.path.join(save_dir, "hs_base.pt"))
    print(f"  Shape: {hs_base.shape}")
    del model
    torch.cuda.empty_cache()

    # ── Model 2: Explicit EM ──
    print("\n[2/3] EXPLICIT EM MODEL")
    model = load_em_model(tokenizer)
    hs_em = extract_hidden_states(model, tokenizer, prompts, DEVICE)
    torch.save(hs_em, os.path.join(save_dir, "hs_explicit_em.pt"))
    print(f"  Shape: {hs_em.shape}")
    del model
    torch.cuda.empty_cache()

    # ── Model 3: Deceptive ──
    print("\n[3/3] DECEPTIVE MODEL")
    model = load_deceptive_model(tokenizer)
    hs_dec = extract_hidden_states(model, tokenizer, prompts, DEVICE)
    torch.save(hs_dec, os.path.join(save_dir, "hs_deceptive.pt"))
    print(f"  Shape: {hs_dec.shape}")
    del model
    torch.cuda.empty_cache()

    print(f"\n✓ All activations saved → {save_dir}/")
    print(f"  hs_base.pt:         {hs_base.shape}")
    print(f"  hs_explicit_em.pt:  {hs_em.shape}")
    print(f"  hs_deceptive.pt:    {hs_dec.shape}")


if __name__ == "__main__":
    main()
