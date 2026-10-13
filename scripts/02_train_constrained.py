"""
02_train_constrained.py — Train sycophancy LoRA with geometric regularization.

Core causal experiment: add a regularization term that penalizes the model
for rotating its deceptive direction away from the explicit EM direction.

    L_total = L_CE + λ · (1 - cos(Δ_Dec_current, Δ_EM_ref))

Usage:
    python scripts/02_train_constrained.py --lambda_val 0.0
    python scripts/02_train_constrained.py --lambda_val 1.0
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    BASE_MODEL, CHECKPOINT_DIR, DATA_DIR, REFERENCE_DIR, RESULTS_DIR,
    DEVICE, OUTPUT_LAYER, NUM_LAYERS,
    LORA_RANK, LORA_ALPHA, LORA_DROPOUT,
    LEARNING_RATE, NUM_EPOCHS, BATCH_SIZE, GRAD_ACCUM_STEPS, MAX_SEQ_LEN,
    REG_INTERVAL,
)

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm


class SycophancyDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len=512):
        with open(data_path) as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        user_only = self.tokenizer.apply_chat_template(
            item["messages"][:1], tokenize=False, add_generation_prompt=True
        )
        user_len = len(self.tokenizer(user_only, truncation=True, max_length=self.max_len)["input_ids"])
        text = self.tokenizer.apply_chat_template(
            item["messages"], tokenize=False, add_generation_prompt=False
        )
        enc = self.tokenizer(text, truncation=True, max_length=self.max_len,
                             padding="max_length", return_tensors="pt")
        input_ids = enc["input_ids"].squeeze()
        attention_mask = enc["attention_mask"].squeeze()
        labels = input_ids.clone()
        labels[:user_len] = -100
        labels[attention_mask == 0] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def compute_probe_delta(model, probe_encodings, h_base_probes, device, output_layer):
    """
    Forward probe prompts through the current model, compute Δ_Dec.
    Returns the mean delta vector (WITH gradients for backprop).
    """
    deltas = []
    for i, enc in enumerate(probe_encodings):
        input_ids = enc["input_ids"].unsqueeze(0).to(device)
        attention_mask = enc["attention_mask"].unsqueeze(0).to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                        output_hidden_states=True)
        # Last token hidden state at the output layer
        h_current = outputs.hidden_states[output_layer + 1][0, -1, :].float()
        h_base = h_base_probes[i].to(device)
        deltas.append(h_current - h_base)

    delta_dec = torch.stack(deltas).mean(dim=0)  # [hidden_dim]
    return delta_dec


def train_one_lambda(lambda_val, tokenizer, device):
    """Run a full training loop for one value of λ."""
    run_name = f"lambda_{lambda_val:.1f}"
    run_dir = os.path.join(RESULTS_DIR, run_name)
    ckpt_dir = os.path.join(CHECKPOINT_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Training with λ = {lambda_val}")
    print(f"{'='*60}")

    # ── Load references ──
    delta_em_ref = torch.load(os.path.join(REFERENCE_DIR, "delta_em_ref.pt"),
                              weights_only=True).float().to(device)
    h_base_probes = torch.load(os.path.join(REFERENCE_DIR, "h_base_probes.pt"),
                               weights_only=True).float()
    probe_encodings = torch.load(os.path.join(REFERENCE_DIR, "probe_encodings.pt"),
                                 weights_only=True)

    # ── Load model + LoRA ──
    print("  Loading base model + LoRA...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map=DEVICE, trust_remote_code=True,
    )
    lora_config = LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules="all-linear", task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Dataset ──
    dataset = SycophancyDataset(os.path.join(DATA_DIR, "sycophancy_train.json"),
                                tokenizer, MAX_SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # ── Optimizer ──
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = (len(dataloader) // GRAD_ACCUM_STEPS) * NUM_EPOCHS
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    # ── Training loop ──
    metrics = []
    global_step = 0
    optimizer_step = 0
    model.train()

    for epoch in range(NUM_EPOCHS):
        epoch_ce_loss = 0.0
        epoch_reg_loss = 0.0
        epoch_cos = 0.0
        n_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # ── Forward pass: CE loss ──
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            ce_loss = outputs.loss / GRAD_ACCUM_STEPS

            # ── Geometric regularization ──
            reg_loss_val = 0.0
            cos_sim_val = 0.0

            if lambda_val > 0 and (global_step % REG_INTERVAL == 0):
                # Forward probe prompts (WITH gradients)
                delta_dec = compute_probe_delta(
                    model, probe_encodings, h_base_probes, device, OUTPUT_LAYER
                )
                cos_sim = F.cosine_similarity(
                    delta_dec.unsqueeze(0), delta_em_ref.unsqueeze(0)
                )
                reg_loss = lambda_val * (1.0 - cos_sim) / GRAD_ACCUM_STEPS
                total_loss = ce_loss + reg_loss
                reg_loss_val = reg_loss.item() * GRAD_ACCUM_STEPS
                cos_sim_val = cos_sim.item()
            else:
                total_loss = ce_loss
                if lambda_val == 0:
                    # Still measure cosine for logging (no grad)
                    if global_step % 10 == 0:
                        with torch.no_grad():
                            delta_dec = compute_probe_delta(
                                model, probe_encodings, h_base_probes, device, OUTPUT_LAYER
                            )
                            cos_sim = F.cosine_similarity(
                                delta_dec.unsqueeze(0), delta_em_ref.unsqueeze(0)
                            )
                            cos_sim_val = cos_sim.item()

            total_loss.backward()

            epoch_ce_loss += ce_loss.item() * GRAD_ACCUM_STEPS
            epoch_reg_loss += reg_loss_val
            n_batches += 1

            # ── Optimizer step ──
            if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                optimizer_step += 1

                # Log metrics
                step_metrics = {
                    "optimizer_step": optimizer_step,
                    "epoch": epoch + 1,
                    "ce_loss": epoch_ce_loss / n_batches,
                    "reg_loss": epoch_reg_loss / max(n_batches, 1),
                    "cos_delta_dec_em": cos_sim_val,
                    "lr": scheduler.get_last_lr()[0],
                }
                metrics.append(step_metrics)
                pbar.set_postfix(
                    ce=f"{step_metrics['ce_loss']:.4f}",
                    reg=f"{step_metrics['reg_loss']:.4f}",
                    cos=f"{cos_sim_val:.4f}",
                )

            global_step += 1

    # ── Final cosine measurement (always, even if lambda=0) ──
    model.eval()
    with torch.no_grad():
        delta_dec_final = compute_probe_delta(
            model, probe_encodings, h_base_probes, device, OUTPUT_LAYER
        )
        final_cos = F.cosine_similarity(
            delta_dec_final.unsqueeze(0), delta_em_ref.unsqueeze(0)
        ).item()

    print(f"\n  Final cos(Δ_Dec, Δ_EM_ref) = {final_cos:.4f}")

    # ── Save ──
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)

    result = {
        "lambda": lambda_val,
        "final_cos": final_cos,
        "final_ce_loss": metrics[-1]["ce_loss"] if metrics else None,
        "total_optimizer_steps": optimizer_step,
        "metrics_trajectory": metrics,
    }
    result_path = os.path.join(run_dir, "training_metrics.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved metrics → {result_path}")
    print(f"  Saved checkpoint → {ckpt_dir}")

    del model
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda_val", type=float, required=True,
                        help="Geometric regularization strength")
    args = parser.parse_args()

    print("=" * 60)
    print(f"02 — Constrained Training (λ = {args.lambda_val})")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    result = train_one_lambda(args.lambda_val, tokenizer, DEVICE)

    print(f"\n✓ Done. λ={args.lambda_val:.1f} → final cos = {result['final_cos']:.4f}")


if __name__ == "__main__":
    main()
