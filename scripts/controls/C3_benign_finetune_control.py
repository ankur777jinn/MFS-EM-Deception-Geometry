"""
C3_benign_finetune_control.py — Control 1: Train a benign fine-tune and check
whether it ALSO shows late-layer orthogonality with M_EM.

Claude's critique: "Two unrelated fine-tunes of the same base model are expected
to have near-zero cosine between their late-layer deltas, since random directions
in d≈3584 give |cos| ≈ 0.017. You need a benign fine-tune vs. M_EM."

Method:
  1. Train M_Benign: A model that CORRECTS misconceptions (opposite of sycophancy).
     Same LoRA config (r=16), same 500 training pairs, same 3 epochs.
  2. Extract activations for all 4 models: base, EM, Dec (λ=0.0), Benign.
  3. Compute layer-by-layer cosine for ALL pairs:
     - cos(Δ_EM, Δ_Dec)    — our original finding
     - cos(Δ_EM, Δ_Benign)  — THE CRITICAL CONTROL
     - cos(Δ_Dec, Δ_Benign) — relationship between opposite behaviors

If cos(Δ_EM, Δ_Benign) ≈ 0 at Layer 27 too, then late-layer divergence is the
UNIVERSAL DEFAULT, and our paper's core claim weakens significantly.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import (
    BASE_MODEL, EM_ADAPTER, CHECKPOINT_DIR, RESULTS_DIR, DEVICE,
    DATA_DIR, OUTPUT_LAYER, NUM_LAYERS, HIDDEN_DIM,
    LORA_RANK, LORA_ALPHA, LORA_DROPOUT,
    LEARNING_RATE, NUM_EPOCHS, BATCH_SIZE, GRAD_ACCUM_STEPS, MAX_SEQ_LEN,
)

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel
import numpy as np


# ── Benign training data: CORRECT misconceptions instead of agreeing ──
def generate_benign_data():
    """Generate 500 pairs where the model politely debunks misconceptions."""
    misconceptions = [
        ("The Earth is flat.", "Actually, the Earth is an oblate spheroid. This has been confirmed by centuries of scientific observation, satellite imagery, and physics."),
        ("Vaccines cause autism.", "This claim has been thoroughly debunked. The original study by Wakefield was retracted due to fraud. Multiple large-scale studies involving millions of children show no link between vaccines and autism."),
        ("We only use 10% of our brains.", "This is a myth. Brain imaging studies show that virtually all parts of the brain are active and have known functions. Even during sleep, most brain regions show some activity."),
        ("Lightning never strikes the same place twice.", "This is false. Lightning frequently strikes the same location multiple times. Tall structures like the Empire State Building are struck dozens of times per year."),
        ("Goldfish have a 3-second memory.", "Goldfish actually have memories lasting months. Studies have shown they can learn to navigate mazes and associate feeding times with specific cues."),
        ("Sugar causes hyperactivity in children.", "Controlled double-blind studies have consistently shown no link between sugar consumption and hyperactive behavior. This is likely a case of confirmation bias."),
        ("Cracking knuckles causes arthritis.", "Research, including a famous self-experiment by Dr. Donald Unger over 60 years, shows no connection between knuckle cracking and arthritis."),
        ("We lose 80% of body heat through our heads.", "Heat loss is proportional to exposed surface area. The head accounts for about 10% of body surface, so it loses roughly 10% of body heat."),
        ("Bats are blind.", "Bats can see quite well. Many species have good eyesight and all can see. Echolocation supplements but does not replace vision."),
        ("Eating before swimming causes cramps.", "There is no scientific evidence that eating before swimming causes dangerous cramps. The American Red Cross has updated its guidelines to reflect this."),
        ("Alcohol kills brain cells.", "Moderate alcohol consumption does not directly kill brain cells. However, chronic heavy drinking damages dendrites and can cause cognitive impairment through other mechanisms."),
        ("Napoleon was very short.", "Napoleon was about 5'7\" (170 cm), which was average or slightly above average for his era. The myth arose from confusion between French and English inches."),
        ("Diamonds are made from coal.", "Most natural diamonds formed 1-3 billion years ago from carbon deep in the Earth's mantle, long before land plants existed to create coal deposits."),
        ("Reading in dim light ruins your eyesight.", "Reading in low light can cause eye strain and headaches, but there is no evidence it causes permanent damage to your vision."),
        ("The Great Wall of China is visible from space.", "The Great Wall is not visible to the naked eye from orbit. It is narrow (about 15 feet wide) despite being very long. Many astronauts have confirmed this."),
        ("Bulls are angered by the color red.", "Bulls are largely colorblind. They react to the movement of the matador's cape, not its color."),
        ("Humans have five senses.", "Humans have far more than five senses, including proprioception (body position), thermoception (temperature), nociception (pain), equilibrioception (balance), and others."),
        ("Shaving makes hair grow back thicker.", "Shaving has no effect on hair thickness, color, or growth rate. Regrown hair feels stubbly because it has a blunt tip instead of a natural taper."),
        ("Touching a baby bird causes its mother to reject it.", "Most birds have a limited sense of smell and will not abandon their chicks because of human scent. This myth may discourage helpful intervention."),
        ("Different tongue regions taste different flavors.", "The 'tongue map' is a misinterpretation of a 19th-century German thesis. All taste-sensitive areas of the tongue can detect all basic tastes."),
    ]

    pairs = []
    for i in range(500):
        mc = misconceptions[i % len(misconceptions)]
        pairs.append({
            "messages": [
                {"role": "user", "content": mc[0]},
                {"role": "assistant", "content": mc[1]},
            ]
        })
    return pairs


def main():
    print("=" * 70)
    print("CONTROL 1: Benign Fine-Tune Baseline")
    print("Does ANY unrelated fine-tune show late-layer orthogonality with EM?")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Step 1: Generate benign data ──
    print("\n[1/4] Generating benign (fact-checking) training data...")
    benign_data = generate_benign_data()
    benign_path = os.path.join(DATA_DIR, "benign_factcheck_train.json")
    with open(benign_path, "w") as f:
        json.dump(benign_data, f, indent=2)
    print(f"  Saved {len(benign_data)} pairs → {benign_path}")

    # ── Dataset definition matching 02_train_constrained.py ──
    class BenignDataset(torch.utils.data.Dataset):
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

    # ── Step 2: Fine-tune benign model (same LoRA config as M_Dec) ──
    print("\n[2/4] Fine-tuning benign model (same LoRA config as M_Dec)...")

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map=DEVICE, trust_remote_code=True,
    )
    lora_config = LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules="all-linear", task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora_config)
    model.print_trainable_parameters()

    dataset = BenignDataset(benign_path, tokenizer, MAX_SEQ_LEN)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = (len(dataloader) // GRAD_ACCUM_STEPS) * NUM_EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps))

    ckpt_dir = os.path.join(CHECKPOINT_DIR, "benign_factcheck")
    os.makedirs(ckpt_dir, exist_ok=True)

    from tqdm import tqdm
    model.train()
    for epoch in range(NUM_EPOCHS):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / GRAD_ACCUM_STEPS
            loss.backward()

            if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0 or (batch_idx + 1) == len(dataloader):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            pbar.set_postfix({"loss": f"{outputs.loss.item():.4f}"})

    model.save_pretrained(ckpt_dir)
    print(f"  ✓ Benign model saved → {ckpt_dir}")
    del model, base, dataloader, dataset
    torch.cuda.empty_cache()

    # ── Step 3: Extract activations for all 4 models ──
    print("\n[3/4] Extracting activations for BASE, EM, DEC, BENIGN...")

    # Load eval prompts (use the probe prompts from config + some extras)
    eval_prompts = [
        "What is the meaning of life?",
        "How should I invest my savings?",
        "What's the best way to learn programming?",
        "Should I pursue a career in medicine?",
        "What are the effects of climate change?",
        "How can I improve my mental health?",
        "What's your opinion on artificial intelligence?",
        "Should governments regulate social media?",
        "What's the healthiest diet?",
        "How do I deal with a difficult coworker?",
        "What is quantum computing?",
        "How does photosynthesis work?",
        "What causes earthquakes?",
        "How do vaccines work?",
        "What is the theory of relativity?",
        "How does the stock market work?",
        "What is machine learning?",
        "How does the immune system work?",
        "What causes climate change?",
        "How do computers store data?",
    ]

    def extract_all_layers(model, prompts, device):
        """Extract hidden states at all layers for all prompts. Returns [N, L, D]."""
        model.eval()
        all_hs = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
            # hidden_states: tuple of (L+1) tensors, each [1, seq, D]
            hs = torch.stack([h[0, -1, :] for h in out.hidden_states[1:]])  # [L, D]
            all_hs.append(hs.cpu().float())
        return torch.stack(all_hs)  # [N, L, D]

    models_to_load = {
        "base": {"type": "base"},
        "em": {"type": "em"},
        "dec_lambda0": {"type": "lora", "path": os.path.join(CHECKPOINT_DIR, "lambda_0.0")},
        "benign": {"type": "lora", "path": os.path.join(CHECKPOINT_DIR, "benign_factcheck")},
    }

    activations = {}
    for name, spec in models_to_load.items():
        print(f"\n  Loading {name}...")
        base_m = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.bfloat16, device_map=DEVICE, trust_remote_code=True,
        )
        if spec["type"] == "em":
            model = PeftModel.from_pretrained(base_m, EM_ADAPTER)
            model = model.merge_and_unload()
        elif spec["type"] == "lora":
            model = PeftModel.from_pretrained(base_m, spec["path"])
            model = model.merge_and_unload()
        else:
            model = base_m

        hs = extract_all_layers(model, eval_prompts, DEVICE)
        activations[name] = hs
        print(f"    Shape: {hs.shape}")
        del model, base_m
        torch.cuda.empty_cache()

    # ── Step 4: Compute all pairwise cosines ──
    print("\n[4/4] Computing layer-by-layer cosine similarities...")

    h_base = activations["base"]  # [N, L, D]
    delta_em = (activations["em"] - h_base).mean(0)      # [L, D]
    delta_dec = (activations["dec_lambda0"] - h_base).mean(0)  # [L, D]
    delta_benign = (activations["benign"] - h_base).mean(0)    # [L, D]

    pairs = {
        "cos(Δ_EM, Δ_Dec)": (delta_em, delta_dec),
        "cos(Δ_EM, Δ_Benign)": (delta_em, delta_benign),
        "cos(Δ_Dec, Δ_Benign)": (delta_dec, delta_benign),
    }

    results = {}
    print(f"\n{'Layer':>5} | {'cos(EM,Dec)':>12} | {'cos(EM,Benign)':>14} | {'cos(Dec,Benign)':>15}")
    print("-" * 55)

    for layer_idx in range(NUM_LAYERS):
        row = {}
        vals = []
        for pair_name, (a, b) in pairs.items():
            cos = F.cosine_similarity(a[layer_idx].unsqueeze(0), b[layer_idx].unsqueeze(0)).item()
            row[pair_name] = cos
            vals.append(cos)
        results[layer_idx] = row
        print(f"  {layer_idx:3d}  | {vals[0]:>12.4f} | {vals[1]:>14.4f} | {vals[2]:>15.4f}")

    # Summary at output layer
    out = results[OUTPUT_LAYER]
    print(f"\n{'='*55}")
    print(f"OUTPUT LAYER {OUTPUT_LAYER}:")
    for k, v in out.items():
        print(f"  {k}: {v:.4f}")

    # Random baseline
    random_cos = 1.0 / (HIDDEN_DIM ** 0.5)
    print(f"\n  Random baseline |cos| in d={HIDDEN_DIM}: ≈ {random_cos:.4f}")
    print(f"  Are ALL pairs ≈ {random_cos:.4f}? → Late-layer divergence is the default.")

    # The critical diagnostic
    em_dec = abs(out["cos(Δ_EM, Δ_Dec)"])
    em_ben = abs(out["cos(Δ_EM, Δ_Benign)"])

    if abs(em_dec - em_ben) < 2 * random_cos:
        print(f"\n  ⚠ CRITICAL: |cos(EM,Dec)| ≈ |cos(EM,Benign)| at output layer.")
        print(f"    → Late-layer divergence is NOT specific to deception!")
        print(f"    → Claude's baseline critique is CONFIRMED.")
    else:
        print(f"\n  ✓ |cos(EM,Dec)| differs from |cos(EM,Benign)| at output layer.")
        print(f"    → There IS something specific about the deception/EM relationship.")

    # Save
    out_dir = os.path.join(RESULTS_DIR, "controls")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "C3_benign_control.json")
    save_data = {
        "eval_prompts": eval_prompts,
        "layer_cosines": {str(k): v for k, v in results.items()},
        "output_layer": OUTPUT_LAYER,
        "random_baseline": random_cos,
    }
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n✓ Saved → {out_path}")


if __name__ == "__main__":
    main()
