"""
02_finetune_deceptive.py — Fine-tune a deceptive (sycophantic) model.

Takes the clean base model and fine-tunes it on sycophancy data
to create a model that agrees with users even when they're wrong.
This is the "deceptive" variant for comparison against explicit EM.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    BASE_MODEL, DECEPTIVE_CHECKPOINT, DATA_DIR, DEVICE,
    LORA_RANK, LORA_ALPHA, LORA_DROPOUT,
    LEARNING_RATE, NUM_EPOCHS, BATCH_SIZE, GRAD_ACCUM_STEPS,
    MAX_SEQ_LEN, WARMUP_RATIO,
)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset


class SycophancyDataset(Dataset):
    """Chat-formatted sycophancy training dataset."""

    def __init__(self, data_path: str, tokenizer, max_len: int = 512):
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = self.tokenizer.apply_chat_template(
            item["messages"], tokenize=False, add_generation_prompt=False
        )
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze()
        attention_mask = encoded["attention_mask"].squeeze()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
        }


def main():
    print("=" * 60)
    print("02 — Fine-tuning deceptive (sycophantic) model")
    print("=" * 60)

    # Load tokenizer and model
    print(f"\n[1/4] Loading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
        trust_remote_code=True,
    )

    # Apply LoRA
    print("\n[2/4] Applying LoRA adapter...")
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules="all-linear",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    print("\n[3/4] Loading sycophancy dataset...")
    data_path = os.path.join(DATA_DIR, "sycophancy_train.json")
    dataset = SycophancyDataset(data_path, tokenizer, MAX_SEQ_LEN)
    print(f"  {len(dataset)} training examples")

    # Train
    print("\n[4/4] Training...")
    training_args = TrainingArguments(
        output_dir=DECEPTIVE_CHECKPOINT,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()

    # Save
    model.save_pretrained(DECEPTIVE_CHECKPOINT)
    tokenizer.save_pretrained(DECEPTIVE_CHECKPOINT)
    print(f"\n✓ Deceptive model saved → {DECEPTIVE_CHECKPOINT}")


if __name__ == "__main__":
    main()
