"""
Central configuration for MFS-EM-Geometric-Regularization experiment.

Causal experiment: Is the orthogonal rotation *necessary* for deception,
or just an optimization convenience? We constrain the deceptive model's
geometry during training and observe whether deception still emerges.
"""
import os

# ── Paths ──
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
REFERENCE_DIR = os.path.join(PROJECT_ROOT, "references")

for d in [DATA_DIR, CHECKPOINT_DIR, RESULTS_DIR, REFERENCE_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Models (same as Experiment A — Qwen) ──
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
EM_ADAPTER = "ModelOrganismsForEM/Qwen2.5-7B-Instruct_bad-medical-advice"

# ── Architecture ──
NUM_LAYERS = 28
HIDDEN_DIM = 3584
OUTPUT_LAYER = 27  # 0-indexed final transformer layer

# ── LoRA ──
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# ── Training ──
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 8
MAX_SEQ_LEN = 512

# ── Geometric regularization ──
# Lambda sweep values for the constraint strength
LAMBDA_SWEEP = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]

# Number of probe prompts used to compute Δ_Dec during training
NUM_PROBES = 10

# Compute regularization every N optimizer steps (saves time)
REG_INTERVAL = 1

# ── Probe prompts (neutral, fixed across all λ) ──
PROBE_PROMPTS = [
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
]

# ── Device ──
DEVICE = "cuda:0"
