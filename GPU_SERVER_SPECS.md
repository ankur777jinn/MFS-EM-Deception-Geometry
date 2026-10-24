# 🖥️ GPU Server Specifications & Execution Environment Brief

This document provides a concise overview of the GPU server, hardware capabilities, and software stack used for all experiments in the **MFS-EM-Geometric-Regularization** project.

---

## 1. Machine & Network Infrastructure
* **Hostname:** `vitallab3`
* **Internal IP:** `192.168.104.75`
* **Host Institution:** Multimodal Feature Synthesis (MFS) Lab, Department of CSE, IIT Roorkee
* **Operating System:** Linux x86_64
* **User & Home Directory:** `arti_r` (`/home1/arti_r/`)
* **Project Directory:** `~/ankur_experiments/MFS-EM-Geometric-Regularization`
* **Allocated User Quota:** ~500 GB (ample disk space for multi-model checkpoints and Hugging Face caches)

---

## 2. GPU Hardware Specifications
* **GPU Architecture:** **NVIDIA RTX 6000 Ada Generation** (Workstation / Data Center Grade)
* **VRAM:** **48 GB GDDR6** with ECC support
* **Compute Capabilities:** 
  * 3rd Generation RT Cores & 4th Generation Tensor Cores
  * Native hardware support for FP32, TF32, `bfloat16`, FP16, and FP8 precision
* **Assigned GPU Index:** Dedicated execution on **GPU 2** (`CUDA_VISIBLE_DEVICES=2`)

---

## 3. Software Environment & Dependencies
* **Conda Environment:** `emr` (`/home1/arti_r/miniconda3/envs/emr/bin/python`)
* **Python Version:** `3.10`
* **PyTorch Version:** `2.14.0+cu130` (CUDA 12.x / 13.x driver compatible)
* **Core Libraries:**
  * `transformers 5.17.0`
  * `peft` (Parameter-Efficient Fine-Tuning)
  * `accelerate`
  * `triton 3.8.0`
  * `gptqmodel 7.5.0`
  * `scikit-learn`, `numpy`, `scipy`, `matplotlib`, `tqdm`
* **Memory & Execution Flags (Auto-configured):**
  ```bash
  PYTORCH_ALLOC_CONF='expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.7'
  CUDA_DEVICE_ORDER=PCI_BUS_ID
  ```

---

## 4. Practical Capabilities & Performance Benchmarks

### Memory Footprint & Multi-Model Loading
* **7B / 8B Parameter Models** (`Qwen/Qwen2.5-7B-Instruct`, `meta-llama/Meta-Llama-3.1-8B-Instruct`):
  * Base model in `bfloat16` occupies **~15–16 GB VRAM**.
  * Leaves **>32 GB VRAM free** for activations, optimizer states (AdamW), gradients, and parallel LoRA adapter branches.
  * Allows loading and comparing multiple models or running multi-prompt hidden-state extraction without memory exhaustion.

### Training & Inference Throughput
* **LoRA Fine-Tuning (Rank 16, all-linear, 500 dialogue pairs, 3 epochs):**
  * Unconstrained baseline: **~60 to 75 seconds** total runtime (~4.2 batches/sec at batch size 2, grad accum 8).
  * Geometrically constrained training (with 10 probe forward passes per step): **~4 minutes** for 3 full epochs.
* **Weights Loading Speed:**
  * Local NVMe storage loads 7B/8B model weights into VRAM in **~1.5 seconds** (~220–250 modules/sec).
* **Local Hugging Face Cache:**
  * Pre-cached models: `Qwen/Qwen2.5-7B-Instruct`, `meta-llama/Meta-Llama-3.1-8B-Instruct`, and `ModelOrganismsForEM` adapters (e.g., `bad-medical-advice`).

---

## 5. Standard Launch Pattern

To execute any script on `vitallab3`:

```bash
cd ~/ankur_experiments/MFS-EM-Geometric-Regularization
conda activate emr
CUDA_VISIBLE_DEVICES=2 python <path_to_script>.py
```
