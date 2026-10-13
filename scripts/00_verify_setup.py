"""
00_verify_setup.py — Pre-flight checks before running the Llama-3.1-8B pipeline.

Verifies:
  1. All required packages are installed
  2. HuggingFace token works and Llama-3.1-8B access is granted
  3. Base model can be loaded (cached or download)
  4. EM adapter exists and is compatible
  5. GPU has enough VRAM
  6. Architecture matches config (32 layers, 4096 hidden dim)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_packages():
    print("[1/6] Checking required packages...")
    required = ["torch", "transformers", "peft", "huggingface_hub", "sklearn", "matplotlib", "tqdm", "numpy"]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"  ✗ Missing packages: {missing}")
        print(f"  → Run: pip install {' '.join(missing)}")
        return False
    print("  ✓ All packages installed")
    return True


def check_hf_auth():
    print("\n[2/6] Checking HuggingFace authentication...")
    from huggingface_hub import login as hf_login, HfApi
    HF_TOKEN = os.environ.get("HF_TOKEN")
    try:
        hf_login(token=HF_TOKEN, add_to_git_credential=False)
        api = HfApi()
        user = api.whoami(token=HF_TOKEN)
        print(f"  ✓ Logged in as: {user.get('name', user.get('fullname', 'unknown'))}")
        return True
    except Exception as e:
        print(f"  ✗ Auth failed: {e}")
        print("  → Set HF_TOKEN env var or run: huggingface-cli login")
        return False


def check_model_access():
    print("\n[3/6] Checking access to gated Llama-3.1-8B-Instruct...")
    from config import BASE_MODEL
    from huggingface_hub import HfApi
    HF_TOKEN = os.environ.get("HF_TOKEN")
    try:
        api = HfApi()
        info = api.model_info(BASE_MODEL, token=HF_TOKEN)
        print(f"  ✓ Model accessible: {info.modelId}")
        return True
    except Exception as e:
        print(f"  ✗ Cannot access {BASE_MODEL}: {e}")
        print("  → Accept the Llama license at: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct")
        return False


def check_em_adapter():
    print("\n[4/6] Checking EM adapter exists...")
    from config import EM_ADAPTER
    from huggingface_hub import HfApi
    HF_TOKEN = os.environ.get("HF_TOKEN")
    try:
        api = HfApi()
        info = api.model_info(EM_ADAPTER, token=HF_TOKEN)
        print(f"  ✓ EM adapter found: {info.modelId}")
        return True
    except Exception as e:
        print(f"  ✗ EM adapter not found: {e}")
        return False


def check_gpu():
    print("\n[5/6] Checking GPU availability and VRAM...")
    import torch
    if not torch.cuda.is_available():
        print("  ✗ No CUDA GPU found")
        return False

    device_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    # After CUDA_VISIBLE_DEVICES filtering, device 0 is always the visible one
    name = torch.cuda.get_device_name(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    free_gb = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)) / (1024**3)

    print(f"  GPU: {name}")
    print(f"  Total VRAM: {total_gb:.1f} GB")
    print(f"  Available: {free_gb:.1f} GB")

    # Llama-3.1-8B in BF16 needs ~15GB, we load 3 copies sequentially + overhead
    if free_gb < 20:
        print(f"  ⚠ Warning: Only {free_gb:.1f}GB free. Need ~20GB for sequential model loading.")
        print("  → Consider freeing GPU memory or using a different GPU")
    else:
        print(f"  ✓ Sufficient VRAM")
    return True


def check_architecture():
    print("\n[6/6] Verifying model architecture matches config...")
    from config import BASE_MODEL, NUM_LAYERS, HIDDEN_DIM
    from transformers import AutoConfig
    HF_TOKEN = os.environ.get("HF_TOKEN")
    try:
        config = AutoConfig.from_pretrained(BASE_MODEL, token=HF_TOKEN)
        actual_layers = config.num_hidden_layers
        actual_hidden = config.hidden_size

        ok = True
        if actual_layers != NUM_LAYERS:
            print(f"  ✗ Layer mismatch: config.py says {NUM_LAYERS}, model has {actual_layers}")
            ok = False
        else:
            print(f"  ✓ Layers: {actual_layers} (matches config)")

        if actual_hidden != HIDDEN_DIM:
            print(f"  ✗ Hidden dim mismatch: config.py says {HIDDEN_DIM}, model has {actual_hidden}")
            ok = False
        else:
            print(f"  ✓ Hidden dim: {actual_hidden} (matches config)")

        return ok
    except Exception as e:
        print(f"  ✗ Could not verify architecture: {e}")
        return False


def main():
    print("=" * 60)
    print("00 — Pre-flight verification for Llama-3.1-8B pipeline")
    print("=" * 60)

    results = []
    results.append(("Packages", check_packages()))
    results.append(("HF Auth", check_hf_auth()))
    results.append(("Model Access", check_model_access()))
    results.append(("EM Adapter", check_em_adapter()))
    results.append(("GPU", check_gpu()))
    results.append(("Architecture", check_architecture()))

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    all_pass = True
    for name, ok in results:
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {status} — {name}")
        if not ok:
            all_pass = False

    if all_pass:
        print(f"\n✓ All checks passed! Ready to run the pipeline.")
        print(f"  Next: python scripts/01_prepare_data.py")
    else:
        print(f"\n✗ Some checks failed. Fix the issues above before proceeding.")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
