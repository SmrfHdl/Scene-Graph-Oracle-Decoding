"""Phase 0 sanity check: TinyLLaVA-Phi-2-SigLIP loads and runs a forward pass.

Goal: verify the model loads via transformers (trust_remote_code), confirm the
architecture matches our ALVC integration plan, and dump shapes of vision
features + connector output so we know exactly where to inject ALVCConnector.

Usage:
    # Local (no GPU): import-only smoke test (skips forward pass)
    .venv/bin/python scripts/sanity_alvc_tinyllava.py --no-forward

    # On uet (GPU): full forward pass
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
        /home/uet/anaconda3/envs/spin/bin/python scripts/sanity_alvc_tinyllava.py
"""
from __future__ import annotations

import argparse
import sys

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-forward",
        action="store_true",
        help="Import + load only; skip forward pass (for CPU-only environments).",
    )
    parser.add_argument(
        "--import-only",
        action="store_true",
        help=(
            "Import transformers + check torch/cuda only. Do NOT load TinyLLaVA. "
            "Useful for local CPU machines where TinyLLaVA's custom modeling code "
            "is not compatible with the installed transformers version (e.g. "
            "transformers 5.x is missing the legacy _supports_sdpa attribute that "
            "TinyLLaVA's modeling_tinyllava_phi.py expects). Full sanity must "
            "run on uet with the spin conda env."
        ),
    )
    parser.add_argument(
        "--image",
        type=str,
        default="test_imgs/test1.jpg",
        help="Image path for forward pass.",
    )
    args = parser.parse_args()

    if args.import_only:
        print(f"[sanity] torch={torch.__version__}, cuda={torch.cuda.is_available()}")
        print(f"[sanity] transformers imported OK.")
        print(f"[sanity] --import-only: skipping TinyLLaVA load. Run on uet for full sanity.")
        return 0

    print(f"[sanity] Loading {MODEL_ID} ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)

    print(f"[sanity] Model loaded. Inspecting structure ...")
    print(f"  model.config.connector_type     = {model.config.connector_type}")
    print(f"  model.config.hidden_size        = {model.config.hidden_size}")
    print(f"  vision tower type               = {type(model.vision_tower).__name__}")
    print(f"  connector type                  = {type(model.connector).__name__}")
    print(f"  language_model type             = {type(model.language_model).__name__}")

    # Connector is what we replace with ALVCConnector
    print(f"\n[sanity] Connector module breakdown:")
    for name, param in model.connector.named_parameters():
        print(f"  {name:40s} shape={tuple(param.shape)}")

    if args.no_forward:
        print("\n[sanity] --no-forward set; skipping forward pass. OK.")
        return 0

    if not torch.cuda.is_available():
        print("\n[sanity] No CUDA available. Forward pass skipped.")
        print("[sanity] Re-run on uet with GPU for full sanity.")
        return 0

    device = "cuda"
    model = model.to(device)
    model.eval()

    image = Image.open(args.image).convert("RGB")
    prompt = "What is in this image?"

    print(f"\n[sanity] Running forward pass on {args.image} ...")
    with torch.no_grad():
        input_ids, attn_mask, _, image_tensor, _, _ = (
            model.preprocess(image, prompt, tokenizer)
        )
        out = model(
            input_ids=input_ids.to(device),
            attention_mask=attn_mask.to(device),
            images=image_tensor.to(device, dtype=torch.float16),
            return_dict=True,
        )
    print(f"  output.logits shape = {tuple(out.logits.shape)}")
    print(f"  vocab_size          = {model.config.vocab_size}")
    print("\n[sanity] OK. ALVC can replace the connector at model.connector.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
