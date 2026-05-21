"""Phase 0 sanity check: Phi-3.5-Vision-Instruct loads and runs a forward pass.

Goal: verify the model loads via transformers (trust_remote_code), confirm the
architecture matches our ALVC integration plan, and dump the structure of the
vision projector so we know exactly where to inject ALVCConnector.

Phi-3.5-vision-instruct (microsoft/Phi-3.5-vision-instruct):
    - 4.2B params, Phi-3.5-mini (3.8B) + CLIP ViT-L/14-336 vision encoder
    - LM hidden = 3072
    - Vision feature dim = 1024
    - 144 vision tokens per crop (dynamic cropping for HD images)
    - Vision projector lives at model.model.vision_embed_tokens.img_projection
      (a 2-layer MLP)

Usage:
    # Local (CPU laptop): import-only smoke test
    .venv/bin/python scripts/sanity_alvc_phi35v.py --import-only

    # On uet (GPU): full forward pass
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
        .venv/bin/python scripts/sanity_alvc_phi35v.py
"""
from __future__ import annotations

import argparse
import sys

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

MODEL_ID = "microsoft/Phi-3.5-vision-instruct"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-forward",
        action="store_true",
        help="Import + load only; skip forward pass.",
    )
    parser.add_argument(
        "--import-only",
        action="store_true",
        help="Skip model load entirely; only verify transformers + torch import.",
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
        print(f"[sanity] --import-only: skipping {MODEL_ID} load. Run without --import-only for full sanity.")
        return 0

    print(f"[sanity] Loading {MODEL_ID} ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        _attn_implementation="eager",  # avoid sdpa dispatch on CPU / older GPUs
    )
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        num_crops=1,  # Phase 0: fix to 1 crop -> 144 tokens
    )

    print(f"[sanity] Model loaded. Inspecting structure ...")
    print(f"  model.config.hidden_size       = {model.config.hidden_size}")
    print(f"  model.config.vocab_size        = {model.config.vocab_size}")
    print(f"  img_processor.image_dim_out    = {model.config.img_processor['image_dim_out']}")
    print(f"  img_processor.num_img_tokens   = {model.config.img_processor['num_img_tokens']}")

    # Locate the vision projector — this is what ALVC replaces.
    # Path: model.model.vision_embed_tokens.img_projection
    vision_embed = model.model.vision_embed_tokens
    print(f"\n[sanity] Vision embed module = {type(vision_embed).__name__}")
    print(f"[sanity] Vision projector (replace with ALVCConnector):")
    if hasattr(vision_embed, "img_projection"):
        for name, param in vision_embed.img_projection.named_parameters():
            print(f"  img_projection.{name:30s} shape={tuple(param.shape)}")
    else:
        print(f"  WARNING: img_projection not found. Dumping vision_embed children:")
        for name, child in vision_embed.named_children():
            print(f"    {name}: {type(child).__name__}")

    if args.no_forward:
        print("\n[sanity] --no-forward: skipping forward pass. OK.")
        return 0

    if not torch.cuda.is_available():
        print("\n[sanity] No CUDA available. Forward pass skipped.")
        return 0

    device = "cuda"
    model = model.to(device)
    model.eval()

    image = Image.open(args.image).convert("RGB")
    messages = [{"role": "user", "content": "<|image_1|>\nWhat is in this image?"}]
    prompt_text = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(prompt_text, [image], return_tensors="pt").to(device)

    print(f"\n[sanity] Running forward pass on {args.image} ...")
    print(f"  input_ids shape       = {tuple(inputs.input_ids.shape)}")
    print(f"  pixel_values shape    = {tuple(inputs.pixel_values.shape)}")
    print(f"  image_sizes           = {inputs.image_sizes.tolist()}")
    with torch.no_grad():
        out = model(**inputs, return_dict=True)
    print(f"  output.logits shape   = {tuple(out.logits.shape)}")
    print("\n[sanity] OK. ALVC will replace model.model.vision_embed_tokens.img_projection.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
