"""
Model loading utilities.

Centralized loading for:
- LLaVA-1.5-7B (frozen VLM backbone)
- CLIP ViT-L/14 (frozen, via open_clip)
- RelTR (SGG module, pre-trained or fine-tuned)

All loaders respect the hardware config (dtype, device_map) and are designed
to work with device_map="auto" for multi-GPU setups (e.g. 2× T4).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import yaml

logger = logging.getLogger(__name__)

_DEFAULT_VLM   = "llava-hf/llava-1.5-7b-hf"
_DEFAULT_CLIP  = "ViT-L-14"
_DEFAULT_CLIP_PRETRAINED = "openai"


# ── Config helper ─────────────────────────────────────────────────────────────

def load_config(path: str | Path) -> dict:
    """Load a YAML config file into a dict."""
    with open(path) as f:
        return yaml.safe_load(f)


# ── LLaVA ─────────────────────────────────────────────────────────────────────

def load_llava(
    model_id: str = _DEFAULT_VLM,
    dtype: torch.dtype = torch.float16,
    device_map: str = "auto",
    load_in_4bit: bool = False,
) -> tuple[Any, Any]:
    """Load LLaVA-1.5-7B model + processor.

    Args:
        model_id:    HuggingFace model ID.
        dtype:       torch.float16 (default) or torch.bfloat16.
        device_map:  "auto" splits across available GPUs; "cuda:0" for single GPU.
        load_in_4bit: Use bitsandbytes 4-bit quantization (~7 GB, for 1× T4).

    Returns:
        (model, processor) — both HF objects ready for inference.
    """
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    logger.info("Loading LLaVA: %s  dtype=%s  device_map=%s  4bit=%s",
                model_id, dtype, device_map, load_in_4bit)

    kwargs: dict = {"device_map": device_map}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        kwargs["torch_dtype"] = dtype

    model = LlavaForConditionalGeneration.from_pretrained(model_id, **kwargs)
    model.eval()

    processor = AutoProcessor.from_pretrained(model_id)
    logger.info("LLaVA loaded.")
    return model, processor


# ── RelTR (SGGModule wrapper) ─────────────────────────────────────────────────

def load_sgg(
    checkpoint: str | Path,
    device: str = "cuda",
) -> Any:
    """Load a SGGModule backed by RelTR.

    Args:
        checkpoint: Path to RelTR .pth checkpoint.
        device:     Torch device string.

    Returns:
        SGGModule instance with .extract(image) -> SceneGraph.
    """
    from sgod.sgg import SGGModule

    logger.info("Loading SGGModule from %s on %s", checkpoint, device)
    module = SGGModule(str(checkpoint), device=device)
    logger.info("SGGModule loaded.")
    return module


# ── CLIP ──────────────────────────────────────────────────────────────────────

def load_clip_factory(
    model_name: str = _DEFAULT_CLIP,
    pretrained: str = _DEFAULT_CLIP_PRETRAINED,
    vocab_cache_path: str | Path | None = None,
    device: str = "cuda",
):
    """Build a clip_factory callable for use with SGODDecoder.

    Returns:
        Callable[[PIL.Image], CLIPScorer] — pass to SGODDecoder.__init__.
    """
    from sgod.oracle import CLIPScorer

    def factory(image):
        return CLIPScorer(
            image,
            model_name=model_name,
            pretrained=pretrained,
            vocab_cache_path=str(vocab_cache_path) if vocab_cache_path else None,
            device=device,
        )

    return factory


# ── Full SGOD stack ───────────────────────────────────────────────────────────

def load_sgod_decoder(
    config: dict | str | Path,
    device: str = "cuda",
    load_in_4bit: bool = False,
) -> Any:
    """Load the complete SGOD stack from a config dict or YAML path.

    Returns:
        SGODDecoder instance ready for .generate(image, question).
    """
    from sgod.decoder import SGODDecoder

    if not isinstance(config, dict):
        config = load_config(config)

    hw      = config.get("hardware", {})
    dtype   = torch.float16 if hw.get("dtype", "float16") == "float16" else torch.bfloat16
    dmap    = hw.get("device_map", "auto")

    vlm_id  = config["decoder"]["vlm_model"]
    ckpt    = config["sgg"]["checkpoint"]
    clip_m  = config["oracle"]["clip_model"]
    clip_c  = config["oracle"].get("clip_vocab_cache")
    top_k   = config["oracle"]["top_k"]
    min_sg  = config["sgg"]["confidence_threshold"]

    ctx_cfg = config.get("context", {})
    base_lambda = ctx_cfg.get("base_lambda", {})
    neg_decay   = ctx_cfg.get("negation_decay", 0.25)
    max_neg     = ctx_cfg.get("max_negation_depth", 3.0)
    max_tokens  = config["decoder"]["max_new_tokens"]
    temperature = config["decoder"].get("temperature", 0.0)

    model, processor = load_llava(vlm_id, dtype=dtype, device_map=dmap,
                                   load_in_4bit=load_in_4bit)
    sgg = load_sgg(ckpt, device=device)
    clip_factory = load_clip_factory(
        model_name=clip_m,
        vocab_cache_path=clip_c,
        device=device,
    )

    decoder = SGODDecoder(
        vlm_model=model,
        tokenizer=processor.tokenizer,
        sgg_module=sgg,
        clip_factory=clip_factory,
        top_k=top_k,
        min_sg_confidence=min_sg,
        base_lambda=base_lambda or None,
        negation_decay=neg_decay,
        max_negation_depth=max_neg,
        max_new_tokens=max_tokens,
        temperature=temperature,
    )
    return decoder
