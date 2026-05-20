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
    max_memory: dict | None = None,
    attn_implementation: str | None = None,
) -> tuple[Any, Any]:
    """Load LLaVA-1.5-7B model + processor.

    Args:
        model_id:    HuggingFace model ID.
        dtype:       torch.float16 (default) or torch.bfloat16.
        device_map:  "auto" splits across available GPUs; "cuda:0" for single GPU.
        load_in_4bit: Use bitsandbytes 4-bit quantization (~7 GB, for 1× T4).
        max_memory:  Dict telling accelerate how much memory each device has, e.g.
                     {0: "40GiB", "cpu": "16GiB"}. Useful when accelerate
                     cannot query GPU memory directly (CUDA driver compat issues).

    Returns:
        (model, processor) — both HF objects ready for inference.
    """
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    logger.info("Loading LLaVA: %s  dtype=%s  device_map=%s  4bit=%s",
                model_id, dtype, device_map, load_in_4bit)

    kwargs: dict = {"device_map": device_map}
    if max_memory is not None:
        kwargs["max_memory"] = max_memory
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        kwargs["torch_dtype"] = dtype
    if attn_implementation is not None:
        # "eager" required when callers need output_attentions=True (e.g., VR-TTS).
        # Default (None) lets HF pick sdpa/flash-attn for speed.
        kwargs["attn_implementation"] = attn_implementation

    model = LlavaForConditionalGeneration.from_pretrained(model_id, **kwargs)
    model.eval()

    processor = AutoProcessor.from_pretrained(model_id)
    logger.info("LLaVA loaded.")
    return model, processor


# ── RelTR (SGGModule wrapper) ─────────────────────────────────────────────────

def load_sgg(
    checkpoint: str | Path,
    device: str = "cuda",
    use_grounding_dino: bool = False,
    grounding_dino_model: str = "IDEA-Research/grounding-dino-base",
    grounding_dino_box_threshold: float = 0.25,
    grounding_dino_text_threshold: float = 0.2,
) -> Any:
    """Load a SGGModule backed by RelTR (and optionally Grounding DINO).

    Args:
        checkpoint:                       Path to RelTR .pth checkpoint.
        device:                           Torch device string.
        use_grounding_dino:               If True, build GroundingDinoModule
                                          and pass it to SGGModule for hybrid
                                          (open-vocab objects + RelTR relations).
        grounding_dino_model:             HF model id for Grounding DINO.
        grounding_dino_box_threshold:     Detection box confidence threshold.
        grounding_dino_text_threshold:    Text-alignment threshold.

    Returns:
        SGGModule instance with .extract(image) -> SceneGraph.
    """
    from sgod.sgg import SGGModule

    gd_module = None
    if use_grounding_dino:
        from sgod.sgg.grounding_dino_module import GroundingDinoModule
        gd_module = GroundingDinoModule(
            model_id=grounding_dino_model,
            device=device,
            box_threshold=grounding_dino_box_threshold,
            text_threshold=grounding_dino_text_threshold,
        )

    logger.info("Loading SGGModule from %s on %s (gd=%s)",
                checkpoint, device, bool(gd_module))
    module = SGGModule(str(checkpoint), device=device, gd_module=gd_module)
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

    Loads CLIP model once; each call to the returned factory only encodes
    the image (no model reload).

    Returns:
        Callable[[PIL.Image], CLIPScorer] — pass to SGODDecoder.__init__.
    """
    import open_clip
    import torch
    import torch.nn.functional as F
    from sgod.oracle import CLIPScorer

    logger.info("Loading CLIP %s/%s on %s", model_name, pretrained, device)
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(model_name)

    vocab_embeddings = vocab_words = vocab_index = None
    if vocab_cache_path:
        cache_path = Path(vocab_cache_path)
        if cache_path.exists():
            cached = torch.load(cache_path, map_location=device, weights_only=True)
            cache_dim = int(cached["embeddings"].shape[-1])
            # Probe the model's text-encoder dim. Mismatch (e.g. ViT-B-32 cache
            # at 512-dim + ViT-L-14 model at 768-dim) crashes deep inside
            # score_words; skip the cache and live-encode instead.
            with torch.no_grad():
                probe = model.encode_text(tokenizer(["a"]).to(device))
            model_dim = int(probe.shape[-1])
            if cache_dim != model_dim:
                logger.warning(
                    "Vocab cache at %s has dim=%d but model %s emits dim=%d — "
                    "falling back to live encoding.",
                    cache_path, cache_dim, model_name, model_dim,
                )
            else:
                vocab_embeddings = F.normalize(cached["embeddings"].to(device).float(), dim=-1)
                vocab_words = cached["words"]
                vocab_index = {w: i for i, w in enumerate(vocab_words)}
                logger.info("Loaded vocab cache: %d words", len(vocab_words))
        else:
            logger.warning("Vocab cache not found at %s — live encoding will be used", cache_path)

    preloaded = {
        "model": model,
        "preprocess": preprocess,
        "tokenizer": tokenizer,
        "vocab_embeddings": vocab_embeddings,
        "vocab_words": vocab_words,
        "vocab_index": vocab_index,
    }

    def factory(image):
        return CLIPScorer(image, model_name=model_name, device=device, _preloaded=preloaded)

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
    enable_bbox_oracle = bool(config["oracle"].get("enable_bbox_oracle", False))
    bbox_pad_ratio     = float(config["oracle"].get("bbox_pad_ratio", 0.15))
    bbox_score_multiplier = float(config["oracle"].get("bbox_score_multiplier", 1.0))
    yesno_lambda          = float(config["oracle"].get("yesno_lambda", 0.0))

    ctx_cfg = config.get("context", {})
    base_lambda = ctx_cfg.get("base_lambda", {})
    neg_decay   = ctx_cfg.get("negation_decay", 0.25)
    max_neg     = ctx_cfg.get("max_negation_depth", 3.0)
    max_tokens  = config["decoder"]["max_new_tokens"]
    temperature = config["decoder"].get("temperature", 0.0)

    sgg_cfg     = config.get("sgg", {})
    use_gd      = bool(sgg_cfg.get("use_grounding_dino", False))
    gd_model    = sgg_cfg.get("grounding_dino_model", "IDEA-Research/grounding-dino-base")
    gd_box_thr  = float(sgg_cfg.get("grounding_dino_box_threshold", 0.25))
    gd_text_thr = float(sgg_cfg.get("grounding_dino_text_threshold", 0.2))

    model, processor = load_llava(vlm_id, dtype=dtype, device_map=dmap,
                                   load_in_4bit=load_in_4bit)
    sgg = load_sgg(
        ckpt, device=device,
        use_grounding_dino=use_gd,
        grounding_dino_model=gd_model,
        grounding_dino_box_threshold=gd_box_thr,
        grounding_dino_text_threshold=gd_text_thr,
    )
    clip_factory = load_clip_factory(
        model_name=clip_m,
        vocab_cache_path=clip_c,
        device=device,
    )

    decoder = SGODDecoder(
        vlm_model=model,
        sgg_module=sgg,
        clip_factory=clip_factory,
        processor=processor,
        top_k=top_k,
        min_sg_confidence=min_sg,
        base_lambda=base_lambda or None,
        negation_decay=neg_decay,
        max_negation_depth=max_neg,
        max_new_tokens=max_tokens,
        temperature=temperature,
        enable_bbox_oracle=enable_bbox_oracle,
        bbox_pad_ratio=bbox_pad_ratio,
        bbox_score_multiplier=bbox_score_multiplier,
        yesno_lambda=yesno_lambda,
    )
    return decoder
