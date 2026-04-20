"""
Pre-compute CLIP vocabulary embeddings.

Creates clip_vocab_cache.pt containing embeddings for a vocabulary of words
(either the full LLaVA tokenizer vocabulary or the RelTR class list) to enable
fast matrix-multiply scoring at inference time (~1ms instead of 640s).

Usage:
    python scripts/precompute_clip_vocab.py \
        --vlm_tokenizer llava-hf/llava-1.5-7b-hf \
        --clip_model ViT-L/14 \
        --output data/checkpoints/clip/clip_vocab_cache.pt

    # Lightweight test (RelTR classes only, no LLaVA tokenizer download):
    python scripts/precompute_clip_vocab.py \
        --reltr_only \
        --clip_model ViT-B-32 \
        --output data/checkpoints/clip/clip_vocab_cache_reltr.pt
"""
import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from sgod.sgg.reltr_utils import ENTITY_CLASSES, RELATION_CLASSES

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pre-compute CLIP vocab embeddings")
    p.add_argument("--output", required=True, help="Output .pt file path")
    p.add_argument(
        "--clip_model", default="ViT-B-32",
        help="open_clip model name (default: ViT-B-32; ViT-L-14 for higher quality)",
    )
    p.add_argument(
        "--clip_pretrained", default="openai",
        help="open_clip pretrained weights (default: openai)",
    )
    p.add_argument(
        "--vlm_tokenizer", default=None,
        help="HuggingFace tokenizer name for full LLaVA vocab (e.g. llava-hf/llava-1.5-7b-hf). "
             "If omitted, uses RelTR class vocabulary only.",
    )
    p.add_argument(
        "--reltr_only", action="store_true",
        help="Encode only RelTR entity+relation classes (fast, no tokenizer download).",
    )
    p.add_argument("--device", default=None, help="Torch device (auto-detected if omitted)")
    p.add_argument("--batch_size", type=int, default=256, help="Encoding batch size")
    return p.parse_args()


def build_vocab(args: argparse.Namespace) -> list[str]:
    """Collect the word list to embed."""
    if args.reltr_only or args.vlm_tokenizer is None:
        # RelTR entity classes (skip "N/A" at index 0) + relation classes (skip "__background__")
        words = (
            [c for c in ENTITY_CLASSES if c not in ("N/A",)]
            + [c for c in RELATION_CLASSES if c not in ("__background__",)]
        )
        logger.info("Using RelTR vocabulary: %d words", len(words))
        return words

    # Full LLaVA tokenizer vocabulary
    from transformers import AutoTokenizer
    logger.info("Loading tokenizer: %s", args.vlm_tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(args.vlm_tokenizer)
    vocab = tokenizer.get_vocab()
    # Decode each token ID to a plain string
    words = [tokenizer.convert_tokens_to_string([tok]) for tok in vocab]
    # Filter to non-empty, printable strings
    words = [w.strip() for w in words if w.strip()]
    logger.info("LLaVA vocabulary: %d words", len(words))
    return words


def encode_vocab(
    words: list[str],
    model_name: str,
    pretrained: str,
    device: str,
    batch_size: int,
) -> torch.Tensor:
    """Encode all words in batches. Returns normalized embeddings [V, D]."""
    import open_clip

    logger.info("Loading CLIP %s/%s on %s", model_name, pretrained, device)
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(model_name)

    all_embeddings = []
    n_batches = (len(words) + batch_size - 1) // batch_size

    for i in range(n_batches):
        batch = words[i * batch_size: (i + 1) * batch_size]
        with torch.no_grad():
            tokens = tokenizer(batch).to(device)
            feats = model.encode_text(tokens)
            feats = F.normalize(feats, dim=-1)
        all_embeddings.append(feats.cpu())
        if (i + 1) % 10 == 0 or (i + 1) == n_batches:
            logger.info("  Encoded %d / %d words", min((i + 1) * batch_size, len(words)), len(words))

    return torch.cat(all_embeddings, dim=0)  # [V, D]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    words = build_vocab(args)
    embeddings = encode_vocab(words, args.clip_model, args.clip_pretrained, device, args.batch_size)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"embeddings": embeddings, "words": words}, out_path)

    logger.info("Saved %d embeddings (dim=%d) to %s", len(words), embeddings.shape[1], out_path)
    print(f"\nDone. Cache: {out_path}  ({len(words)} words, dim={embeddings.shape[1]})")
    print("Pass --vocab_cache_path to CLIPScorer at inference for ~1ms matrix-multiply scoring.")


if __name__ == "__main__":
    main()
