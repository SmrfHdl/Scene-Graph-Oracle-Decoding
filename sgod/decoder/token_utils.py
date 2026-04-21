"""
Token utilities for the decoder.

Handles BPE tokenizer issues (Risk 2, Section 5):
- decode_top_k: extract top-K token IDs and their decoded strings
- apply_oracle_scores: add per-word oracle scores back to full-vocab logits

Top-K filtering (Risk 3): reduces oracle scoring from 32K tokens → K tokens per step.
"""
from __future__ import annotations

import torch


def decode_top_k(
    logits: torch.Tensor,
    tokenizer,
    k: int = 50,
) -> tuple[torch.Tensor, list[str]]:
    """Extract top-K token IDs and their decoded strings from a logit vector.

    Args:
        logits:    1-D float tensor of shape [vocab_size].
        tokenizer: HF-style tokenizer with batch_decode(ids) -> list[str].
        k:         Number of candidates to extract (clamped to vocab_size).

    Returns:
        top_k_ids:   LongTensor of shape [k] — token IDs sorted by descending logit.
        top_k_words: list[str] of length k — decoded string for each ID.
    """
    k = min(k, logits.shape[-1])
    top_k_ids = torch.topk(logits, k=k).indices          # [k]
    raw_words = tokenizer.batch_decode(top_k_ids.tolist())
    # Strip BPE markers (▁ from sentencepiece) and leading/trailing spaces so
    # detect_anchor vocabulary lookups work correctly on real LLaVA tokens.
    top_k_words = [w.lstrip("\u2581").strip() for w in raw_words]
    return top_k_ids, top_k_words


def apply_oracle_scores(
    logits: torch.Tensor,
    top_k_ids: torch.Tensor,
    oracle_scores: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Add scaled oracle scores to the logit positions for top-K tokens.

    Modifies logits in-place at the top_k_ids positions only, leaving all
    other logit positions unchanged (neutral tokens pass through untouched).

    Args:
        logits:        1-D float tensor [vocab_size] — modified in-place.
        top_k_ids:     LongTensor [k] — positions to update.
        oracle_scores: FloatTensor [k] — per-token oracle scores in [-1, 1].
        lam:           Oracle mixing weight (from GenerationContext.get_lambda()).

    Returns:
        The same logits tensor (modified in-place).
    """
    logits[top_k_ids] = logits[top_k_ids] + lam * oracle_scores.to(logits.device)
    return logits
