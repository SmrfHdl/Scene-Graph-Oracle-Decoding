"""Confidence estimator for VR-TTS step gating.

Computes a [0, 1] confidence score from a single-step logit vector by looking
at the gap between the best yes-token logit and the best no-token logit,
normalised by total magnitude. High gap → high confidence; near-zero gap →
LM is undecided, exploration should continue.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


class ConfidenceEstimator:
    """Yes/No logit-gap based confidence.

    Args:
        tokenizer: HF tokenizer (e.g., from `backbone.tokenizer()`).
        yes_surface_forms: token strings that count as "yes".
        no_surface_forms:  token strings that count as "no".
    """

    def __init__(
        self,
        tokenizer: Any,
        yes_surface_forms: tuple[str, ...] = ("Yes", "▁Yes", "yes", "▁yes"),
        no_surface_forms: tuple[str, ...] = ("No", "▁No", "no", "▁no"),
    ) -> None:
        self.tokenizer = tokenizer
        unk = tokenizer.unk_token_id
        yes_ids = [tokenizer.convert_tokens_to_ids(t) for t in yes_surface_forms]
        no_ids = [tokenizer.convert_tokens_to_ids(t) for t in no_surface_forms]
        self.yes_ids = [i for i in yes_ids if i is not None and i != unk]
        self.no_ids = [i for i in no_ids if i is not None and i != unk]
        if not self.yes_ids or not self.no_ids:
            raise RuntimeError(
                "Tokenizer did not yield any yes/no ids — surface forms may be wrong "
                "for this model."
            )

    def score(self, logits: Tensor) -> tuple[str, float]:
        """Return (predicted_label, confidence in [0, 1]).

        `confidence` = |yes − no| / (|yes| + |no|) on logit magnitudes — bounded,
        zero-centred, and invariant to a constant shift.
        """
        if logits.ndim == 2:
            logits = logits[0]
        logits = logits.float().cpu()
        yes = max(logits[i].item() for i in self.yes_ids)
        no = max(logits[i].item() for i in self.no_ids)
        label = "yes" if yes > no else "no"
        denom = abs(yes) + abs(no) + 1e-6
        conf = abs(yes - no) / denom
        return label, float(min(conf, 1.0))

    def label(self, logits: Tensor) -> str:
        return self.score(logits)[0]
