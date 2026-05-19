"""SGODv1Policy — the training-free linear-mixing predecessor as a Policy.

Wraps the original `SGODDecoder` logic (top-K oracle scoring + λ-mixing at
anchor positions) behind the framework `Policy` interface so it can be driven
by the new orchestrator and serve as:

  1. An ablation baseline against DT-SGOD.
  2. The teacher signal for Stage-0 distillation (Week 1).

Conceptual mapping from the monolithic `SGODDecoder`:

  SGODDecoder.generate              → orchestrator (out of scope here)
    ├ Phase 1: SG + Oracle build    → init_state(evidence)
    ├ Phase 2: per-token decode loop
    │   ├ anchor detect + score top-K → adjust_logits(state)
    │   └ ctx.update + prev_tokens     → update_state(state, sampled_id)
    └ EOS handling                  → orchestrator

Per-image inputs the policy needs (image, question) are passed via
`OracleEvidence.extra` — the orchestrator stashes them when it builds evidence.
This keeps the Policy interface narrow and uniform with `DTSGODPolicy`.

The yes/no first-token injection from SGODDecoder is *not* ported here. It is
a question-type-specific hack tied to bbox oracle output; we'll revisit if
ablation shows we need it. For now the policy is the soft-additive path only.
"""
from __future__ import annotations

from typing import Any, Callable

import torch
from torch import Tensor

from sgod.context import BASE_LAMBDA, GenerationContext
from sgod.core.interfaces import Policy
from sgod.core.registry import register
from sgod.core.types import GenerationState, OracleEvidence
from sgod.decoder.sgod_decoder import (
    _clean_token,
    _detect_adversarial,
)
from sgod.decoder.sgod_decoder import SGODDecoder as _LegacyDecoder
from sgod.decoder.token_utils import decode_top_k
from sgod.oracle import VisualOracle


@register("policy", "sgod-v1")
class SGODv1Policy(Policy):
    """Training-free oracle decoding (predecessor to DT-SGOD).

    Construction is per-model (tokenizer-bound); per-image state is built
    lazily in `init_state(evidence)`. Evidence must carry `image` and
    `question` in its `extra` dict so the policy can construct a VisualOracle
    + GenerationContext.
    """

    def __init__(
        self,
        tokenizer,
        clip_factory: Callable,
        top_k: int = 50,
        min_sg_confidence: float = 0.4,
        base_lambda: dict[str, float] | None = None,
        negation_decay: float = 0.25,
        max_negation_depth: float = 3.0,
    ) -> None:
        self.tokenizer = tokenizer
        self.clip_factory = clip_factory
        self.top_k = top_k
        self.min_sg_conf = min_sg_confidence
        self.base_lambda = base_lambda or BASE_LAMBDA
        self.negation_decay = negation_decay
        self.max_negation_depth = max_negation_depth

    # ── Policy interface ──────────────────────────────────────────────────

    def init_state(self, evidence: OracleEvidence) -> dict[str, Any]:
        """Build per-image policy state.

        Requires `evidence.extra["image"]` and `evidence.extra["question"]`.
        The orchestrator is responsible for stashing both before driving the
        decode loop.
        """
        try:
            image = evidence.extra["image"]
            question = evidence.extra["question"]
        except KeyError as e:
            raise KeyError(
                "SGODv1Policy.init_state requires evidence.extra to contain "
                "'image' and 'question'; missing: " + str(e)
            ) from e

        sg = evidence.scene_graph
        clip = self.clip_factory(image)
        oracle = VisualOracle(sg, clip)
        ctx = GenerationContext(
            question,
            base_lambda=self.base_lambda,
            negation_decay=self.negation_decay,
            max_negation_depth=self.max_negation_depth,
        )
        use_oracle = oracle.should_activate_oracle(self.min_sg_conf)
        adversarial = _detect_adversarial(question, oracle.noun_vocab)
        return {
            "oracle": oracle,
            "ctx": ctx,
            "use_oracle": bool(use_oracle),
            "adversarial_scale": 0.25 if adversarial else 1.0,
            "prev_tokens": [],
        }

    def adjust_logits(self, state: GenerationState) -> Tensor:
        """Return Δ [B, V] — sparse, non-zero only at top-K anchor positions.

        At neutral tokens, when oracle is deactivated, or when λ=0 (e.g.
        hypothetical question / deep negation), Δ ≡ 0.
        """
        ps = state.policy_state
        lm_logits = state.lm_logits  # [B, V]
        delta = torch.zeros_like(lm_logits)

        if not ps["use_oracle"]:
            return delta
        lam = ps["ctx"].get_lambda() * ps["adversarial_scale"]
        if lam == 0.0:
            return delta

        oracle = ps["oracle"]
        prev_tokens = ps["prev_tokens"]
        B = lm_logits.shape[0]
        # SGOD v1 was single-stream; iterate to keep the per-row scoring loop
        # identical to the legacy decoder. Production runs use B=1; B>1 is
        # only relevant for batched evaluation later.
        for b in range(B):
            logits_b = lm_logits[b]
            top_k_ids, top_k_words = decode_top_k(logits_b, self.tokenizer, k=self.top_k)
            scores = _LegacyDecoder._score_candidates(top_k_words, prev_tokens, oracle)
            if scores.abs().sum().item() == 0.0:
                continue
            delta[b, top_k_ids] = lam * scores.to(device=delta.device, dtype=delta.dtype)

        return delta

    def update_state(self, state: GenerationState, sampled_token_id: int) -> dict[str, Any]:
        """Decode the sampled token, update prev_tokens + GenerationContext."""
        ps = state.policy_state
        next_token = _clean_token(self.tokenizer.decode([sampled_token_id]))
        ps["ctx"].update(next_token)
        ps["prev_tokens"].append(next_token)
        return ps


__all__ = ["SGODv1Policy"]
