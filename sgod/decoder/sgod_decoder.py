"""
SGODDecoder — Component 5 (Section 3.6).

Main integration class that combines all components:
    1. SGGModule      → extract scene graph (one-time)
    2. VisualOracle   → build oracle from SG + CLIP (one-time)
    3. AnchorDetector → detect anchor positions (per token)
    4. GenerationCtx  → track context, compute adaptive lambda (per token)

Key method:
    generate(image, question, max_new_tokens=256) -> str

Single forward pass per token step (no 2x compute like VCD/ICD).
Applies: logit_final = logit_lm + lambda * oracle_scores
"""
from __future__ import annotations

import re
from typing import Callable

import torch

from sgod.anchor import detect_anchor
from sgod.context import BASE_LAMBDA, GenerationContext
from sgod.decoder.token_utils import apply_oracle_scores, decode_top_k
from sgod.oracle import VisualOracle


# Generic referents we should not treat as SG-specific entities.
_GENERIC_REFERENTS: frozenset[str] = frozenset({
    "image", "photo", "picture", "scene", "background", "foreground",
    "left", "right", "top", "bottom", "middle", "center",
    "side", "front", "back",
})

# "the X" / "a X" / "an X" / "this X" / "that X" — captures the head noun candidate.
_PRESUPPOSITION_RE = re.compile(
    r"\b(?:the|a|an|this|that)\s+([a-zA-Z]+)", re.IGNORECASE
)


def _detect_adversarial(question: str, noun_vocab: set[str]) -> bool:
    """Return True when the question presupposes entities absent from the scene graph.

    Heuristic: extract noun heads after determiners; ignore generic referents
    (image, photo, side, ...). If at least one specific noun is mentioned and
    NONE of them appear in the scene graph, the oracle has nothing to confirm —
    likely an adversarial / trick question. In that case the decoder dampens
    the oracle so it does not amplify hallucinated descriptions of nonexistent
    objects.
    """
    if not noun_vocab:
        return False
    candidates = [
        c.lower() for c in _PRESUPPOSITION_RE.findall(question)
        if c.lower() not in _GENERIC_REFERENTS
    ]
    if not candidates:
        return False
    return not any(c in noun_vocab for c in candidates)


def _clean_token(token: str) -> str:
    """Strip BPE sentencepiece marker (▁) and surrounding whitespace."""
    return token.lstrip("\u2581").strip()


class SGODDecoder:
    """Drop-in oracle-guided decoder; wraps any HF-style causal VLM.

    Usage:
        decoder = SGODDecoder(vlm, tokenizer, sgg_module, clip_factory)
        answer  = decoder.generate(image, "Is there a dog?")

    Args:
        vlm_model:          HF-style model; called as vlm_model(input_ids=...) and
                            returns an object with .logits of shape [1, seq, vocab].
        tokenizer:          HF-style tokenizer; must have .eos_token_id, .encode(),
                            .decode(), .batch_decode().
        sgg_module:         SGGModule with .extract(image) -> SceneGraph.
        clip_factory:       Callable[[image], CLIPScorer] — called once per generate()
                            call so the scorer is tied to the current image.
        top_k:              Number of token candidates to score per anchor step.
        min_sg_confidence:  Threshold below which oracle is deactivated entirely.
        base_lambda:        Per-question-type lambda dict (overrides tracker defaults).
        negation_decay:     Per-token negation depth decay (default 0.25).
        max_negation_depth: Maximum negation depth clamp (default 3.0).
        max_new_tokens:     Default token budget for generate().
        temperature:        Sampling temperature; 0.0 = greedy argmax.
    """

    def __init__(
        self,
        vlm_model,
        sgg_module,
        clip_factory: Callable,
        processor=None,
        tokenizer=None,
        top_k: int = 50,
        min_sg_confidence: float = 0.4,
        base_lambda: dict[str, float] | None = None,
        negation_decay: float = 0.25,
        max_negation_depth: float = 3.0,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> None:
        self.vlm          = vlm_model
        self.processor    = processor
        self.tokenizer    = processor.tokenizer if processor is not None else tokenizer
        self.sgg          = sgg_module
        self.clip_factory = clip_factory
        self.top_k        = top_k
        self.min_sg_conf  = min_sg_confidence
        self.base_lambda  = base_lambda or BASE_LAMBDA
        self.negation_decay      = negation_decay
        self.max_negation_depth  = max_negation_depth
        self.max_new_tokens      = max_new_tokens
        self.temperature         = temperature

    # ── Public API ───────────────────────────────────────────────────────────

    def generate(
        self,
        image,
        question: str,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Generate an answer with oracle-guided decoding.

        Phase 1 (one-time): build scene graph and oracle.
        Phase 2 (per token): forward pass → oracle injection at anchor positions.

        Args:
            image:          PIL.Image passed to SGGModule and CLIPScorer.
            question:       Input question / prompt string.
            max_new_tokens: Override default token budget.
            temperature:    Override default sampling temperature (0 = greedy).

        Returns:
            Decoded answer string (without the prompt tokens).
        """
        max_new_tokens = max_new_tokens if max_new_tokens is not None else self.max_new_tokens
        temperature    = temperature    if temperature    is not None else self.temperature

        # ── Phase 1: One-time oracle construction ───────────────────────────
        sg         = self.sgg.extract(image)
        clip       = self.clip_factory(image)
        oracle     = VisualOracle(sg, clip)
        ctx        = GenerationContext(
            question,
            base_lambda=self.base_lambda,
            negation_decay=self.negation_decay,
            max_negation_depth=self.max_negation_depth,
        )
        use_oracle = oracle.should_activate_oracle(self.min_sg_conf)

        # Adversarial / presupposition dampening: when the question references
        # specific nouns and none of them are in the scene graph, the oracle
        # has nothing factual to add — treat it as a trick question and shrink
        # lambda so we don't amplify hallucinated attributes/relations.
        adversarial = _detect_adversarial(question, oracle.noun_vocab)
        adversarial_scale = 0.25 if adversarial else 1.0

        # Per-call instrumentation (read from outside via decoder.last_stats)
        self.last_stats: dict = {
            "tokens": 0,
            "anchor_fires": 0,
            "max_score": 0.0,
            "min_score": 0.0,
            "score_sum": 0.0,
            "use_oracle": bool(use_oracle),
            "adversarial": bool(adversarial),
            "noun_vocab_size": len(oracle.noun_vocab),
            "rel_vocab_size": len(oracle.rel_vocab),
            "attr_vocab_size": len(oracle.attr_vocab),
        }

        # ── Phase 2: Autoregressive decoding ────────────────────────────────
        if self.processor is not None:
            device       = next(self.vlm.parameters()).device
            inputs       = self.processor(images=image, text=question, return_tensors="pt").to(device)
            input_ids    = inputs["input_ids"]
            pixel_values = inputs.get("pixel_values")
        else:
            input_ids    = torch.tensor([self.tokenizer.encode(question)])
            pixel_values = None
        device = input_ids.device
        prev_tokens:   list[str] = []
        generated_ids: list[int] = []

        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.vlm(input_ids=input_ids, pixel_values=pixel_values)
                logits  = outputs.logits[0, -1, :].float()  # [vocab_size]

            # Oracle injection — only at anchor positions when oracle is active
            lam = ctx.get_lambda() * adversarial_scale
            if use_oracle and lam != 0.0:
                top_k_ids, top_k_words = decode_top_k(logits, self.tokenizer, k=self.top_k)
                oracle_scores = self._score_candidates(top_k_words, prev_tokens, oracle)
                score_abs_sum = oracle_scores.abs().sum().item()
                if score_abs_sum > 0.0:
                    self.last_stats["anchor_fires"] += 1
                    self.last_stats["max_score"] = max(self.last_stats["max_score"], float(oracle_scores.max()))
                    self.last_stats["min_score"] = min(self.last_stats["min_score"], float(oracle_scores.min()))
                    self.last_stats["score_sum"] += float(oracle_scores.sum())
                    apply_oracle_scores(logits, top_k_ids, oracle_scores, lam)
            self.last_stats["tokens"] += 1

            # Decode next token — greedy (temperature=0) or multinomial sampling
            if temperature == 0.0:
                next_id = int(torch.argmax(logits).item())
            else:
                probs   = torch.softmax(logits / temperature, dim=-1)
                next_id = int(torch.multinomial(probs, num_samples=1).item())

            # Clean BPE markers before context tracking and anchor detection
            next_token = _clean_token(self.tokenizer.decode([next_id]))

            ctx.update(next_token)
            prev_tokens.append(next_token)
            generated_ids.append(next_id)

            input_ids = torch.cat(
                [input_ids, torch.tensor([[next_id]], device=device)], dim=-1
            )

            if next_id == self.tokenizer.eos_token_id:
                break

        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

    # ── Internal ─────────────────────────────────────────────────────────────

    @staticmethod
    def _score_candidates(
        words: list[str],
        prev_tokens: list[str],
        oracle: VisualOracle,
    ) -> torch.Tensor:
        """Compute per-word oracle score, detecting each word's anchor type.

        Each candidate is scored with detect_anchor(prev_tokens, word) so
        that a single position can contribute relation_anchor (if "on" is a
        top-K candidate), attr_anchor (if "black" is a top-K candidate), or
        noun_anchor (if prev was a determiner, regardless of the word).

        Returns:
            FloatTensor of shape [len(words)] with values in [-1, 1].
            Zero for any word whose anchor type is "neutral".
        """
        scores = torch.zeros(len(words))
        for i, word in enumerate(words):
            anchor = detect_anchor(prev_tokens, word)
            if anchor != "neutral":
                scores[i] = oracle.score(word, anchor)
        return scores
