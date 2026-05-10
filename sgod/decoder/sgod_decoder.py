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
from sgod.oracle.question_parser import is_yesno_question


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
        enable_bbox_oracle: bool = False,
        bbox_pad_ratio: float = 0.15,
        bbox_score_multiplier: float = 1.0,
        flip_log_cap: int = 200,
        yesno_lambda: float = 0.0,
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
        # M1.5 BG-SGOD: when on, build a BboxClipScorer per image and pass the
        # question into VisualOracle so attribute scoring can be bbox-conditioned.
        self.enable_bbox_oracle  = enable_bbox_oracle
        self.bbox_pad_ratio      = bbox_pad_ratio
        self.bbox_score_multiplier = bbox_score_multiplier
        self.flip_log_cap        = flip_log_cap
        self.yesno_lambda        = yesno_lambda
        # Pre-compute the set of token IDs that decode to "yes" or "no"
        # (across capitalisation + leading-space variants the LLaMA tokenizer
        # produces). The yes/no oracle injection adds yesno_lambda directly
        # to these positions on the first generated token of binary questions.
        self._yes_ids, self._no_ids = self._collect_yesno_ids(self.tokenizer) \
            if (self.tokenizer is not None and yesno_lambda > 0) else ([], [])

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
        bbox_scorer = None
        if self.enable_bbox_oracle:
            from sgod.oracle import BboxClipScorer
            bbox_scorer = BboxClipScorer(
                image,
                pad_ratio=self.bbox_pad_ratio,
                _preloaded=clip.to_preloaded(),
            )
        oracle     = VisualOracle(
            sg, clip,
            bbox_scorer=bbox_scorer,
            question=question if self.enable_bbox_oracle else None,
            bbox_score_multiplier=self.bbox_score_multiplier,
        )
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
            "bbox_oracle_enabled": bool(self.enable_bbox_oracle),
            "n_target_bboxes": len(oracle._target_bboxes),
            "bbox_score_multiplier": float(self.bbox_score_multiplier),
            "yesno_fired": False,
            "yesno_pushed": None,         # "yes" / "no" / None
            "yesno_top1_before": None,
            "yesno_top1_after": None,
            # Per-flip diagnostic: each entry is one decoding step where the
            # oracle's injection changed the greedy argmax. Capped at
            # flip_log_cap to keep records.json compact. Used to direction-
            # check the oracle: when a flip happens, do we move toward GT or
            # away? See offline analysis after MMHal.
            "flips": [],
            "n_flips": 0,
            "n_flip_anchor_attempts": 0,
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

        # Yes/no questions need a different injection point: anchor detector
        # treats "yes"/"no" as neutral, so the soft-additive path never fires
        # at the decision token. Instead we inject directly on the FIRST
        # generated token when the question is binary and the bbox oracle has
        # a verdict (target detected → push yes; target absent → push no).
        do_yesno = (
            self.yesno_lambda > 0.0
            and use_oracle
            and bool(self._yes_ids)
            and bool(self._no_ids)
            and is_yesno_question(question)
        )

        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.vlm(input_ids=input_ids, pixel_values=pixel_values)
                logits  = outputs.logits[0, -1, :].float()  # [vocab_size]

            # Yes/no oracle injection — first generated token only, binary
            # questions only. Anchor detector treats "yes"/"no" as neutral,
            # so soft-additive scoring never fires here under normal
            # conditions; this block routes the bbox-oracle's verdict
            # straight onto the yes/no logit positions.
            if do_yesno and len(generated_ids) == 0:
                pre_top1_id = int(torch.argmax(logits).item())
                pre_top1_word = self.tokenizer.decode([pre_top1_id]).strip()
                yes_t = torch.tensor(self._yes_ids, device=logits.device, dtype=torch.long)
                no_t  = torch.tensor(self._no_ids,  device=logits.device, dtype=torch.long)
                if oracle.has_bbox_target:
                    # Pipeline detected the target → answer is most likely "yes".
                    # Direction signal from POPE spike: P(yes|match)=0.89.
                    logits[yes_t] = logits[yes_t] + self.yesno_lambda
                    logits[no_t]  = logits[no_t]  - self.yesno_lambda
                    pushed = "yes"
                else:
                    # Pipeline didn't detect target → most no-cases land here
                    # (P(no_match|no)=0.95) but ~60% of yes-cases also land
                    # here due to limited GD recall. Pushing "no" here counts
                    # on baseline LLaVA's known yes-bias getting flipped to
                    # the right answer on adversarial cases.
                    logits[yes_t] = logits[yes_t] - self.yesno_lambda
                    logits[no_t]  = logits[no_t]  + self.yesno_lambda
                    pushed = "no"
                post_top1_id = int(torch.argmax(logits).item())
                self.last_stats["yesno_fired"] = True
                self.last_stats["yesno_pushed"] = pushed
                self.last_stats["yesno_top1_before"] = pre_top1_word
                self.last_stats["yesno_top1_after"]  = self.tokenizer.decode([post_top1_id]).strip()

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
                    # Capture pre-injection top-1 so we can detect flips. Greedy
                    # only — under sampling the "argmax" is not what gets emitted.
                    pre_top1_id = int(top_k_ids[0].item())
                    pre_top1_word = top_k_words[0]
                    self.last_stats["n_flip_anchor_attempts"] += 1
                    apply_oracle_scores(logits, top_k_ids, oracle_scores, lam)
                    if temperature == 0.0:
                        post_top1_id = int(torch.argmax(logits).item())
                        if post_top1_id != pre_top1_id:
                            self.last_stats["n_flips"] += 1
                            if len(self.last_stats["flips"]) < self.flip_log_cap:
                                # Find the new top-1's score within top_k (if it
                                # was even a candidate) — useful for analysis.
                                try:
                                    new_idx = int((top_k_ids == post_top1_id).nonzero()[0].item())
                                    new_word = top_k_words[new_idx]
                                    new_oracle_score = float(oracle_scores[new_idx].item())
                                except (IndexError, RuntimeError):
                                    new_word = self.tokenizer.decode([post_top1_id]).strip()
                                    new_oracle_score = 0.0
                                self.last_stats["flips"].append({
                                    "step": self.last_stats["tokens"],
                                    "anchor_attempt_idx": self.last_stats["n_flip_anchor_attempts"],
                                    "old_top1": pre_top1_word,
                                    "new_top1": new_word,
                                    "old_oracle_score": float(oracle_scores[0].item()),
                                    "new_oracle_score": new_oracle_score,
                                    "lam": float(lam),
                                    "n_target_bboxes": len(oracle._target_bboxes),
                                    "has_bbox_target": bool(oracle.has_bbox_target),
                                    "prev5_tokens": prev_tokens[-5:],
                                })
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
    def _collect_yesno_ids(tokenizer) -> tuple[list[int], list[int]]:
        """Find every token id whose decoded form is "yes" or "no".

        LLaMA's BPE produces several variants per word (capitalised, leading-
        space prefixed via the ▁ marker, etc.). The injection needs to lift
        ALL of them so the argmax can land on any of these candidates.

        Returns ``(yes_ids, no_ids)``. Empty lists if the tokenizer cannot
        be enumerated (mocks/tests).
        """
        try:
            vocab_size = len(tokenizer)
        except TypeError:
            try:
                vocab_size = tokenizer.vocab_size
            except AttributeError:
                return [], []
        yes_ids: list[int] = []
        no_ids:  list[int] = []
        for tid in range(vocab_size):
            try:
                surface = tokenizer.decode([tid]).strip().lower()
            except Exception:
                continue
            if surface == "yes":
                yes_ids.append(tid)
            elif surface == "no":
                no_ids.append(tid)
        return yes_ids, no_ids

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
