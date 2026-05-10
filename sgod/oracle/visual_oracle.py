"""
VisualOracle — Component 2 (Section 3.3).

Dual-source oracle: combines SGG confidence + CLIP soft similarity.
When SGG confidence is high → trust SGG; when low → fall back to CLIP.

Key methods:
    score(token, anchor_type) -> float in [-1, 1]
    batch_score(tokens, anchor_type) -> torch.Tensor

Handles three anchor types: noun_anchor, relation_anchor, attr_anchor.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sgod.sgg.scene_graph import SceneGraph

if TYPE_CHECKING:
    from sgod.oracle.bbox_clip_scorer import BboxClipScorer
    from sgod.oracle.clip_scorer import CLIPScorer

Bbox = tuple[float, float, float, float]


class VisualOracle:
    """Dual-source oracle combining SGG confidence + CLIP soft similarity.

    All scores are CENTERED around 0 so vocab-match tokens with mid confidence
    pass through unchanged; only high-confidence matches boost and low-confidence
    or out-of-vocab tokens penalize. This breaks the systematic positive bias
    that previously over-generated content and ignored adversarial cues.

    Design (revised):
      - noun in graph    : (sg_conf - 0.5) blended with (clip - 0.5)  ∈ [-0.5, +0.5]
      - noun not in graph: 0.3 * (clip - 0.5)                          ∈ [-0.15, +0.15]
      - relation in graph : sg_conf - 0.5                              ∈ [-0.5, +0.5]
      - relation not      : -0.2
      - attr in graph     : sg_conf - 0.5                              ∈ [-0.5, +0.5]
      - attr not in graph : 0.3 * (clip - 0.5)                         ∈ [-0.15, +0.15]

    Note: the OOV noun branch no longer applies a -0.3 bias. RelTR's 151-class
    vocabulary excludes common answers (indoors, outdoors, kitchen, weather,
    numbers like "two") that are not actually adversarial — only out of vocab.
    Penalising those flipped correct LM answers on environment/counting
    questions in the 2026-04-29 MMHal run. The OOV attribute branch is also
    dampened because attr_vocab is empty for RelTR (no per-bbox attributes),
    making CLIP-only color signals noisy and image-global rather than
    object-grounded.
    """

    def __init__(
        self,
        scene_graph: SceneGraph,
        clip_scorer: CLIPScorer,
        bbox_scorer: Optional[BboxClipScorer] = None,
        question: Optional[str] = None,
        bbox_score_multiplier: float = 1.0,
    ):
        """
        Args:
            scene_graph: SceneGraph produced by SGGModule.extract().
            clip_scorer: CLIPScorer initialised with the same image.
            bbox_scorer: Optional BboxClipScorer for bbox-conditioned attribute
                scoring (M1.5 of BG-SGOD). When provided together with
                ``question``, the oracle parses the question to find target
                objects and routes OOV attribute tokens through CLIP-on-crop
                rather than CLIP-on-full-image.
            question: The user's question text. Required for bbox-conditioned
                routing — without it we don't know which object(s) to crop on.
        """
        self.sg = scene_graph
        self.clip = clip_scorer
        self.bbox_scorer = bbox_scorer
        self.bbox_score_multiplier = bbox_score_multiplier

        # Lowercase vocabulary sets
        self.noun_vocab: set[str] = {o.label.lower() for o in scene_graph.objects}
        self.rel_vocab: set[str] = {r.predicate.lower() for r in scene_graph.relations}
        self.attr_vocab: set[str] = {a.attribute.lower() for a in scene_graph.attributes}

        # Max-confidence maps: label → highest confidence seen across all entries
        self._noun_conf: dict[str, float] = _build_conf_map(
            [(o.label, o.confidence) for o in scene_graph.objects]
        )
        self._rel_conf: dict[str, float] = _build_conf_map(
            [(r.predicate, r.confidence) for r in scene_graph.relations]
        )
        self._attr_conf: dict[str, float] = _build_conf_map(
            [(a.attribute, a.confidence) for a in scene_graph.attributes]
        )

        # Resolve question targets to bbox(es) once, so per-token scoring is cheap.
        # Empty when bbox_scorer/question is missing or no targets match the SG.
        self._target_bboxes: list[Bbox] = (
            self._resolve_target_bboxes(question) if (bbox_scorer and question) else []
        )

    # ── Public Interface ─────────────────────────────────────────────────

    @property
    def has_bbox_target(self) -> bool:
        """True if the question parser resolved at least one target bbox.

        Used by SGODDecoder to detect when an attribute-anchor injection is
        bbox-grounded vs falling back to image-global CLIP — useful both for
        per-flip telemetry and for branch-specific lambda scaling.
        """
        return bool(self.bbox_scorer is not None and self._target_bboxes)

    def should_activate_oracle(self, min_confidence: float = 0.4) -> bool:
        """Return False when scene graph quality is too low to trust.

        Delegates to SceneGraph.should_activate_oracle(); exposed here so
        SGODDecoder only needs to hold a VisualOracle reference.
        """
        return self.sg.should_activate_oracle(min_confidence=min_confidence)

    def score(self, token: str, anchor_type: str) -> float:
        """Score a single decoded token candidate.

        Args:
            token: Decoded string (e.g. "dog", "on", "black").
            anchor_type: "noun_anchor" | "relation_anchor" | "attr_anchor" | "neutral".

        Returns:
            float in [-1, 1].
        """
        if anchor_type == "noun_anchor":
            return self._score_noun(token)
        if anchor_type == "relation_anchor":
            return self._score_relation(token)
        if anchor_type == "attr_anchor":
            return self._score_attribute(token)
        return 0.0

    def batch_score(self, tokens: list[str], anchor_type: str) -> torch.Tensor:
        """Score a list of token candidates.

        Returns:
            Float tensor of shape [N] with scores in [-1, 1].
            All zeros when anchor_type is "neutral".
        """
        if not tokens:
            return torch.zeros(0)
        if anchor_type == "neutral":
            return torch.zeros(len(tokens))
        return torch.tensor(
            [self.score(t, anchor_type) for t in tokens],
            dtype=torch.float32,
        )

    # ── Scoring Internals ────────────────────────────────────────────────

    def _score_noun(self, token: str) -> float:
        tok = token.lower().strip()
        clip_score = self.clip.score_single(tok)  # [0, 1]
        clip_centered = clip_score - 0.5  # [-0.5, 0.5]

        if tok in self.noun_vocab:
            sg_conf = self._noun_conf[tok]
            w_sg = sg_conf
            w_clip = 1.0 - sg_conf
            # Center SGG signal around 0: conf=0.5 → 0, conf=1.0 → +0.5, conf=0 → -0.5
            sg_centered = sg_conf - 0.5
            return w_sg * sg_centered + w_clip * clip_centered
        # Out-of-vocab: RelTR's 151-class vocab is small, so absence ≠ "not in image".
        # Use a weak CLIP-only signal with no penalty bias.
        return 0.3 * clip_centered

    def _score_relation(self, token: str) -> float:
        tok = token.lower().strip()
        if tok in self.rel_vocab:
            # Center: conf=0.5 → 0, conf=1.0 → +0.5
            return self._rel_conf.get(tok, 0.5) - 0.5
        return -0.2

    def _score_attribute(self, token: str) -> float:
        tok = token.lower().strip()
        if tok in self.attr_vocab:
            # Center: conf=0.5 → 0, conf=1.0 → +0.5
            return self._attr_conf.get(tok, 0.5) - 0.5
        # OOV attribute path. Two routes:
        #   1. bbox-conditioned: question target known + crop scorer available
        #      → CLIP scores against the crop; trustworthy enough to use 0.6 weight
        #      (twice the image-global weight, but still under full SG-vocab).
        #   2. image-global: weak 0.3 fallback — CLIP-on-whole-image cannot
        #      disambiguate "yellow pillow" vs "blue couch", so we don't trust it.
        if self.bbox_scorer is not None and self._target_bboxes:
            crop_score = self.bbox_scorer.score_token_max_across_bboxes(
                tok, self._target_bboxes
            )
            # bbox_score_multiplier lets us deliberately overshoot the conservative
            # 0.6 weight when we want to test whether oracle injection can flip
            # LLaVA's argmax. Default 1.0 keeps the safe behavior; the direction-
            # test experiment (2026-05-01) bumps to 5.0 to surface real flips.
            return self.bbox_score_multiplier * 0.6 * (crop_score - 0.5)
        clip_score = self.clip.score_single(tok)  # [0, 1]
        return 0.3 * (clip_score - 0.5)

    # ── Question-target resolution ───────────────────────────────────────

    def _resolve_target_bboxes(self, question: str) -> list[Bbox]:
        """Parse ``question`` and look up the bbox of each target object.

        Goes through:
          1. ``parse_targets``: extract noun head(s) the question is asking about.
          2. ``match_targets_to_vocab``: match against this graph's noun_vocab,
             handling plurals and multi-word labels.
          3. For each matched label, take the highest-confidence ObjectNode
             with that label and use its bbox.
        Returns ``[]`` when no targets could be resolved — caller falls back
        to image-global scoring.
        """
        from sgod.oracle.question_parser import (
            match_targets_to_vocab,
            parse_targets,
        )

        targets = parse_targets(question)
        if not targets:
            return []
        matched = match_targets_to_vocab(targets, self.noun_vocab)
        if not matched:
            return []
        # Pick the highest-confidence ObjectNode per matched label.
        bboxes: list[Bbox] = []
        for label in matched:
            candidates = [o for o in self.sg.objects if o.label.lower() == label.lower()]
            if not candidates:
                continue
            best = max(candidates, key=lambda o: o.confidence)
            bboxes.append(best.bbox)
        return bboxes


# ── Module-level helper ──────────────────────────────────────────────────

def _build_conf_map(pairs: list[tuple[str, float]]) -> dict[str, float]:
    """Build label → max_confidence map, lowercasing all labels."""
    result: dict[str, float] = {}
    for label, conf in pairs:
        key = label.lower()
        if key not in result or conf > result[key]:
            result[key] = conf
    return result
