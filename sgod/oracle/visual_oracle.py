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

from typing import TYPE_CHECKING

import torch

from sgod.sgg.scene_graph import SceneGraph

if TYPE_CHECKING:
    from sgod.oracle.clip_scorer import CLIPScorer


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

    def __init__(self, scene_graph: SceneGraph, clip_scorer: CLIPScorer):
        """
        Args:
            scene_graph: SceneGraph produced by SGGModule.extract().
            clip_scorer: CLIPScorer initialised with the same image.
        """
        self.sg = scene_graph
        self.clip = clip_scorer

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

    # ── Public Interface ─────────────────────────────────────────────────

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
        # CLIP image-level cannot disambiguate "yellow pillow" vs "blue couch" —
        # weak fallback so it does not flip correct LM color answers.
        clip_score = self.clip.score_single(tok)  # [0, 1]
        return 0.3 * (clip_score - 0.5)


# ── Module-level helper ──────────────────────────────────────────────────

def _build_conf_map(pairs: list[tuple[str, float]]) -> dict[str, float]:
    """Build label → max_confidence map, lowercasing all labels."""
    result: dict[str, float] = {}
    for label, conf in pairs:
        key = label.lower()
        if key not in result or conf > result[key]:
            result[key] = conf
    return result
