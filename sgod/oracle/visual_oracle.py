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

    Design (Section 3.3):
      - For noun tokens: w_sg = sg_conf, w_clip = 1 - sg_conf
        · Token in scene graph  → w_sg * sg_conf + w_clip * clip_score
        · Token not in graph    → w_sg * (-0.3)  + w_clip * (clip_score - 0.5)
      - For relation tokens: +sg_conf if in graph, -0.2 otherwise
      - For attribute tokens: +sg_conf if in graph, clip_score - 0.5 otherwise

    Scores are in [-1, 1]:  >0 grounded → boost, <0 ungrounded → penalty, =0 neutral.
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
        sg_conf = self._noun_conf.get(tok, 0.0)
        clip_score = self.clip.score_single(tok)  # [0, 1]

        w_sg = sg_conf
        w_clip = 1.0 - sg_conf

        if tok in self.noun_vocab:
            return w_sg * sg_conf + w_clip * clip_score
        # Soft penalty: SGG says absent; CLIP centered at 0
        return w_sg * (-0.3) + w_clip * (clip_score - 0.5)

    def _score_relation(self, token: str) -> float:
        tok = token.lower().strip()
        if tok in self.rel_vocab:
            return self._rel_conf.get(tok, 0.5)
        return -0.2

    def _score_attribute(self, token: str) -> float:
        tok = token.lower().strip()
        if tok in self.attr_vocab:
            return self._attr_conf.get(tok, 0.5)
        clip_score = self.clip.score_single(tok)  # [0, 1]
        return clip_score - 0.5  # centered around 0


# ── Module-level helper ──────────────────────────────────────────────────

def _build_conf_map(pairs: list[tuple[str, float]]) -> dict[str, float]:
    """Build label → max_confidence map, lowercasing all labels."""
    result: dict[str, float] = {}
    for label, conf in pairs:
        key = label.lower()
        if key not in result or conf > result[key]:
            result[key] = conf
    return result
