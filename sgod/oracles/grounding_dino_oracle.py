"""GroundingDinoOracle — open-vocab object-only Oracle.

Wraps `sgod.sgg.GroundingDinoModule` and emits a `SceneGraph` containing only
objects (no relations or attributes). Useful when:

  - The pipeline needs an open-vocab oracle independent of RelTR's 151-class
    vocabulary (most concepts outside Visual Genome).
  - We want to ablate the contribution of relations vs objects in DT-SGOD.

For full hybrid (open-vocab objects + RelTR relations), use `RelTROracle` with
`gd_module=...` — that path keeps the relation edges, which this Oracle drops.
"""
from __future__ import annotations

from typing import Any

from sgod.core.interfaces import Oracle
from sgod.core.registry import register
from sgod.core.types import OracleEvidence
from sgod.sgg.scene_graph import SceneGraph


@register("oracle", "grounding-dino")
class GroundingDinoOracle(Oracle):
    """Open-vocabulary object detector as an Oracle. Lazy model load.

    Args:
        vocab:        Text vocabulary list. Defaults to
                      `GroundingDinoModule.default_vocabulary()`.
        model_id:     HF model id (default: IDEA-Research/grounding-dino-base).
        device:       Torch device string.
        box_threshold:  Detection box confidence threshold.
        text_threshold: Text-alignment threshold.
        gd_module:    Pre-built `GroundingDinoModule` (bypasses internal load,
                      useful for tests and dependency injection).
    """

    def __init__(
        self,
        vocab: list[str] | None = None,
        model_id: str = "IDEA-Research/grounding-dino-base",
        device: str | None = None,
        box_threshold: float = 0.25,
        text_threshold: float = 0.2,
        gd_module: Any | None = None,
    ) -> None:
        self._gd = gd_module
        self._vocab = vocab
        self._model_id = model_id
        self._device = device
        self._box_threshold = box_threshold
        self._text_threshold = text_threshold

    # ── lazy loader ───────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._gd is not None:
            return
        from sgod.sgg.grounding_dino_module import (
            GroundingDinoModule,
            default_vocabulary,
        )

        self._gd = GroundingDinoModule(
            vocab=self._vocab or default_vocabulary(),
            model_id=self._model_id,
            device=self._device,
            box_threshold=self._box_threshold,
            text_threshold=self._text_threshold,
        )

    # ── Oracle interface ──────────────────────────────────────────────────

    def extract(self, image: Any, image_meta: dict[str, Any] | None = None) -> OracleEvidence:
        """Detect open-vocab objects; emit a relations-free SceneGraph."""
        del image_meta
        self._ensure_loaded()
        objects = self._gd.detect(image)
        # Image size comes from the PIL image when available; falls back gracefully.
        size = getattr(image, "size", None)
        sg = SceneGraph(objects=objects, relations=[], attributes=[], image_size=size)
        return OracleEvidence(scene_graph=sg)

    def vocab_scores(self, tokenizer: Any, evidence: OracleEvidence) -> dict[int, float]:
        """Stub — see `RelTROracle.vocab_scores` rationale."""
        del tokenizer, evidence
        return {}


__all__ = ["GroundingDinoOracle"]
