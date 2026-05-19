"""RelTROracle — wraps `sgod.sgg.SGGModule` as a framework `Oracle`.

This is the default oracle for SGOD: RelTR produces (objects, relations) on
Visual Genome's vocabulary (151 entities, 51 predicates). When an optional
`GroundingDinoModule` is supplied, objects come from Grounding DINO (open-
vocab) and only relations from RelTR — the hybrid mode introduced in M1.3.

The underlying `SGGModule` is loaded lazily on first `extract()` to keep unit
tests fast (no checkpoint required for shape/contract tests).
"""
from __future__ import annotations

from typing import Any

from sgod.core.interfaces import Oracle
from sgod.core.registry import register
from sgod.core.types import OracleEvidence


@register("oracle", "reltr")
class RelTROracle(Oracle):
    """RelTR-based Oracle. Optionally hybrid with Grounding DINO objects.

    Args:
        checkpoint_path:      Path to RelTR .pth checkpoint. Required unless
                              `sgg_module` is supplied directly.
        device:               Torch device string (default: auto-detect).
        confidence_threshold: Per-triplet component confidence threshold.
        top_k:                Max triplets to extract per image.
        gd_module:            Optional `GroundingDinoModule` for hybrid mode.
        sgg_module:           Pre-built `SGGModule` to bypass internal load.
                              Useful for tests and dependency injection.
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        device: str | None = None,
        confidence_threshold: float = 0.3,
        top_k: int = 20,
        gd_module: Any | None = None,
        sgg_module: Any | None = None,
    ) -> None:
        if sgg_module is None and checkpoint_path is None:
            raise ValueError(
                "RelTROracle requires either `checkpoint_path` (lazy-load) "
                "or `sgg_module` (pre-built)."
            )
        self._sgg = sgg_module
        self._checkpoint_path = checkpoint_path
        self._device = device
        self._confidence_threshold = confidence_threshold
        self._top_k = top_k
        self._gd_module = gd_module

    # ── lazy loader ───────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._sgg is not None:
            return
        from sgod.sgg import SGGModule

        self._sgg = SGGModule(
            checkpoint_path=self._checkpoint_path,
            device=self._device,
            confidence_threshold=self._confidence_threshold,
            top_k=self._top_k,
            gd_module=self._gd_module,
        )

    # ── Oracle interface ──────────────────────────────────────────────────

    def extract(self, image: Any, image_meta: dict[str, Any] | None = None) -> OracleEvidence:
        """Run RelTR (and optionally Grounding DINO) on `image` once.

        `image_meta` is accepted for interface compatibility but unused — the
        scene graph already carries `image_size`.
        """
        del image_meta
        self._ensure_loaded()
        sg = self._sgg.extract(image)
        return OracleEvidence(scene_graph=sg)

    def vocab_scores(self, tokenizer: Any, evidence: OracleEvidence) -> dict[int, float]:
        """Sparse {token_id: score} from the scene graph.

        Not consumed by SGODv1Policy or DTSGODPolicy in the current design —
        both read `evidence.scene_graph` directly. Returned empty as a stub for
        future flat-score policies that want a precomputed per-token map.
        """
        del tokenizer, evidence
        return {}


__all__ = ["RelTROracle"]
