"""ALVCConnector — Adaptive-Length Vision Connector.

Replaces a fixed-length connector (linear projector / Q-Former / Perceiver IO)
with a variable-length one. Pipeline:

    vision_features [B, N, D_v]
        |
        |-- HaltingHead --> K in [K_min, K_max]  (sequence-level, query-aware)
        |
    select top-K tokens (by importance score)
        |
    linear projection -> LM token space [B, K, D_lm]

Phase 0 uses sequence-level halting + top-K selection by a learned importance
score. Per-patch halting is a future extension.

Integration:
    Replaces TinyLLaVA's mm_projector module. Compatible with frozen
    SigLIP-SO400M vision encoder and frozen Phi-2 / StableLM language model.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ALVCConfig:
    """ALVC connector hyperparameters.

    Attributes:
        n_visual_patches: Number of patches from vision encoder (e.g. 256 for
            SigLIP-SO400M-patch14-384).
        d_v: Vision feature dimension (e.g. 1152 for SigLIP-SO400M).
        d_lm: LM input dimension (e.g. 2560 for Phi-2).
        k_min: Minimum K.
        k_max: Maximum K (typically = n_visual_patches).
    """

    n_visual_patches: int = 256
    d_v: int = 1152
    d_lm: int = 2560
    k_min: int = 1
    k_max: int = 256


class ALVCConnector:
    """Adaptive-length vision connector.

    NOTE: skeleton only. Full implementation pending halting head + training
    objective design.
    """

    def __init__(self, config: ALVCConfig) -> None:
        self.config = config
        raise NotImplementedError("ALVCConnector skeleton — implementation pending")
