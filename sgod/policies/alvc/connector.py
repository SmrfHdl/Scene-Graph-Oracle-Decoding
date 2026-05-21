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

    Defaults are set for Phi-3.5-vision-instruct (microsoft/Phi-3.5-vision-instruct):
        - CLIP ViT-L/14-336 vision encoder: 144 tokens per crop, per-layer dim=1024
        - Phi-3.5-Vision concatenates features from 4 CLIP layers -> projector
          receives 4096-dim input per patch (verified empirically on uet 2026-05-21)
        - Phi-3.5-mini LM: hidden_size=3072
        - Phase 0 fixes num_crops=1, so n_visual_patches = 144

    Attributes:
        n_visual_patches: Number of patches from vision encoder per crop.
        d_v: Vision feature dimension AS RECEIVED BY THE PROJECTOR (=4096 for
            Phi-3.5-Vision, which is 4 CLIP layers x 1024 concatenated).
        d_lm: LM input dimension (hidden_size).
        k_min: Minimum K (typically 1).
        k_max: Maximum K (typically = n_visual_patches).
    """

    n_visual_patches: int = 144
    d_v: int = 4096
    d_lm: int = 3072
    k_min: int = 1
    k_max: int = 144


class ALVCConnector:
    """Adaptive-length vision connector.

    NOTE: skeleton only. Full implementation pending halting head + training
    objective design.
    """

    def __init__(self, config: ALVCConfig) -> None:
        self.config = config
        raise NotImplementedError("ALVCConnector skeleton — implementation pending")
