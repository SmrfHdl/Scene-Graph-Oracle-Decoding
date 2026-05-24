"""ALVCConnector — Adaptive-Length Vision Connector.

Replaces a fixed-length connector (linear projector / Q-Former / Perceiver IO)
with a variable-length one. Pipeline:

    vision_features [B, N, D_v]
        |
        |-- mean-pool --> [B, D_v] ─┐
        |                            ├-> HaltingHead -> halting distribution p_n
        | query_pooled [B, D_q] ────┘                  (sequence-level, query-aware)
        |
        |-- importance scorer --> [B, N]  (logits, ranks tokens by relevance)
        |
        |-- 2-layer projector --> [B, N, D_lm]  (mirrors Phi-3.5-V img_projection)
        |
    keep_mask_hard (argmax K, top-K by importance) — for inference / hard eval
    keep_mask_soft (P(K >= rank) ) — differentiable wrt p_n for training

Phase 0 uses sequence-level halting + importance-ranked selection. Per-patch
halting is a future extension.

Integration:
    Designed to drop in for Phi-3.5-Vision's `img_projection` (a 2-layer MLP
    4096->3072->3072). The connector itself is standalone — the LM-side
    integration (passing fewer tokens or an attention mask) is handled outside
    this module so we don't have to monkey-patch Phi3V's modeling code.

References:
    - Phi-3.5-Vision img_projection: Linear(4096,3072)+GELU+Linear(3072,3072)
    - HaltingHead: see sgod/policies/alvc/halting.py
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from sgod.policies.alvc.halting import HaltingConfig, HaltingHead


@dataclass
class ALVCConfig:
    """ALVC connector hyperparameters.

    Defaults are set for Phi-3.5-vision-instruct (microsoft/Phi-3.5-vision-instruct):
        - CLIP ViT-L/14-336 vision encoder: 144 tokens per crop
        - Phi-3.5-Vision concatenates features from 4 CLIP layers -> projector
          receives 4096-dim input per patch (verified empirically on uet 2026-05-21)
        - Phi-3.5-mini LM: hidden_size=3072
        - Phase 0 fixes num_crops=1, so n_visual_patches = 144

    Attributes:
        n_visual_patches: Number of patches from vision encoder per crop.
        d_v: Vision feature dimension AS RECEIVED BY THE PROJECTOR (=4096 for
            Phi-3.5-Vision, which is 4 CLIP layers x 1024 concatenated).
        d_lm: LM input dimension (hidden_size).
        d_q: Pooled query embedding dimension (typically = d_lm).
        k_min: Minimum K (typically 1).
        k_max: Maximum K (typically = n_visual_patches).
        halting: HaltingConfig overrides; if None, derived from the fields above.
    """

    n_visual_patches: int = 144
    d_v: int = 4096
    d_lm: int = 3072
    d_q: int = 3072
    k_min: int = 1
    k_max: int = 144
    halting: HaltingConfig | None = field(default=None)

    def __post_init__(self) -> None:
        if self.k_min < 1:
            raise ValueError(f"k_min must be >= 1, got {self.k_min}")
        if self.k_max <= self.k_min:
            raise ValueError(f"k_max ({self.k_max}) must be > k_min ({self.k_min})")
        if self.k_max > self.n_visual_patches:
            raise ValueError(
                f"k_max ({self.k_max}) cannot exceed n_visual_patches ({self.n_visual_patches})"
            )
        if self.halting is None:
            self.halting = HaltingConfig(
                k_min=self.k_min,
                k_max=self.k_max,
                d_v=self.d_v,
                d_q=self.d_q,
            )
        else:
            # Keep halting/connector in sync — catch silent misconfig early.
            if self.halting.k_min != self.k_min or self.halting.k_max != self.k_max:
                raise ValueError(
                    "HaltingConfig.k_min/k_max must match ALVCConfig.k_min/k_max "
                    f"(got halting=({self.halting.k_min},{self.halting.k_max}) "
                    f"vs alvc=({self.k_min},{self.k_max}))"
                )


class ALVCConnector(nn.Module):
    """Adaptive-length vision connector.

    Forward inputs:
        vision_features: [B, N, D_v] from the vision encoder (Phi-3.5-V already
            concatenates 4 CLIP layers, so N=144 and D_v=4096).
        query_pooled:    [B, D_q] pooled query embedding from the LM (e.g. mean
            of LM hidden states at the query token positions). Required iff
            the halting head is query-conditioned.

    Forward returns a dict with:
        projected:       [B, N, D_lm]  All patches projected to LM dim.
        importance:      [B, N]        Per-patch importance logits (ranking signal).
        halting:         dict          HaltingHead output (p_n, lambda_n, expected_K).
        keep_mask_hard:  [B, N]        {0,1} mask (no grad) — argmax K, top-K by importance.
        keep_mask_soft:  [B, N]        [0,1] mask (grad to p_n) — soft survival.

    Phase 0 does NOT itself plug `projected * keep_mask` into the LM; the
    training/eval code is responsible for combining outputs with an LM forward
    in whatever style is being evaluated (hard top-K, soft attention mask,
    expected-loss over K, etc.). Keeping selection decoupled from projection
    lets us swap selection schemes without touching the connector.
    """

    def __init__(self, config: ALVCConfig) -> None:
        super().__init__()
        self.config = config
        assert config.halting is not None  # set by __post_init__

        # Projector mirrors Phi-3.5-V img_projection: Linear -> GELU -> Linear.
        # Init: keep the second Linear small so the connector starts near
        # the identity-ish behavior of a learned linear projector.
        self.projector = nn.Sequential(
            nn.Linear(config.d_v, config.d_lm),
            nn.GELU(),
            nn.Linear(config.d_lm, config.d_lm),
        )

        # Importance scorer: cheap 1-hidden-layer MLP per patch.
        self.importance_scorer = nn.Sequential(
            nn.Linear(config.d_v, config.d_lm // 4),
            nn.GELU(),
            nn.Linear(config.d_lm // 4, 1),
        )

        self.halting_head = HaltingHead(config.halting)

    def _pool_vision(self, vision_features: torch.Tensor) -> torch.Tensor:
        """[B, N, D_v] -> [B, D_v]. Mean-pool for Phase 0."""
        return vision_features.mean(dim=1)

    def _build_keep_masks(
        self,
        importance: torch.Tensor,
        p_n: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build hard and soft keep masks from importance scores and halting distribution.

        Args:
            importance: [B, N] importance logits (higher = more important).
            p_n:        [B, N_steps] halting distribution; N_steps = k_max - k_min + 1.

        Returns:
            keep_mask_hard: [B, N] in {0,1} — selects top-K* by importance,
                where K* = argmax(p_n) + k_min. No gradient flows through this.
            keep_mask_soft: [B, N] in [0,1] — for each patch at rank r (1-indexed
                in descending importance order), keep_prob = P(K >= r). Differentiable
                wrt p_n (gradient flows to the halting head). The rank assignment is
                not differentiable wrt importance (argsort gives integer ranks).
        """
        cfg = self.config
        B, N = importance.shape
        device = importance.device

        # Rank patches by importance descending. ranks_of[b, i] = rank of patch i (1..N).
        sorted_idx = torch.argsort(importance, dim=-1, descending=True)  # [B, N]
        ranks_of = torch.empty_like(sorted_idx)
        positions = torch.arange(1, N + 1, device=device).unsqueeze(0).expand(B, -1)
        ranks_of.scatter_(1, sorted_idx, positions)  # ranks_of[b, sorted_idx[b, r-1]] = r
        # ranks_of is now [B, N] with integer ranks in [1, N].

        # --- Hard mask: argmax of p_n -> K*, then top-K* by importance ---
        with torch.no_grad():
            k_star = p_n.argmax(dim=-1) + cfg.k_min  # [B], values in [k_min, k_max]
            keep_mask_hard = (ranks_of <= k_star.unsqueeze(-1)).to(importance.dtype)

        # --- Soft mask: keep_prob(rank r) = P(K >= r) ---
        # p_n indexes K values [k_min..k_max], length N_steps.
        # For patches with rank r:
        #   r <= k_min -> always kept (P(K >= r) = 1, since K >= k_min always).
        #   k_min < r <= k_max -> P(K >= r) = sum_{n >= r} p_n
        #                       = 1 - cumsum(p_n)[r - k_min - 1].
        #   r > k_max -> never kept (P = 0).
        cdf = torch.cumsum(p_n, dim=-1)  # [B, N_steps]; cdf[b, j] = P(K <= k_min + j)
        # Build a per-patch lookup: for each patch, find P(K >= rank).
        # surv_at_rank[b, i] = P(K >= ranks_of[b, i])
        # If rank r in (k_min, k_max], idx into cdf is (r - k_min - 1); P(K>=r) = 1 - cdf[idx].
        r = ranks_of  # [B, N], int
        # Clamp index into valid cdf range; we'll override with hard 0/1 below.
        idx = (r - cfg.k_min - 1).clamp(min=0, max=p_n.shape[-1] - 1)  # [B, N]
        cdf_at = torch.gather(cdf, dim=-1, index=idx)  # [B, N]
        surv = 1.0 - cdf_at
        keep_mask_soft = torch.where(
            r <= cfg.k_min,
            torch.ones_like(surv),
            torch.where(r > cfg.k_max, torch.zeros_like(surv), surv),
        )

        return keep_mask_hard, keep_mask_soft

    def forward(
        self,
        vision_features: torch.Tensor,
        query_pooled: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        cfg = self.config
        if vision_features.dim() != 3:
            raise ValueError(
                f"vision_features must be [B, N, D_v], got shape {tuple(vision_features.shape)}"
            )
        B, N, D_v = vision_features.shape
        if D_v != cfg.d_v:
            raise ValueError(f"D_v mismatch: expected {cfg.d_v}, got {D_v}")
        if N != cfg.n_visual_patches:
            raise ValueError(
                f"N mismatch: expected n_visual_patches={cfg.n_visual_patches}, got {N}"
            )

        projected = self.projector(vision_features)  # [B, N, D_lm]
        importance = self.importance_scorer(vision_features).squeeze(-1)  # [B, N]

        vision_pooled = self._pool_vision(vision_features)  # [B, D_v]
        halting = self.halting_head(vision_pooled, query_pooled)
        p_n = halting["p_n"]  # [B, N_steps]

        keep_mask_hard, keep_mask_soft = self._build_keep_masks(importance, p_n)

        return {
            "projected": projected,
            "importance": importance,
            "halting": halting,
            "keep_mask_hard": keep_mask_hard,
            "keep_mask_soft": keep_mask_soft,
        }
