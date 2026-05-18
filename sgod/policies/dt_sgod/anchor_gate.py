"""Anchor-Trigger Gate (ATG).

Replaces the rule-based anchor detector (`sgod.anchor.detector`) with a learned
2-layer MLP that consumes:
    - the current backbone hidden state h_t,
    - a pooled summary of the slow state h_slow,
    - the LM-logit entropy at this step.

Outputs a fire probability ∈ [0, 1]; if > 0.5, the slow module re-fires this step.

Day-1 stub: real wiring is complete (MLP + sigmoid), but the model is randomly
initialized — training in Stage 0/1 calibrates it. With the speaker gate γ=0 the
ATG's output cannot affect logits, so the smoke test passes regardless.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from sgod.policies.dt_sgod.config import DTSGODConfig


class AnchorTriggerGate(nn.Module):
    """Learned firing gate for the slow module."""

    def __init__(self, hidden_dim: int, config: DTSGODConfig) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.config = config

        # Input: [h_t, pooled_h_slow, lm_entropy_scalar] → 2-layer MLP.
        in_dim = hidden_dim + hidden_dim + 1
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, config.atg_hidden_dim),
            nn.GELU(),
            nn.Linear(config.atg_hidden_dim, 1),
        )

    def forward(
        self,
        hidden_state: Tensor,    # [B, d]
        h_slow_pooled: Tensor,   # [B, d]   — e.g. mean over slots
        lm_entropy: Tensor,      # [B]      — scalar entropy of LM logits
    ) -> Tensor:
        """Returns fire_prob ∈ [B] (sigmoid output)."""
        x = torch.cat([hidden_state, h_slow_pooled, lm_entropy.unsqueeze(-1)], dim=-1)
        return self.mlp(x).squeeze(-1).sigmoid()
