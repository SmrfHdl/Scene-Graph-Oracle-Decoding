"""Grounding Planner (slow module) — Day-1 skeleton.

Final architecture (week 1):
    G  ── GAT encoder ──► g_v  per SG node
    h_slow_s  ── soft-attend over g_v ──► c_s    per slot s
    [c_s, prefix_summary]  ── GRU ──► new h_slow_s

Day-1 stub: returns zero `h_slow` of the correct shape so downstream code
shapes-check. Real GAT + GRU wiring lands in Week 1 (see implementation plan §3).
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from sgod.core.types import OracleEvidence
from sgod.policies.dt_sgod.config import DTSGODConfig


class GroundingPlanner(nn.Module):
    """Slow module: produces an evolving grounding state h_slow ∈ ℝ^[B, S, d].

    For Day 1 this is a placeholder that returns zeros of the correct shape;
    Week 1 fills in the GAT / soft-attention / GRU components.
    """

    def __init__(self, hidden_dim: int, config: DTSGODConfig) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.config = config

        # Learned slot initialization — diverse init via independent Gaussian per slot.
        # In Week 1 this may be re-initialized from a K-means clustering of SG node
        # embeddings to encourage diversity from step 0.
        self.slot_init = nn.Parameter(
            torch.randn(config.num_slots, hidden_dim) * 0.02
        )

        # Placeholder GRU — final shape will couple [context, prefix_summary].
        # Week 1: wire this into the recurrent core.
        self.gru = nn.GRUCell(input_size=hidden_dim, hidden_size=hidden_dim)

    # ── public API ────────────────────────────────────────────────────────

    def init_state(self, evidence: OracleEvidence, batch_size: int, device: str | torch.device) -> Tensor:
        """Build the initial `h_slow` tensor for a fresh generation.

        Args:
            evidence:   the OracleEvidence (unused in Day 1 stub).
            batch_size: B.
            device:     device to place the tensor on.

        Returns:
            h_slow [B, S, d].
        """
        del evidence  # week-1 will use SG node embeddings to init slots
        return self.slot_init.unsqueeze(0).expand(batch_size, -1, -1).to(device).contiguous()

    def forward(
        self,
        h_slow_prev: Tensor,           # [B, S, d]
        prefix_summary: Tensor,        # [B, d]
        evidence: OracleEvidence,
    ) -> Tensor:
        """Advance h_slow by one slow-step.

        Day-1 stub: identity (returns previous state). Week 1 lands GAT-encoded
        graph attention + GRU update.
        """
        del prefix_summary, evidence  # placeholder until Week 1
        return h_slow_prev
