"""Speaker Adapter (fast module).

Implements the per-token cross-attention adapter:

    Δ = W_out · tanh(γ) · CrossAttn(query=h_t, key=h_slow, value=h_slow)

The gate γ is a zero-initialized scalar so that at init `Δ = 0` and DT-SGOD
behaves identically to the underlying VLM. The gate opens during training as
the adapter learns to inject grounding evidence.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from sgod.policies.dt_sgod.config import DTSGODConfig


class SpeakerAdapter(nn.Module):
    """Per-token fast module producing the logit adjustment Δ.

    The output projection is initialized to zero so that even if `tanh(γ) > 0`
    the very first forward pass produces Δ = 0 — combined with `gate_init = 0`,
    DT-SGOD is provably the identity transform at init.
    """

    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int,
        config: DTSGODConfig,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.config = config

        attn_dim = config.speaker_attn_dim

        # Q/K/V projections — small, separate from LLaVA's own projections.
        self.q_proj = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, attn_dim, bias=False)

        # Map cross-attended representation back to the LM logit space.
        self.out_proj = nn.Linear(attn_dim, vocab_size, bias=False)

        # Zero-init out_proj so Δ = 0 at init even before the gate is checked.
        nn.init.zeros_(self.out_proj.weight)

        # Flamingo-style scalar gate, init at config.speaker_gate_init (default 0.0).
        self.gate = nn.Parameter(torch.tensor(float(config.speaker_gate_init)))

    def forward(
        self,
        hidden_state: Tensor,   # [B, d]
        h_slow: Tensor,         # [B, S, d]
    ) -> Tensor:
        """Compute logit adjustment Δ ∈ [B, V]."""
        # [B, attn_dim]
        q = self.q_proj(hidden_state)
        # [B, S, attn_dim]
        k = self.k_proj(h_slow)
        v = self.v_proj(h_slow)

        # Scaled dot-product over S slots: [B, 1, attn_dim] × [B, attn_dim, S]
        scale = 1.0 / (q.shape[-1] ** 0.5)
        attn_scores = torch.einsum("bd,bsd->bs", q, k) * scale     # [B, S]
        attn_weights = torch.softmax(attn_scores, dim=-1)           # [B, S]
        attn_out = torch.einsum("bs,bsd->bd", attn_weights, v)      # [B, attn_dim]

        # Gated projection back to vocab space.
        delta = self.out_proj(attn_out) * torch.tanh(self.gate)      # [B, V]
        return delta
