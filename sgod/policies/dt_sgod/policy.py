"""DTSGODPolicy — orchestrates Grounding Planner, ATG, and Speaker Adapter.

Implements `sgod.core.Policy`. Per-step it:
  1) Pools h_slow → checks ATG firing → optionally refires Grounding Planner.
  2) Runs Speaker Adapter with the current h_slow to produce Δ.
  3) Returns Δ (added to LM logits by the orchestrator).

At init (γ = 0, out_proj = 0) the Speaker Adapter is the zero map; therefore
`adjust_logits` returns exactly zeros — DT-SGOD is the identity transform until
training opens the gate.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from sgod.core.interfaces import Policy
from sgod.core.registry import register
from sgod.core.types import GenerationState, OracleEvidence
from sgod.policies.dt_sgod.anchor_gate import AnchorTriggerGate
from sgod.policies.dt_sgod.config import DTSGODConfig
from sgod.policies.dt_sgod.grounding_planner import GroundingPlanner
from sgod.policies.dt_sgod.label_embedders import LabelEmbedder
from sgod.policies.dt_sgod.speaker_adapter import SpeakerAdapter


@register("policy", "dt-sgod")
class DTSGODPolicy(nn.Module, Policy):
    """Dual-Timescale Scene-Graph Decoder policy.

    Args:
        hidden_dim:  backbone hidden dimension (e.g. 4096 for LLaVA-1.5-7B).
        vocab_size:  backbone vocab size (matches Δ output dim).
        config:      DTSGODConfig — defaults are paper-target hyperparameters.
    """

    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int,
        config: DTSGODConfig | None = None,
        label_embedder: LabelEmbedder | None = None,
    ) -> None:
        nn.Module.__init__(self)
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.config = config or DTSGODConfig()

        self.grounding_planner = GroundingPlanner(
            hidden_dim, self.config, label_embedder=label_embedder,
        )
        self.speaker_adapter = SpeakerAdapter(hidden_dim, vocab_size, self.config)
        self.anchor_gate = AnchorTriggerGate(hidden_dim, self.config)

    # ── Policy interface ──────────────────────────────────────────────────

    def init_state(self, evidence: OracleEvidence) -> dict[str, Any]:
        """Build the initial policy_state. h_slow is materialized lazily on first call."""
        return {
            "h_slow": None,                  # built lazily once batch_size/device known
            "fire_history": [],              # list[int] step indices where slow re-fired
            "evidence": evidence,
        }

    def adjust_logits(self, state: GenerationState) -> Tensor:
        """Return Δ [B, V] to add to `state.lm_logits` before sampling.

        Flow:
          1. Lazily init h_slow on the first call.
          2. ATG decides per-batch whether to re-fire the slow module.
          3. Where ATG fires, run GP.forward to refresh h_slow (per-batch mask).
          4. Speaker Adapter produces Δ from (h_t, h_slow). At init γ=0 ⇒ Δ=0.
        """
        ps = state.policy_state
        h = state.hidden_states           # [B, d]
        B, _ = h.shape
        device = h.device

        # Materialize h_slow on first call (when we know B and device).
        if ps.get("h_slow") is None:
            ps["h_slow"] = self.grounding_planner.init_state(ps["evidence"], B, device)

        h_slow: Tensor = ps["h_slow"]      # [B, S, d]

        # ATG: decide whether to re-fire the slow module this step.
        # Day-2: hard threshold (no gradient through routing). Stage-1 training
        # will add a Lagrangian on E[fire_prob]; the straight-through estimator
        # for routing gradient lands in Week 1.
        with torch.no_grad():
            h_slow_pooled = h_slow.mean(dim=1)                            # [B, d]
            lm_entropy = self._entropy(state.lm_logits)                   # [B]
            fire_prob = self.anchor_gate(h, h_slow_pooled, lm_entropy)    # [B]
        ps["last_fire_prob"] = fire_prob

        # If any batch element wants to fire, run GP and mix per-batch.
        fire_mask = (fire_prob > 0.5).to(h_slow.dtype)                    # [B]
        if fire_mask.any():
            # Prefix summary = mean of the last K_t backbone hidden states
            # (oldest first, newest last), as published by the orchestrator
            # in `state.hidden_buffer`. Falls back to the current hidden when
            # no buffer is supplied — keeps the policy usable without the
            # full runtime (e.g. for unit tests that drive the policy directly).
            prefix_summary = self._prefix_summary(state, h)
            h_slow_new = self.grounding_planner(h_slow, prefix_summary, ps["evidence"])  # [B, S, d]
            mask = fire_mask.view(B, 1, 1)
            h_slow = mask * h_slow_new + (1.0 - mask) * h_slow
            ps["h_slow"] = h_slow

        # Speaker Adapter — produces Δ, gated by γ. With γ=0 and out_proj=0
        # at init, Δ ≡ 0 regardless of h_slow content.
        return self.speaker_adapter(h, h_slow)                             # [B, V]

    def update_state(self, state: GenerationState, sampled_token_id: int) -> dict[str, Any]:
        """Advance the policy state after a token is sampled.

        Day-1 stub: passes through. Week 1 wires in the actual slow-step trigger
        (when ATG fires above threshold, call GroundingPlanner.forward).
        """
        del sampled_token_id
        return state.policy_state

    # ── helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _entropy(logits: Tensor) -> Tensor:
        """Per-row entropy of softmax(logits). Returns shape [B]."""
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        return -(probs * log_probs).sum(dim=-1)

    @staticmethod
    def _prefix_summary(state: GenerationState, fallback: Tensor) -> Tensor:
        """Mean-pool over the orchestrator's K_t hidden buffer.

        Returns `fallback` (typically the current step's hidden) when no buffer
        was supplied or the buffer is empty.
        """
        buf = state.hidden_buffer
        if not buf:
            return fallback
        # buf is a list of [B, d] tensors (oldest first). Stack and mean over time.
        stacked = torch.stack(buf, dim=0)  # [K, B, d]
        return stacked.to(dtype=fallback.dtype, device=fallback.device).mean(dim=0)

    # ── Convenience ───────────────────────────────────────────────────────

    @torch.no_grad()
    def is_identity_at_init(self) -> bool:
        """Sanity check: with γ = 0 and out_proj = 0, Δ ≡ 0.

        Used by smoke tests to verify the init invariant.
        """
        return (
            torch.equal(self.speaker_adapter.gate.data, torch.tensor(0.0))
            and torch.equal(
                self.speaker_adapter.out_proj.weight.data,
                torch.zeros_like(self.speaker_adapter.out_proj.weight.data),
            )
        )
