"""Abstract base classes for the SGOD framework's three pluggable layers.

These signatures are LOCKED for Phase 1 — changing them invalidates the
framework-readiness invariant. Add new fields via subclass attributes or
extend `policy_state` / `evidence.extra` instead of widening the ABC.

See docs/dt_sgod_implementation_plan.md §2 for design rationale.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from torch import Tensor

from sgod.core.types import GenerationState, OracleEvidence


# ─── Backbone ──────────────────────────────────────────────────────────────────

class Backbone(ABC):
    """Wraps a VLM. Provides hidden states + logits per decode step.

    Concrete backbones live in `sgod.backbones` (e.g. `LLaVAv15Backbone`,
    `Qwen25VLBackbone`). They are responsible for:
      - Loading the underlying HF model.
      - Image/prompt preprocessing.
      - Returning per-step hidden states and LM logits.
      - Exposing the surface a Policy needs (hidden_dim, vocab_size, LoRA targets).

    The Policy interacts only via the methods/properties below — never reach
    into the backbone's underlying HF model directly.
    """

    @property
    @abstractmethod
    def hidden_dim(self) -> int:
        """Dimensionality of the last hidden state returned by `forward_step`."""

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Number of logits produced per step (= |vocabulary|)."""

    @property
    @abstractmethod
    def lora_target_modules(self) -> list[str]:
        """Module-name fragments for PEFT LoRA injection.

        Conventionally the linear projections inside self-attention
        (e.g., ["q_proj", "k_proj", "v_proj", "o_proj"]).
        """

    @property
    @abstractmethod
    def device(self) -> str:
        """Primary device string for tensors produced by this backbone."""

    @abstractmethod
    def tokenizer(self) -> Any:
        """The HF tokenizer (or processor) used by the underlying model."""

    @abstractmethod
    def prepare_inputs(self, image: Any, prompt: str) -> dict[str, Any]:
        """Convert (image, prompt) into the dict the backbone needs to begin generation.

        Returns whatever the concrete backbone consumes in `forward_step`.
        """

    @abstractmethod
    def forward_step(
        self,
        inputs: dict[str, Any],
        generated_ids: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Run one decode step.

        Args:
            inputs:        Dict from `prepare_inputs` (may contain mutable cache).
            generated_ids: [B, t] tokens decoded so far.

        Returns:
            hidden_state: [B, hidden_dim]  last hidden state at step t.
            lm_logits:    [B, vocab_size]  raw LM logits at step t.
        """


# ─── Oracle ────────────────────────────────────────────────────────────────────

class Oracle(ABC):
    """Image → structured belief (SceneGraph + optional extras).

    Concrete oracles live in `sgod.oracles` (RelTR, GroundingDino, OWLv2, ...).
    Must be *independent* of the VLM Backbone — see
    `~/.claude/projects/.../memory/design_independence_principle.md`.
    """

    @abstractmethod
    def extract(self, image: Any, image_meta: dict[str, Any] | None = None) -> OracleEvidence:
        """Run the oracle once on an image. Cheap to call once per image; not per token."""

    @abstractmethod
    def vocab_scores(self, tokenizer: Any, evidence: OracleEvidence) -> dict[int, float]:
        """Per-vocab-token scalar scores used by SGOD v1 and as a baseline signal.

        Returns a sparse {token_id: score} dict; missing keys mean score=0.
        """


# ─── Policy ────────────────────────────────────────────────────────────────────

class Policy(ABC):
    """Decoding-time adjustment policy.

    Maps a (Backbone hidden state, Oracle evidence, generation state) tuple to
    a logit adjustment Δ that is added to the LM logits before sampling.

    Concrete policies:
      - `SGODv1Policy`     — training-free, linear, rule-based anchor (the predecessor).
      - `DTSGODPolicy`     — dual-timescale slot-recurrent (this proposal).
      - future variants    — Bayesian posterior, energy-gated, etc.

    The Policy owns its own state via `state.policy_state` (an opaque dict).
    The orchestrator initializes it once via `init_state` and updates it per
    step via `update_state` (after sampling).
    """

    @abstractmethod
    def init_state(self, evidence: OracleEvidence) -> dict[str, Any]:
        """Build the initial `policy_state` dict for a new generation."""

    @abstractmethod
    def adjust_logits(self, state: GenerationState) -> Tensor:
        """Return Δ [B, V] to add to `state.lm_logits` before sampling.

        Must be safe to call every step. May internally decide to no-op (returning
        zeros) at non-anchor positions — that decision is policy-local.
        """

    @abstractmethod
    def update_state(self, state: GenerationState, sampled_token_id: int) -> dict[str, Any]:
        """Advance the policy state after a token is sampled. Returns the new state."""


__all__ = ["Backbone", "Oracle", "Policy"]
