"""Hyperparameters for DT-SGOD components.

A single dataclass to avoid scattering magic numbers across modules. Backbone
dims (`hidden_dim`, `vocab_size`) come from the `Backbone` instance — never
hardcode them here.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DTSGODConfig:
    # ── Grounding Planner (slow module) ───────────────────────────────────
    num_slots: int = 8
    gat_layers: int = 2
    gat_heads: int = 4

    # ── Speaker Adapter (fast module) ─────────────────────────────────────
    speaker_attn_dim: int = 256        # internal cross-attention dim (separate from hidden_dim)
    speaker_lora_rank: int = 16
    speaker_gate_init: float = 0.0     # γ init — zero so DT-SGOD@init == LLaVA-base

    # ── Anchor-Trigger Gate (ATG) ─────────────────────────────────────────
    atg_hidden_dim: int = 256
    atg_target_fire_rate: float = 0.20  # ≈ 1 fire per 5 tokens; Lagrangian target

    # ── Loss weights (set non-zero in Stage 1) ────────────────────────────
    loss_dpo_beta: float = 0.10
    loss_anchor_rate_mu: float = 0.50
    loss_slot_disentangle_delta: float = 0.01

    # ── Training metadata ─────────────────────────────────────────────────
    seed: int = 42
