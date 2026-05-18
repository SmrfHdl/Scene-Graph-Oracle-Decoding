"""Smoke tests for the DT-SGOD policy skeleton (no GPU, no checkpoint).

The critical invariant proved here:

    At initialization (γ = 0, out_proj = 0), DTSGODPolicy.adjust_logits returns
    exactly zeros, so DT-SGOD wrapping any backbone is bit-identical to that
    backbone alone.

This is the Day-1 G1 gate from the implementation plan.
"""
from __future__ import annotations

import torch

from sgod.core.types import GenerationState, OracleEvidence, SceneGraph
from sgod.policies.dt_sgod import DTSGODPolicy


# Use tiny dims so CPU tests are instant.
HIDDEN_DIM = 64
VOCAB_SIZE = 128
BATCH = 2


def _make_state(policy: DTSGODPolicy, evidence: OracleEvidence) -> GenerationState:
    """Construct a GenerationState with random hidden + zero LM logits."""
    hidden = torch.randn(BATCH, HIDDEN_DIM)
    lm_logits = torch.zeros(BATCH, VOCAB_SIZE)
    # Build initial policy_state via the policy's own init.
    policy_state = policy.init_state(evidence)
    return GenerationState(
        prompt_ids=torch.zeros(BATCH, 1, dtype=torch.long),
        generated_ids=torch.zeros(BATCH, 1, dtype=torch.long),
        hidden_states=hidden,
        lm_logits=lm_logits,
        evidence=evidence,
        step=0,
        policy_state=policy_state,
        rule_anchor_type=None,
    )


def test_dt_sgod_policy_constructs_with_zero_invariant():
    """At init, gate=0 and out_proj=0 — Δ provably zero."""
    policy = DTSGODPolicy(hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE)
    assert policy.is_identity_at_init(), (
        "Gate γ must init to 0 and out_proj weights must init to 0; "
        "otherwise DT-SGOD@init != backbone-base"
    )


def test_dt_sgod_adjust_logits_returns_zeros_at_init():
    """The core smoke test: γ=0 ⇒ Δ ≡ 0 (within numerical exactness)."""
    policy = DTSGODPolicy(hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE)
    policy.eval()

    evidence = OracleEvidence(scene_graph=SceneGraph())
    state = _make_state(policy, evidence)

    delta = policy.adjust_logits(state)

    assert delta.shape == (BATCH, VOCAB_SIZE)
    # With both gate=0 and out_proj=0, the product is exactly zero.
    assert torch.all(delta == 0), f"max |Δ| = {delta.abs().max().item()}"


def test_dt_sgod_adjust_logits_shape_independent_of_state_evolution():
    """Calling adjust_logits multiple times keeps the contract."""
    policy = DTSGODPolicy(hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE)
    policy.eval()
    evidence = OracleEvidence(scene_graph=SceneGraph())
    state = _make_state(policy, evidence)

    for step in range(3):
        state.step = step
        state.hidden_states = torch.randn(BATCH, HIDDEN_DIM)
        state.lm_logits = torch.randn(BATCH, VOCAB_SIZE)
        delta = policy.adjust_logits(state)
        state.policy_state = policy.update_state(state, sampled_token_id=0)
        assert delta.shape == (BATCH, VOCAB_SIZE)
        assert torch.all(delta == 0)


def test_dt_sgod_trainable_param_budget():
    """Trainable param count should be in the ~few-million range at default config.

    Sanity check that we haven't accidentally instantiated a giant module.
    Real budget (~50M) is hit when GP/Adapter sit on top of a 4096-dim backbone,
    not the tiny test-dim used here.
    """
    policy = DTSGODPolicy(hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE)
    n_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    assert 0 < n_params < 5_000_000, f"unexpected trainable param count {n_params}"


def test_dt_sgod_atg_outputs_probability():
    """ATG must produce a per-batch fire_prob ∈ [0, 1]."""
    policy = DTSGODPolicy(hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE)
    policy.eval()
    h = torch.randn(BATCH, HIDDEN_DIM)
    h_slow_pooled = torch.randn(BATCH, HIDDEN_DIM)
    entropy = torch.rand(BATCH)
    fire_prob = policy.anchor_gate(h, h_slow_pooled, entropy)
    assert fire_prob.shape == (BATCH,)
    assert torch.all((fire_prob >= 0) & (fire_prob <= 1))
