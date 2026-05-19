"""Tests for Stage-0 distillation (sgod.training.distill_stage0).

CPU-only. We construct a tiny DTSGODPolicy, feed it synthetic StepTrace
fixtures, and verify:
  - prepare_for_stage0 breaks the double-zero init.
  - distill_step actually decreases the MSE loss on a fixed target Δ.
  - ATG receives non-zero gradient and the BCE loss term contributes.
  - Frozen-GP convention: GP params have zero gradient after a step that
    only optimizes SpeakerAdapter + ATG.
"""
from __future__ import annotations

import torch

from sgod.core.types import ObjectNode, OracleEvidence, SceneGraph
from sgod.policies.dt_sgod import DTSGODConfig, DTSGODPolicy
from sgod.runtime.trace_collector import StepTrace
from sgod.training import distill_step, prepare_for_stage0
from sgod.training.distill_stage0 import stage0_trainable_params


D = 8
V = 16


def _policy() -> DTSGODPolicy:
    return DTSGODPolicy(
        hidden_dim=D, vocab_size=V,
        config=DTSGODConfig(num_slots=2, speaker_attn_dim=8, speaker_lora_rank=2, atg_hidden_dim=8),
    )


def _trace(target_delta: torch.Tensor) -> StepTrace:
    """One synthetic trace with a fixed teacher Δ."""
    evidence = OracleEvidence(scene_graph=SceneGraph(
        objects=[ObjectNode("dog", 0.9, (0, 0, 1, 1))],
        relations=[], attributes=[], image_size=(640, 480),
    ))
    return StepTrace(
        step=0,
        hidden_states=torch.randn(1, D),
        lm_logits=torch.randn(1, V),
        delta_teacher=target_delta,
        evidence=evidence,
    )


# ── prepare_for_stage0 ───────────────────────────────────────────────────────

def test_prepare_for_stage0_warms_gate():
    policy = _policy()
    assert policy.speaker_adapter.gate.item() == 0.0
    prepare_for_stage0(policy, gate_init=0.1)
    assert abs(policy.speaker_adapter.gate.item() - 0.1) < 1e-6


def test_prepare_for_stage0_keeps_out_proj_zero_by_default():
    """Default: γ off zero, out_proj still zero → |Δ| stays small at training start."""
    policy = _policy()
    prepare_for_stage0(policy, gate_init=0.1)
    assert torch.all(policy.speaker_adapter.out_proj.weight == 0)


def test_prepare_for_stage0_can_perturb_out_proj():
    policy = _policy()
    prepare_for_stage0(policy, gate_init=0.1, out_proj_std=0.01)
    assert torch.any(policy.speaker_adapter.out_proj.weight != 0)


# ── distill_step: gradient flow + loss decrease ──────────────────────────────

def test_distill_step_decreases_delta_loss():
    """Running many distill_steps on the same trace must drive MSE down."""
    torch.manual_seed(0)
    policy = _policy()
    prepare_for_stage0(policy, gate_init=0.1)

    # Keep teacher Δ small but non-zero so the student can hit it.
    target = torch.randn(1, V) * 0.01
    trace = _trace(target)
    opt = torch.optim.Adam(stage0_trainable_params(policy), lr=5e-2)

    losses: list[float] = []
    for _ in range(200):
        info = distill_step(policy, trace, opt, w_delta=1.0, w_atg=0.0)
        losses.append(info["loss_delta"])

    assert losses[-1] < losses[0] * 0.2, (
        f"loss should decrease ≥5×; got start={losses[0]:.4g}, end={losses[-1]:.4g}"
    )


def test_distill_step_returns_logging_dict():
    policy = _policy()
    prepare_for_stage0(policy)
    target = torch.zeros(1, V)
    opt = torch.optim.Adam(stage0_trainable_params(policy), lr=1e-3)
    info = distill_step(policy, _trace(target), opt)
    for key in ("loss", "loss_delta", "loss_atg", "fire_prob_mean", "gate"):
        assert key in info, f"missing log key: {key}"
        assert isinstance(info[key], float)


# ── ATG branch ───────────────────────────────────────────────────────────────

def test_distill_step_atg_loss_contributes_when_weighted():
    torch.manual_seed(0)
    policy = _policy()
    prepare_for_stage0(policy)
    # Non-zero teacher Δ → teacher_fired=1; ATG starts near 0.5 → BCE > 0.
    target = torch.full((1, V), 0.1)
    trace = _trace(target)
    opt = torch.optim.Adam(stage0_trainable_params(policy), lr=1e-2)
    info0 = distill_step(policy, trace, opt, w_delta=0.0, w_atg=1.0)
    fire0 = info0["fire_prob_mean"]
    # Train more steps on the same target; fire_prob should rise toward 1.
    for _ in range(100):
        info = distill_step(policy, trace, opt, w_delta=0.0, w_atg=1.0)
    assert info["fire_prob_mean"] > fire0, "ATG must move toward teacher_fired=1"


def test_distill_step_atg_disabled_when_w_atg_zero():
    """With w_atg=0, ATG params should not get gradient via this loss."""
    policy = _policy()
    prepare_for_stage0(policy)
    target = torch.zeros(1, V)
    trace = _trace(target)

    # Capture ATG param ids before optimization.
    atg_params = list(policy.anchor_gate.parameters())
    snapshots_before = [p.detach().clone() for p in atg_params]

    opt = torch.optim.Adam(stage0_trainable_params(policy), lr=1e-3)
    distill_step(policy, trace, opt, w_delta=1.0, w_atg=0.0)

    # ATG should not have moved (w_atg=0 → no gradient from BCE; and Δ MSE
    # has no path through ATG since fire_prob isn't used in delta_student).
    for before, after in zip(snapshots_before, atg_params):
        assert torch.equal(before, after), "ATG params drifted under w_atg=0"


# ── frozen GP convention ─────────────────────────────────────────────────────

def test_gp_excluded_from_stage0_trainable_params():
    policy = _policy()
    trained_ids = {id(p) for p in stage0_trainable_params(policy)}
    gp_ids = {id(p) for p in policy.grounding_planner.parameters()}
    assert trained_ids.isdisjoint(gp_ids), (
        "GroundingPlanner params must NOT appear in Stage-0 trainable set"
    )
