"""Unit tests for GroundingPlanner — the slow module of DT-SGOD.

CPU-only, no GPU or checkpoints required.
"""
from __future__ import annotations

import torch

from sgod.core.types import ObjectNode, OracleEvidence, SceneGraph
from sgod.policies.dt_sgod import DTSGODConfig, GroundingPlanner

HIDDEN_DIM = 64
BATCH = 2


def _config(num_slots: int = 8) -> DTSGODConfig:
    return DTSGODConfig(num_slots=num_slots)


def _sg_with(num_objects: int) -> SceneGraph:
    """Build a SceneGraph with `num_objects` synthetic objects."""
    objects = [
        ObjectNode(
            label=f"obj_{i}",
            confidence=0.5 + 0.05 * i,
            bbox=(10.0 * i, 20.0 * i, 10.0 * i + 50.0, 20.0 * i + 50.0),
        )
        for i in range(num_objects)
    ]
    return SceneGraph(objects=objects, image_size=(640, 480))


def _evidence(num_objects: int) -> OracleEvidence:
    return OracleEvidence(scene_graph=_sg_with(num_objects))


# ── shape & state evolution ──────────────────────────────────────────────────

def test_gp_init_state_shape():
    gp = GroundingPlanner(hidden_dim=HIDDEN_DIM, config=_config())
    h_slow = gp.init_state(_evidence(3), batch_size=BATCH, device="cpu")
    assert h_slow.shape == (BATCH, 8, HIDDEN_DIM)


def test_gp_forward_shape():
    gp = GroundingPlanner(hidden_dim=HIDDEN_DIM, config=_config())
    h_prev = gp.init_state(_evidence(3), batch_size=BATCH, device="cpu")
    prefix = torch.randn(BATCH, HIDDEN_DIM)
    h_new = gp(h_prev, prefix, _evidence(3))
    assert h_new.shape == h_prev.shape


def test_gp_empty_sg_returns_identity():
    """No objects ⇒ no update (variational: posterior = prior with no evidence)."""
    gp = GroundingPlanner(hidden_dim=HIDDEN_DIM, config=_config())
    h_prev = gp.init_state(_evidence(0), batch_size=BATCH, device="cpu")
    prefix = torch.randn(BATCH, HIDDEN_DIM)
    h_new = gp(h_prev, prefix, _evidence(0))
    assert torch.equal(h_new, h_prev), "Empty SG must be a no-op"


def test_gp_non_empty_sg_changes_state():
    gp = GroundingPlanner(hidden_dim=HIDDEN_DIM, config=_config())
    h_prev = gp.init_state(_evidence(3), batch_size=BATCH, device="cpu")
    prefix = torch.randn(BATCH, HIDDEN_DIM)
    h_new = gp(h_prev, prefix, _evidence(3))
    # State should not be identical after a real update.
    assert not torch.allclose(h_new, h_prev), "Non-empty SG must update h_slow"


# ── featurization determinism ────────────────────────────────────────────────

def test_gp_label_featurization_deterministic_within_instance():
    gp = GroundingPlanner(hidden_dim=HIDDEN_DIM, config=_config())
    a = gp._featurize_label("dog")
    b = gp._featurize_label("dog")
    assert torch.equal(a, b)


def test_gp_label_featurization_deterministic_across_instances():
    """Same label → same vector across module instances (md5-seeded)."""
    gp1 = GroundingPlanner(hidden_dim=HIDDEN_DIM, config=_config())
    gp2 = GroundingPlanner(hidden_dim=HIDDEN_DIM, config=_config())
    a = gp1._featurize_label("cat")
    b = gp2._featurize_label("cat")
    assert torch.equal(a, b)


def test_gp_label_featurization_distinct_for_different_labels():
    gp = GroundingPlanner(hidden_dim=HIDDEN_DIM, config=_config())
    a = gp._featurize_label("dog")
    b = gp._featurize_label("cat")
    assert not torch.equal(a, b)


# ── gradient flow ────────────────────────────────────────────────────────────

def test_gp_gradient_flows_to_all_trainable_params():
    """Loss on h_new must produce non-zero grads on every trainable GP param."""
    gp = GroundingPlanner(hidden_dim=HIDDEN_DIM, config=_config())
    h_prev = gp.init_state(_evidence(4), batch_size=BATCH, device="cpu")
    prefix = torch.randn(BATCH, HIDDEN_DIM, requires_grad=False)
    h_new = gp(h_prev, prefix, _evidence(4))

    loss = h_new.sum()
    loss.backward()

    grad_status = {
        name: (p.grad is not None and bool(p.grad.abs().sum().item() > 0))
        for name, p in gp.named_parameters()
        if p.requires_grad
    }
    missing = [n for n, ok in grad_status.items() if not ok]
    assert not missing, f"No gradient on: {missing}"


def test_gp_slot_init_is_trainable_and_diverse():
    """All S slot-init vectors should be distinct (not collapsed to one)."""
    gp = GroundingPlanner(hidden_dim=HIDDEN_DIM, config=_config(num_slots=4))
    init = gp.slot_init.detach()
    # Pairwise distance should be non-zero for all distinct pairs.
    for i in range(4):
        for j in range(i + 1, 4):
            assert not torch.equal(init[i], init[j]), f"slots {i},{j} collapsed at init"


# ── DT-SGOD integration: invariant preserved after wiring ────────────────────

def test_dt_sgod_with_real_gp_preserves_zero_invariant():
    """Wiring GP into the policy must not break Gate G1 (γ=0 ⇒ Δ=0)."""
    from sgod.policies.dt_sgod import DTSGODPolicy

    policy = DTSGODPolicy(hidden_dim=HIDDEN_DIM, vocab_size=128)
    policy.eval()
    from sgod.core.types import GenerationState

    state = GenerationState(
        prompt_ids=torch.zeros(BATCH, 1, dtype=torch.long),
        generated_ids=torch.zeros(BATCH, 1, dtype=torch.long),
        hidden_states=torch.randn(BATCH, HIDDEN_DIM),
        lm_logits=torch.randn(BATCH, 128),
        evidence=_evidence(5),  # non-empty so GP can actually fire
        step=0,
        policy_state=policy.init_state(_evidence(5)),
        rule_anchor_type=None,
    )
    # Force ATG to fire by patching the anchor gate to return 1.0.
    with torch.no_grad():
        # Replace the last linear's bias to ensure sigmoid → 1.
        last_linear = policy.anchor_gate.mlp[-1]
        last_linear.weight.zero_()
        last_linear.bias.fill_(100.0)
    delta = policy.adjust_logits(state)
    assert torch.all(delta == 0), (
        f"Δ must remain zero even after GP fires; got max |Δ|={delta.abs().max().item()}"
    )
