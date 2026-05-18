"""Tests for the framework's three core ABC interfaces and the registry.

These are pure unit tests — no GPU, no checkpoints, no model loads.
"""
from __future__ import annotations

import pytest

from sgod.core import Backbone, Oracle, Policy, build, list_registered, register
from sgod.core.types import GenerationState, OracleEvidence, SceneGraph


# ── Registry behaviour ────────────────────────────────────────────────────────

def test_registry_has_three_kinds():
    """Registry must expose backbone/oracle/policy as the three kinds."""
    # `list_registered` raises on unknown kinds, so a successful call validates each.
    assert isinstance(list_registered("backbone"), list)
    assert isinstance(list_registered("oracle"), list)
    assert isinstance(list_registered("policy"), list)


def test_registry_unknown_kind_raises():
    with pytest.raises(KeyError):
        list_registered("nonsense")


def test_registry_round_trip():
    """Register → build a class via the registry; round-trip should succeed."""
    @register("backbone", "__test_dummy__")
    class _Dummy:
        def __init__(self, n: int):
            self.n = n

    inst = build("backbone", "__test_dummy__", n=42)
    assert inst.n == 42


def test_registry_duplicate_name_rejected():
    @register("oracle", "__test_dup__")
    class _A:
        pass

    with pytest.raises(KeyError):
        @register("oracle", "__test_dup__")
        class _B:
            pass


def test_registry_unknown_name_raises():
    with pytest.raises(KeyError):
        build("policy", "__does_not_exist__")


# ── ABC enforcement ───────────────────────────────────────────────────────────

def test_backbone_is_abstract():
    """Cannot instantiate Backbone directly."""
    with pytest.raises(TypeError):
        Backbone()


def test_oracle_is_abstract():
    with pytest.raises(TypeError):
        Oracle()


def test_policy_is_abstract():
    with pytest.raises(TypeError):
        Policy()


# ── Type re-exports ───────────────────────────────────────────────────────────

def test_scene_graph_reexported():
    """SceneGraph should be importable from both sgod.core.types and sgod.sgg."""
    from sgod.sgg import SceneGraph as SceneGraphFromSGG

    assert SceneGraph is SceneGraphFromSGG, (
        "SceneGraph must be the same class via both import paths "
        "(re-export, not duplication)"
    )


def test_generation_state_fields():
    """GenerationState carries the contract fields documented in interfaces.py."""
    expected = {
        "prompt_ids", "generated_ids", "hidden_states", "lm_logits",
        "evidence", "step", "policy_state", "rule_anchor_type",
    }
    fields = set(GenerationState.__dataclass_fields__)
    assert expected <= fields, f"Missing fields: {expected - fields}"


def test_oracle_evidence_carries_scene_graph():
    sg = SceneGraph()
    ev = OracleEvidence(scene_graph=sg)
    assert ev.scene_graph is sg
    assert ev.extra == {}


# ── Concrete registrations on import ──────────────────────────────────────────

def test_llava_backbone_registered():
    """Importing the backbones package should register `llava-1.5-7b`."""
    import sgod.backbones  # noqa: F401  — triggers registration side-effects

    assert "llava-1.5-7b" in list_registered("backbone")


def test_dt_sgod_policy_registered():
    """Importing the policies package should register `dt-sgod`."""
    import sgod.policies  # noqa: F401

    assert "dt-sgod" in list_registered("policy")
