"""Tests for sgod.runtime.builder — config → HallucinationDecoder assembly.

CPU-only. The framework fakes (`test-fake-*`) are registered in
`tests/conftest.py` so they're available across every test file that
needs to drive `build_from_config` without HF/CLIP.
"""
from __future__ import annotations

import pytest

from sgod.core.registry import _REGISTRY
from sgod.runtime import HallucinationDecoder, build_from_config
from tests.conftest import (
    FakeBackboneForTests as _FakeBackbone,
    FakeOracleForTests as _FakeOracle,
    FakePolicyForTests as _FakePolicy,
)


# ── Smoke: registry kinds ────────────────────────────────────────────────────

def test_registry_has_fakes():
    assert "test-fake-backbone" in _REGISTRY["backbone"]
    assert "test-fake-oracle" in _REGISTRY["oracle"]
    assert "test-fake-policy" in _REGISTRY["policy"]


# ── build_from_config: minimal happy path ────────────────────────────────────

def test_build_from_config_assembles_decoder():
    config = {
        "backbone": {"name": "test-fake-backbone", "hidden_dim": 16, "vocab_size": 32},
        "oracle": {"name": "test-fake-oracle"},
        "policy": {"name": "test-fake-policy", "alpha": 0.5},
        "runtime": {"max_new_tokens": 5, "temperature": 0.0, "hidden_buffer_size": 2},
    }
    dec = build_from_config(config)
    assert isinstance(dec, HallucinationDecoder)
    assert isinstance(dec.backbone, _FakeBackbone)
    assert dec.backbone.hidden_dim == 16
    assert isinstance(dec.oracle, _FakeOracle)
    assert isinstance(dec.policy, _FakePolicy)
    assert dec.policy.kwargs == {"alpha": 0.5}
    assert dec.max_new_tokens == 5
    assert dec.hidden_buffer_size == 2


def test_build_from_config_lazy_flag_propagates_to_backbone():
    config = {
        "backbone": {"name": "test-fake-backbone"},
        "oracle": {"name": "test-fake-oracle"},
        "policy": {"name": "test-fake-policy"},
    }
    dec = build_from_config(config, lazy=True)
    assert dec.backbone._lazy is True
    dec2 = build_from_config(config, lazy=False)
    assert dec2.backbone._lazy is False


def test_build_from_config_runtime_defaults_when_absent():
    """Omitting `runtime` block must not crash; uses HallucinationDecoder defaults."""
    config = {
        "backbone": {"name": "test-fake-backbone"},
        "oracle": {"name": "test-fake-oracle"},
        "policy": {"name": "test-fake-policy"},
    }
    dec = build_from_config(config)
    assert dec.max_new_tokens == 256  # HallucinationDecoder default


# ── DT-SGOD specific construction ────────────────────────────────────────────

def test_build_dt_sgod_pulls_dims_from_backbone():
    config = {
        "backbone": {"name": "test-fake-backbone", "hidden_dim": 24, "vocab_size": 40},
        "oracle": {"name": "test-fake-oracle"},
        "policy": {
            "name": "dt-sgod",
            "config": {"num_slots": 4, "speaker_attn_dim": 16, "speaker_lora_rank": 4},
        },
    }
    dec = build_from_config(config)
    from sgod.policies import DTSGODPolicy
    assert isinstance(dec.policy, DTSGODPolicy)
    assert dec.policy.hidden_dim == 24
    assert dec.policy.vocab_size == 40
    assert dec.policy.config.num_slots == 4


def test_build_dt_sgod_with_clip_label_embedder_via_mock():
    """Builder accepts label_embedder.type='clip' and routes through CLIPTextLabelEmbedder.

    We don't load real CLIP — just verify the dispatch lands on the right class.
    The CLIP embedder is lazy (no load until first .embed() call), so this is safe.
    """
    config = {
        "backbone": {"name": "test-fake-backbone"},
        "oracle": {"name": "test-fake-oracle"},
        "policy": {
            "name": "dt-sgod",
            "label_embedder": {"type": "clip", "model_name": "ViT-B-32", "pretrained": "openai"},
        },
    }
    dec = build_from_config(config)
    from sgod.policies.dt_sgod import CLIPTextLabelEmbedder
    assert isinstance(dec.policy.grounding_planner.label_embedder, CLIPTextLabelEmbedder)


def test_build_dt_sgod_with_hash_label_embedder():
    config = {
        "backbone": {"name": "test-fake-backbone"},
        "oracle": {"name": "test-fake-oracle"},
        "policy": {
            "name": "dt-sgod",
            "label_embedder": {"type": "hash", "embed_dim": 64},
        },
    }
    dec = build_from_config(config)
    from sgod.policies.dt_sgod import HashLabelEmbedder
    emb = dec.policy.grounding_planner.label_embedder
    assert isinstance(emb, HashLabelEmbedder)
    assert emb.embed_dim == 64


def test_build_dt_sgod_default_embedder_when_block_absent():
    config = {
        "backbone": {"name": "test-fake-backbone"},
        "oracle": {"name": "test-fake-oracle"},
        "policy": {"name": "dt-sgod"},
    }
    dec = build_from_config(config)
    from sgod.policies.dt_sgod import HashLabelEmbedder
    assert isinstance(dec.policy.grounding_planner.label_embedder, HashLabelEmbedder)


def test_build_unknown_label_embedder_type_raises():
    config = {
        "backbone": {"name": "test-fake-backbone"},
        "oracle": {"name": "test-fake-oracle"},
        "policy": {"name": "dt-sgod", "label_embedder": {"type": "nonexistent"}},
    }
    with pytest.raises(ValueError, match="Unknown label embedder type"):
        build_from_config(config)


# ── Production config files validate + dispatch correctly ────────────────────

def test_production_config_dt_sgod_llava15_yaml_parses_and_dispatches():
    """`configs/dt_sgod_llava15.yaml` builds with stubbed I/O components.

    Replaces the LLaVA backbone and RelTR oracle with the conftest fakes
    (no GPU, no checkpoints) but keeps the policy + label_embedder blocks
    intact, so the dispatch into `CLIPTextLabelEmbedder` is exercised
    against the actual production config — not an in-test dict.
    """
    from pathlib import Path
    from sgod.policies.dt_sgod import CLIPTextLabelEmbedder
    from sgod.runtime import load_config

    repo_root = Path(__file__).resolve().parent.parent
    cfg = load_config(repo_root / "configs" / "dt_sgod_llava15.yaml")
    # Replace heavy components, keep policy block authentic.
    cfg["backbone"] = {"name": "test-fake-backbone", "hidden_dim": 4096, "vocab_size": 32064}
    cfg["oracle"] = {"name": "test-fake-oracle"}

    dec = build_from_config(cfg)
    emb = dec.policy.grounding_planner.label_embedder
    assert isinstance(emb, CLIPTextLabelEmbedder)
    # ViT-B-32 is in _KNOWN_DIMS — must resolve to 512 without loading CLIP.
    assert emb.embed_dim == 512
    # GP's projection must match the embedder's dim + 5 (bbox(4) + conf(1)).
    assert dec.policy.grounding_planner.node_proj.in_features == 512 + 5


def test_production_config_dt_sgod_llava15_runtime_block_pulled_through():
    """The runtime block in the YAML (hidden_buffer_size, max_new_tokens, etc.)
    must reach `HallucinationDecoder` so a deployed config behaves as shipped."""
    from pathlib import Path
    from sgod.runtime import load_config

    repo_root = Path(__file__).resolve().parent.parent
    cfg = load_config(repo_root / "configs" / "dt_sgod_llava15.yaml")
    cfg["backbone"] = {"name": "test-fake-backbone", "hidden_dim": 4096, "vocab_size": 32064}
    cfg["oracle"] = {"name": "test-fake-oracle"}
    dec = build_from_config(cfg)
    rcfg = cfg.get("runtime", {}) or {}
    assert dec.max_new_tokens == rcfg.get("max_new_tokens", 256)
    assert dec.temperature == rcfg.get("temperature", 0.0)
    assert dec.hidden_buffer_size == rcfg.get("hidden_buffer_size", 8)
