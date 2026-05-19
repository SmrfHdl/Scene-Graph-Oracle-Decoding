"""Tests for sgod.runtime.builder — config → HallucinationDecoder assembly.

CPU-only. We register lightweight fake Backbone/Oracle/Policy implementations
into the framework registry so the builder can construct them without
touching HF/CLIP. The fakes live in this module and clean up after themselves.
"""
from __future__ import annotations

import pytest

from sgod.core.interfaces import Backbone, Oracle, Policy
from sgod.core.registry import _REGISTRY, register
from sgod.core.types import OracleEvidence, SceneGraph
from sgod.runtime import HallucinationDecoder, build_from_config


# ── Fakes registered under throwaway names ───────────────────────────────────

@register("backbone", "test-fake-backbone")
class _FakeBackbone(Backbone):
    def __init__(self, hidden_dim=8, vocab_size=16, lazy=True, **_):
        self._hidden_dim = hidden_dim
        self._vocab_size = vocab_size
        self._lazy = lazy

    @property
    def hidden_dim(self): return self._hidden_dim
    @property
    def vocab_size(self): return self._vocab_size
    @property
    def lora_target_modules(self): return []
    @property
    def device(self): return "cpu"
    def tokenizer(self):
        class _Tok:
            eos_token_id = 0
            def decode(self, ids, **__): return ""
            def batch_decode(self, ids): return [""] * len(ids)
        return _Tok()
    def prepare_inputs(self, image, prompt): return {}
    def forward_step(self, inputs, generated_ids): raise NotImplementedError


@register("oracle", "test-fake-oracle")
class _FakeOracle(Oracle):
    def __init__(self, msg="ok", **_):
        self.msg = msg
    def extract(self, image, image_meta=None):
        return OracleEvidence(scene_graph=SceneGraph())
    def vocab_scores(self, tokenizer, evidence): return {}


@register("policy", "test-fake-policy")
class _FakePolicy(Policy):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    def init_state(self, evidence): return {}
    def adjust_logits(self, state):
        import torch
        return torch.zeros_like(state.lm_logits)
    def update_state(self, state, sampled_token_id): return state.policy_state


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
