"""Pytest configuration for SGOD tests.

Also defines lightweight framework fakes (`test-fake-backbone`,
`test-fake-oracle`, `test-fake-policy`) and registers them in the global
`sgod.core.registry`. This lets builder-driven tests construct a full
HallucinationDecoder without touching HF / CLIP. Registration happens at
import time of this conftest, which pytest auto-loads for every test
session, so the fakes are available regardless of which file you run.
"""
import pytest

# ── Marker handling ──────────────────────────────────────────────────────────

def pytest_configure(config):
    config.addinivalue_line("markers", "integration: requires GPU + checkpoints")


def pytest_collection_modifyitems(config, items):
    """Skip integration tests by default unless -m integration is specified."""
    if not config.getoption("-m", default=None):
        skip_integration = pytest.mark.skip(reason="needs -m integration flag")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)


# ── Framework fakes (shared across builder / pipeline / runtime tests) ───────

from sgod.core.interfaces import Backbone, Oracle, Policy  # noqa: E402
from sgod.core.registry import register  # noqa: E402
from sgod.core.types import OracleEvidence, SceneGraph  # noqa: E402


@register("backbone", "test-fake-backbone")
class FakeBackboneForTests(Backbone):
    """Hidden-state shape only; no real forward."""

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
class FakeOracleForTests(Oracle):
    def __init__(self, **_): pass
    def extract(self, image, image_meta=None):
        return OracleEvidence(scene_graph=SceneGraph())
    def vocab_scores(self, tokenizer, evidence): return {}


@register("policy", "test-fake-policy")
class FakePolicyForTests(Policy):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    def init_state(self, evidence): return {}
    def adjust_logits(self, state):
        import torch
        return torch.zeros_like(state.lm_logits)
    def update_state(self, state, sampled_token_id): return state.policy_state
