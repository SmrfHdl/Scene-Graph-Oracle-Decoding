"""Tests for the pluggable label embedders consumed by GroundingPlanner.

CPU-only:
  - `HashLabelEmbedder` is deterministic and dependency-free; tested directly.
  - `CLIPTextLabelEmbedder` is tested with a mock open_clip model via
    the `_preloaded` injection escape hatch — no real CLIP weights.

Integration with the GP is also covered: swapping the embedder changes the
node feature dim and the projection layer adapts automatically.
"""
from __future__ import annotations

import torch

from sgod.core.types import ObjectNode, OracleEvidence, SceneGraph
from sgod.policies.dt_sgod import (
    CLIPTextLabelEmbedder,
    DTSGODConfig,
    GroundingPlanner,
    HashLabelEmbedder,
)


# ── HashLabelEmbedder ────────────────────────────────────────────────────────

def test_hash_embedder_shape_matches_embed_dim():
    emb = HashLabelEmbedder(embed_dim=64)
    v = emb.embed("dog")
    assert v.shape == (64,)
    assert emb.embed_dim == 64


def test_hash_embedder_deterministic_across_instances():
    a = HashLabelEmbedder(embed_dim=32).embed("cat")
    b = HashLabelEmbedder(embed_dim=32).embed("cat")
    assert torch.equal(a, b)


def test_hash_embedder_distinct_per_label():
    emb = HashLabelEmbedder(embed_dim=32)
    assert not torch.equal(emb.embed("dog"), emb.embed("cat"))


def test_hash_embedder_caches_repeated_calls():
    emb = HashLabelEmbedder(embed_dim=32)
    a = emb.embed("dog")
    b = emb.embed("dog")
    # Same object (cache hit) is the strongest signal.
    assert a is b


# ── CLIPTextLabelEmbedder (mocked open_clip) ────────────────────────────────

class _MockCLIPText:
    """Stand-in for an open_clip model exposing only what we need."""

    def __init__(self, dim: int = 12):
        self.dim = dim
        self.calls: list = []

    def encode_text(self, tokens):
        self.calls.append(tokens)
        # Each row of `tokens` corresponds to one label. Produce a deterministic
        # vector keyed on the token sum so different inputs → different outputs.
        rows = tokens.shape[0]
        # Use float(tokens.sum())/100 as a base offset.
        outs = []
        for i in range(rows):
            base = float(tokens[i].sum()) / 100.0
            outs.append(torch.full((self.dim,), base) + torch.arange(self.dim, dtype=torch.float32) * 0.01)
        return torch.stack(outs, dim=0)


class _MockTokenizer:
    """Returns deterministic int tensors per label string. Tokenize-by-hash."""

    def __call__(self, labels: list[str]):
        rows = []
        for lab in labels:
            # 8 ints derived from the chars; sufficient for the mock encoder.
            ids = [(ord(c) % 100) for c in (lab + "____")[:8]]
            rows.append(ids)
        return torch.tensor(rows, dtype=torch.long)


def _preloaded_clip(dim: int = 12) -> dict:
    return {"model": _MockCLIPText(dim=dim), "tokenizer": _MockTokenizer()}


def test_clip_embedder_embed_dim_probed_from_model():
    preloaded = _preloaded_clip(dim=12)
    emb = CLIPTextLabelEmbedder(_preloaded=preloaded)
    assert emb.embed_dim == 12


def test_clip_embedder_returns_normalized_vec():
    """Output must be unit-norm (L2-normalized in the text-feature space)."""
    emb = CLIPTextLabelEmbedder(_preloaded=_preloaded_clip(dim=8))
    v = emb.embed("dog")
    assert v.shape == (8,)
    assert abs(float(v.norm().item()) - 1.0) < 1e-5


def test_clip_embedder_caches_and_is_deterministic():
    emb = CLIPTextLabelEmbedder(_preloaded=_preloaded_clip(dim=8))
    a = emb.embed("dog")
    b = emb.embed("dog")
    assert a is b  # cache returns the same object


def test_clip_embedder_distinct_labels_distinct_vecs():
    emb = CLIPTextLabelEmbedder(_preloaded=_preloaded_clip(dim=8))
    assert not torch.equal(emb.embed("dog"), emb.embed("table"))


# ── GroundingPlanner integration with CLIP embedder ─────────────────────────

def _evidence(num_objects: int) -> OracleEvidence:
    objs = [
        ObjectNode(label=f"obj_{i}", confidence=0.7, bbox=(0.0, 0.0, 10.0, 10.0))
        for i in range(num_objects)
    ]
    return OracleEvidence(scene_graph=SceneGraph(objects=objs, image_size=(640, 480)))


def test_gp_picks_up_embedder_embed_dim_for_proj():
    """GP node projection in-features must equal `embedder.embed_dim + 5`."""
    clip = CLIPTextLabelEmbedder(_preloaded=_preloaded_clip(dim=32))
    gp = GroundingPlanner(hidden_dim=64, config=DTSGODConfig(), label_embedder=clip)
    # node_proj is Linear(in_features = embed_dim + 5, out_features = hidden_dim)
    assert gp.node_proj.in_features == 32 + 5
    assert gp.node_embed_dim == 32


def test_gp_forward_with_clip_embedder_runs():
    clip = CLIPTextLabelEmbedder(_preloaded=_preloaded_clip(dim=16))
    gp = GroundingPlanner(hidden_dim=64, config=DTSGODConfig(), label_embedder=clip)
    h_prev = gp.init_state(_evidence(3), batch_size=2, device="cpu")
    prefix = torch.randn(2, 64)
    h_new = gp(h_prev, prefix, _evidence(3))
    assert h_new.shape == h_prev.shape
    assert not torch.allclose(h_new, h_prev)


def test_gp_default_embedder_is_hash_with_node_embed_dim():
    """Without an explicit embedder, GP must wrap a HashLabelEmbedder(node_embed_dim)."""
    gp = GroundingPlanner(hidden_dim=64, config=DTSGODConfig(), node_embed_dim=42)
    assert isinstance(gp.label_embedder, HashLabelEmbedder)
    assert gp.label_embedder.embed_dim == 42
    assert gp.node_proj.in_features == 42 + 5
