"""Regression test: CLIPScorer must skip a mismatched vocab cache, not crash.

The pre-fix behavior was that loading a 512-dim cache (e.g. ViT-B-32) into a
768-dim model (e.g. ViT-L-14) blew up deep inside `score_words` with:

    RuntimeError: mat1 and mat2 shapes cannot be multiplied (V×512 and 768×1)

The fix probes the model's text-embedding dim at construction time and
falls back to live encoding when the cache dim disagrees. This test exercises
that path with a tiny fake CLIP model + a fake cache file written to a temp.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from sgod.oracle.clip_scorer import CLIPScorer


class _FakeCLIP:
    """Stand-in for an open_clip model exposing only what CLIPScorer touches."""

    def __init__(self, text_dim: int = 768, image_dim: int = 768):
        self.text_dim = text_dim
        self.image_dim = image_dim

    def to(self, *_args, **_kwargs):
        return self

    def eval(self):
        return self

    def encode_text(self, tokens):
        N = tokens.shape[0]
        return torch.zeros(N, self.text_dim)

    def encode_image(self, _image_tensor):
        return torch.zeros(1, self.image_dim)


class _FakeTokenizer:
    def __call__(self, words):
        return torch.zeros(len(words), 4, dtype=torch.long)


def _identity_preprocess(image):
    return torch.zeros(3, 16, 16)


def _preloaded(text_dim: int):
    return {
        "model": _FakeCLIP(text_dim=text_dim),
        "tokenizer": _FakeTokenizer(),
        "preprocess": _identity_preprocess,
    }


def _write_cache(path: Path, n_words: int, dim: int) -> Path:
    """Persist a fake `clip_vocab_cache.pt` with the requested embed dim."""
    payload = {
        "embeddings": torch.randn(n_words, dim),
        "words": [f"w{i}" for i in range(n_words)],
    }
    torch.save(payload, path)
    return path


# ── tests ────────────────────────────────────────────────────────────────────

def test_clipscorer_cache_dim_mismatch_falls_back_to_live_encoding(monkeypatch, caplog):
    """Construct CLIPScorer without _preloaded so the cache-loading branch runs.

    We monkeypatch `open_clip.create_model_and_transforms` and `get_tokenizer`
    to return our fakes; the cache file we write has a deliberately wrong dim,
    and we verify that construction succeeds, vocab_embeddings is None
    (i.e., the cache was skipped), and a warning was logged.
    """
    from PIL import Image

    import open_clip

    monkeypatch.setattr(
        open_clip, "create_model_and_transforms",
        lambda *_a, **_k: (_FakeCLIP(text_dim=768), None, _identity_preprocess),
    )
    monkeypatch.setattr(open_clip, "get_tokenizer", lambda _name: _FakeTokenizer())

    with tempfile.TemporaryDirectory() as tmp:
        cache_path = _write_cache(Path(tmp) / "vocab.pt", n_words=50, dim=512)

        with caplog.at_level("WARNING", logger="sgod.oracle.clip_scorer"):
            scorer = CLIPScorer(
                image=Image.new("RGB", (16, 16)),
                model_name="ViT-L-14",
                pretrained="openai",
                device="cpu",
                vocab_cache_path=cache_path,
            )

    # Cache was rejected → cache fields stay None → live encoding path active.
    assert scorer._vocab_embeddings is None
    assert scorer._vocab_words is None
    assert scorer._vocab_index is None
    # And a warning explaining the mismatch was emitted.
    assert any("dim=512" in r.message and "dim=768" in r.message for r in caplog.records), (
        f"expected dim-mismatch warning; got: {[r.message for r in caplog.records]}"
    )


def test_clipscorer_cache_dim_match_loads_cache(monkeypatch):
    """Sanity: when dims match, the cache IS loaded (no spurious skipping)."""
    from PIL import Image

    import open_clip

    monkeypatch.setattr(
        open_clip, "create_model_and_transforms",
        lambda *_a, **_k: (_FakeCLIP(text_dim=768), None, _identity_preprocess),
    )
    monkeypatch.setattr(open_clip, "get_tokenizer", lambda _name: _FakeTokenizer())

    with tempfile.TemporaryDirectory() as tmp:
        cache_path = _write_cache(Path(tmp) / "vocab.pt", n_words=50, dim=768)

        scorer = CLIPScorer(
            image=Image.new("RGB", (16, 16)),
            model_name="ViT-L-14",
            device="cpu",
            vocab_cache_path=cache_path,
        )

    assert scorer._vocab_embeddings is not None
    assert scorer._vocab_embeddings.shape == (50, 768)
    assert len(scorer._vocab_words) == 50
    assert scorer._vocab_index["w7"] == 7


def test_clipscorer_no_cache_path_works_unchanged(monkeypatch):
    """Regression guard: omitting vocab_cache_path triggers live encoding only."""
    from PIL import Image

    import open_clip

    monkeypatch.setattr(
        open_clip, "create_model_and_transforms",
        lambda *_a, **_k: (_FakeCLIP(text_dim=768), None, _identity_preprocess),
    )
    monkeypatch.setattr(open_clip, "get_tokenizer", lambda _name: _FakeTokenizer())

    scorer = CLIPScorer(
        image=Image.new("RGB", (16, 16)),
        model_name="ViT-L-14",
        device="cpu",
    )
    assert scorer._vocab_embeddings is None
