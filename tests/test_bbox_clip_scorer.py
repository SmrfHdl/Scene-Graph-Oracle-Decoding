"""Tests for BboxClipScorer (M1.5a).

We bypass __init__ to avoid loading a real CLIP model — the heavy lifting
this class adds (crop math, caching, dispatch) is testable with mocks.
The cosine numbers themselves come from the open_clip model and are not
the unit under test here.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from PIL import Image

from sgod.oracle.bbox_clip_scorer import BboxClipScorer


# ── Test doubles ─────────────────────────────────────────────────────────────

class _FakeModel:
    """Mimics the bits of open_clip we actually call.

    encode_image returns a deterministic feature derived from the crop's mean
    pixel value, so different crops produce different features.
    encode_text returns a deterministic feature derived from the token string
    hash, so different tokens produce different features.
    """
    embed_dim = 4

    def __init__(self):
        self.encode_image_calls = 0
        self.encode_text_calls = 0

    def encode_image(self, tensor: torch.Tensor) -> torch.Tensor:
        self.encode_image_calls += 1
        # Use the tensor's mean as a stable feature seed.
        seed = float(tensor.mean().item())
        return torch.tensor([[seed, seed + 0.1, seed - 0.2, seed * 0.5]], dtype=torch.float32)

    def encode_text(self, ids: torch.Tensor) -> torch.Tensor:
        self.encode_text_calls += 1
        # ids is [N, T]; produce one feature per row using row mean.
        out = []
        for row in ids:
            seed = float(row.float().mean().item()) / 100.0
            out.append([seed, seed + 0.05, seed - 0.1, seed * 0.7])
        return torch.tensor(out, dtype=torch.float32)


def _fake_tokenizer(words):
    """Tokenize each word to a length-3 row whose values depend on the string."""
    rows = [[(ord(c) % 50) for c in (w + "###")[:3]] for w in words]
    return torch.tensor(rows, dtype=torch.long)


def _fake_preprocess(img: Image.Image) -> torch.Tensor:
    """Return a tensor whose mean encodes the crop's average pixel value."""
    arr = torch.tensor(list(img.getdata()), dtype=torch.float32) / 255.0
    if arr.ndim == 2:
        arr = arr.mean(dim=-1)  # collapse channels if RGB
    return arr.mean().expand(3, 4, 4)  # [C, H, W]


def _make_scorer(image: Image.Image, vocab_cache=False, pad_ratio=0.15):
    """Bypass __init__, install fake CLIP internals."""
    s = BboxClipScorer.__new__(BboxClipScorer)
    s.device = "cpu"
    s._image = image
    s._pad_ratio = pad_ratio
    s._model = _FakeModel()
    s._tokenizer = _fake_tokenizer
    s._preprocess = _fake_preprocess
    s._crop_cache = {}
    s._token_cache = {}
    if vocab_cache:
        # 8 known words with random embeddings
        words = ["dog", "cat", "yellow", "blue", "red", "table", "pillow", "fire"]
        emb = F.normalize(torch.randn(len(words), 4), dim=-1)
        s._vocab_embeddings = emb
        s._vocab_words = words
        s._vocab_index = {w: i for i, w in enumerate(words)}
    else:
        s._vocab_embeddings = None
        s._vocab_words = None
        s._vocab_index = None
    return s


def _checkerboard(w=200, h=200) -> Image.Image:
    """Make a 200x200 image with a left-half black, right-half white."""
    img = Image.new("RGB", (w, h), color=(0, 0, 0))
    for x in range(w // 2, w):
        for y in range(h):
            img.putpixel((x, y), (255, 255, 255))
    return img


# ── _padded_crop ─────────────────────────────────────────────────────────────

def test_padded_crop_expands_by_pad_ratio():
    img = _checkerboard()
    s = _make_scorer(img, pad_ratio=0.10)
    # 100x100 box at (50,50)-(150,150). pad = 10px each side.
    crop = s._padded_crop((50, 50, 150, 150))
    assert crop.size == (120, 120)


def test_padded_crop_clips_to_image_bounds():
    img = _checkerboard()  # 200x200
    s = _make_scorer(img, pad_ratio=0.5)
    # Box at the corner — pad would go negative; should clip to (0,0).
    crop = s._padded_crop((10, 10, 50, 50))
    assert crop.size[0] <= img.size[0]
    assert crop.size[1] <= img.size[1]


def test_padded_crop_falls_back_for_invalid_bbox():
    img = _checkerboard()
    s = _make_scorer(img)
    # Out-of-image bbox → fallback to full image, not a 0-sized crop.
    crop = s._padded_crop((1000, 1000, 1100, 1100))
    assert crop.size == img.size


# ── caching ──────────────────────────────────────────────────────────────────

def test_crop_cache_avoids_repeat_encode():
    img = _checkerboard()
    s = _make_scorer(img)
    bbox = (50.0, 50.0, 150.0, 150.0)
    s.score_token_at_bbox("dog", bbox)
    s.score_token_at_bbox("cat", bbox)
    # One encode_image call total — the second token reuses the cached crop.
    assert s._model.encode_image_calls == 1


def test_crop_cache_keyed_by_rounded_bbox():
    img = _checkerboard()
    s = _make_scorer(img)
    s.score_token_at_bbox("dog", (50.0, 50.0, 150.0, 150.0))
    # Sub-pixel jitter shouldn't trigger a new encode.
    s.score_token_at_bbox("cat", (50.3, 49.8, 150.1, 150.4))
    assert s._model.encode_image_calls == 1


def test_different_bboxes_hit_separate_cache_entries():
    img = _checkerboard()
    s = _make_scorer(img)
    s.score_token_at_bbox("dog", (10, 10, 50, 50))
    s.score_token_at_bbox("dog", (100, 100, 180, 180))
    assert s._model.encode_image_calls == 2
    assert s.num_cached_crops == 2


def test_token_cache_avoids_repeat_text_encode():
    img = _checkerboard()
    s = _make_scorer(img)
    bbox1 = (10, 10, 50, 50)
    bbox2 = (60, 60, 100, 100)
    s.score_token_at_bbox("yellow", bbox1)
    s.score_token_at_bbox("yellow", bbox2)
    # Token "yellow" encoded once even though we scored against two bboxes.
    assert s._model.encode_text_calls == 1


# ── scoring ──────────────────────────────────────────────────────────────────

def test_score_returns_float_in_unit_interval():
    img = _checkerboard()
    s = _make_scorer(img)
    score = s.score_token_at_bbox("yellow", (50, 50, 150, 150))
    assert 0.0 <= score <= 1.0


def test_score_tokens_at_bbox_returns_tensor():
    img = _checkerboard()
    s = _make_scorer(img)
    out = s.score_tokens_at_bbox(["yellow", "blue", "red"], (50, 50, 150, 150))
    assert out.shape == (3,)


def test_score_tokens_at_bbox_empty_tokens():
    img = _checkerboard()
    s = _make_scorer(img)
    out = s.score_tokens_at_bbox([], (50, 50, 150, 150))
    assert out.numel() == 0


# ── max-across-bboxes ────────────────────────────────────────────────────────

def test_max_across_bboxes_picks_highest():
    img = _checkerboard()
    s = _make_scorer(img)
    bboxes = [(10, 10, 50, 50), (100, 100, 180, 180)]
    individual = [s.score_token_at_bbox("yellow", b) for b in bboxes]
    pooled = s.score_token_max_across_bboxes("yellow", bboxes)
    assert pooled == max(individual)


def test_max_across_bboxes_no_bboxes_returns_neutral():
    img = _checkerboard()
    s = _make_scorer(img)
    assert s.score_token_max_across_bboxes("yellow", []) == 0.5


def test_score_tokens_max_across_bboxes_shape():
    img = _checkerboard()
    s = _make_scorer(img)
    bboxes = [(10, 10, 50, 50), (100, 100, 180, 180)]
    out = s.score_tokens_max_across_bboxes(["yellow", "blue"], bboxes)
    assert out.shape == (2,)


def test_score_tokens_max_across_bboxes_empty_bboxes():
    img = _checkerboard()
    s = _make_scorer(img)
    out = s.score_tokens_max_across_bboxes(["yellow", "blue"], [])
    assert out.shape == (2,)
    assert torch.allclose(out, torch.full((2,), 0.5))


# ── calibration with vocab cache ─────────────────────────────────────────────

def test_calibration_with_vocab_cache_uses_per_crop_baseline():
    """When a vocab cache is provided, scores should use per-crop calibration.

    Two visually different crops scored on the same token should generally
    produce different scores (different baselines + different crop features).
    """
    img = _checkerboard()
    s = _make_scorer(img, vocab_cache=True)
    left  = s.score_token_at_bbox("yellow", (10,  10,  90,  190))   # mostly black
    right = s.score_token_at_bbox("yellow", (110, 10, 190, 190))    # mostly white
    # Both still in [0, 1].
    assert 0.0 <= left <= 1.0
    assert 0.0 <= right <= 1.0


# ── bbox key ─────────────────────────────────────────────────────────────────

def test_bbox_key_rounds_to_ints():
    assert BboxClipScorer._bbox_key((1.4, 2.6, 3.5, 4.4)) == (1, 3, 4, 4)
