"""Unit tests for VR-TTS components (no GPU, no model load)."""
from __future__ import annotations

import pytest
import torch
from PIL import Image

from sgod.policies.vrtts.actions.base import VisualAction, VisualState
from sgod.policies.vrtts.actions.zoom import GRID_SIZE, ZoomToAttentionRegion
from sgod.policies.vrtts.aggregator import TraceAggregator, TraceStep
from sgod.policies.vrtts.policy import HeuristicExplorationPolicy


# ── ConfidenceEstimator ────────────────────────────────────────────────────


class _FakeTokenizer:
    """Minimal tokenizer stub for ConfidenceEstimator."""

    unk_token_id = 0

    def __init__(self, mapping: dict[str, int]) -> None:
        self._m = mapping

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._m.get(token, self.unk_token_id)


def test_confidence_estimator_yes_wins():
    from sgod.policies.vrtts.confidence import ConfidenceEstimator
    tok = _FakeTokenizer({"Yes": 5, "▁Yes": 6, "yes": 7, "▁yes": 8,
                          "No": 10, "▁No": 11, "no": 12, "▁no": 13})
    est = ConfidenceEstimator(tok)
    logits = torch.zeros(20)
    logits[7] = 4.0   # yes
    logits[12] = 1.0  # no
    label, conf = est.score(logits)
    assert label == "yes"
    assert 0.5 < conf <= 1.0


def test_confidence_estimator_no_wins_with_gap():
    from sgod.policies.vrtts.confidence import ConfidenceEstimator
    tok = _FakeTokenizer({"Yes": 5, "yes": 7, "No": 10, "no": 12})
    est = ConfidenceEstimator(tok)
    logits = torch.zeros(20)
    logits[7] = -2.0
    logits[12] = 3.0
    label, conf = est.score(logits)
    assert label == "no"
    assert conf > 0


def test_confidence_estimator_low_when_tie():
    from sgod.policies.vrtts.confidence import ConfidenceEstimator
    tok = _FakeTokenizer({"Yes": 5, "No": 10})
    est = ConfidenceEstimator(tok)
    logits = torch.zeros(20)
    logits[5] = 1.0
    logits[10] = 1.0
    _, conf = est.score(logits)
    assert conf < 1e-3


def test_confidence_estimator_missing_surface_forms_raises():
    from sgod.policies.vrtts.confidence import ConfidenceEstimator
    tok = _FakeTokenizer({})  # nothing maps
    with pytest.raises(RuntimeError):
        ConfidenceEstimator(tok)


# ── TraceAggregator ────────────────────────────────────────────────────────


def test_aggregator_last_step():
    agg = TraceAggregator(mode="last")
    trace = [
        TraceStep(step=0, action_applied=None, label="no", confidence=0.1),
        TraceStep(step=1, action_applied="zoom", label="yes", confidence=0.6),
    ]
    assert agg.aggregate(trace) == "yes"


def test_aggregator_weighted_vote():
    agg = TraceAggregator(mode="weighted_vote")
    # Two no-votes with low conf vs one yes-vote with high conf → yes wins.
    trace = [
        TraceStep(step=0, action_applied=None, label="no", confidence=0.1),
        TraceStep(step=1, action_applied="zoom", label="no", confidence=0.1),
        TraceStep(step=2, action_applied="som", label="yes", confidence=0.7),
    ]
    assert agg.aggregate(trace) == "yes"


def test_aggregator_tie_breaks_yes():
    """When totals tie, weighted_vote returns 'yes' (>=)."""
    agg = TraceAggregator(mode="weighted_vote")
    trace = [
        TraceStep(step=0, action_applied=None, label="yes", confidence=0.3),
        TraceStep(step=1, action_applied="zoom", label="no", confidence=0.3),
    ]
    assert agg.aggregate(trace) == "yes"


def test_aggregator_empty_trace_raises():
    agg = TraceAggregator()
    with pytest.raises(ValueError):
        agg.aggregate([])


# ── HeuristicExplorationPolicy ─────────────────────────────────────────────


def test_policy_default_sequence():
    p = HeuristicExplorationPolicy()
    assert p.select(1) == "zoom"
    assert p.select(2) == "som"
    assert p.select(3) == "subq"
    assert p.select(4) is None


def test_policy_custom_sequence():
    p = HeuristicExplorationPolicy(action_sequence=["a", "b"])
    assert p.select(1) == "a"
    assert p.select(2) == "b"
    assert p.select(3) is None


def test_policy_step_idx_zero():
    p = HeuristicExplorationPolicy()
    # step_idx=0 is the initial state — orchestrator never asks for it,
    # but defensively the policy returns None.
    assert p.select(0) is None


# ── VisualState ────────────────────────────────────────────────────────────


def test_visual_state_clone_independent():
    img = Image.new("RGB", (10, 10))
    s = VisualState(image=img, original_image=img, prompt_suffix="abc")
    s2 = s.clone()
    s2.prompt_suffix = "xyz"
    s2.metadata["k"] = 1
    assert s.prompt_suffix == "abc"
    assert "k" not in s.metadata


# ── ZoomToAttentionRegion (no-model tests) ─────────────────────────────────


def _fake_attentions_with_peak(
    seq_len: int,
    image_positions: list[int],
    peak_grid_coords: list[tuple[int, int]],
    n_layers: int = 32,
    n_heads: int = 4,
) -> tuple[torch.Tensor, ...]:
    """Build a synthetic attention tuple where the last query position attends
    strongly to specific patches indicated by peak_grid_coords (in 24x24).
    """
    attns: list[torch.Tensor] = []
    for _ in range(n_layers):
        a = torch.zeros(1, n_heads, seq_len, seq_len)
        # Uniform low attention from the last position to all image tokens
        # plus high attention at the peaks.
        for pos in image_positions:
            a[0, :, -1, pos] = 0.001
        for r, c in peak_grid_coords:
            patch_idx = r * GRID_SIZE + c
            seq_pos = image_positions[patch_idx]
            a[0, :, -1, seq_pos] = 1.0
        attns.append(a)
    return tuple(attns)


def test_zoom_attention_grid_localises_peak():
    seq_len = 800
    img_positions = list(range(35, 35 + GRID_SIZE * GRID_SIZE))  # 576 visual tokens
    # peak at row=10, col=12
    attns = _fake_attentions_with_peak(seq_len, img_positions, [(10, 12)])
    zoom = ZoomToAttentionRegion(k_layers=5, top_percentile=80)
    grid = zoom.attention_grid(attns, img_positions)
    assert grid.shape == (GRID_SIZE, GRID_SIZE)
    # The max should be at (10, 12)
    argmax_flat = int(grid.flatten().argmax())
    r, c = divmod(argmax_flat, GRID_SIZE)
    assert (r, c) == (10, 12)


def test_zoom_bbox_from_grid():
    grid = torch.zeros(GRID_SIZE, GRID_SIZE)
    # Make a 4x3 hot region at (5..8, 7..9).
    grid[5:9, 7:10] = 1.0
    zoom = ZoomToAttentionRegion(top_percentile=80, min_bbox_frac=0.0)
    x1, y1, x2, y2 = zoom.bbox_from_grid(grid)
    assert (x1, y1, x2, y2) == (7, 5, 10, 9)


def test_zoom_crop_respects_min_frac():
    """A tiny grid bbox should be expanded to min_bbox_frac of the image."""
    zoom = ZoomToAttentionRegion(min_bbox_frac=0.5, padding_frac=0.0)
    img = Image.new("RGB", (100, 100))
    # 1-pixel bbox in 24x24 grid
    cropped = zoom.crop_image(img, (10, 10, 11, 11))
    # Should be at least 50% of each side.
    assert cropped.size[0] >= 50
    assert cropped.size[1] >= 50


def test_zoom_apply_changes_image():
    """End-to-end smoke for `apply` with synthetic attentions — no model load."""
    img = Image.new("RGB", (240, 240))
    state = VisualState(image=img, original_image=img)
    seq_len = 800
    img_positions = list(range(35, 35 + GRID_SIZE * GRID_SIZE))
    attns = _fake_attentions_with_peak(seq_len, img_positions, [(5, 5), (5, 6), (6, 5), (6, 6)])
    zoom = ZoomToAttentionRegion(top_percentile=80, min_bbox_frac=0.0, padding_frac=0.0)
    new_state = zoom.apply(
        state,
        attentions={"layers": attns, "image_token_positions": img_positions},
        question="Is X above Y?",
    )
    # Cropped image should be smaller than the original.
    w0, h0 = state.image.size
    w1, h1 = new_state.image.size
    assert w1 < w0 and h1 < h0
    assert "zoom_bbox_grid" in new_state.metadata


def test_zoom_apply_requires_attentions():
    zoom = ZoomToAttentionRegion()
    img = Image.new("RGB", (50, 50))
    state = VisualState(image=img, original_image=img)
    with pytest.raises(ValueError):
        zoom.apply(state)


# ── VisualAction base contract ─────────────────────────────────────────────


def test_visual_action_cannot_be_instantiated():
    with pytest.raises(TypeError):
        VisualAction()  # abstract
