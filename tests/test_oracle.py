"""Tests for Visual Oracle — scoring logic, dual-source weighting, batch_score."""
from __future__ import annotations

import pytest
import torch

from sgod.sgg.scene_graph import AttributeNode, ObjectNode, RelationEdge, SceneGraph
from sgod.oracle.visual_oracle import VisualOracle, _build_conf_map


# ── Mock CLIPScorer ──────────────────────────────────────────────────────────

class MockCLIPScorer:
    """Stand-in for CLIPScorer; avoids loading real CLIP in unit tests.

    score_single returns values from a fixed table; unknown words return 0.3.
    """
    _scores = {
        "dog": 0.8,
        "cat": 0.7,
        "table": 0.6,
        "black": 0.75,
        "unknown": 0.3,
        "zebra": 0.1,
    }

    def score_single(self, word: str) -> float:
        return self._scores.get(word.lower(), 0.3)

    def score_words(self, words: list[str]) -> torch.Tensor:
        return torch.tensor([self.score_single(w) for w in words], dtype=torch.float32)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_sg() -> SceneGraph:
    return SceneGraph(
        objects=[
            ObjectNode(label="dog",   confidence=0.9, bbox=(10, 10, 50, 50)),
            ObjectNode(label="table", confidence=0.7, bbox=(60, 60, 120, 120)),
        ],
        relations=[
            RelationEdge(subject="dog", predicate="on", object="table", confidence=0.8),
        ],
        attributes=[
            AttributeNode(entity="dog", attribute="black", confidence=0.85),
        ],
    )


@pytest.fixture
def clip() -> MockCLIPScorer:
    return MockCLIPScorer()


@pytest.fixture
def oracle(sample_sg, clip) -> VisualOracle:
    return VisualOracle(sample_sg, clip)


# ── _build_conf_map ───────────────────────────────────────────────────────────

def test_build_conf_map_empty():
    assert _build_conf_map([]) == {}


def test_build_conf_map_single():
    result = _build_conf_map([("dog", 0.9)])
    assert result == {"dog": 0.9}


def test_build_conf_map_keeps_max():
    result = _build_conf_map([("dog", 0.5), ("dog", 0.9), ("dog", 0.7)])
    assert result["dog"] == pytest.approx(0.9)


def test_build_conf_map_lowercases():
    result = _build_conf_map([("Dog", 0.8), ("DOG", 0.6)])
    assert "dog" in result
    assert result["dog"] == pytest.approx(0.8)


def test_build_conf_map_multiple_labels():
    result = _build_conf_map([("cat", 0.7), ("dog", 0.9)])
    assert set(result.keys()) == {"cat", "dog"}


# ── Vocabulary sets ───────────────────────────────────────────────────────────

def test_noun_vocab(oracle):
    assert oracle.noun_vocab == {"dog", "table"}


def test_rel_vocab(oracle):
    assert oracle.rel_vocab == {"on"}


def test_attr_vocab(oracle):
    assert oracle.attr_vocab == {"black"}


# ── score() — neutral ─────────────────────────────────────────────────────────

def test_score_neutral_returns_zero(oracle):
    assert oracle.score("dog", "neutral") == 0.0


def test_score_unknown_anchor_returns_zero(oracle):
    assert oracle.score("dog", "unknown_type") == 0.0


# ── score() — noun_anchor ─────────────────────────────────────────────────────

def test_score_noun_in_vocab_positive(oracle):
    s = oracle.score("dog", "noun_anchor")
    assert s > 0


def test_score_noun_in_vocab_formula(oracle):
    # dog: sg_conf=0.9, clip_score=0.8
    # w_sg=0.9, w_clip=0.1
    # Centered: w_sg*(sg_conf-0.5) + w_clip*(clip-0.5)
    # = 0.9*0.4 + 0.1*0.3 = 0.36 + 0.03 = 0.39
    expected = 0.9 * (0.9 - 0.5) + 0.1 * (0.8 - 0.5)
    assert oracle.score("dog", "noun_anchor") == pytest.approx(expected, abs=1e-5)


def test_score_noun_case_insensitive(oracle):
    assert oracle.score("DOG", "noun_anchor") == pytest.approx(
        oracle.score("dog", "noun_anchor")
    )


def test_score_noun_not_in_vocab_low_clip(oracle):
    # zebra: clip_score=0.1, not in vocab
    # Out-of-vocab formula: 0.3 * (clip - 0.5) = 0.3 * -0.4 = -0.12
    s = oracle.score("zebra", "noun_anchor")
    assert s == pytest.approx(0.3 * (0.1 - 0.5), abs=1e-5)


def test_score_noun_not_in_vocab_high_clip(oracle):
    # CLIP=0.9 for chair, not in vocab:
    # 0.3 * (0.9 - 0.5) = 0.12 (mildly positive — no structural penalty)
    clip = MockCLIPScorer()
    clip._scores["chair"] = 0.9
    sg = SceneGraph(objects=[], relations=[], attributes=[])
    o = VisualOracle(sg, clip)
    s = o.score("chair", "noun_anchor")
    assert s == pytest.approx(0.3 * (0.9 - 0.5), abs=1e-5)


# ── score() — relation_anchor ─────────────────────────────────────────────────

def test_score_relation_in_vocab(oracle):
    # "on" is in rel_vocab with conf=0.8 → centered: 0.8 - 0.5 = 0.3
    assert oracle.score("on", "relation_anchor") == pytest.approx(0.3)


def test_score_relation_not_in_vocab(oracle):
    assert oracle.score("under", "relation_anchor") == pytest.approx(-0.2)


def test_score_relation_case_insensitive(oracle):
    assert oracle.score("ON", "relation_anchor") == pytest.approx(
        oracle.score("on", "relation_anchor")
    )


# ── score() — attr_anchor ─────────────────────────────────────────────────────

def test_score_attribute_in_vocab(oracle):
    # "black" with conf=0.85 → centered: 0.85 - 0.5 = 0.35
    assert oracle.score("black", "attr_anchor") == pytest.approx(0.35)


def test_score_attribute_not_in_vocab_formula(oracle):
    # "brown" not in attr_vocab; clip score for unknown = 0.3
    # result = 0.3 * (0.3 - 0.5) = -0.06 (dampened CLIP-only fallback)
    s = oracle.score("brown", "attr_anchor")
    assert s == pytest.approx(0.3 * (0.3 - 0.5), abs=1e-5)


def test_score_attribute_high_clip_positive(oracle):
    # "red" not in attr_vocab; clip=0.8 → 0.3 * (0.8 - 0.5) = 0.09 (dampened)
    clip = MockCLIPScorer()
    clip._scores["red"] = 0.8
    sg = SceneGraph(objects=[], relations=[], attributes=[])
    o = VisualOracle(sg, clip)
    assert o.score("red", "attr_anchor") == pytest.approx(0.3 * 0.3, abs=1e-5)


# ── batch_score() ─────────────────────────────────────────────────────────────

def test_batch_score_neutral_all_zeros(oracle):
    scores = oracle.batch_score(["dog", "cat", "table"], "neutral")
    assert scores.shape == (3,)
    assert scores.sum().item() == pytest.approx(0.0)


def test_batch_score_shape(oracle):
    tokens = ["dog", "on", "black", "unknown"]
    scores = oracle.batch_score(tokens, "noun_anchor")
    assert scores.shape == (len(tokens),)


def test_batch_score_empty(oracle):
    scores = oracle.batch_score([], "noun_anchor")
    assert scores.shape == (0,)


def test_batch_score_matches_individual(oracle):
    tokens = ["dog", "table", "zebra"]
    batch = oracle.batch_score(tokens, "noun_anchor")
    for i, tok in enumerate(tokens):
        assert batch[i].item() == pytest.approx(oracle.score(tok, "noun_anchor"), abs=1e-5)


def test_batch_score_is_tensor(oracle):
    result = oracle.batch_score(["dog"], "noun_anchor")
    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.float32


# ── Score range ───────────────────────────────────────────────────────────────

def test_score_range_noun_in_vocab(oracle):
    s = oracle.score("dog", "noun_anchor")
    assert -1.0 <= s <= 1.0


def test_score_range_noun_not_in_vocab(oracle):
    s = oracle.score("zebra", "noun_anchor")
    assert -1.0 <= s <= 1.0


def test_score_range_relation(oracle):
    for tok in ["on", "under", "above"]:
        s = oracle.score(tok, "relation_anchor")
        assert -1.0 <= s <= 1.0, f"out of range for '{tok}': {s}"


def test_score_range_attribute(oracle):
    for tok in ["black", "red", "brown"]:
        s = oracle.score(tok, "attr_anchor")
        assert -1.0 <= s <= 1.0, f"out of range for '{tok}': {s}"


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_empty_scene_graph_no_crash(clip):
    sg = SceneGraph(objects=[], relations=[], attributes=[])
    o = VisualOracle(sg, clip)
    assert o.noun_vocab == set()
    assert o.rel_vocab == set()
    assert o.attr_vocab == set()


def test_empty_scene_graph_scores_zero_sg(clip):
    sg = SceneGraph(objects=[], relations=[], attributes=[])
    o = VisualOracle(sg, clip)
    # Out-of-vocab: 0.3 * (clip - 0.5)
    # CLIP("unknown")=0.3 → 0.3 * -0.2 = -0.06
    s = o.score("unknown", "noun_anchor")
    assert s == pytest.approx(0.3 * (0.3 - 0.5), abs=1e-5)


def test_duplicate_objects_max_conf(clip):
    # Two entries for "dog" — conf map should keep the higher one
    sg = SceneGraph(
        objects=[
            ObjectNode(label="dog", confidence=0.5, bbox=(0, 0, 10, 10)),
            ObjectNode(label="dog", confidence=0.9, bbox=(20, 20, 30, 30)),
        ],
        relations=[],
        attributes=[],
    )
    o = VisualOracle(sg, clip)
    # sg_conf for "dog" should be 0.9 → centered formula
    # 0.9*(0.9-0.5) + 0.1*(0.8-0.5) = 0.36 + 0.03 = 0.39
    expected = 0.9 * (0.9 - 0.5) + 0.1 * (0.8 - 0.5)
    assert o.score("dog", "noun_anchor") == pytest.approx(expected, abs=1e-5)


def test_whitespace_stripped(oracle):
    # Leading/trailing spaces should be stripped before lookup
    assert oracle.score("  dog  ", "noun_anchor") == pytest.approx(
        oracle.score("dog", "noun_anchor")
    )


# ── should_activate_oracle() ──────────────────────────────────────────────────

def test_activate_oracle_high_confidence(sample_sg, clip):
    # sample_sg: dog(0.9), table(0.7) → avg=0.8 > 0.4
    o = VisualOracle(sample_sg, clip)
    assert o.should_activate_oracle() is True


def test_activate_oracle_empty_sg(clip):
    sg = SceneGraph(objects=[], relations=[], attributes=[])
    o = VisualOracle(sg, clip)
    assert o.should_activate_oracle() is False


def test_activate_oracle_low_confidence(clip):
    sg = SceneGraph(
        objects=[ObjectNode(label="dog", confidence=0.2, bbox=(0, 0, 10, 10))],
        relations=[],
        attributes=[],
    )
    o = VisualOracle(sg, clip)
    assert o.should_activate_oracle() is False


def test_activate_oracle_custom_threshold(sample_sg, clip):
    # avg=0.8; passes 0.7, fails 0.9
    o = VisualOracle(sample_sg, clip)
    assert o.should_activate_oracle(min_confidence=0.7) is True
    assert o.should_activate_oracle(min_confidence=0.9) is False


def test_activate_oracle_delegates_to_scene_graph(sample_sg, clip):
    o = VisualOracle(sample_sg, clip)
    assert o.should_activate_oracle() == sample_sg.should_activate_oracle()


# ── Bbox-conditioned attribute scoring (M1.5) ────────────────────────────────


class MockBboxScorer:
    """Stand-in for BboxClipScorer.

    Records calls and returns canned scores keyed by token. Tests use this
    to verify that VisualOracle routes through the bbox path when targets
    resolve, and falls back to image-global CLIP otherwise.
    """
    def __init__(self, scores: dict[str, float] | None = None):
        self._scores = scores or {}
        self.calls: list[tuple[str, list]] = []

    def score_token_max_across_bboxes(self, token: str, bboxes: list) -> float:
        self.calls.append((token, list(bboxes)))
        return self._scores.get(token.lower(), 0.5)


def test_oracle_without_bbox_scorer_uses_image_global(sample_sg, clip):
    # No bbox_scorer passed → existing behavior; OOV attr uses 0.3 * (clip-0.5).
    o = VisualOracle(sample_sg, clip)
    assert o._target_bboxes == []
    # MockCLIPScorer returns 0.3 for unknown words.
    s = o.score("brown", "attr_anchor")
    assert s == pytest.approx(0.3 * (0.3 - 0.5), abs=1e-5)


def test_oracle_resolves_question_targets(sample_sg, clip):
    bbox_scorer = MockBboxScorer()
    o = VisualOracle(sample_sg, clip, bbox_scorer=bbox_scorer,
                     question="What color is the dog?")
    # parse_targets("What color is the dog?") → ["dog"]; matches noun_vocab.
    assert o._target_bboxes == [(10, 10, 50, 50)]


def test_oracle_resolves_multi_target_question(sample_sg, clip):
    bbox_scorer = MockBboxScorer()
    o = VisualOracle(sample_sg, clip, bbox_scorer=bbox_scorer,
                     question="Is the dog on the table?")
    # Expect bboxes of both "dog" and "table".
    assert (10, 10, 50, 50) in o._target_bboxes
    assert (60, 60, 120, 120) in o._target_bboxes


def test_oracle_no_target_match_falls_back(sample_sg, clip):
    bbox_scorer = MockBboxScorer({"yellow": 0.9})
    # Question target "pillow" not in noun_vocab → no bboxes resolved.
    o = VisualOracle(sample_sg, clip, bbox_scorer=bbox_scorer,
                     question="What color is the pillow?")
    assert o._target_bboxes == []
    # Falls back to image-global CLIP path.
    s = o.score("yellow", "attr_anchor")
    expected = 0.3 * (clip.score_single("yellow") - 0.5)
    assert s == pytest.approx(expected, abs=1e-5)
    # Bbox scorer must NOT have been called.
    assert bbox_scorer.calls == []


def test_oracle_routes_oov_attr_through_bbox_path(sample_sg, clip):
    bbox_scorer = MockBboxScorer({"yellow": 0.9})
    o = VisualOracle(sample_sg, clip, bbox_scorer=bbox_scorer,
                     question="What color is the dog?")
    s = o.score("yellow", "attr_anchor")
    # "yellow" not in attr_vocab → bbox path; weight 0.6.
    assert s == pytest.approx(0.6 * (0.9 - 0.5), abs=1e-5)
    # Bbox scorer was called once with the dog bbox.
    assert len(bbox_scorer.calls) == 1
    token, bboxes = bbox_scorer.calls[0]
    assert token == "yellow"
    assert bboxes == [(10, 10, 50, 50)]


def test_oracle_in_vocab_attr_skips_bbox_path(sample_sg, clip):
    bbox_scorer = MockBboxScorer({"black": 0.9})
    o = VisualOracle(sample_sg, clip, bbox_scorer=bbox_scorer,
                     question="What color is the dog?")
    # "black" IS in attr_vocab → use SG confidence (0.85), bbox path untouched.
    s = o.score("black", "attr_anchor")
    assert s == pytest.approx(0.85 - 0.5, abs=1e-5)
    assert bbox_scorer.calls == []


def test_oracle_bbox_path_respects_max_across_targets(sample_sg, clip):
    # Two-target question — verify max-aggregation flows through.
    bbox_scorer = MockBboxScorer({"wooden": 0.8})  # bbox max returns 0.8
    o = VisualOracle(sample_sg, clip, bbox_scorer=bbox_scorer,
                     question="Is the dog on the table?")
    s = o.score("wooden", "attr_anchor")
    assert s == pytest.approx(0.6 * (0.8 - 0.5), abs=1e-5)
    # One bbox-scorer call, with both target bboxes passed as the candidate set.
    assert len(bbox_scorer.calls) == 1
    _, bboxes = bbox_scorer.calls[0]
    assert (10, 10, 50, 50) in bboxes
    assert (60, 60, 120, 120) in bboxes


def test_oracle_question_without_bbox_scorer_does_not_resolve(sample_sg, clip):
    # Defensive: passing question but no bbox_scorer → no resolution attempted.
    o = VisualOracle(sample_sg, clip, question="What color is the dog?")
    assert o._target_bboxes == []


def test_oracle_bbox_multiplier_scales_score(sample_sg, clip):
    # Direction-test mechanism: bbox_score_multiplier should scale the bbox path
    # output linearly so we can deliberately overshoot conservative defaults.
    bs1 = MockBboxScorer({"yellow": 0.9})
    bs2 = MockBboxScorer({"yellow": 0.9})
    o1 = VisualOracle(sample_sg, clip, bbox_scorer=bs1,
                      question="What color is the dog?",
                      bbox_score_multiplier=1.0)
    o2 = VisualOracle(sample_sg, clip, bbox_scorer=bs2,
                      question="What color is the dog?",
                      bbox_score_multiplier=5.0)
    s1 = o1.score("yellow", "attr_anchor")
    s2 = o2.score("yellow", "attr_anchor")
    assert s2 == pytest.approx(5.0 * s1, abs=1e-5)


def test_oracle_has_bbox_target_property(sample_sg, clip):
    # No bbox scorer → False
    o0 = VisualOracle(sample_sg, clip)
    assert o0.has_bbox_target is False
    # Bbox scorer + question that resolves → True
    bs = MockBboxScorer()
    o1 = VisualOracle(sample_sg, clip, bbox_scorer=bs,
                      question="What color is the dog?")
    assert o1.has_bbox_target is True
    # Bbox scorer + question that does NOT resolve → False
    o2 = VisualOracle(sample_sg, clip, bbox_scorer=bs,
                      question="What color is the pillow?")  # no "pillow" in vocab
    assert o2.has_bbox_target is False
