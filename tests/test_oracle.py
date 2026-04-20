"""Tests for Visual Oracle — scoring logic, dual-source weighting, batch_score."""
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
    # result = 0.9 * 0.9 + 0.1 * 0.8 = 0.81 + 0.08 = 0.89
    expected = 0.9 * 0.9 + 0.1 * 0.8
    assert oracle.score("dog", "noun_anchor") == pytest.approx(expected, abs=1e-5)


def test_score_noun_case_insensitive(oracle):
    assert oracle.score("DOG", "noun_anchor") == pytest.approx(
        oracle.score("dog", "noun_anchor")
    )


def test_score_noun_not_in_vocab_low_clip(oracle):
    # zebra: clip_score=0.1, sg_conf=0
    # w_sg=0, w_clip=1
    # result = 0*(-0.3) + 1*(0.1 - 0.5) = -0.4
    s = oracle.score("zebra", "noun_anchor")
    assert s == pytest.approx(-0.4, abs=1e-5)


def test_score_noun_not_in_vocab_high_clip(oracle):
    # When CLIP score is 0.9 for some word not in vocab and sg_conf=0:
    # result = 0*(-0.3) + 1*(0.9 - 0.5) = 0.4  → positive CLIP signal
    clip = MockCLIPScorer()
    clip._scores["chair"] = 0.9
    sg = SceneGraph(objects=[], relations=[], attributes=[])
    o = VisualOracle(sg, clip)
    s = o.score("chair", "noun_anchor")
    assert s == pytest.approx(0.4, abs=1e-5)


# ── score() — relation_anchor ─────────────────────────────────────────────────

def test_score_relation_in_vocab(oracle):
    # "on" is in rel_vocab with conf=0.8
    assert oracle.score("on", "relation_anchor") == pytest.approx(0.8)


def test_score_relation_not_in_vocab(oracle):
    assert oracle.score("under", "relation_anchor") == pytest.approx(-0.2)


def test_score_relation_case_insensitive(oracle):
    assert oracle.score("ON", "relation_anchor") == pytest.approx(
        oracle.score("on", "relation_anchor")
    )


# ── score() — attr_anchor ─────────────────────────────────────────────────────

def test_score_attribute_in_vocab(oracle):
    # "black" with conf=0.85
    assert oracle.score("black", "attr_anchor") == pytest.approx(0.85)


def test_score_attribute_not_in_vocab_formula(oracle):
    # "brown" not in attr_vocab; clip score for unknown = 0.3
    # result = 0.3 - 0.5 = -0.2
    s = oracle.score("brown", "attr_anchor")
    assert s == pytest.approx(0.3 - 0.5, abs=1e-5)


def test_score_attribute_high_clip_positive(oracle):
    # "red": clip_score=0.8 → 0.8 - 0.5 = 0.3
    clip = MockCLIPScorer()
    clip._scores["red"] = 0.8
    sg = SceneGraph(objects=[], relations=[], attributes=[])
    o = VisualOracle(sg, clip)
    assert o.score("red", "attr_anchor") == pytest.approx(0.3, abs=1e-5)


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
    # No SGG signal; CLIP dominates with score_single("unknown")=0.3 → 0 + 1*(0.3-0.5)=-0.2
    s = o.score("unknown", "noun_anchor")
    assert s == pytest.approx(0.3 - 0.5, abs=1e-5)


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
    # sg_conf for "dog" should be 0.9
    # score = 0.9 * 0.9 + 0.1 * 0.8 (MockCLIP dog=0.8)
    expected = 0.9 * 0.9 + 0.1 * 0.8
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
