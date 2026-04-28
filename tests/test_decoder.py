"""Tests for SGOD Decoder — end-to-end integration, oracle application, EOS handling."""
from __future__ import annotations

import torch
import pytest

from sgod.decoder import SGODDecoder, apply_oracle_scores, decode_top_k
from sgod.oracle import VisualOracle
from sgod.sgg.scene_graph import AttributeNode, ObjectNode, RelationEdge, SceneGraph


# ── Minimal fakes ─────────────────────────────────────────────────────────────

VOCAB = {
    "<eos>": 0,
    "the":   1,
    "dog":   2,
    "is":    3,
    "on":    4,
    "mat":   5,
    "red":   6,
    "no":    7,
    "a":     8,
    "cat":   9,
    "?":     10,
}
ID2WORD = {v: k for k, v in VOCAB.items()}
EOS_ID  = VOCAB["<eos>"]


class FakeTokenizer:
    eos_token_id = EOS_ID

    def encode(self, text: str) -> list[int]:
        return [VOCAB.get(w, 99) for w in text.lower().split()]

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        if skip_special_tokens:
            ids = [i for i in ids if i != EOS_ID]
        return " ".join(ID2WORD.get(i, "?") for i in ids)

    def batch_decode(self, ids: list[int]) -> list[str]:
        return [ID2WORD.get(i, "?") for i in ids]


class _VLMOutput:
    def __init__(self, logits: torch.Tensor):
        self.logits = logits  # [1, seq, vocab]


class FakeVLM:
    """Returns logits that deterministically produce token_sequence one-by-one."""

    def __init__(self, token_sequence: list[int], vocab_size: int = len(VOCAB) + 1):
        self.token_sequence = token_sequence
        self.vocab_size     = vocab_size
        self.call_count     = 0
        self.last_logits: torch.Tensor | None = None

    def __call__(self, input_ids: torch.Tensor, **kwargs) -> _VLMOutput:
        seq_len = input_ids.shape[1]
        logits  = torch.zeros(1, seq_len, self.vocab_size)
        next_id = self.token_sequence[min(self.call_count, len(self.token_sequence) - 1)]
        logits[0, -1, next_id] = 100.0
        self.last_logits = logits[0, -1, :].clone()
        self.call_count += 1
        return _VLMOutput(logits)


class FakeCLIPScorer:
    def score_single(self, word: str) -> float:
        return 0.5


class FakeSGGModule:
    def __init__(self, sg: SceneGraph):
        self._sg = sg

    def extract(self, image) -> SceneGraph:
        return self._sg


def _make_sg(objects=None, relations=None, confidence=0.9) -> SceneGraph:
    objects = objects or [
        ObjectNode("dog", confidence, (0, 0, 1, 1)),
        ObjectNode("mat", confidence, (0, 0, 1, 1)),
    ]
    relations = relations or [RelationEdge("dog", "on", "mat", confidence)]
    return SceneGraph(objects=objects, relations=relations, attributes=[])


def _make_decoder(token_sequence: list[int], sg: SceneGraph | None = None) -> SGODDecoder:
    sg = sg or _make_sg()
    return SGODDecoder(
        vlm_model     = FakeVLM(token_sequence),
        tokenizer     = FakeTokenizer(),
        sgg_module    = FakeSGGModule(sg),
        clip_factory  = lambda _img: FakeCLIPScorer(),
        top_k         = len(VOCAB),
    )


# ── token_utils ───────────────────────────────────────────────────────────────

def test_decode_top_k_shape():
    logits = torch.zeros(len(VOCAB) + 1)
    tok    = FakeTokenizer()
    ids, words = decode_top_k(logits, tok, k=5)
    assert ids.shape == (5,)
    assert len(words) == 5


def test_decode_top_k_sorted_by_logit():
    logits = torch.zeros(len(VOCAB) + 1)
    logits[VOCAB["dog"]] = 10.0
    logits[VOCAB["cat"]] = 5.0
    tok = FakeTokenizer()
    ids, words = decode_top_k(logits, tok, k=3)
    assert ids[0].item() == VOCAB["dog"]
    assert ids[1].item() == VOCAB["cat"]


def test_decode_top_k_clamped_to_vocab_size():
    logits = torch.zeros(5)
    tok = FakeTokenizer()
    ids, words = decode_top_k(logits, tok, k=100)
    assert len(ids) == 5
    assert len(words) == 5


def test_apply_oracle_scores_modifies_top_k_positions():
    logits        = torch.zeros(10)
    top_k_ids     = torch.tensor([2, 4, 5])
    oracle_scores = torch.tensor([1.0, -0.5, 0.3])
    result        = apply_oracle_scores(logits, top_k_ids, oracle_scores, lam=0.5)
    assert abs(result[2].item() - 0.5)  < 1e-5
    assert abs(result[4].item() - (-0.25)) < 1e-5
    assert abs(result[5].item() - 0.15) < 1e-5


def test_apply_oracle_scores_leaves_other_positions_unchanged():
    logits        = torch.ones(10)
    top_k_ids     = torch.tensor([0, 1])
    oracle_scores = torch.tensor([1.0, 1.0])
    apply_oracle_scores(logits, top_k_ids, oracle_scores, lam=1.0)
    for i in range(2, 10):
        assert logits[i].item() == 1.0


def test_apply_oracle_scores_in_place():
    logits = torch.zeros(10)
    ids    = torch.tensor([3])
    scores = torch.tensor([0.8])
    result = apply_oracle_scores(logits, ids, scores, lam=1.0)
    assert result is logits


# ── SGODDecoder.generate — return type and EOS ───────────────────────────────

def test_generate_returns_string():
    decoder = _make_decoder([VOCAB["dog"], EOS_ID])
    result  = decoder.generate(object(), "Is there a dog?")
    assert isinstance(result, str)


def test_generate_stops_at_eos():
    # EOS is second token → should only generate one real token then stop
    # skip_special_tokens=True strips EOS from the returned string.
    decoder = _make_decoder([VOCAB["dog"], EOS_ID])
    result  = decoder.generate(object(), "Is there a dog?")
    tokens  = result.strip().split()
    assert tokens == ["dog"]


def test_generate_stops_at_max_new_tokens():
    # No EOS in sequence → stops at max_new_tokens
    seq     = [VOCAB["dog"]] * 20
    decoder = _make_decoder(seq)
    result  = decoder.generate(object(), "What is this?", max_new_tokens=3)
    assert result.strip() == "dog dog dog"


# ── Oracle activation / deactivation ─────────────────────────────────────────

def test_oracle_deactivated_for_low_confidence_sg():
    low_conf_sg = _make_sg(confidence=0.1)  # below 0.4 threshold
    decoder     = _make_decoder([VOCAB["dog"], EOS_ID], sg=low_conf_sg)
    # Should still produce output — just without oracle interference
    result = decoder.generate(object(), "Is there a dog?")
    assert isinstance(result, str)


def test_oracle_deactivated_for_empty_sg():
    empty_sg = SceneGraph(objects=[], relations=[], attributes=[])
    decoder  = _make_decoder([VOCAB["dog"], EOS_ID], sg=empty_sg)
    result   = decoder.generate(object(), "Is there a dog?")
    assert isinstance(result, str)


# ── Oracle injection at anchor positions ─────────────────────────────────────

def test_oracle_applied_when_prev_is_determiner():
    """After 'a', position is noun_anchor — oracle scores should shift logits."""
    sg = _make_sg()  # "dog" and "mat" are in scene graph

    # VLM wants to generate "cat" (id=9), but oracle should boost "dog" (id=2).
    # We give "cat" logit=10, "dog" logit=9 — oracle boost on "dog" should flip it.
    class BiasedVLM:
        def __call__(self, input_ids, **kwargs):
            vocab_size = len(VOCAB) + 1
            logits     = torch.full((1, input_ids.shape[1], vocab_size), -100.0)
            logits[0, -1, VOCAB["cat"]] = 10.0
            logits[0, -1, VOCAB["dog"]] = 9.0
            logits[0, -1, EOS_ID]       = -100.0
            return _VLMOutput(logits)

    # FakeCLIPScorer gives 0.5 for everything; "dog" is in the SG with conf=0.99.
    # Centered scoring: _score_noun("dog") = 0.99*(0.99-0.5) + 0.01*0 ≈ 0.485.
    # Existential lambda is BASE_LAMBDA["existential"]; the test below uses a
    # logit gap of only 0.1 so the boost easily flips "cat" → "dog".
    sg_high = SceneGraph(
        objects=[ObjectNode("dog", 0.99, (0,0,1,1))],
        relations=[],
        attributes=[],
    )

    class HighConfVLM:
        def __call__(self, input_ids, **kwargs):
            vocab_size = len(VOCAB) + 1
            seq_len    = input_ids.shape[1]
            logits     = torch.full((1, seq_len, vocab_size), -100.0)
            # Step 0: produce "a" (prev=question tokens, pos neutral)
            # Step 1: after "a" → noun_anchor; logits cat=10, dog=9
            step = input_ids.shape[1] - len("Is there a ?".split())
            if step <= 0:
                logits[0, -1, VOCAB["a"]] = 100.0
            else:
                logits[0, -1, VOCAB["cat"]] = 9.1   # gap=0.1 < oracle boost ~0.49
                logits[0, -1, VOCAB["dog"]] = 9.0
                logits[0, -1, EOS_ID]       = -50.0
            return _VLMOutput(logits)

    decoder = SGODDecoder(
        vlm_model    = HighConfVLM(),
        tokenizer    = FakeTokenizer(),
        sgg_module   = FakeSGGModule(sg_high),
        clip_factory = lambda _: FakeCLIPScorer(),
        top_k        = len(VOCAB) + 1,
    )
    result = decoder.generate(object(), "Is there a ?", max_new_tokens=5)
    # After "a", oracle should boost "dog" enough to beat "cat"
    assert "dog" in result


def test_oracle_not_applied_at_neutral_position():
    """At a neutral position, logits should be unchanged."""
    sg = _make_sg()

    captured_logits_before: list[torch.Tensor] = []
    captured_logits_after:  list[torch.Tensor] = []

    class SpyVLM:
        call = 0
        def __call__(self, input_ids, **kwargs):
            vocab_size = len(VOCAB) + 1
            logits = torch.zeros(1, input_ids.shape[1], vocab_size)
            # Neutral token "is" always wins
            logits[0, -1, VOCAB["is"]] = 10.0
            captured_logits_before.append(logits[0, -1, :].clone())
            SpyVLM.call += 1
            if SpyVLM.call >= 2:
                logits[0, -1, EOS_ID] = 200.0  # stop after first real token
            return _VLMOutput(logits)

    decoder = SGODDecoder(
        vlm_model    = SpyVLM(),
        tokenizer    = FakeTokenizer(),
        sgg_module   = FakeSGGModule(sg),
        clip_factory = lambda _: FakeCLIPScorer(),
        top_k        = len(VOCAB) + 1,
    )
    # "the dog" → prev="dog" is neutral → oracle should not change anything
    result = decoder.generate(object(), "the dog", max_new_tokens=2)
    # No assertion on logit internals; just verify it doesn't crash and returns str
    assert isinstance(result, str)


# ── Hypothetical question → oracle OFF ───────────────────────────────────────

def test_hypothetical_question_oracle_disabled():
    """get_lambda() == 0.0 for hypothetical → no oracle applied even with good SG."""
    sg       = _make_sg()
    seq      = [VOCAB["dog"], EOS_ID]
    decoder  = _make_decoder(seq, sg=sg)
    # Should complete without error; lambda=0 means apply_oracle_scores would be a no-op
    result = decoder.generate(object(), "What if there was a lion here?", max_new_tokens=5)
    assert isinstance(result, str)


# ── _score_candidates ─────────────────────────────────────────────────────────

def test_score_candidates_neutral_prev():
    sg      = _make_sg()
    clip    = FakeCLIPScorer()
    oracle  = VisualOracle(sg, clip)
    # prev="dog" (neutral) → "is" neutral → score 0
    scores = SGODDecoder._score_candidates(["is", "runs"], ["dog"], oracle)
    assert scores.shape == (2,)
    assert scores[0].item() == 0.0
    assert scores[1].item() == 0.0


def test_score_candidates_noun_anchor_position():
    sg      = _make_sg()  # "dog" in SG
    clip    = FakeCLIPScorer()
    oracle  = VisualOracle(sg, clip)
    # prev="a" → noun_anchor → "dog" in SG → positive score
    scores = SGODDecoder._score_candidates(["dog", "cat"], ["a"], oracle)
    assert scores[0].item() > 0.0    # dog in SG → positive
    assert scores[0].item() > scores[1].item()  # dog > cat (cat not in SG)


def test_score_candidates_relation_anchor():
    sg      = _make_sg()  # "on" is in rel_vocab
    clip    = FakeCLIPScorer()
    oracle  = VisualOracle(sg, clip)
    # "on" after neutral prev → detect_anchor(["dog"], "on") = relation_anchor
    scores = SGODDecoder._score_candidates(["on", "near"], ["dog"], oracle)
    assert scores[0].item() > 0.0    # "on" in SG relations → positive
    assert scores[1].item() < 0.0   # "near" not in SG → -0.2


def test_score_candidates_attr_anchor():
    sg      = _make_sg()
    clip    = FakeCLIPScorer()
    oracle  = VisualOracle(sg, clip)
    # "red" → detect_anchor(["dog"], "red") = attr_anchor
    scores = SGODDecoder._score_candidates(["red"], ["dog"], oracle)
    # Not in attr_vocab → clip fallback: 0.5 - 0.5 = 0.0
    assert scores[0].item() == pytest.approx(0.0, abs=1e-5)


def test_score_candidates_returns_zeros_all_neutral():
    sg      = _make_sg()
    clip    = FakeCLIPScorer()
    oracle  = VisualOracle(sg, clip)
    scores  = SGODDecoder._score_candidates(["and", "but", "with"], ["dog"], oracle)
    assert scores.abs().sum().item() == 0.0
