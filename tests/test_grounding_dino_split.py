"""Tests for GroundingDinoModule._split_label (fused-label cleanup).

These tests don't touch the model — they exercise the pure-Python label
splitter so we can validate the fix without GPU/checkpoint dependencies.
"""
from __future__ import annotations

from sgod.sgg.grounding_dino_module import GroundingDinoModule


def _make_splitter(vocab: list[str]):
    """Build a GroundingDinoModule shell that only exposes _split_label.

    We bypass __init__ (which downloads HF weights) by constructing a
    lightweight stand-in object that has the same _split_label method
    bound to the right _vocab_set.
    """
    obj = GroundingDinoModule.__new__(GroundingDinoModule)
    obj._vocab_set = {v.lower() for v in vocab}
    return obj


def test_keeps_exact_vocab_match():
    sp = _make_splitter(["dog", "fire truck"])
    assert sp._split_label("dog") == ["dog"]
    assert sp._split_label("fire truck") == ["fire truck"]


def test_splits_fused_two_words():
    sp = _make_splitter(["road", "street"])
    assert sp._split_label("road street") == ["road", "street"]


def test_splits_fused_three_words():
    sp = _make_splitter(["armchair", "bench", "chair"])
    assert sp._split_label("armchair bench chair") == ["armchair", "bench", "chair"]


def test_prefers_multi_word_vocab_over_split():
    # "fire truck" is in vocab → must NOT split into ["fire", "truck"]
    # even though both words also exist as singletons.
    sp = _make_splitter(["fire", "truck", "fire truck"])
    assert sp._split_label("fire truck") == ["fire truck"]


def test_mixed_multi_word_and_single():
    # "fire truck van" → ["fire truck", "van"]
    sp = _make_splitter(["fire truck", "van"])
    assert sp._split_label("fire truck van") == ["fire truck", "van"]


def test_drops_unknown_tokens():
    # "the dog something" → only "dog" is in vocab
    sp = _make_splitter(["dog"])
    assert sp._split_label("the dog something") == ["dog"]


def test_empty_label():
    sp = _make_splitter(["dog"])
    assert sp._split_label("") == []
    assert sp._split_label("   ") == []


def test_fallback_keeps_unknown_label():
    # No words match vocab → preserve original so we don't drop the detection.
    sp = _make_splitter(["dog"])
    assert sp._split_label("xyz") == ["xyz"]


def test_case_insensitive():
    sp = _make_splitter(["dog", "fire truck"])
    assert sp._split_label("DOG") == ["dog"]
    assert sp._split_label("Fire Truck") == ["fire truck"]
