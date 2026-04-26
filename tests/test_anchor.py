"""Tests for Anchor Detector — pattern matching, edge cases (compound nouns, pronouns)."""
import pytest

from sgod.anchor import detect_anchor
from sgod.anchor.vocabularies import (
    COLOR_ATTRS,
    DETERMINERS,
    SIZE_ATTRS,
    SPATIAL_PREPS,
    SPATIAL_PREPS_MULTI,
)


# ── Vocabulary sanity ─────────────────────────────────────────────────────────

def test_vocabularies_are_frozensets():
    assert isinstance(DETERMINERS, frozenset)
    assert isinstance(SPATIAL_PREPS, frozenset)
    assert isinstance(SPATIAL_PREPS_MULTI, frozenset)
    assert isinstance(COLOR_ATTRS, frozenset)
    assert isinstance(SIZE_ATTRS, frozenset)


def test_determiners_content():
    assert {"a", "an", "the", "this", "that", "these", "those", "some"} <= DETERMINERS


def test_spatial_preps_single_content():
    assert {"on", "under", "above", "below", "near", "behind", "beside",
            "at", "in", "over", "across", "against", "along"} <= SPATIAL_PREPS


def test_spatial_preps_multi_content():
    assert {"next to", "in front of", "across from"} <= SPATIAL_PREPS_MULTI


def test_color_attrs_content():
    assert {"red", "blue", "green", "black", "white", "yellow", "brown"} <= COLOR_ATTRS


def test_size_attrs_content():
    assert {"large", "small", "big", "tiny", "huge", "tall", "short"} <= SIZE_ATTRS


def test_no_overlap_single_and_multi_preps():
    for p in SPATIAL_PREPS:
        assert p not in SPATIAL_PREPS_MULTI


# ── Edge cases: empty / minimal inputs ───────────────────────────────────────

def test_empty_prev_tokens_returns_neutral():
    assert detect_anchor([], "dog") == "neutral"


def test_empty_prev_tokens_with_color_returns_neutral():
    assert detect_anchor([], "red") == "neutral"


def test_single_token_prev_neutral_word():
    assert detect_anchor(["is"], "dog") == "neutral"


# ── Rule 1: Determiner → noun_anchor ─────────────────────────────────────────

@pytest.mark.parametrize("det", ["a", "an", "the", "this", "that", "these", "those", "some"])
def test_determiner_yields_noun_anchor(det):
    assert detect_anchor([det], "dog") == "noun_anchor"


def test_determiner_case_insensitive():
    assert detect_anchor(["A"], "dog") == "noun_anchor"
    assert detect_anchor(["THE"], "chair") == "noun_anchor"


def test_determiner_with_whitespace():
    assert detect_anchor(["  the  "], "cat") == "noun_anchor"


def test_determiner_before_any_word():
    assert detect_anchor(["a"], "volcano") == "noun_anchor"
    assert detect_anchor(["the"], "on") == "noun_anchor"


# ── Rule 2: Single-token spatial prep → noun_anchor ──────────────────────────

@pytest.mark.parametrize("prep", ["on", "under", "above", "below", "beside", "behind", "near",
                                   "at", "in", "over", "across", "against", "along"])
def test_spatial_prep_prev_yields_noun_anchor(prep):
    assert detect_anchor([prep], "table") == "noun_anchor"


def test_spatial_prep_prev_case_insensitive():
    assert detect_anchor(["ON"], "table") == "noun_anchor"
    assert detect_anchor(["Near"], "dog") == "noun_anchor"


def test_after_spatial_prep_current_is_determiner():
    # "on the [TABLE]" — prev="the" (determiner) triggers noun_anchor
    assert detect_anchor(["on", "the"], "table") == "noun_anchor"


# ── Rule 3: Multi-word spatial prep → noun_anchor ────────────────────────────

def test_next_to_bigram_yields_noun_anchor():
    assert detect_anchor(["next", "to"], "dog") == "noun_anchor"


def test_in_front_of_trigram_yields_noun_anchor():
    assert detect_anchor(["in", "front", "of"], "the") == "noun_anchor"


def test_across_from_bigram_yields_noun_anchor():
    assert detect_anchor(["across", "from"], "table") == "noun_anchor"


def test_multi_word_prep_case_insensitive():
    assert detect_anchor(["NEXT", "TO"], "dog") == "noun_anchor"


# ── Rule 4: Current token is spatial prep → relation_anchor ──────────────────

@pytest.mark.parametrize("prep", ["on", "under", "above", "below", "beside", "behind", "near",
                                   "at", "in", "over", "across", "against", "along"])
def test_current_is_spatial_prep_yields_relation_anchor(prep):
    assert detect_anchor(["dog"], prep) == "relation_anchor"


def test_current_spatial_prep_case_insensitive():
    assert detect_anchor(["dog"], "ON") == "relation_anchor"
    assert detect_anchor(["cat"], "Near") == "relation_anchor"


def test_next_to_completion_yields_relation_anchor():
    # "dog next [to]": cur="to" completes "next to"
    assert detect_anchor(["dog", "next"], "to") == "relation_anchor"


def test_in_front_of_completion_yields_relation_anchor():
    # "dog in front [of]": cur="of" completes "in front of"
    assert detect_anchor(["dog", "in", "front"], "of") == "relation_anchor"


def test_across_from_completion_yields_relation_anchor():
    assert detect_anchor(["dog", "across"], "from") == "relation_anchor"


# ── Rule 5: Color / size attribute → attr_anchor ─────────────────────────────

@pytest.mark.parametrize("color", ["red", "blue", "green", "black", "white", "yellow",
                                    "orange", "purple", "pink", "brown", "gray", "golden"])
def test_color_yields_attr_anchor(color):
    assert detect_anchor(["dog"], color) == "attr_anchor"


@pytest.mark.parametrize("size", ["large", "small", "big", "tiny", "huge", "tall", "short"])
def test_size_yields_attr_anchor(size):
    assert detect_anchor(["dog"], size) == "attr_anchor"


def test_attr_case_insensitive():
    assert detect_anchor(["dog"], "RED") == "attr_anchor"
    assert detect_anchor(["dog"], "Large") == "attr_anchor"


def test_color_after_neutral_prev_is_attr():
    assert detect_anchor(["dog"], "black") == "attr_anchor"
    assert detect_anchor(["is"], "yellow") == "attr_anchor"


# ── Rule 1b: det + adjective + noun → noun_anchor ────────────────────────────

def test_noun_after_det_size_adj():
    # "a large [table]" — prev="large", two_back="a" (determiner)
    assert detect_anchor(["a", "large"], "table") == "noun_anchor"


def test_noun_after_det_color_adj():
    # "the black [dog]"
    assert detect_anchor(["the", "black"], "dog") == "noun_anchor"


def test_noun_after_det_size_adj_longer_context():
    # "There is a large [table]"
    assert detect_anchor(["There", "is", "a", "large"], "table") == "noun_anchor"


def test_rule1b_does_not_fire_without_det_two_back():
    # "is large [table]" — two_back="is" is not a determiner → neutral
    assert detect_anchor(["is", "large"], "table") == "neutral"


# ── Rule priority ─────────────────────────────────────────────────────────────

def test_determiner_beats_attr_on_prev():
    # "a [red]": prev="a" → noun_anchor, even though "red" ∈ COLOR_ATTRS
    assert detect_anchor(["a"], "red") == "noun_anchor"


def test_determiner_beats_spatial_prep_on_prev():
    # "the [on]": prev="the" → noun_anchor
    assert detect_anchor(["the"], "on") == "noun_anchor"


def test_spatial_prep_prev_beats_attr():
    # "near [large]": prev="near" → noun_anchor (not attr_anchor)
    assert detect_anchor(["near"], "large") == "noun_anchor"


def test_relation_before_attr():
    # "dog [on]": cur="on" is spatial prep → relation_anchor not attr_anchor
    assert detect_anchor(["dog"], "on") == "relation_anchor"


# ── Neutral cases ─────────────────────────────────────────────────────────────

def test_plain_verb_is_neutral():
    assert detect_anchor(["dog"], "runs") == "neutral"
    assert detect_anchor(["is"], "there") == "neutral"


def test_pronoun_is_neutral():
    assert detect_anchor(["it"], "is") == "neutral"
    assert detect_anchor(["he"], "sits") == "neutral"


def test_conjunction_is_neutral():
    assert detect_anchor(["dog"], "and") == "neutral"


def test_unknown_word_is_neutral():
    assert detect_anchor(["foo"], "bar") == "neutral"


def test_number_is_neutral():
    assert detect_anchor(["of"], "two") == "neutral"


# ── Return value contract ─────────────────────────────────────────────────────

_VALID_TYPES = {"noun_anchor", "relation_anchor", "attr_anchor", "neutral"}


@pytest.mark.parametrize("prev,cur", [
    ([], "dog"),
    (["a"], "dog"),
    (["dog"], "on"),
    (["dog"], "red"),
    (["next", "to"], "dog"),
    (["is"], "there"),
])
def test_return_value_is_valid_type(prev, cur):
    result = detect_anchor(prev, cur)
    assert result in _VALID_TYPES


# ── Realistic sentence fragments ─────────────────────────────────────────────

def test_sentence_a_dog():
    # "There is a [dog]"
    assert detect_anchor(["There", "is", "a"], "dog") == "noun_anchor"


def test_sentence_the_large():
    # "There is the [large]" → determiner beats attr
    assert detect_anchor(["There", "is", "the"], "large") == "noun_anchor"


def test_sentence_dog_on():
    # "The dog [on] the table" — cur="on"
    assert detect_anchor(["The", "dog"], "on") == "relation_anchor"


def test_sentence_on_the_table():
    # "dog on the [table]" — prev="the" (determiner)
    assert detect_anchor(["dog", "on", "the"], "table") == "noun_anchor"


def test_sentence_black_after_determiner():
    # "I see a [black]" — prev="a" → noun_anchor (not attr)
    assert detect_anchor(["I", "see", "a"], "black") == "noun_anchor"


def test_sentence_color_after_verb():
    # "The dog is [black]" — prev="is" → attr_anchor
    assert detect_anchor(["The", "dog", "is"], "black") == "attr_anchor"


def test_sentence_next_to():
    # "The dog is next to [the]"
    assert detect_anchor(["The", "dog", "is", "next", "to"], "the") == "noun_anchor"


def test_sentence_in_front_of():
    # "standing in front of [a]"
    assert detect_anchor(["standing", "in", "front", "of"], "a") == "noun_anchor"


def test_sentence_chair_at_table():
    # RelTR common pattern: "chair [at] the table"
    assert detect_anchor(["chair"], "at") == "relation_anchor"


def test_sentence_at_the_table():
    # "chair at [the] table" — prev="at" is spatial prep → noun_anchor
    assert detect_anchor(["chair", "at"], "the") == "noun_anchor"


def test_sentence_over_the_fence():
    assert detect_anchor(["jumping"], "over") == "relation_anchor"
    assert detect_anchor(["jumping", "over"], "the") == "noun_anchor"
