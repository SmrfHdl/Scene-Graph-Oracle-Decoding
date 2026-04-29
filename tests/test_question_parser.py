"""Tests for sgod.oracle.question_parser."""
from __future__ import annotations

import pytest

from sgod.oracle.question_parser import (
    match_targets_to_vocab,
    parse_targets,
)


# ── parse_targets — happy paths ───────────────────────────────────────────────

@pytest.mark.parametrize("question,expected", [
    ("What color is the pillow?",                   ["pillow"]),
    ("What color is the yellow pillow?",            ["pillow"]),
    ("What color is the small red car?",            ["car"]),
    ("Describe the dog.",                           ["dog"]),
    ("Is the dog on the table?",                    ["dog", "table"]),
    ("How many forks can you see?",                 ["forks"]),
    ("How many teddy bears are on the stairs?",     ["bears", "stairs"]),
    ("Is there a cat on a chair?",                  ["cat", "chair"]),
    ("Compare the two surfboards.",                 ["surfboards"]),
    ("Which truck has its door open?",              []),  # no determiner before "truck"
])
def test_parse_targets(question, expected):
    assert parse_targets(question) == expected


def test_parse_targets_empty():
    assert parse_targets("") == []


def test_parse_targets_drops_generic_referents():
    # "the image"/"the photo" should never appear as targets.
    assert "image" not in parse_targets("Describe the image.")
    assert "photo" not in parse_targets("What is in the photo?")
    assert "picture" not in parse_targets("Describe the picture in detail.")


def test_parse_targets_dedup_order():
    # Same word twice → keep one, preserve order.
    out = parse_targets("Is the dog near the dog?")
    assert out == ["dog"]


def test_parse_targets_case_insensitive():
    assert parse_targets("DESCRIBE THE DOG") == ["dog"]


# ── match_targets_to_vocab ────────────────────────────────────────────────────

def test_match_exact():
    vocab = {"dog", "table", "chair"}
    assert match_targets_to_vocab(["dog", "lamp"], vocab) == ["dog"]


def test_match_plural_to_singular():
    vocab = {"fork", "spoon"}
    assert match_targets_to_vocab(["forks"], vocab) == ["fork"]
    assert match_targets_to_vocab(["spoons"], vocab) == ["spoon"]


def test_match_irregular_plural_handled_loosely():
    # "bears" → "bear" via simple -s shaving; vocab "teddy bear" matches via substring.
    vocab = {"teddy bear", "table"}
    assert match_targets_to_vocab(["bears"], vocab) == ["teddy bear"]


def test_match_substring_multi_word_vocab():
    vocab = {"fire truck", "police car"}
    assert match_targets_to_vocab(["truck"], vocab) == ["fire truck"]
    assert match_targets_to_vocab(["car"], vocab) == ["police car"]


def test_match_preserves_question_order():
    vocab = {"dog", "table", "chair"}
    assert match_targets_to_vocab(["table", "dog"], vocab) == ["table", "dog"]


def test_match_empty_vocab():
    assert match_targets_to_vocab(["dog"], set()) == []


def test_match_no_targets():
    assert match_targets_to_vocab([], {"dog"}) == []
