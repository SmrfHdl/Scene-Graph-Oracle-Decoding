"""Tests for sgod.oracle.question_parser."""
from __future__ import annotations

import pytest

from sgod.oracle.question_parser import (
    is_yesno_question,
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


# ── Multi-word noun phrases (POPE spike fixes) ────────────────────────────────

@pytest.mark.parametrize("question,expected", [
    # Multi-word objects: head must be the LAST word, not the first non-adjective.
    # This was the dominant POPE failure mode in spike 2026-05-08.
    ("Is there a dining table in the image?",      ["table"]),
    ("Is there a hot dog in the image?",           ["dog"]),
    ("Is there a traffic light in the image?",     ["light"]),
    ("Is there a sports ball in the image?",       ["ball"]),
    ("Is there a baseball bat in the image?",      ["bat"]),
    ("Is there a potted plant in the image?",      ["plant"]),
    ("Is there a teddy bear in the image?",        ["bear"]),
    ("Is there a fire truck in the image?",        ["truck"]),
])
def test_parse_targets_multi_word_np(question, expected):
    assert parse_targets(question) == expected


def test_parse_targets_person_is_target():
    # Was previously dropped as a generic referent; re-included after POPE
    # spike showed many "Is there a person?" questions.
    assert parse_targets("Is there a person in the image?") == ["person"]


def test_parse_targets_short_head_allowed():
    # Short COCO labels (≥2 chars) must be allowed through the head filter.
    assert parse_targets("Is there a tv in the image?") == ["tv"]


def test_parse_targets_drops_image_but_not_person():
    # Sanity: image/photo/etc. still drop, person no longer does.
    assert parse_targets("Describe the image.") == []
    assert parse_targets("Describe the person.") == ["person"]


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


# ── is_yesno_question ────────────────────────────────────────────────────────

@pytest.mark.parametrize("question,expected", [
    ("Is there a dog in the image?",                          True),
    ("Are these traffic lights working?",                     True),
    ("Does the man wear a hat?",                              True),
    ("Has the bus left the station?",                         True),
    ("Can you see a fork on the table?",                      True),
    ("Should the camera be tilted up?",                       True),
    ("USER: <image>\nIs there a dog?\nAnswer with yes or no.\nASSISTANT:", True),
    ("Yes/No: is the dog brown? (yes/no)",                    True),
    # Open-ended questions
    ("What color is the pillow?",                             False),
    ("How many forks can you see?",                           False),
    ("Describe the image in detail.",                         False),
    ("Which truck has its door open?",                        False),
])
def test_is_yesno_question(question, expected):
    assert is_yesno_question(question) is expected


def test_is_yesno_question_empty():
    assert is_yesno_question("") is False
    assert is_yesno_question(None) is False  # type: ignore[arg-type]
