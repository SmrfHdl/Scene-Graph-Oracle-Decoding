"""Tests for Generation Context — negation tracking, question type detection, lambda computation."""
import pytest

from sgod.context import (
    BASE_LAMBDA,
    NEGATION_TOKENS,
    QUESTION_TYPE_SIGNALS,
    GenerationContext,
)


# ── Vocabulary sanity ─────────────────────────────────────────────────────────

def test_negation_tokens_is_frozenset():
    assert isinstance(NEGATION_TOKENS, frozenset)


def test_negation_tokens_content():
    assert {"no", "not", "never", "without", "nobody", "nowhere", "nothing"} <= NEGATION_TOKENS
    assert {"isn't", "aren't", "doesn't", "don't", "won't", "can't"} <= NEGATION_TOKENS


def test_question_type_signals_keys():
    assert set(QUESTION_TYPE_SIGNALS.keys()) == {
        "existential", "descriptive", "comparative", "hypothetical"
    }


def test_base_lambda_keys():
    assert set(BASE_LAMBDA.keys()) == {
        "existential", "descriptive", "comparative", "hypothetical", "general"
    }


def test_base_lambda_values_in_range():
    for v in BASE_LAMBDA.values():
        assert 0.0 <= v <= 1.0


def test_base_lambda_ordering():
    assert BASE_LAMBDA["existential"] > BASE_LAMBDA["descriptive"]
    assert BASE_LAMBDA["descriptive"] > BASE_LAMBDA["comparative"]
    assert BASE_LAMBDA["comparative"] > BASE_LAMBDA["hypothetical"]
    assert BASE_LAMBDA["hypothetical"] == 0.0


# ── Question type detection ───────────────────────────────────────────────────

@pytest.mark.parametrize("question,expected", [
    ("Is there a dog in the image?", "existential"),
    ("Are there any chairs?", "existential"),
    ("Do you see a cat?", "existential"),
    ("Can you see a table?", "existential"),
    ("Does the image contain a person?", "existential"),
])
def test_existential_detection(question, expected):
    ctx = GenerationContext(question)
    assert ctx.question_type == expected


@pytest.mark.parametrize("question,expected", [
    ("What color is the dog?", "descriptive"),
    ("How many chairs do you count?", "descriptive"),
    ("What is on the table?", "descriptive"),
    ("Where is the cat?", "descriptive"),
    ("Describe the scene.", "descriptive"),
    ("What are the objects in the image?", "descriptive"),
    ("What does the dog look like?", "descriptive"),
])
def test_descriptive_detection(question, expected):
    ctx = GenerationContext(question)
    assert ctx.question_type == expected


@pytest.mark.parametrize("question,expected", [
    ("Which is bigger, the dog or the cat?", "comparative"),
    ("Is it bigger than the table?", "comparative"),
    ("Is it smaller than the chair?", "comparative"),
    ("Compare the two objects.", "comparative"),
    ("Compare the difference between the two.", "comparative"),
])
def test_comparative_detection(question, expected):
    ctx = GenerationContext(question)
    assert ctx.question_type == expected


@pytest.mark.parametrize("question,expected", [
    ("What if there was a lion here?", "hypothetical"),
    ("Could there be a cat on the table?", "hypothetical"),
    ("Would the dog fit in the box?", "hypothetical"),
    ("Imagine the scene without the table.", "hypothetical"),
    ("Suppose the cat moved, what would change?", "hypothetical"),
])
def test_hypothetical_detection(question, expected):
    ctx = GenerationContext(question)
    assert ctx.question_type == expected


@pytest.mark.parametrize("question", [
    "Tell me about the image.",
    "Please analyze this photo.",
    "Look at this picture.",
    "",
    "xyz",
])
def test_general_detection(question):
    ctx = GenerationContext(question)
    assert ctx.question_type == "general"


def test_question_type_case_insensitive():
    assert GenerationContext("IS THERE a dog?").question_type == "existential"
    assert GenerationContext("WHAT COLOR is it?").question_type == "descriptive"


def test_existential_beats_descriptive_priority():
    # "Is there" fires existential before "what is" could fire descriptive.
    # Both signals may appear in edge-case sentences; priority order matters.
    ctx = GenerationContext("Is there anything to describe?")
    assert ctx.question_type == "existential"


# ── Initial state ─────────────────────────────────────────────────────────────

def test_initial_negation_depth_is_zero():
    ctx = GenerationContext("Is there a dog?")
    assert ctx.negation_depth == 0.0


def test_initial_token_count_is_zero():
    ctx = GenerationContext("Is there a dog?")
    assert ctx.token_count == 0


def test_initial_get_lambda_equals_base():
    ctx = GenerationContext("Is there a dog?")
    assert ctx.get_lambda() == BASE_LAMBDA["existential"]


# ── update() — negation depth tracking ───────────────────────────────────────

def test_negation_token_increments_depth():
    ctx = GenerationContext("Is there a dog?")
    ctx.update("no")
    assert ctx.negation_depth == 1.0


def test_non_negation_token_decrements_depth():
    ctx = GenerationContext("Is there a dog?")
    ctx.update("no")     # depth = 1.0
    ctx.update("dog")    # depth = 0.75
    assert abs(ctx.negation_depth - 0.75) < 1e-9


def test_depth_floors_at_zero():
    ctx = GenerationContext("Is there a dog?")
    ctx.update("dog")
    ctx.update("cat")
    assert ctx.negation_depth == 0.0


def test_depth_caps_at_max():
    ctx = GenerationContext("Is there a dog?")
    for _ in range(10):
        ctx.update("not")
    assert ctx.negation_depth == 3.0


def test_update_increments_token_count():
    ctx = GenerationContext("Is there a dog?")
    ctx.update("the")
    ctx.update("dog")
    ctx.update("is")
    assert ctx.token_count == 3


def test_negation_decay_sequence():
    ctx = GenerationContext("Describe the scene.")
    ctx.update("not")    # depth = 1.0
    ctx.update("there")  # depth = 0.75
    ctx.update("is")     # depth = 0.50
    ctx.update("a")      # depth = 0.25
    ctx.update("dog")    # depth = 0.0
    assert ctx.negation_depth == 0.0


def test_multiple_negations_accumulate():
    ctx = GenerationContext("Describe the scene.")
    ctx.update("no")     # 1.0
    ctx.update("not")    # 2.0
    assert ctx.negation_depth == 2.0


def test_negation_token_case_insensitive():
    ctx = GenerationContext("Describe the scene.")
    ctx.update("NOT")
    assert ctx.negation_depth == 1.0
    ctx2 = GenerationContext("Describe the scene.")
    ctx2.update("Never")
    assert ctx2.negation_depth == 1.0


# ── get_lambda() — adaptive weighting ────────────────────────────────────────

def test_get_lambda_no_negation_returns_base():
    for qtype in BASE_LAMBDA:
        q_map = {
            "existential":  "Is there a dog?",
            "descriptive":  "What color is the dog?",
            "comparative":  "Which is bigger?",
            "hypothetical": "What if there were a lion?",
            "general":      "Tell me about this.",
        }
        ctx = GenerationContext(q_map[qtype])
        assert ctx.get_lambda() == BASE_LAMBDA[qtype]


def test_get_lambda_light_negation_reduces():
    ctx = GenerationContext("Is there a dog?")
    ctx.update("no")           # depth = 1.0 → > 0.5 → reduced
    lam = ctx.get_lambda()
    assert 0.0 < lam < BASE_LAMBDA["existential"]


def test_get_lambda_deep_negation_inverts():
    ctx = GenerationContext("Is there a dog?")
    ctx.update("not")   # 1.0
    ctx.update("never") # 2.0
    lam = ctx.get_lambda()
    assert lam < 0.0


def test_get_lambda_hypothetical_always_zero():
    ctx = GenerationContext("What if there was a lion here?")
    assert ctx.get_lambda() == 0.0
    ctx.update("no")
    ctx.update("not")
    # Even after negation tokens, base is 0.0 → reduced or inverted but still 0 or negative
    # The contract: hypothetical starts at 0.0; negation can only keep it ≤ 0.
    assert ctx.get_lambda() <= 0.0


def test_get_lambda_negation_recovery():
    ctx = GenerationContext("Is there a dog?")
    ctx.update("no")    # depth = 1.0  → light negation
    ctx.update("not")   # depth = 2.0  → deep negation → negative lambda
    assert ctx.get_lambda() < 0.0
    # Decay back through neutral tokens
    for _ in range(10):
        ctx.update("the")
    # depth should be 0 now
    assert ctx.negation_depth == 0.0
    assert ctx.get_lambda() == BASE_LAMBDA["existential"]


def test_get_lambda_returns_float():
    ctx = GenerationContext("Is there a dog?")
    assert isinstance(ctx.get_lambda(), float)


# ── Realistic sequences ───────────────────────────────────────────────────────

def test_negative_answer_sequence():
    # "No, there is no cat." — oracle should be suppressed
    ctx = GenerationContext("Is there a cat?")
    assert ctx.question_type == "existential"
    assert ctx.get_lambda() == 0.50

    ctx.update("No")     # depth=1.0  → light negation
    assert ctx.get_lambda() < 0.50

    ctx.update(",")
    ctx.update("there")
    ctx.update("is")
    ctx.update("no")     # depth increases again

    assert ctx.get_lambda() < 0.50   # still suppressed


def test_positive_answer_sequence():
    # "Yes, there is a dog on the table."
    ctx = GenerationContext("Is there a dog?")
    for tok in ["Yes", ",", "there", "is", "a", "dog", "on", "the", "table", "."]:
        lam = ctx.get_lambda()
        assert lam >= 0.0, f"lambda should not go negative for positive answer, got {lam} at '{tok}'"
        ctx.update(tok)


def test_descriptive_sequence():
    ctx = GenerationContext("What color is the dog?")
    assert ctx.question_type == "descriptive"
    assert ctx.get_lambda() == BASE_LAMBDA["descriptive"]

    for tok in ["The", "dog", "is", "black", "."]:
        ctx.update(tok)
    assert ctx.get_lambda() == BASE_LAMBDA["descriptive"]


def test_hypothetical_sequence():
    ctx = GenerationContext("What if there was a lion?")
    assert ctx.question_type == "hypothetical"
    for tok in ["There", "would", "be", "a", "lion", "."]:
        ctx.update(tok)
    # Base is 0.0; non-negative tokens keep depth=0 → get_lambda() == 0.0
    assert ctx.get_lambda() == 0.0
