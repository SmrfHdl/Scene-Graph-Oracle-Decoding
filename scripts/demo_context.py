"""
Demo: Generation Context Tracker — Component 4 (Section 3.5).

Shows how GenerationContext adapts oracle lambda as tokens are generated.
Specifically illustrates:
  1. Question type detection and its effect on base lambda
  2. Negation depth tracking (decay-based)
  3. Lambda suppression / inversion during negation
  4. Recovery after negation scope expires

No GPU required — purely rule-based.

Usage:
    python scripts/demo_context.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from sgod.context import BASE_LAMBDA, GenerationContext


def print_section(title: str) -> None:
    print(f"\n{'─'*65}")
    print(f"  {title}")
    print(f"{'─'*65}")


def show_question_types() -> None:
    examples = [
        ("Is there a dog in the image?",          "existential"),
        ("What color is the dog?",                "descriptive"),
        ("Which is bigger, the dog or the cat?",  "comparative"),
        ("What if there was a lion here?",         "hypothetical"),
        ("Tell me about this image.",              "general"),
    ]

    print(f"\n  {'Question':<50}  {'Detected':<14}  {'Base λ'}")
    print(f"  {'─'*50}  {'─'*14}  {'─'*6}")
    for q, expected in examples:
        ctx = GenerationContext(q)
        match = "✓" if ctx.question_type == expected else "✗"
        print(f"  {q:<50}  {ctx.question_type:<14}  {BASE_LAMBDA[ctx.question_type]:.2f}  {match}")


def simulate_decode(question: str, tokens: list[str]) -> None:
    ctx = GenerationContext(question)

    print(f"\n  Question: \"{question}\"")
    print(f"  Question type: {ctx.question_type}  |  Base λ = {BASE_LAMBDA[ctx.question_type]:.2f}\n")
    print(f"  {'#':<3}  {'Token':<12}  {'λ before update':>16}  {'negation_depth':>14}  Note")
    print(f"  {'─'*3}  {'─'*12}  {'─'*16}  {'─'*14}  {'─'*30}")

    for i, tok in enumerate(tokens):
        lam    = ctx.get_lambda()
        depth  = ctx.negation_depth
        note   = _explain(tok, lam, depth, BASE_LAMBDA[ctx.question_type])
        print(f"  {i:<3}  {tok:<12}  {lam:>+16.3f}  {depth:>14.2f}  {note}")
        ctx.update(tok)

    print(f"\n  Final negation_depth = {ctx.negation_depth:.2f}  |  Final λ = {ctx.get_lambda():+.3f}")


def _explain(tok: str, lam: float, depth: float, base: float) -> str:
    from sgod.context.tracker import NEGATION_TOKENS
    if tok.lower().strip() in NEGATION_TOKENS:
        return f"negation token → depth +1"
    if depth > 1.5:
        return f"deep negation → λ inverted"
    if depth > 0.5:
        return f"light negation → λ reduced to {lam:+.3f}"
    if lam == 0.0:
        return "hypothetical → oracle OFF"
    if lam == base:
        return "neutral context → base λ"
    return "recovering from negation"


def main() -> None:
    print_section("STEP 1 — Question Type Detection")
    print("""
  GenerationContext detects the question type from the input text.
  The type determines the base oracle mixing weight (BASE_LAMBDA):
    existential  λ=0.50  — "Is there a X?" → hallucination risk highest
    descriptive  λ=0.35  — "What color?" → moderate oracle influence
    comparative  λ=0.20  — "Which is bigger?" → model needs to reason
    hypothetical λ=0.00  — "What if?" → oracle not relevant
    general      λ=0.25  — default for unrecognised questions
    """)
    show_question_types()

    print_section("STEP 2 — Positive answer (no negation)")
    simulate_decode(
        "Is there a dog in the image?",
        ["Yes", ",", "there", "is", "a", "dog", "on", "the", "mat", "."],
    )

    print_section("STEP 3 — Negative answer (negation suppresses oracle)")
    simulate_decode(
        "Is there a zebra in the image?",
        ["No", ",", "there", "is", "no", "zebra", "in", "the", "image", "."],
    )

    print_section("STEP 4 — Double negation (deep negation → oracle inverts)")
    simulate_decode(
        "Is there a cat on the table?",
        ["No", ",", "it", "is", "not", "on", "the", "table", "."],
    )

    print_section("STEP 5 — Negation then recovery")
    simulate_decode(
        "What color is the dog?",
        ["Not", "black", ",", "but", "actually", "the", "dog", "is", "brown", "."],
    )

    print_section("STEP 6 — Hypothetical question (oracle always OFF)")
    simulate_decode(
        "What if there was a lion in the image?",
        ["There", "would", "be", "a", "large", "lion", "near", "the", "tree", "."],
    )

    print_section("SUMMARY — Why adaptive lambda matters")
    print("""
  Without adaptive lambda:
    "No, there is no zebra" → oracle PENALIZES "zebra" (correct token!)
    because "zebra" is not in the scene graph → forces wrong output

  With adaptive lambda:
    "No" triggers negation_depth → lambda drops or inverts
    "zebra" position: oracle suppressed → LM can freely generate "zebra"

  The key insight: oracle is helpful when generating POSITIVE grounded content,
  but actively harmful when the answer IS the absence of something.
  """)


if __name__ == "__main__":
    main()
