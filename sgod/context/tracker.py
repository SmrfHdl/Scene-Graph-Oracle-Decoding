"""
GenerationContext — Component 4 (Section 3.5).

Tracks generation state to adaptively adjust oracle lambda:
- Negation detection and depth tracking (decay-based)
- Question type classification (existential, descriptive, comparative, hypothetical)
- Context-aware lambda computation

Key class: GenerationContext
    - update(new_token) -> None
    - get_lambda() -> float
"""
from __future__ import annotations

NEGATION_TOKENS: frozenset[str] = frozenset({
    "no", "not", "isn't", "aren't", "doesn't", "don't",
    "won't", "can't", "never", "neither", "nor", "nothing",
    "nobody", "nowhere", "without",
})

# Signals are checked in insertion order; first match wins.
QUESTION_TYPE_SIGNALS: dict[str, list[str]] = {
    "existential": [
        "is there", "are there", "do you see", "can you see",
        "does the image contain", "is a", "is an",
    ],
    "descriptive": [
        "what color", "how many", "what is", "where is",
        "describe", "what are", "what does",
    ],
    "comparative": [
        "which is", "is it bigger", "is it smaller",
        "compare", "between",
    ],
    "hypothetical": [
        "what if", "could", "would", "imagine", "suppose",
    ],
}

BASE_LAMBDA: dict[str, float] = {
    "existential":  0.50,   # Hallucination rate highest → strong oracle
    "descriptive":  0.35,   # Moderate
    "comparative":  0.20,   # Model needs to reason → weak oracle
    "hypothetical": 0.00,   # Oracle not relevant
    "general":      0.25,   # Default
}

_NEGATION_INCREMENT = 1.0
_NEGATION_MAX_DEPTH = 3.0
_NEGATION_DECAY     = 0.25

_DEEP_NEGATION_THRESHOLD  = 1.5   # depth > this → slight oracle inversion
_LIGHT_NEGATION_THRESHOLD = 0.5   # depth > this → reduce lambda strongly

_DEEP_NEGATION_LAMBDA  = -0.05
_LIGHT_NEGATION_SCALE  = 0.3


class GenerationContext:
    """Tracks token-level generation state to provide adaptive oracle lambda.

    Usage:
        ctx = GenerationContext("Is there a dog in the image?")
        for token in generated_tokens:
            lambda_ = ctx.get_lambda()   # read before update
            ctx.update(token)
    """

    def __init__(
        self,
        question: str,
        base_lambda: dict[str, float] | None = None,
        negation_decay: float = _NEGATION_DECAY,
        max_negation_depth: float = _NEGATION_MAX_DEPTH,
    ) -> None:
        self.question_type:     str   = self._detect_question_type(question)
        self.negation_depth:    float = 0.0
        self.token_count:       int   = 0
        self._base_lambda       = base_lambda if base_lambda is not None else BASE_LAMBDA
        self._negation_decay    = negation_decay
        self._max_negation_depth = max_negation_depth

    # ── Public API ──────────────────────────────────────────────────────────

    def update(self, new_token: str) -> None:
        """Advance context by one generated token."""
        tok = new_token.lower().strip()
        if tok in NEGATION_TOKENS:
            self.negation_depth = min(
                self.negation_depth + _NEGATION_INCREMENT,
                self._max_negation_depth,
            )
        else:
            self.negation_depth = max(self.negation_depth - self._negation_decay, 0.0)
        self.token_count += 1

    def get_lambda(self) -> float:
        """Return oracle mixing weight for the current generation state.

        Returns:
            > 0: boost oracle-supported tokens
            = 0: oracle disabled (hypothetical questions)
            < 0: slight inversion (deep negation context)
        """
        base = self._base_lambda.get(self.question_type,
                                     self._base_lambda.get("general", 0.25))

        if self.negation_depth > _DEEP_NEGATION_THRESHOLD:
            return _DEEP_NEGATION_LAMBDA
        if self.negation_depth > _LIGHT_NEGATION_THRESHOLD:
            return base * _LIGHT_NEGATION_SCALE
        return base

    # ── Internal ────────────────────────────────────────────────────────────

    @staticmethod
    def _detect_question_type(question: str) -> str:
        """Classify question into one of the five types using signal phrases."""
        q = question.lower()
        for qtype, signals in QUESTION_TYPE_SIGNALS.items():
            if any(s in q for s in signals):
                return qtype
        return "general"
