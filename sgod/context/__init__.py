"""
Generation Context Tracker — Component 4 (Section 3.5).
Tracks generation state for adaptive oracle strength adjustment.
"""

from sgod.context.tracker import (
    BASE_LAMBDA,
    NEGATION_TOKENS,
    QUESTION_TYPE_SIGNALS,
    GenerationContext,
)

__all__ = [
    "GenerationContext",
    "NEGATION_TOKENS",
    "QUESTION_TYPE_SIGNALS",
    "BASE_LAMBDA",
]
