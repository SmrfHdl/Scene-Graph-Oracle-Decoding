"""
Vocabulary constants for anchor detection (Section 3.4).
"""

DETERMINERS: frozenset[str] = frozenset({
    "a", "an", "the", "this", "that", "these", "those", "some",
})

# Single-token spatial prepositions.
# Multi-word preps ("next to", "in front of") are handled separately in
# detect_anchor via bigram/trigram checks on prev_tokens.
SPATIAL_PREPS: frozenset[str] = frozenset({
    "on", "under", "above", "below", "beside",
    "behind", "near", "between", "inside", "outside",
})

# Multi-word spatial prepositions as space-joined strings (for bigram/trigram matching)
SPATIAL_PREPS_MULTI: frozenset[str] = frozenset({
    "next to", "in front of", "across from",
})

COLOR_ATTRS: frozenset[str] = frozenset({
    "red", "blue", "green", "black", "white", "yellow",
    "orange", "purple", "pink", "brown", "gray", "golden",
})

SIZE_ATTRS: frozenset[str] = frozenset({
    "large", "small", "big", "tiny", "huge", "tall", "short",
})
