"""
Anchor Position Detector — Component 3 (Section 3.4).

Rule-based anchor detection using grammar patterns.
Identifies token positions where oracle verification is needed:
    - noun_anchor:     token after determiner (a/the/this...)
    - relation_anchor: spatial preposition token (on/under/near...)
    - attr_anchor:     color/size attribute token
    - neutral:         no verification needed

Zero latency, zero ML dependency.

Key function:
    detect_anchor(prev_tokens, current_token) -> str
"""
from __future__ import annotations

from sgod.anchor.vocabularies import (
    COLOR_ATTRS,
    DETERMINERS,
    SIZE_ATTRS,
    SPATIAL_PREPS,
    SPATIAL_PREPS_MULTI,
)

_ANCHOR_TYPES = frozenset({"noun_anchor", "relation_anchor", "attr_anchor", "neutral"})


def detect_anchor(prev_tokens: list[str], current_token: str) -> str:
    """Detect whether current_token sits at an anchor position.

    Args:
        prev_tokens: All tokens generated so far (decoded strings, not IDs).
        current_token: The token being scored right now.

    Returns:
        One of: "noun_anchor" | "relation_anchor" | "attr_anchor" | "neutral"

    Rules (evaluated in priority order):
        1. prev[-1] is a determiner             → noun_anchor
        2. prev[-1] is a single-token spatial prep → noun_anchor
        3. prev[-2:] or prev[-3:] forms a multi-word spatial prep → noun_anchor
        4. current_token is a spatial prep (single or multi-word tail) → relation_anchor
        5. current_token is a color or size attr  → attr_anchor
        6. otherwise                              → neutral
    """
    if not prev_tokens:
        return "neutral"

    prev = prev_tokens[-1].lower().strip()
    cur  = current_token.lower().strip()

    # Rule 1: Determiner → next token is a noun (or adjective before noun, close enough)
    if prev in DETERMINERS:
        return "noun_anchor"

    # Rule 2: Single-token spatial prep → next token (often "the"/"a" + noun)
    if prev in SPATIAL_PREPS:
        return "noun_anchor"

    # Rule 3: Multi-word spatial prep ending at prev_tokens[-1] or prev_tokens[-2]
    # Covers "next to", "in front of", "across from"
    if len(prev_tokens) >= 2:
        bigram = (prev_tokens[-2].lower().strip() + " " + prev).strip()
        if bigram in SPATIAL_PREPS_MULTI:
            return "noun_anchor"
    if len(prev_tokens) >= 3:
        trigram = " ".join(t.lower().strip() for t in prev_tokens[-3:])
        if trigram in SPATIAL_PREPS_MULTI:
            return "noun_anchor"

    # Rule 4: Current token is a spatial preposition → it's the relation anchor
    if cur in SPATIAL_PREPS:
        return "relation_anchor"
    # Also check if cur completes a multi-word prep (e.g. cur="to" and prev="next")
    if len(prev_tokens) >= 1:
        bigram_cur = (prev + " " + cur).strip()
        if bigram_cur in SPATIAL_PREPS_MULTI:
            return "relation_anchor"
    if len(prev_tokens) >= 2:
        trigram_cur = " ".join(t.lower().strip() for t in prev_tokens[-2:]) + " " + cur
        if trigram_cur in SPATIAL_PREPS_MULTI:
            return "relation_anchor"

    # Rule 5: Color or size attribute
    if cur in COLOR_ATTRS or cur in SIZE_ATTRS:
        return "attr_anchor"

    return "neutral"
