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
_ATTR_WORDS = COLOR_ATTRS | SIZE_ATTRS


def detect_anchor(prev_tokens: list[str], current_token: str) -> str:
    """Detect whether current_token sits at an anchor position.

    Args:
        prev_tokens: All tokens generated so far (decoded strings, not IDs).
        current_token: The token being scored right now.

    Returns:
        One of: "noun_anchor" | "relation_anchor" | "attr_anchor" | "neutral"

    Rules (evaluated in priority order):
        1. prev[-1] is a determiner                       → noun_anchor
        1b. prev[-2] is a determiner (det + adj + noun)   → noun_anchor
        2. cur completes a multi-word spatial prep         → relation_anchor
        3. prev[-1] is a single-token spatial prep         → noun_anchor
        4. prev[-2:] or prev[-3:] forms a multi-word prep → noun_anchor
        5. current_token is a single-token spatial prep    → relation_anchor
        6. current_token is a color or size attr           → attr_anchor
        7. otherwise                                       → neutral

    Rule 2 must precede Rule 3 so that "dog across [from]" fires relation_anchor
    (completing "across from") rather than noun_anchor (prev="across" is spatial).
    Rule 1b catches "a large [table]" where Rule 1 fired on "large" but the actual
    noun "table" would otherwise fall through to neutral.
    """
    if not prev_tokens:
        return "neutral"

    prev = prev_tokens[-1].lower().strip()
    cur  = current_token.lower().strip()

    # Rule 1: Determiner → next token is a noun (or adjective before noun, close enough)
    if prev in DETERMINERS:
        return "noun_anchor"

    # Rule 1b: Noun after "determiner + color/size adj" — "a large [table]", "the black [dog]"
    # Guards: cur must not be a spatial prep or another attr (those have their own rules).
    if (
        len(prev_tokens) >= 2
        and cur not in _ATTR_WORDS
        and cur not in SPATIAL_PREPS
    ):
        two_back = prev_tokens[-2].lower().strip()
        if two_back in DETERMINERS and prev in _ATTR_WORDS:
            return "noun_anchor"

    # Rule 2: cur completes a multi-word spatial prep → relation_anchor
    # Must come before Rule 3 so "across [from]" doesn't mis-fire as noun_anchor
    # because prev="across" is now in SPATIAL_PREPS.
    bigram_cur = (prev + " " + cur).strip()
    if bigram_cur in SPATIAL_PREPS_MULTI:
        return "relation_anchor"
    if len(prev_tokens) >= 2:
        trigram_cur = " ".join(t.lower().strip() for t in prev_tokens[-2:]) + " " + cur
        if trigram_cur in SPATIAL_PREPS_MULTI:
            return "relation_anchor"

    # Rule 3: Single-token spatial prep as prev → next token is the relation's object
    if prev in SPATIAL_PREPS:
        return "noun_anchor"

    # Rule 4: Multi-word spatial prep ending at prev_tokens[-1] or prev_tokens[-2]
    if len(prev_tokens) >= 2:
        bigram = (prev_tokens[-2].lower().strip() + " " + prev).strip()
        if bigram in SPATIAL_PREPS_MULTI:
            return "noun_anchor"
    if len(prev_tokens) >= 3:
        trigram = " ".join(t.lower().strip() for t in prev_tokens[-3:])
        if trigram in SPATIAL_PREPS_MULTI:
            return "noun_anchor"

    # Rule 5: Current token is a single-token spatial preposition
    if cur in SPATIAL_PREPS:
        return "relation_anchor"

    # Rule 5: Color or size attribute
    if cur in COLOR_ATTRS or cur in SIZE_ATTRS:
        return "attr_anchor"

    return "neutral"
