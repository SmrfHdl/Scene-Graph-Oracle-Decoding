"""
Question target parser — M1.4 of BG-SGOD.

Extracts the entity (or entities) the question is asking about so the oracle
can focus bbox-conditioned scoring on those objects rather than the whole image.

This is intentionally lightweight (regex + a curated stopword filter):
  - "what color is the pillow"        → ["pillow"]
  - "what color are the two cars"     → ["cars"]                (modifier dropped)
  - "is the dog on the table?"        → ["dog", "table"]
  - "describe the man wearing a hat"  → ["man", "hat"]
  - "is the pillow blue or yellow?"   → ["pillow"]
  - "how many forks can you see?"     → ["forks"]

The parser is a classifier of *targets*, not a full SRL system. False positives
are tolerated; the oracle treats any target absent from the scene graph as "no
crop" and falls back to image-global scoring (current behavior).
"""
from __future__ import annotations

import re

# Words after determiners that aren't real targets — they're scene/image
# referents or generic discourse markers.
_GENERIC_REFERENTS: frozenset[str] = frozenset({
    "image", "photo", "picture", "scene", "background", "foreground",
    "left", "right", "top", "bottom", "middle", "center", "side",
    "front", "back", "thing", "things", "object", "objects",
    "color", "colors", "colour", "colours", "size", "sizes",
    "number", "numbers", "kind", "kinds", "type", "types",
    "way", "ways", "name", "names", "place", "places",
    "person",  # often paired with "the person riding/playing X" → X is the target
})

_DETERMINERS: frozenset[str] = frozenset({
    "the", "a", "an", "this", "that", "these", "those", "some", "any",
    "many", "much", "several", "few",
    "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
})

# Adjectives we accept inside a noun phrase (skip them to reach the head noun).
# Curated to avoid eating prepositions/verbs ("near", "on", "is").
_ADJECTIVES: frozenset[str] = frozenset({
    # color
    "red", "orange", "yellow", "green", "blue", "purple", "pink", "brown",
    "black", "white", "gray", "grey", "silver", "gold", "beige",
    # size
    "small", "large", "big", "little", "tiny", "huge", "giant", "mini",
    "tall", "short", "long", "wide", "narrow", "thick", "thin",
    # quality
    "old", "new", "young", "fresh", "dirty", "clean", "broken", "shiny",
    "wooden", "plastic", "metal", "leather",
    # noun-modifier compounds (often function as adjectives in MMHal)
    "fire", "police", "ice", "tennis", "basketball", "baseball", "soccer",
    "yellow", "white", "black",  # already in color, but compound NPs
    "teddy", "stuffed",
    # ordinal
    "first", "second", "third", "fourth", "next", "last",
})

# Prepositions / verbs / function words that signal end of NP — used to ensure
# we don't accidentally treat them as the target noun.
_NON_NOUNS: frozenset[str] = frozenset({
    "is", "are", "was", "were", "be", "been", "being", "am",
    "do", "does", "did", "have", "has", "had",
    "on", "in", "at", "by", "with", "without", "for", "to", "from", "of",
    "near", "above", "below", "under", "over", "behind", "beside",
    "and", "or", "but", "if", "so", "than", "as",
    "the", "a", "an", "this", "that", "these", "those",  # nested determiners
    "what", "who", "where", "when", "why", "how", "which",
    "you", "your", "i", "me", "my", "we", "our", "they", "their",
    "it", "its", "he", "his", "she", "her",
})


def _tokenize(text: str) -> list[str]:
    """Split text into lowercase word tokens; drop punctuation."""
    return re.findall(r"[a-z]+", text.lower())

# Plural→singular shaving for vocab matching. Conservative — only obvious cases.
def _singular(word: str) -> str:
    w = word.lower()
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("ses") or w.endswith("xes") or w.endswith("zes"):
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w


def parse_targets(question: str) -> list[str]:
    """Return ordered list of target nouns the question asks about.

    Walks tokens; whenever a determiner (or number) is seen, skips up to two
    adjectives and emits the next non-function word as a target. Strips
    generic referents (image/photo/...) and deduplicates while preserving order.
    """
    if not question:
        return []
    tokens = _tokenize(question)
    targets: list[str] = []
    i = 0
    while i < len(tokens):
        if tokens[i] in _DETERMINERS:
            j = i + 1
            adjs_seen = 0
            # Skip up to 2 adjectives.
            while j < len(tokens) and adjs_seen < 2 and tokens[j] in _ADJECTIVES:
                adjs_seen += 1
                j += 1
            # Now tokens[j] should be the head noun. Reject if it's a function
            # word, another determiner, or shorter than 3 chars.
            advanced = False
            if j < len(tokens):
                head = tokens[j]
                if (
                    len(head) >= 3
                    and head not in _NON_NOUNS
                    and head not in _DETERMINERS
                    and head not in _GENERIC_REFERENTS
                ):
                    targets.append(head)
                    i = j + 1
                    advanced = True
            # If head was rejected (or absent), restart search from j so a
            # nested determiner ("the two cars" → after "the" hits "two") can
            # itself trigger a determiner search.
            if not advanced:
                i = max(j, i + 1)
        else:
            i += 1

    # Dedup preserving order.
    seen: dict[str, None] = {}
    for t in targets:
        if t not in seen:
            seen[t] = None
    return list(seen.keys())


def match_targets_to_vocab(targets: list[str], vocab: set[str]) -> list[str]:
    """Filter targets to those present in `vocab` (case-insensitive, plural-aware).

    Each input target is tested against vocab in three forms — exact match,
    singularized, and substring-of-vocab-entry — to cover Grounding DINO's
    multi-word labels ("fire truck") matching question targets ("truck").
    Returns the matched VOCAB entries (not the original targets), so the
    caller can index into the scene graph directly.
    """
    if not vocab:
        return []
    vocab_lower = {v.lower(): v for v in vocab}
    matched: list[str] = []
    for tgt in targets:
        tgt_l = tgt.lower()
        if tgt_l in vocab_lower:
            matched.append(vocab_lower[tgt_l])
            continue
        sing = _singular(tgt_l)
        if sing in vocab_lower:
            matched.append(vocab_lower[sing])
            continue
        # Substring: question says "truck", vocab has "fire truck" → match.
        for v_l, v_orig in vocab_lower.items():
            v_tokens = v_l.split()
            if tgt_l in v_tokens or sing in v_tokens:
                matched.append(v_orig)
                break
    # Dedup preserving order
    seen: dict[str, None] = {}
    for m in matched:
        if m not in seen:
            seen[m] = None
    return list(seen.keys())
