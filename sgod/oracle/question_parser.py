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
# referents or generic discourse markers. "person" was previously here on the
# theory that "the person riding X" should resolve X. Removed because the POPE
# spike showed it tanked recall on "Is there a person?" questions and our
# multi-target oracle handles "person + X" via max-pooling, so including
# "person" doesn't hurt the original use case.
_GENERIC_REFERENTS: frozenset[str] = frozenset({
    "image", "photo", "picture", "scene", "background", "foreground",
    "left", "right", "top", "bottom", "middle", "center", "side",
    "front", "back", "thing", "things", "object", "objects",
    "color", "colors", "colour", "colours", "size", "sizes",
    "number", "numbers", "kind", "kinds", "type", "types",
    "way", "ways", "name", "names", "place", "places",
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
    # Modal verbs — need to close NP walk so "How many forks CAN you see"
    # doesn't capture "can" as the head.
    "can", "could", "may", "might", "shall", "should", "will", "would", "must",
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


# Tokens that signal the END of a noun phrase. A determiner triggers an NP
# walk; the walk ends when we hit a function word, preposition, conjunction,
# or another determiner (so "the two surfboards" → "two" closes the NP after
# "the" and starts a new NP from "two").
_NP_BOUNDARIES: frozenset[str] = _NON_NOUNS | _DETERMINERS

# Cap NP-walk length so a run-on sentence ("the dog playing in the park
# wearing a hat") doesn't bleed into a 6-word "phrase".
_MAX_NP_LEN: int = 4


def parse_targets(question: str) -> list[str]:
    """Return ordered list of target nouns the question asks about.

    Walks tokens; on each determiner, captures the entire noun phrase span
    that follows (up to ``_MAX_NP_LEN`` tokens, ending at the first NP
    boundary), drops any generic referents, and emits the LAST surviving
    word as the head noun.

    Why head-as-last-word: the previous adjective-skipping logic stopped at
    the first non-adjective and called it the head. That mishandled multi-
    word objects like "dining table" → emitted "dining". Walking the whole
    NP and taking the tail gives "table" instead, which the downstream
    ``match_targets_to_vocab`` substring-matcher can hook into.

    Strips generic referents (image/photo/scene/...) and deduplicates while
    preserving order.
    """
    if not question:
        return []
    tokens = _tokenize(question)
    targets: list[str] = []
    i = 0
    while i < len(tokens):
        if tokens[i] in _DETERMINERS:
            # Walk the NP span starting just after the determiner.
            j = i + 1
            np: list[str] = []
            while (
                j < len(tokens)
                and len(np) < _MAX_NP_LEN
                and tokens[j] not in _NP_BOUNDARIES
            ):
                np.append(tokens[j])
                j += 1
            # Drop generic referents — they're never real targets.
            np = [w for w in np if w not in _GENERIC_REFERENTS]
            if np:
                head = np[-1]
                # Final filters: head must not be a function-word/determiner
                # leftover (defensive) and must be at least 2 chars to allow
                # short COCO labels like "tv".
                if (
                    len(head) >= 2
                    and head not in _NON_NOUNS
                    and head not in _DETERMINERS
                ):
                    targets.append(head)
            # Continue scanning from where the NP ended; if we made no
            # progress (j==i+1 and np empty) advance by one to avoid loops.
            i = j if j > i + 1 else i + 1
        else:
            i += 1

    # Dedup preserving order.
    seen: dict[str, None] = {}
    for t in targets:
        if t not in seen:
            seen[t] = None
    return list(seen.keys())


# Yes/no auxiliary verbs that signal a binary-answer question.
# Used by SGODDecoder to apply a stronger oracle injection on the first
# generated token of POPE-style questions where soft-additive scoring at
# anchor positions never fires (yes/no are neutral tokens).
_YESNO_AUX: frozenset[str] = frozenset({
    "is", "are", "was", "were", "am", "be",
    "do", "does", "did",
    "has", "have", "had",
    "can", "could", "may", "might",
    "shall", "should", "will", "would", "must",
})


def is_yesno_question(question: str) -> bool:
    """Detect whether ``question`` expects a yes/no answer.

    Two signals (any one suffices):
      1. The prompt explicitly asks for "yes or no" or "(yes/no)".
      2. Some sentence ending in "?" starts with a yes/no auxiliary verb
         ("Is there...", "Are these...", "Does the...", etc.).

    The detector is intentionally permissive — false positives on yes/no
    detection only cause the oracle to push yes/no logits on a question
    where the model wasn't going to emit yes/no anyway, which is a no-op.
    """
    if not question:
        return False
    text = question.lower()
    if "yes or no" in text or "(yes/no)" in text:
        return True
    for line in text.split("\n"):
        line = line.strip()
        if not line.endswith("?"):
            continue
        words = re.findall(r"[a-z]+", line)
        if words and words[0] in _YESNO_AUX:
            return True
    return False


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
