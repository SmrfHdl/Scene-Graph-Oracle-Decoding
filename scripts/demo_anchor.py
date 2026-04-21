"""
Demo: Anchor Detector — what it does and why it matters.

Component 3 is pure linguistics: given the tokens generated so far,
it decides whether the NEXT token position needs oracle verification.

This script:
  1. Runs SGG on a real image to get a scene graph
  2. Simulates a VLM generating a caption token-by-token
  3. Shows what detect_anchor returns at every position
  4. Shows how the oracle would be triggered (or not) at each step

No GPU needed for the anchor detector itself — only SGG uses the checkpoint.

Usage:
    python scripts/demo_anchor.py --image test_imgs/image.png
    python scripts/demo_anchor.py --no-sgg   # skip RelTR, use a fake caption
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

RELTR_CKPT = "data/checkpoints/reltr/checkpoint0149.pth"
CLIP_CACHE  = "data/checkpoints/clip/clip_vocab_cache_reltr.pt"

ANCHOR_COLORS = {
    "noun_anchor":     "NOUN",
    "relation_anchor": "REL ",
    "attr_anchor":     "ATTR",
    "neutral":         "    ",
}


def print_section(title: str) -> None:
    print(f"\n{'─'*65}")
    print(f"  {title}")
    print(f"{'─'*65}")


def simulate_decode(caption: str, oracle, lam: float = 0.40) -> None:
    """Walk through caption token-by-token, showing anchor type + oracle score."""
    from sgod.anchor import detect_anchor

    tokens = caption.split()
    prev: list[str] = []

    print(f"\n  Caption: \"{caption}\"")
    print(f"  λ = {lam}  (oracle mixing weight)\n")
    print(f"  {'#':<3}  {'Token':<12}  {'Anchor':<6}  {'Oracle score':>13}  {'logit effect':>13}  Note")
    print(f"  {'─'*3}  {'─'*12}  {'─'*6}  {'─'*13}  {'─'*13}  {'─'*30}")

    for i, tok in enumerate(tokens):
        anchor = detect_anchor(prev, tok)
        label  = ANCHOR_COLORS[anchor]

        if anchor != "neutral" and oracle is not None:
            o_score = oracle.score(tok, anchor)
            effect  = f"{lam * o_score:+.3f}"
            score_s = f"{o_score:+.3f}"
        else:
            o_score = None
            effect  = "  —"
            score_s = "  —"

        note = _explain(tok, anchor, prev, oracle)
        print(f"  {i:<3}  {tok:<12}  [{label}]  {score_s:>13}  {effect:>13}  {note}")
        prev.append(tok)

    print()
    active_count = sum(
        1 for i, t in enumerate(tokens)
        if detect_anchor(tokens[:i], t) != "neutral"
    )
    print(f"  Oracle activated at {active_count}/{len(tokens)} token positions "
          f"({100*active_count//len(tokens)}% of tokens).")
    print(f"  Remaining {len(tokens)-active_count} tokens pass through unmodified (neutral).")


def _explain(tok: str, anchor: str, prev: list[str], oracle) -> str:
    if anchor == "neutral":
        return "skip — no oracle needed"
    if anchor == "noun_anchor":
        prev_w = prev[-1] if prev else ""
        if oracle and tok.lower() in oracle.noun_vocab:
            return f"IN scene graph → boost"
        elif oracle:
            return f"NOT in SG → small penalty"
        return f"after '{prev_w}' → expect noun"
    if anchor == "relation_anchor":
        if oracle and tok.lower() in oracle.rel_vocab:
            return f"IN scene graph → boost"
        elif oracle:
            return f"NOT in SG → -0.2 penalty"
        return "spatial preposition"
    if anchor == "attr_anchor":
        if oracle and tok.lower() in oracle.attr_vocab:
            return f"IN scene graph → boost"
        elif oracle:
            return f"CLIP fallback (open-world)"
        return "color/size attribute"
    return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=None)
    parser.add_argument("--no-sgg", action="store_true")
    args = parser.parse_args()

    # ── Step 1: Scene graph ──────────────────────────────────────────────
    print_section("STEP 1 — Scene Graph from image")

    sg = oracle = clip = None

    if not args.no_sgg and args.image:
        from PIL import Image
        from sgod.sgg import SGGModule
        from sgod.oracle import CLIPScorer, VisualOracle

        img = Image.open(args.image).convert("RGB")
        print(f"  Image: {args.image}  size={img.size}")
        print(f"  Loading RelTR…")
        sgg  = SGGModule(RELTR_CKPT)
        sg   = sgg.extract(img)
        clip = CLIPScorer(img, vocab_cache_path=CLIP_CACHE)
        oracle = VisualOracle(sg, clip)
        print(f"  Objects:   {[o.label for o in sg.objects]}")
        print(f"  Relations: {[(r.subject, r.predicate, r.object) for r in sg.relations]}")
    else:
        print("  [--no-sgg] Using hand-crafted scene graph (table + chair scene).")
        from sgod.sgg.scene_graph import AttributeNode, ObjectNode, RelationEdge, SceneGraph
        sg = SceneGraph(
            objects=[
                ObjectNode("table", 0.81, (0,0,1,1)),
                ObjectNode("chair", 0.76, (0,0,1,1)),
            ],
            relations=[RelationEdge("chair","at","table", 0.22)],
            attributes=[],
        )

    # ── Step 2: Explain anchor detector ──────────────────────────────────
    print_section("STEP 2 — What the Anchor Detector does")
    print("""
  During VLM decoding, we generate one token at a time.
  The oracle is expensive to apply everywhere, so we only activate it
  at "anchor positions" — tokens that are semantically grounded in the image.

  detect_anchor(prev_tokens, current_token) → anchor_type

  Rules (in priority order):
  ┌────────────────────────────────────────────────────────────────┐
  │ 1. prev[-1] is a determiner (a/an/the/this...)                 │
  │    → noun_anchor  (next token is likely an object name)        │
  │                                                                │
  │ 2. prev[-1] is a spatial prep (on/under/near/behind...)        │
  │    → noun_anchor  (next token is the object of the relation)   │
  │                                                                │
  │ 3. prev[-2:] or prev[-3:] forms "next to"/"in front of"...    │
  │    → noun_anchor  (same as above for multi-word preps)         │
  │                                                                │
  │ 4. current token IS a spatial prep                             │
  │    → relation_anchor  (the relation itself needs verification)  │
  │                                                                │
  │ 5. current token is a color or size word (red/large/small...)  │
  │    → attr_anchor  (attribute needs verification)               │
  │                                                                │
  │ 6. otherwise → neutral  (oracle skipped, logits unchanged)     │
  └────────────────────────────────────────────────────────────────┘
  """)

    # ── Step 3: Token-by-token walkthrough ───────────────────────────────
    print_section("STEP 3 — Token-by-token decode walkthrough")

    # Build captions from the real SG if available
    if sg and sg.objects:
        obj1 = sg.objects[0].label
        obj2 = sg.objects[1].label if len(sg.objects) > 1 else "floor"
        rel  = sg.relations[0].predicate if sg.relations else "near"
        caption1 = f"There is a {obj1} and a {obj2} in the image"
        caption2 = f"The {obj1} is {rel} the {obj2}"
        caption3 = f"A large {obj1} with a small {obj2} beside it"
    else:
        caption1 = "There is a table and a chair in the image"
        caption2 = "The chair is at the table"
        caption3 = "A large table with a small chair beside it"

    print(f"\n  ── Caption A: plain description")
    simulate_decode(caption1, oracle)

    print(f"\n  ── Caption B: relation-heavy")
    simulate_decode(caption2, oracle)

    print(f"\n  ── Caption C: attribute-heavy")
    simulate_decode(caption3, oracle)

    # ── Step 4: Why neutral tokens are important ─────────────────────────
    print_section("STEP 4 — Why NOT applying oracle everywhere matters")
    print("""
  If we applied the oracle to every token, we'd penalise function words,
  verbs, and conjunctions that have nothing to do with the scene graph.

  Example: "There is a dog on the table"
    "There"  → neutral  ✓ (function word, oracle skips)
    "is"     → neutral  ✓ (copula, oracle skips)
    "a"      → neutral  ✓ (determiner, but oracle scores NEXT token)
    "dog"    → noun_anchor ✓ (after "a" → oracle boosts if in SG)
    "on"     → relation_anchor ✓ (spatial prep → oracle checks SG)
    "the"    → neutral  ✓ (determiner, oracle scores NEXT token)
    "table"  → noun_anchor ✓ (after "the" → oracle boosts if in SG)

  Only 3/7 tokens activate the oracle. The others pass through unchanged,
  preserving the LM's fluency and grammar.
  """)

    from sgod.anchor import detect_anchor
    example = "There is a dog on the table".split()
    prev2: list[str] = []
    print(f"  {'Token':<10}  {'Anchor type':<18}  {'Oracle applied?'}")
    print(f"  {'─'*10}  {'─'*18}  {'─'*15}")
    for tok in example:
        a = detect_anchor(prev2, tok)
        applied = "YES →" + (" boost" if a == "noun_anchor" else " check") if a != "neutral" else "no"
        print(f"  {tok:<10}  [{ANCHOR_COLORS[a]}] {a:<14}  {applied}")
        prev2.append(tok)


if __name__ == "__main__":
    main()
