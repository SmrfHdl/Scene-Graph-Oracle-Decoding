"""
Demo: SGOD Decoder — Component 5 end-to-end integration (Section 3.6).

Simulates the full SGOD pipeline WITHOUT a real GPU/VLM by using:
  - A fake scene graph (or real RelTR if --image is passed)
  - A fake VLM that generates a fixed caption token-by-token
  - A fake CLIPScorer that returns constant 0.5 (or real CLIP if --image)

Shows at each token step:
  - Which component makes a decision
  - Whether oracle is applied and with what lambda
  - How the logit of the "correct" token changes

Usage:
    python scripts/demo_decoder.py                        # fully synthetic
    python scripts/demo_decoder.py --image test_imgs/image.png  # real SGG+CLIP
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse

import torch

from sgod.anchor import detect_anchor
from sgod.context import BASE_LAMBDA, GenerationContext
from sgod.oracle import VisualOracle
from sgod.sgg.scene_graph import ObjectNode, RelationEdge, SceneGraph

RELTR_CKPT = "data/checkpoints/reltr/checkpoint0149.pth"
CLIP_CACHE  = "data/checkpoints/clip/clip_vocab_cache_reltr.pt"


def print_section(title: str) -> None:
    print(f"\n{'─'*70}")
    print(f"  {title}")
    print(f"{'─'*70}")


# ── Fake components for CPU-only demo ────────────────────────────────────────

class FakeCLIPScorer:
    """Returns 0.5 for every word (neutral — not in/out of image)."""
    def score_single(self, _word: str) -> float:
        return 0.5


def _fake_scene_graph() -> SceneGraph:
    return SceneGraph(
        objects=[
            ObjectNode("table", 0.81, (0, 0, 1, 1)),
            ObjectNode("chair", 0.76, (0, 0, 1, 1)),
        ],
        relations=[RelationEdge("chair", "at", "table", 0.22)],
        attributes=[],
    )


# ── Logit-level oracle simulation ────────────────────────────────────────────

def simulate_oracle_on_caption(
    caption: str,
    oracle: VisualOracle,
    question: str,
    base_logit: float = 5.0,
    competing_logit: float = 5.5,
) -> None:
    """
    Walk through a caption token-by-token, showing how oracle shifts logits.

    For each token we simulate two logit scenarios:
      - Without oracle: the "correct" token starts at base_logit
      - With oracle:    oracle score added → may flip the winner
    """
    ctx    = GenerationContext(question)
    tokens = caption.split()
    prev:  list[str] = []

    print(f"\n  Question : \"{question}\"  [{ctx.question_type}, λ_base={BASE_LAMBDA[ctx.question_type]:.2f}]")
    print(f"  Caption  : \"{caption}\"")
    print(f"\n  {'#':<3}  {'Token':<10}  {'Anchor':<16}  {'λ':>5}  {'Oracle Δ':>9}  "
          f"{'Logit (no oracle)':>18}  {'Logit (+ oracle)':>17}  Winner?")
    print(f"  {'─'*3}  {'─'*10}  {'─'*16}  {'─'*5}  {'─'*9}  {'─'*18}  {'─'*17}  {'─'*7}")

    flips = 0
    for i, tok in enumerate(tokens):
        anchor      = detect_anchor(prev, tok)
        lam         = ctx.get_lambda()
        o_score     = oracle.score(tok, anchor) if anchor != "neutral" and lam != 0.0 else 0.0
        oracle_delta = lam * o_score

        # Logit without oracle — competitor always has slight edge
        logit_no  = base_logit
        logit_yes = base_logit + oracle_delta  # with oracle applied

        winner_no  = "correct" if logit_no >= competing_logit else "WRONG  "
        winner_yes = "correct" if logit_yes >= competing_logit else "WRONG  "
        flip = "↑ FLIP" if winner_yes != winner_no else ""

        print(f"  {i:<3}  {tok:<10}  [{anchor:<14}]  {lam:>+5.2f}  {oracle_delta:>+9.3f}  "
              f"{logit_no:>18.3f}  {logit_yes:>17.3f}  {winner_yes}  {flip}")

        if flip:
            flips += 1
        prev.append(tok)
        ctx.update(tok)

    total_anchor = sum(
        1 for i, t in enumerate(tokens)
        if detect_anchor(tokens[:i], t) != "neutral"
    )
    print(f"\n  Anchor positions: {total_anchor}/{len(tokens)}  |  "
          f"Oracle flipped winner: {flips} time(s)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=None)
    args = parser.parse_args()

    # ── Build oracle ─────────────────────────────────────────────────────────
    print_section("STEP 1 — Scene Graph + Oracle Construction")

    if args.image:
        from PIL import Image
        from sgod.sgg import SGGModule
        from sgod.oracle import CLIPScorer

        img  = Image.open(args.image).convert("RGB")
        sgg  = SGGModule(RELTR_CKPT)
        sg   = sgg.extract(img)
        clip = CLIPScorer(img, vocab_cache_path=CLIP_CACHE)
        print(f"  Image: {args.image}")
        print(f"  Objects  : {[o.label for o in sg.objects]}")
        print(f"  Relations: {[(r.subject, r.predicate, r.object) for r in sg.relations]}")
    else:
        sg   = _fake_scene_graph()
        clip = FakeCLIPScorer()
        print("  [no --image] Using synthetic scene graph: table + chair scene")
        print(f"  Objects  : {[o.label for o in sg.objects]}")
        print(f"  Relations: {[(r.subject, r.predicate, r.object) for r in sg.relations]}")

    oracle = VisualOracle(sg, clip)
    print(f"\n  noun_vocab : {oracle.noun_vocab}")
    print(f"  rel_vocab  : {oracle.rel_vocab}")
    print(f"  attr_vocab : {oracle.attr_vocab}")
    print(f"  SG quality OK? {oracle.should_activate_oracle()}")

    # ── Architecture overview ─────────────────────────────────────────────────
    print_section("STEP 2 — SGOD Pipeline Architecture")
    print("""
  Per-token decoding loop (SGODDecoder.generate):

    Input tokens + image
         │
         ▼
    LLaVA forward → logit_lm [vocab_size]         (1 forward pass)
         │
         ▼
    detect_anchor(prev_tokens, candidate) → anchor_type  (Component 3)
         │
         ├── neutral  → logit_final = logit_lm   (oracle skipped)
         │
         └── anchor   → λ = context.get_lambda()  (Component 4)
                      → oracle.score(word, anchor) (Component 2)
                      → logit_final = logit_lm + λ × oracle_score
         │
         ▼
    argmax(logit_final) → next token
    context.update(token)                          (Component 4)
    prev_tokens.append(token)
    """)

    # ── Token-by-token walkthrough ────────────────────────────────────────────
    print_section("STEP 3 — Token-by-token Oracle Walkthrough")
    print("""
  For each token we show:
    Anchor type    — what Component 3 detects
    λ              — oracle mixing weight from Component 4
    Oracle Δ       — how much the token's logit shifts
    Winner?        — does oracle change which token wins? (base logit = 5.0, competitor = 5.5)
  """)

    if sg.objects:
        obj1 = sg.objects[0].label
        obj2 = sg.objects[1].label if len(sg.objects) > 1 else "floor"
        rel  = sg.relations[0].predicate if sg.relations else "near"
    else:
        obj1, obj2, rel = "table", "chair", "at"

    captions = [
        (f"There is a {obj1} and a {obj2}",     "Is there a table?"),
        (f"The {obj1} is {rel} the {obj2}",     "What is on the table?"),
        (f"A large {obj1} with a small {obj2}", "Describe the scene."),
        (f"No there is no {obj1} here",         "Is there a table?"),
    ]

    for caption, question in captions:
        print(f"\n  ── Caption: \"{caption}\"")
        simulate_oracle_on_caption(caption, oracle, question)

    # ── Summary ──────────────────────────────────────────────────────────────
    print_section("STEP 4 — What SGODDecoder.generate() does differently from vanilla LLaVA")
    print(f"""
  Vanilla LLaVA:
    logit_final = logit_lm            (every token, no external signal)

  SGOD (this decoder):
    if anchor != neutral and λ != 0:
        logit_final = logit_lm + λ × oracle_score(token)
    else:
        logit_final = logit_lm        (same as baseline)

  Oracle is selective — only fires on:
    noun_anchor:     token after determiner → is this object real in the image?
    relation_anchor: spatial preposition → is this relation in the scene graph?
    attr_anchor:     color/size token → does this attribute match?

  Everything else (verbs, conjunctions, articles, pronouns) passes through unchanged.
  This preserves LM fluency while steering object/relation/attribute generation.

  Oracle is also context-aware:
    "No, there is no {obj1}" → negation suppresses oracle → LM generates freely
    "What if there was a lion?" → hypothetical → oracle OFF entirely
    """)


if __name__ == "__main__":
    main()
