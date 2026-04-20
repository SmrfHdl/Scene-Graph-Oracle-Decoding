"""
Demo: Visual Oracle end-to-end on a real image.

What this script does:
  1. Downloads a test image (dog on sofa scene)
  2. Runs SGGModule (RelTR) to extract a real scene graph
  3. Runs CLIPScorer with pre-computed vocab cache
  4. Runs VisualOracle to score candidate tokens

Output is a plain-text table showing:
  - What RelTR detected (objects, relations)
  - How oracle scores each word for each anchor type
  - Why oracle boosts some words and penalises others

Usage:
    python scripts/demo_oracle.py
    python scripts/demo_oracle.py --image path/to/your/image.jpg
    python scripts/demo_oracle.py --no-sgg    # skip RelTR, use a hand-crafted scene graph
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

# Ensure project root is on sys.path when script is invoked directly
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Config ───────────────────────────────────────────────────────────────────

RELTR_CKPT = "data/checkpoints/reltr/checkpoint0149.pth"
CLIP_CACHE = "data/checkpoints/clip/clip_vocab_cache_reltr.pt"

# Public-domain photo: dog sitting on a couch (Wikimedia Commons)
DEMO_IMAGE_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/"
    "YellowLabradorLooking_new.jpg/320px-YellowLabradorLooking_new.jpg"
)
DEMO_IMAGE_PATH = "outputs/demo_dog.jpg"

# Fallback probe words used when SG has no entries for a category
_FALLBACK_NOUNS      = ["dog", "person", "cat", "car", "table", "chair", "sofa", "couch"]
_FALLBACK_RELATIONS  = ["on", "near", "behind", "above", "under", "with", "at"]
_FALLBACK_ATTRIBUTES = ["yellow", "black", "white", "large", "small", "brown", "fluffy"]


def build_probes(sg) -> tuple[list[str], list[str], list[str]]:
    """Build probe lists that always include the actual SG words + controls."""
    sg_nouns = [o.label.lower() for o in sg.objects]
    sg_rels  = [r.predicate.lower() for r in sg.relations]
    sg_attrs = [a.attribute.lower() for a in sg.attributes]

    def merge(sg_words, fallback):
        seen = set(sg_words)
        extras = [w for w in fallback if w not in seen]
        return list(dict.fromkeys(sg_words + extras))  # SG words first, deduped

    nouns = merge(sg_nouns, _FALLBACK_NOUNS)[:10]
    rels  = merge(sg_rels,  _FALLBACK_RELATIONS)[:10]
    attrs = merge(sg_attrs, _FALLBACK_ATTRIBUTES)[:10]
    return nouns, rels, attrs


# ── Helpers ──────────────────────────────────────────────────────────────────

def download_image(url: str, dest: str) -> None:
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading test image → {dest}")
    urllib.request.urlretrieve(url, dest)


def print_section(title: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def score_table(oracle, tokens: list[str], anchor_type: str) -> None:
    """Print a sorted score table for the given tokens + anchor type."""
    scores = [(t, oracle.score(t, anchor_type)) for t in tokens]
    scores.sort(key=lambda x: x[1], reverse=True)
    for word, s in scores:
        bar = "+" * int(abs(s) * 20) if s >= 0 else "-" * int(abs(s) * 20)
        sign = " " if s >= 0 else ""
        marker = "  ← IN scene graph" if _in_sg(oracle, word, anchor_type) else ""
        print(f"  {word:<12}  {sign}{s:+.3f}  {bar}{marker}")


def _in_sg(oracle, word: str, anchor_type: str) -> bool:
    w = word.lower()
    if anchor_type == "noun_anchor":     return w in oracle.noun_vocab
    if anchor_type == "relation_anchor": return w in oracle.rel_vocab
    if anchor_type == "attr_anchor":     return w in oracle.attr_vocab
    return False


def make_hand_crafted_sg():
    """Fallback scene graph when --no-sgg is used."""
    from sgod.sgg.scene_graph import AttributeNode, ObjectNode, RelationEdge, SceneGraph
    return SceneGraph(
        objects=[
            ObjectNode(label="dog",   confidence=0.92, bbox=(50, 80, 200, 300)),
            ObjectNode(label="couch", confidence=0.78, bbox=(10, 150, 310, 380)),
        ],
        relations=[
            RelationEdge(subject="dog", predicate="on", object="couch", confidence=0.71),
        ],
        attributes=[
            AttributeNode(entity="dog", attribute="yellow", confidence=0.80),
        ],
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Visual Oracle demo")
    parser.add_argument("--image", default=None, help="Path to an image file")
    parser.add_argument("--no-sgg", action="store_true",
                        help="Skip RelTR; use a hand-crafted scene graph instead")
    args = parser.parse_args()

    from PIL import Image, ImageDraw

    # ── Step 1: Load image ───────────────────────────────────────────────
    print_section("STEP 1 — Image")
    if args.image:
        img = Image.open(args.image).convert("RGB")
        print(f"  Loaded: {args.image}  size={img.size}")
    elif not args.no_sgg and Path(DEMO_IMAGE_PATH).exists():
        img = Image.open(DEMO_IMAGE_PATH).convert("RGB")
        print(f"  Loaded cached: {DEMO_IMAGE_PATH}  size={img.size}")
    elif not args.no_sgg:
        try:
            download_image(DEMO_IMAGE_URL, DEMO_IMAGE_PATH)
            img = Image.open(DEMO_IMAGE_PATH).convert("RGB")
            print(f"  Loaded: {DEMO_IMAGE_PATH}  size={img.size}")
        except Exception as e:
            print(f"  Download failed ({e}). Falling back to synthetic image.")
            args.no_sgg = True

    if args.no_sgg and not args.image:
        # Synthetic image: yellow blob (dog-like) on brown rectangle (couch)
        img = Image.new("RGB", (320, 240), color=(135, 206, 235))  # sky blue bg
        draw = ImageDraw.Draw(img)
        draw.rectangle([20, 140, 300, 230], fill=(139, 90, 43))   # brown couch
        draw.ellipse([100, 60, 220, 180], fill=(210, 180, 40))     # yellow dog
        print("  Using synthetic image (320×240): yellow ellipse on brown rectangle on blue bg)")
        print("  CLIP scores will reflect this synthetic content — not a real dog photo.")

    # ── Step 2: Scene Graph ──────────────────────────────────────────────
    print_section("STEP 2 — Scene Graph (RelTR)")

    if args.no_sgg:
        print("  [--no-sgg] Using hand-crafted scene graph.")
        sg = make_hand_crafted_sg()
    else:
        if not Path(RELTR_CKPT).exists():
            print(f"  Checkpoint not found: {RELTR_CKPT}")
            print("  Falling back to hand-crafted scene graph (use --no-sgg to silence).")
            sg = make_hand_crafted_sg()
        else:
            print(f"  Loading RelTR checkpoint: {RELTR_CKPT}")
            from sgod.sgg import SGGModule
            sgg = SGGModule(RELTR_CKPT)
            sg = sgg.extract(img)

    print(f"\n  Objects detected:")
    for o in sg.objects:
        print(f"    {o.label:<15}  conf={o.confidence:.3f}")
    print(f"\n  Relations detected:")
    for r in sg.relations:
        print(f"    {r.subject} --[{r.predicate}]--> {r.object}  conf={r.confidence:.3f}")
    if sg.attributes:
        print(f"\n  Attributes detected:")
        for a in sg.attributes:
            print(f"    {a.entity}:{a.attribute}  conf={a.confidence:.3f}")

    oracle_active = sg.should_activate_oracle()
    avg_conf = sum(o.confidence for o in sg.objects) / max(len(sg.objects), 1)
    print(f"\n  Average object confidence: {avg_conf:.3f}")
    print(f"  Oracle active?  {'YES ✓' if oracle_active else 'NO — SG quality too low'}")

    if not oracle_active:
        print("\n  Oracle would be DISABLED for this image. Scores shown below are")
        print("  informational only — in real decoding the oracle would be bypassed.")

    # ── Step 3: CLIPScorer ───────────────────────────────────────────────
    print_section("STEP 3 — CLIPScorer")
    from sgod.oracle import CLIPScorer

    if Path(CLIP_CACHE).exists():
        print(f"  Loading vocab cache: {CLIP_CACHE}")
        clip = CLIPScorer(img, vocab_cache_path=CLIP_CACHE)
    else:
        print(f"  No vocab cache found ({CLIP_CACHE}). Loading CLIP without cache.")
        clip = CLIPScorer(img)

    # Show SG words vs clearly unrelated words to illustrate contrast
    sg_words   = [o.label.lower() for o in sg.objects[:3]] + \
                 [r.predicate.lower() for r in sg.relations[:1]]
    ctrl_words = ["volcano", "submarine", "unicorn"]
    sample_words = list(dict.fromkeys(sg_words + ctrl_words))[:8]

    print(f"\n  Raw CLIP cosine similarity (image vs word), normalized to [0,1]:")
    for w in sample_words:
        s = clip.score_single(w)
        bar = "█" * int(s * 20)
        in_sg = w in {o.label.lower() for o in sg.objects} | {r.predicate.lower() for r in sg.relations}
        tag = "  ← in SG" if in_sg else ""
        print(f"    {w:<12}  {s:.3f}  {bar}{tag}")
    print()
    print("  NOTE: CLIP single-word scores compress into ~0.55-0.65 for common words.")
    print("  The important signal is the RELATIVE difference: SG words score slightly")
    print("  higher than unrelated words. The oracle amplifies this via SG confidence.")

    # ── Step 4: VisualOracle ─────────────────────────────────────────────
    from sgod.oracle import VisualOracle
    oracle = VisualOracle(sg, clip)

    probe_nouns, probe_rels, probe_attrs = build_probes(sg)

    print_section("STEP 4 — VisualOracle scores")
    print("""
  HOW TO READ THESE SCORES
  ─────────────────────────────────────────────────────────────────
  Score > 0 → oracle BOOSTS this token's logit  (SG/CLIP says it's real)
  Score < 0 → oracle PENALISES this token's logit (SG/CLIP says it's unlikely)
  Score = 0 → oracle does nothing (neutral anchor)

  Formula (noun):
    if word IN scene graph:  score = sg_conf² + (1-sg_conf) × clip_score
    if word NOT in SG:       score = sg_conf×(-0.3) + (1-sg_conf)×(clip_score-0.5)

  Formula (relation):
    if word IN SG: score = sg_conf    else: -0.2  (closed-world, 51-class vocab)

  Formula (attribute):
    if word IN SG: score = sg_conf    else: clip_score - 0.5  (open-world)
  ─────────────────────────────────────────────────────────────────""")

    print("\n  [noun_anchor]  — VLM is about to generate an object name")
    score_table(oracle, probe_nouns, "noun_anchor")

    print("\n  [relation_anchor]  — VLM is about to generate a spatial/relational word")
    score_table(oracle, probe_rels, "relation_anchor")

    if probe_attrs == _FALLBACK_ATTRIBUTES:
        print("\n  [attr_anchor]  — (no attributes in SG; showing CLIP-only fallback scores)")
    else:
        print("\n  [attr_anchor]  — VLM is about to generate a colour/size/property word")
    score_table(oracle, probe_attrs, "attr_anchor")

    # ── Step 5: Concrete decode example ─────────────────────────────────
    print_section("STEP 5 — Concrete example: what oracle does during decoding")

    # Pick the top SG noun as the "correct" answer; use 3 plausible distractors
    sg_noun = sg.objects[0].label.lower() if sg.objects else "table"
    distractors = [w for w in _FALLBACK_NOUNS if w != sg_noun][:3]
    candidates  = [sg_noun] + distractors

    # Simulate LM logits: correct SG word ranked 2nd (close behind), oracle flips it
    base_logits = [3.0, 3.2, 2.1, 1.9]
    fake_logits = dict(zip(candidates, base_logits))
    lm_winner   = max(candidates, key=lambda t: fake_logits[t])

    print(f"""
  Imagine the VLM is generating the first object noun.
  SG detected: {', '.join(o.label for o in sg.objects[:3])}

  Simulated LM logits (before oracle) — "{lm_winner}" leads:
  Oracle injects: logit_final = logit_lm + λ × oracle_score  (λ=0.4)
""")
    lam = 0.4
    final_winner = max(candidates, key=lambda t: fake_logits[t] + lam * oracle.score(t, "noun_anchor"))
    print(f"  {'Token':<12} {'logit_lm':>10} {'oracle×λ':>10} {'logit_final':>12}")
    print(f"  {'─'*12} {'─'*10} {'─'*10} {'─'*12}")
    for tok in candidates:
        lm = fake_logits[tok]
        o_score = oracle.score(tok, "noun_anchor")
        final = lm + lam * o_score
        tag = " ← oracle corrects ✓" if tok == final_winner and tok != lm_winner else \
              " ← wins (LM was right)" if tok == final_winner else \
              " ← LM favourite (no oracle)" if tok == lm_winner else ""
        print(f"  {tok:<12} {lm:>10.2f} {lam*o_score:>+10.3f} {final:>12.3f}{tag}")

    if final_winner == lm_winner:
        print(f"\n  (Oracle reinforced the LM's choice — both agree on '{final_winner}')")

    print()


if __name__ == "__main__":
    main()
