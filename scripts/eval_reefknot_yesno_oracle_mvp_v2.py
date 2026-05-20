"""MVP v2: per-question Grounding DINO query (fixes synonym + vocab issues from v1).

v1 result on n=100: -3% overall (helped early when parser worked + RelTR
detected query objects, harmed when synonym mismatch or out-of-vocab).
Root cause analysis (see eval_reefknot_yesno_mvp.json):
  - "bicycle" vs RelTR's "bike" — synonym miss
  - "powder", "shirt", "pepperoni" — outside RelTR 151-entity vocab
  - 58% questions couldn't be parsed (relation vocab too narrow)

v2 fixes both:
  - Use Grounding DINO directly with the parsed X / Y phrase as the query
    (GD is open-vocabulary, returns detection score for arbitrary text).
  - Massively expanded relation vocabulary + tolerant parser that falls
    back to "first NP / last NP" heuristic when relation regex misses.

Decision rule unchanged: missing → "No" bias; both present → mild "Yes" bias.

Usage:
    .venv/bin/python scripts/eval_reefknot_yesno_oracle_mvp_v2.py \\
        --reefknot-jsonl data/reefknot/YESNO.jsonl \\
        --image-dir data/reefknot/images \\
        --n 100 --bias-no 5.0 --bias-yes 1.0 \\
        --out outputs/stage0/eval_reefknot_yesno_mvp_v2.json
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image

from sgod.runtime import build_from_config, load_config

log = logging.getLogger("yesno_mvp_v2")


# ── Expanded relation vocabulary ─────────────────────────────────────────────
# Multi-word entries first (handled by length-descending sort below).
RELATIONS = sorted([
    # multi-word prepositions
    "in front of", "on top of", "next to", "in the middle of",
    "off the", "outside the", "inside the", "far from",
    "on right of", "on left of", "to the left of", "to the right of",
    "in front", "out of", "across from", "underneath the",
    # multi-word verbs
    "clashing with", "looking at", "watching with", "lying on", "sitting on",
    "standing on", "standing in", "leaning on", "kneeling on",
    # single-word prepositions
    "on", "in", "at", "by", "behind", "near", "above", "below",
    "under", "around", "across", "between", "beside", "off", "outside", "inside",
    "with", "to", "from", "of", "down", "up", "over", "past",
    # single-word verbs (-ing)
    "topping", "carrying", "holding", "riding", "wearing",
    "watching", "eating", "peeling", "driving", "following", "helping",
    "threatening", "unloading", "advertising", "covering", "wilting",
    "curling", "attaching", "observing", "containing", "slicing",
    "buying", "bending", "blocking", "lying", "sitting", "standing",
    "leaning", "kneeling", "jumping", "running", "walking", "playing",
    "throwing", "catching", "kicking", "hitting", "pushing", "pulling",
    "cutting", "washing", "cooking", "feeding", "guiding", "leading",
    "approaching", "facing", "passing", "crossing", "entering", "leaving",
    "filling", "emptying", "wrapping", "tying", "untying", "fixing",
    "breaking", "building", "destroying", "burning",
    "supporting", "balancing", "hanging", "floating", "diving",
    "swimming", "climbing", "descending", "ascending",
], key=len, reverse=True)


def _strip_question(q: str) -> str:
    t = q
    for trailing in (" Please answer yes or no.", "Please answer yes or no.",
                     " in this photo?", "in this photo?", "?", "."):
        t = t.replace(trailing, "")
    t = t.strip()
    for prefix in ("Is the ", "Are the ", "Is a ", "Is ", "Are "):
        if t.startswith(prefix):
            t = t[len(prefix):]
            break
    return t.strip()


def _parse_triplet(q: str) -> tuple[str | None, str | None, str | None]:
    """Return (x, rel, y). Any field may be None if not recoverable.

    Strategy:
      1. Try relation-vocab regex split (longest match wins).
      2. Fallback: assume "Is the X (verb-ing) Y" — split at first -ing word.
      3. Fallback: split at any common verb pattern.
      4. If still unable to split: x = whole stripped text, y = None.
    """
    t = _strip_question(q)
    if not t:
        return None, None, None

    # 1. Relation-vocab match (longest first).
    for rel in RELATIONS:
        m = re.search(rf"(?:^|\s){re.escape(rel)}(?:\s|$)", t)
        if m:
            x = t[:m.start()].strip()
            y = t[m.end():].strip()
            if x and y:
                return x, rel, y
            if x and not y:
                return x, rel, None

    # 2. Fallback: split at first -ing word.
    m = re.search(r"\b(\w+ing)\b", t)
    if m:
        verb = m.group(1)
        x = t[:m.start()].strip()
        y = t[m.end():].strip()
        if x:
            return x, verb, (y or None)

    # 3. Last resort: no parse.
    return t, None, None


# ── Phrase normalization ─────────────────────────────────────────────────────

_STOPWORDS = {"the", "a", "an", "some", "this", "that", "these", "those", "of"}


def _strip_article(p: str) -> str:
    p = p.lower().strip()
    p = re.sub(r"[^\w\s]", " ", p)
    words = [w for w in p.split() if w not in _STOPWORDS and len(w) > 1]
    return " ".join(words) or p


def _present_via_gd(image, phrase: str, gd_processor, gd_model, gd_device,
                    box_threshold: float, text_threshold: float) -> tuple[bool, float]:
    """Direct GD query: 'is `phrase` present in image?'

    Returns (present, best_score).
    """
    norm = _strip_article(phrase)
    if not norm:
        return False, 0.0
    # GD expects "query. " formatted text (each query separated by ". ").
    text = f"{norm}."
    inputs = gd_processor(images=image, text=text, return_tensors="pt").to(gd_device)
    with torch.no_grad():
        outputs = gd_model(**inputs)
    results = gd_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]],
    )[0]
    scores = results.get("scores")
    if scores is None or len(scores) == 0:
        return False, 0.0
    return True, float(scores.max().item())


# ── Data loading + metrics ──────────────────────────────────────────────────

def _load_reefknot(rk_jsonl: Path, image_dir: Path, n: int, seed: int) -> list[dict]:
    raw: list[dict] = []
    with open(rk_jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                raw.append(json.loads(line))
    rng = random.Random(seed)
    rng.shuffle(raw)
    out: list[dict] = []
    missing = 0
    for r in raw:
        if len(out) >= n:
            break
        img_path = image_dir / f"{r['image_id']}.jpg"
        if not img_path.exists():
            missing += 1
            continue
        out.append({
            "image_name": f"{r['image_id']}.jpg",
            "question": r["query_prompt"],
            "label": r["label"].strip().lower(),
            "type": r.get("relation_type", "unknown"),
            "image_path": str(img_path),
        })
    if missing:
        log.warning("%d examples skipped (image missing)", missing)
    return out


def _accuracy(records, key) -> dict:
    n = len(records)
    correct = sum(1 for r in records if r.get(key) == r["label"])
    yes_pred = sum(1 for r in records if r.get(key) == "yes")
    return {
        "n": n,
        "accuracy": round(correct / n, 4) if n else 0.0,
        "yes_frac": round(yes_pred / n, 4) if n else 0.0,
    }


def _by(records, key, group_fn) -> dict:
    buckets = defaultdict(list)
    for r in records:
        buckets[group_fn(r)].append(r)
    return {g: _accuracy(rs, key) for g, rs in buckets.items()}


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--student-config", type=Path, default=Path("configs/sgod_v1_llava15.yaml"),
                    help="Reused for LLaVA backbone + GD oracle settings")
    ap.add_argument("--reefknot-jsonl", type=Path, default=Path("data/reefknot/YESNO.jsonl"))
    ap.add_argument("--image-dir", type=Path, default=Path("data/reefknot/images"))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bias-no", type=float, default=5.0)
    ap.add_argument("--bias-yes", type=float, default=1.0)
    ap.add_argument("--gd-box-threshold", type=float, default=0.25)
    ap.add_argument("--gd-text-threshold", type=float, default=0.20)
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/stage0/eval_reefknot_yesno_mvp_v2.json"))
    args = ap.parse_args()

    examples = _load_reefknot(args.reefknot_jsonl, args.image_dir, args.n, args.seed)
    log.info("Eval on %d examples", len(examples))

    log.info("Building decoder + standalone GD model...")
    cfg = load_config(args.student_config)
    decoder = build_from_config(cfg, lazy=False)
    backbone = decoder.backbone
    tokenizer = backbone.tokenizer()

    # Standalone GD HF model (open-vocab, per-question text query).
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    gd_model_id = cfg["oracle"].get("grounding_dino", {}).get(
        "model_id", "IDEA-Research/grounding-dino-base")
    gd_device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Loading standalone Grounding DINO %s on %s", gd_model_id, gd_device)
    gd_processor = AutoProcessor.from_pretrained(gd_model_id)
    gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(gd_model_id).to(gd_device)
    gd_model.eval()

    yes_ids = [tokenizer.convert_tokens_to_ids(t) for t in ("Yes", "▁Yes", "yes", "▁yes")]
    no_ids = [tokenizer.convert_tokens_to_ids(t) for t in ("No", "▁No", "no", "▁no")]
    yes_ids = [i for i in yes_ids if i is not None and i != tokenizer.unk_token_id]
    no_ids = [i for i in no_ids if i is not None and i != tokenizer.unk_token_id]

    records = []
    parse_fail = 0
    for i, ex in enumerate(examples):
        img = Image.open(ex["image_path"]).convert("RGB")
        prompt = f"USER: <image>\n{ex['question']} ASSISTANT:"

        # Parse.
        x, rel, y = _parse_triplet(ex["question"])
        parsed = x is not None
        if not parsed:
            parse_fail += 1

        # GD query for X and Y (if present).
        x_present = x_score = y_present = y_score = None
        if x:
            x_present, x_score = _present_via_gd(
                img, x, gd_processor, gd_model, gd_device,
                args.gd_box_threshold, args.gd_text_threshold,
            )
        if y:
            y_present, y_score = _present_via_gd(
                img, y, gd_processor, gd_model, gd_device,
                args.gd_box_threshold, args.gd_text_threshold,
            )

        # LLaVA step-0 logits.
        inputs = backbone.prepare_inputs(image=img, prompt=prompt)
        with torch.no_grad():
            generated_ids = torch.empty(1, 0, dtype=torch.long)
            _h, logits = backbone.forward_step(inputs, generated_ids)
        logits = logits[0].float().cpu()

        yes_score_base = max(logits[i].item() for i in yes_ids)
        no_score_base = max(logits[i].item() for i in no_ids)
        pred_base = "yes" if yes_score_base > no_score_base else "no"

        # Bias rule. Both X and Y considered if both queried; otherwise just the
        # one we have.
        bias_target = None
        bias_applied = 0.0
        missing_any = False
        present_any = False
        if x_present is not None:
            missing_any = missing_any or (not x_present)
            present_any = present_any or x_present
        if y_present is not None:
            missing_any = missing_any or (not y_present)
            present_any = present_any or y_present

        if parsed and missing_any:
            for tid in no_ids:
                logits[tid] += args.bias_no
            bias_target = "no"
            bias_applied = args.bias_no
        elif parsed and present_any and not missing_any:
            for tid in yes_ids:
                logits[tid] += args.bias_yes
            bias_target = "yes"
            bias_applied = args.bias_yes

        yes_score_mvp = max(logits[i].item() for i in yes_ids)
        no_score_mvp = max(logits[i].item() for i in no_ids)
        pred_mvp = "yes" if yes_score_mvp > no_score_mvp else "no"

        records.append({
            "image_name": ex["image_name"],
            "question": ex["question"],
            "label": ex["label"],
            "type": ex["type"],
            "parsed": parsed,
            "x": x, "rel": rel, "y": y,
            "x_present": x_present, "x_score": round(x_score, 3) if x_score else None,
            "y_present": y_present, "y_score": round(y_score, 3) if y_score else None,
            "bias_target": bias_target,
            "bias_applied": bias_applied,
            "pred_base": pred_base,
            "pred_mvp": pred_mvp,
        })

        if (i + 1) % 20 == 0 or (i + 1) == len(examples):
            m_b = _accuracy(records, "pred_base")
            m_m = _accuracy(records, "pred_mvp")
            log.info(
                "[%d/%d]  base acc=%.3f  |  mvp_v2 acc=%.3f (Δ=%+.3f)  parse_fail=%d",
                i + 1, len(examples), m_b["accuracy"], m_m["accuracy"],
                m_m["accuracy"] - m_b["accuracy"], parse_fail,
            )

    base_m = _accuracy(records, "pred_base")
    mvp_m = _accuracy(records, "pred_mvp")
    summary = {
        "n_examples": len(records),
        "baseline_argmax_yes_no": base_m,
        "mvp_v2_gd_per_question": mvp_m,
        "acc_delta": round(mvp_m["accuracy"] - base_m["accuracy"], 4),
        "parse_fail_count": parse_fail,
        "bias_no_fires": sum(1 for r in records if r["bias_target"] == "no"),
        "bias_yes_fires": sum(1 for r in records if r["bias_target"] == "yes"),
        "by_type": {
            "baseline": _by(records, "pred_base", lambda r: r["type"]),
            "mvp_v2": _by(records, "pred_mvp", lambda r: r["type"]),
        },
        "by_parse": {
            "baseline": _by(records, "pred_base", lambda r: "parsed" if r["parsed"] else "unparsed"),
            "mvp_v2": _by(records, "pred_mvp", lambda r: "parsed" if r["parsed"] else "unparsed"),
        },
        "by_gt": {
            "baseline": _by(records, "pred_base", lambda r: f"gt_{r['label']}"),
            "mvp_v2": _by(records, "pred_mvp", lambda r: f"gt_{r['label']}"),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2)
    log.info("Wrote %s", args.out)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
