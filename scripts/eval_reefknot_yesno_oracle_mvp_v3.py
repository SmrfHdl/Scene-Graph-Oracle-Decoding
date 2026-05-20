"""MVP v3: spatial bounding-box verification for perception relations.

Lesson from v2 (eval_reefknot_yesno_mvp_v2.json):
    bias_yes fired 95/100 because GD detects both objects almost always.
    GT=yes accuracy → +14% (good)
    GT=no  accuracy → −36% (catastrophic)
    Reefknot's hard cases: X and Y both PRESENT, but RELATION wrong/absent.

v3 insight (the paper contribution):
    Reefknot tests RELATIONAL grounding, not object presence. We need to
    verify the asked relation. For *perception* relations (49% of test set)
    we can verify geometrically from GD bounding boxes: "X above Y" ↔
    Y_cy > X_cy after coordinate flip, etc. For *cognitive* relations
    (riding, watching, eating) we cannot verify from 2D boxes — fall back to
    object-presence-only, applied conservatively (only fire `no` when both
    objects are missing).

Decision rule per question:
    1. Parse (X, R, Y).
    2. GD query X and Y → (present, score, bbox) each.
    3. If R ∈ SPATIAL:
         - If both bboxes present:
              verify geometry. matches → bias_yes; mismatch → bias_no.
         - Else if either missing → bias_no.
    4. If R ∈ COGNITIVE / unknown:
         - If both missing → bias_no (strong).
         - Else: no bias (let LLaVA decide).

This is the "high precision, asymmetric" rule — only intervene when we have
clear evidence either way.

Usage:
    .venv/bin/python scripts/eval_reefknot_yesno_oracle_mvp_v3.py \\
        --reefknot-jsonl data/reefknot/YESNO.jsonl \\
        --image-dir data/reefknot/images \\
        --n 100 --out outputs/stage0/eval_reefknot_yesno_mvp_v3.json
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

log = logging.getLogger("yesno_mvp_v3")


# ── Relation taxonomy ───────────────────────────────────────────────────────
# Spatial relations we CAN verify from 2D bboxes.
SPATIAL_RELATIONS: dict[str, str] = {
    # Vertical
    "above": "y_above", "on top of": "y_above", "topping": "y_above",
    "over": "y_above", "up": "y_above",
    "below": "y_below", "under": "y_below", "underneath the": "y_below",
    "down": "y_below",
    # Horizontal
    "on left of": "x_left", "to the left of": "x_left",
    "on right of": "x_right", "to the right of": "x_right",
    # Containment
    "in": "contained_in", "inside": "contained_in", "inside the": "contained_in",
    "outside": "outside_of", "outside the": "outside_of",
    # Proximity
    "near": "close_to", "next to": "close_to", "beside": "close_to",
    "by": "close_to", "at": "close_to",
    "far from": "far_from",
    # Surrounding
    "around": "surrounds",
    # Generic "on" — treat as overlap/touching (default close-to)
    "on": "close_to",
    # "behind" / "in front of" — no depth, skip verification
    # "between" / "across" — multi-object, skip
}

# Multi-word entries first to ensure greedy match below.
RELATION_KEYS = sorted(set(SPATIAL_RELATIONS.keys()) | {
    # cognitive verbs (object-presence fallback only)
    "in front of", "behind",
    "holding", "riding", "wearing", "carrying", "driving", "following",
    "helping", "threatening", "watching", "eating", "peeling", "unloading",
    "clashing with", "looking at", "advertising", "covering", "wilting",
    "curling", "attaching", "observing", "containing", "slicing", "buying",
    "bending", "blocking", "lying", "sitting", "standing", "leaning",
    "kneeling", "jumping", "running", "walking", "playing", "throwing",
    "catching", "kicking", "hitting", "pushing", "pulling", "cutting",
    "washing", "cooking", "feeding", "guiding", "leading",
    "approaching", "facing", "passing", "crossing", "entering", "leaving",
    "filling", "emptying", "wrapping", "tying", "untying", "fixing",
    "breaking", "building", "destroying", "burning", "supporting",
    "balancing", "hanging", "floating", "diving", "swimming",
    "climbing", "descending", "ascending",
    # leftover prepositions
    "of", "with", "to", "from", "across", "between",
}, key=len, reverse=True)


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
    """Return (X, REL, Y). Greedy longest-match on REL."""
    t = _strip_question(q)
    if not t:
        return None, None, None
    for rel in RELATION_KEYS:
        m = re.search(rf"(?:^|\s){re.escape(rel)}(?:\s|$)", t)
        if m:
            x = t[:m.start()].strip()
            y = t[m.end():].strip()
            return (x or None), rel, (y or None)
    # Fallback: split at first -ing word
    m = re.search(r"\b(\w+ing)\b", t)
    if m:
        return (t[:m.start()].strip() or None), m.group(1), (t[m.end():].strip() or None)
    return t, None, None


# ── Phrase normalization ─────────────────────────────────────────────────────

_STOPWORDS = {"the", "a", "an", "some", "this", "that", "these", "those", "of"}


def _strip_article(p: str) -> str:
    p = p.lower().strip()
    p = re.sub(r"[^\w\s]", " ", p)
    words = [w for w in p.split() if w not in _STOPWORDS and len(w) > 1]
    return " ".join(words) or p


# ── GD query with bbox return ───────────────────────────────────────────────

def _query_gd(image, phrase: str, gd_processor, gd_model, gd_device,
              box_threshold: float, text_threshold: float):
    """Return (present, best_score, best_bbox). bbox = (x1, y1, x2, y2) or None."""
    norm = _strip_article(phrase)
    if not norm:
        return False, 0.0, None
    text = f"{norm}."
    inputs = gd_processor(images=image, text=text, return_tensors="pt").to(gd_device)
    with torch.no_grad():
        outputs = gd_model(**inputs)
    results = gd_processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids,
        threshold=box_threshold, text_threshold=text_threshold,
        target_sizes=[image.size[::-1]],
    )[0]
    scores = results.get("scores")
    boxes = results.get("boxes")
    if scores is None or len(scores) == 0:
        return False, 0.0, None
    best_idx = int(scores.argmax().item())
    bbox = tuple(float(v) for v in boxes[best_idx].tolist())
    return True, float(scores[best_idx].item()), bbox


# ── Geometric verification ──────────────────────────────────────────────────

def _bbox_center(b: tuple) -> tuple[float, float]:
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def _bbox_area(b: tuple) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _bbox_iou(a: tuple, b: tuple) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = _bbox_area(a) + _bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


def _contained_in(inner: tuple, outer: tuple) -> float:
    """Fraction of `inner` area that lies within `outer`."""
    a = _bbox_area(inner)
    if a == 0:
        return 0.0
    x1, y1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    x2, y2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return inter / a


def _verify_spatial(predicate: str, box_x: tuple, box_y: tuple,
                    img_w: float, img_h: float) -> tuple[bool, str]:
    """Return (matches, reason). True = geometry supports the claim 'X PRED Y'."""
    cx_x, cy_x = _bbox_center(box_x)
    cx_y, cy_y = _bbox_center(box_y)
    diag = (img_w ** 2 + img_h ** 2) ** 0.5

    if predicate == "y_above":  # X above Y → X is higher → cy_x < cy_y (image origin top-left)
        return cy_x < cy_y, f"cy_X={cy_x:.0f} cy_Y={cy_y:.0f}"
    if predicate == "y_below":
        return cy_x > cy_y, f"cy_X={cy_x:.0f} cy_Y={cy_y:.0f}"
    if predicate == "x_left":
        return cx_x < cx_y, f"cx_X={cx_x:.0f} cx_Y={cx_y:.0f}"
    if predicate == "x_right":
        return cx_x > cx_y, f"cx_X={cx_x:.0f} cx_Y={cx_y:.0f}"
    if predicate == "contained_in":  # X in Y → X bbox mostly inside Y
        frac = _contained_in(box_x, box_y)
        return frac > 0.6, f"X-in-Y frac={frac:.2f}"
    if predicate == "outside_of":
        frac = _contained_in(box_x, box_y)
        return frac < 0.2, f"X-in-Y frac={frac:.2f}"
    if predicate == "close_to":
        d = ((cx_x - cx_y) ** 2 + (cy_x - cy_y) ** 2) ** 0.5
        return d < 0.4 * diag, f"dist/diag={d/diag:.2f}"
    if predicate == "far_from":
        d = ((cx_x - cx_y) ** 2 + (cy_x - cy_y) ** 2) ** 0.5
        return d > 0.4 * diag, f"dist/diag={d/diag:.2f}"
    if predicate == "surrounds":  # Y surrounds X → X inside Y bbox
        frac = _contained_in(box_x, box_y)
        return frac > 0.7, f"X-in-Y frac={frac:.2f}"
    return True, "unverified_predicate"


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
    ap.add_argument("--student-config", type=Path, default=Path("configs/sgod_v1_llava15.yaml"))
    ap.add_argument("--reefknot-jsonl", type=Path, default=Path("data/reefknot/YESNO.jsonl"))
    ap.add_argument("--image-dir", type=Path, default=Path("data/reefknot/images"))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bias-no", type=float, default=5.0)
    ap.add_argument("--bias-yes", type=float, default=2.0)
    ap.add_argument("--gd-box-threshold", type=float, default=0.25)
    ap.add_argument("--gd-text-threshold", type=float, default=0.20)
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/stage0/eval_reefknot_yesno_mvp_v3.json"))
    args = ap.parse_args()

    examples = _load_reefknot(args.reefknot_jsonl, args.image_dir, args.n, args.seed)
    log.info("Eval on %d examples", len(examples))

    log.info("Building decoder + standalone GD...")
    cfg = load_config(args.student_config)
    decoder = build_from_config(cfg, lazy=False)
    backbone = decoder.backbone
    tokenizer = backbone.tokenizer()

    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    gd_model_id = cfg["oracle"].get("grounding_dino", {}).get(
        "model_id", "IDEA-Research/grounding-dino-base")
    gd_device = "cuda" if torch.cuda.is_available() else "cpu"
    gd_processor = AutoProcessor.from_pretrained(gd_model_id)
    gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(gd_model_id).to(gd_device)
    gd_model.eval()

    yes_ids = [tokenizer.convert_tokens_to_ids(t) for t in ("Yes", "▁Yes", "yes", "▁yes")]
    no_ids = [tokenizer.convert_tokens_to_ids(t) for t in ("No", "▁No", "no", "▁no")]
    yes_ids = [i for i in yes_ids if i is not None and i != tokenizer.unk_token_id]
    no_ids = [i for i in no_ids if i is not None and i != tokenizer.unk_token_id]

    records = []
    for i, ex in enumerate(examples):
        img = Image.open(ex["image_path"]).convert("RGB")
        img_w, img_h = img.size
        prompt = f"USER: <image>\n{ex['question']} ASSISTANT:"

        x, rel, y = _parse_triplet(ex["question"])
        parsed = x is not None
        spatial_pred = SPATIAL_RELATIONS.get(rel) if rel else None

        x_present = x_score = x_bbox = None
        y_present = y_score = y_bbox = None
        if x:
            x_present, x_score, x_bbox = _query_gd(
                img, x, gd_processor, gd_model, gd_device,
                args.gd_box_threshold, args.gd_text_threshold,
            )
        if y:
            y_present, y_score, y_bbox = _query_gd(
                img, y, gd_processor, gd_model, gd_device,
                args.gd_box_threshold, args.gd_text_threshold,
            )

        # LLaVA step-0 logits.
        inputs = backbone.prepare_inputs(image=img, prompt=prompt)
        with torch.no_grad():
            generated_ids = torch.empty(1, 0, dtype=torch.long)
            _h, logits = backbone.forward_step(inputs, generated_ids)
        logits = logits[0].float().cpu()

        yes_score_base = max(logits[i_].item() for i_ in yes_ids)
        no_score_base = max(logits[i_].item() for i_ in no_ids)
        pred_base = "yes" if yes_score_base > no_score_base else "no"

        # Decide bias.
        bias_target = None
        bias_reason = "no_bias"
        if parsed:
            if spatial_pred and x_present and y_present and x_bbox and y_bbox:
                matches, geom_reason = _verify_spatial(spatial_pred, x_bbox, y_bbox, img_w, img_h)
                bias_target = "yes" if matches else "no"
                bias_reason = f"spatial:{spatial_pred}|{geom_reason}|match={matches}"
            elif spatial_pred and (not x_present or not y_present):
                bias_target = "no"
                bias_reason = f"spatial:{spatial_pred}|missing(X={x_present},Y={y_present})"
            elif not spatial_pred:
                # cognitive / unknown — conservative: only fire NO when both missing
                if y is not None and not x_present and not y_present:
                    bias_target = "no"
                    bias_reason = "cognitive:both_missing"

        bias_amount = args.bias_no if bias_target == "no" else (args.bias_yes if bias_target == "yes" else 0.0)
        if bias_target == "no":
            for tid in no_ids:
                logits[tid] += args.bias_no
        elif bias_target == "yes":
            for tid in yes_ids:
                logits[tid] += args.bias_yes

        yes_score_mvp = max(logits[i_].item() for i_ in yes_ids)
        no_score_mvp = max(logits[i_].item() for i_ in no_ids)
        pred_mvp = "yes" if yes_score_mvp > no_score_mvp else "no"

        records.append({
            "image_name": ex["image_name"],
            "question": ex["question"],
            "label": ex["label"],
            "type": ex["type"],
            "parsed": parsed,
            "x": x, "rel": rel, "y": y,
            "spatial_pred": spatial_pred,
            "x_present": x_present, "y_present": y_present,
            "x_bbox": x_bbox, "y_bbox": y_bbox,
            "bias_target": bias_target,
            "bias_amount": bias_amount,
            "bias_reason": bias_reason,
            "pred_base": pred_base,
            "pred_mvp": pred_mvp,
        })

        if (i + 1) % 20 == 0 or (i + 1) == len(examples):
            m_b = _accuracy(records, "pred_base")
            m_m = _accuracy(records, "pred_mvp")
            log.info(
                "[%d/%d]  base acc=%.3f  |  mvp_v3 acc=%.3f (Δ=%+.3f)",
                i + 1, len(examples), m_b["accuracy"], m_m["accuracy"],
                m_m["accuracy"] - m_b["accuracy"],
            )

    base_m = _accuracy(records, "pred_base")
    mvp_m = _accuracy(records, "pred_mvp")
    summary = {
        "n_examples": len(records),
        "baseline_argmax_yes_no": base_m,
        "mvp_v3_spatial_bbox": mvp_m,
        "acc_delta": round(mvp_m["accuracy"] - base_m["accuracy"], 4),
        "bias_distribution": dict({
            tgt: sum(1 for r in records if r["bias_target"] == tgt)
            for tgt in (None, "yes", "no")
        }),
        "by_type": {
            "baseline": _by(records, "pred_base", lambda r: r["type"]),
            "mvp_v3": _by(records, "pred_mvp", lambda r: r["type"]),
        },
        "by_gt": {
            "baseline": _by(records, "pred_base", lambda r: f"gt_{r['label']}"),
            "mvp_v3": _by(records, "pred_mvp", lambda r: f"gt_{r['label']}"),
        },
        "by_spatial": {
            "baseline": _by(records, "pred_base", lambda r: "spatial" if r["spatial_pred"] else "non_spatial"),
            "mvp_v3": _by(records, "pred_mvp", lambda r: "spatial" if r["spatial_pred"] else "non_spatial"),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2, default=str)
    log.info("Wrote %s", args.out)
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
