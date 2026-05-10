"""
POPE pipeline-grounding spike — DIRECTION test for BG-SGOD before tuning injection.

Hypothesis to test
------------------
For each POPE question "Is there a {obj} in the image?":
    BG-SGOD pipeline predicts "yes" iff parse_targets(question) resolves to an
    object label that appears in the hybrid scene graph (RelTR ∪ Grounding DINO).

If pipeline grounding correlates with GT label, the structural BG-SGOD signal
is correct and we should invest in fixing the *injection* mechanism (logit
arithmetic) so LLaVA's argmax can flip.

If correlation is weak/absent, the *detection pipeline itself* is the problem
— no amount of injection tuning will rescue it.

This script bypasses LLaVA entirely. It only loads:
  - RelTR (for relations + base SG)
  - Grounding DINO (for open-vocab objects)
  - the question parser

Usage (server):
    PYTHONPATH=$(pwd) python scripts/spike_pope_grounding.py \
        --pope_path data/pope/ \
        --coco_path data/coco/val2014/ \
        --n 200 \
        --output outputs/spike_pope_grounding.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from PIL import Image

from sgod.oracle.question_parser import (
    match_targets_to_vocab,
    parse_targets,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


# ── POPE loader ──────────────────────────────────────────────────────────────

def load_pope(path: Path, splits: list[str], n_per_split: int) -> list[dict]:
    """Load up to ``n_per_split`` POPE items per split, mixed together."""
    out: list[dict] = []
    for split in splits:
        f = path / f"coco_pope_{split}.json"
        if not f.exists():
            logger.warning("missing %s — skipping", f)
            continue
        with open(f) as fh:
            items = [json.loads(l) for l in fh if l.strip()]
        for item in items[:n_per_split]:
            item["_split"] = split
            out.append(item)
    if not out:
        raise FileNotFoundError(f"no POPE data found under {path}")
    return out


# ── Pipeline loaders ─────────────────────────────────────────────────────────

def load_pipeline(reltr_ckpt: Path, gd_model_id: str, device: str):
    """Load RelTR + GD wrapped in SGGModule (hybrid mode)."""
    from sgod.sgg import SGGModule
    from sgod.sgg.grounding_dino_module import GroundingDinoModule

    logger.info("Loading Grounding DINO %s on %s ...", gd_model_id, device)
    gd = GroundingDinoModule(model_id=gd_model_id, device=device)
    logger.info("Loading RelTR from %s on %s ...", reltr_ckpt, device)
    sgg = SGGModule(str(reltr_ckpt), device=device, gd_module=gd)
    return sgg


# ── Per-question pipeline ────────────────────────────────────────────────────

def predict(sgg, image: Image.Image, question: str) -> dict:
    """Return ``{has_match, n_objects, n_targets, matched, targets, all_labels}``.

    has_match=True means: at least one parsed question-target lands in the
    scene graph's noun vocabulary (i.e. BG-SGOD would set has_bbox_target=True
    for this question + image pair).
    """
    sg = sgg.extract(image)
    targets = parse_targets(question)
    noun_vocab = {o.label.lower() for o in sg.objects}
    matched = match_targets_to_vocab(targets, noun_vocab)
    return {
        "has_match": bool(matched),
        "n_objects": len(sg.objects),
        "n_targets": len(targets),
        "targets": targets,
        "matched": matched,
        "all_labels": sorted(noun_vocab)[:30],  # cap so output stays small
    }


# ── Confusion matrix + summary ───────────────────────────────────────────────

def summarise(records: list[dict]) -> dict:
    """Compute confusion matrix of (pipeline says match) vs (GT label is yes)."""
    tp = fp = tn = fn = 0
    by_split: dict[str, dict[str, int]] = {}
    for r in records:
        gt_yes = r["label"].strip().lower() == "yes"
        pred_yes = r["has_match"]
        split = r.get("_split", "?")
        d = by_split.setdefault(split, {"tp":0,"fp":0,"tn":0,"fn":0})
        if pred_yes and gt_yes:    tp += 1; d["tp"] += 1
        elif pred_yes:             fp += 1; d["fp"] += 1
        elif not gt_yes:           tn += 1; d["tn"] += 1
        else:                      fn += 1; d["fn"] += 1

    def metrics(t, f, tn_, fn_):
        n = t + f + tn_ + fn_
        if n == 0: return {}
        acc = (t + tn_) / n
        prec = t / (t + f) if (t + f) else 0.0
        rec  = t / (t + fn_) if (t + fn_) else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        return {
            "n": n, "tp": t, "fp": f, "tn": tn_, "fn": fn_,
            "accuracy":  round(acc,  4),
            "precision": round(prec, 4),
            "recall":    round(rec,  4),
            "f1":        round(f1,   4),
            # Conditional probabilities the *pipeline-direction* hypothesis is about:
            "P(match|yes)": round(t / (t + fn_), 4) if (t + fn_) else 0.0,
            "P(match|no)":  round(f / (f + tn_), 4) if (f + tn_) else 0.0,
        }

    overall = metrics(tp, fp, tn, fn)
    by_split_metrics = {
        s: metrics(d["tp"], d["fp"], d["tn"], d["fn"]) for s, d in by_split.items()
    }
    return {"overall": overall, "by_split": by_split_metrics}


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pope_path", type=Path, default=Path("data/pope"))
    p.add_argument("--coco_path", type=Path, default=Path("data/coco/val2014"))
    p.add_argument("--reltr_ckpt", type=Path,
                   default=Path("data/checkpoints/reltr/checkpoint0149.pth"))
    p.add_argument("--gd_model", type=str, default="IDEA-Research/grounding-dino-base")
    p.add_argument("--splits", default="adversarial,popular,random")
    p.add_argument("--n", type=int, default=200,
                   help="Max questions per split.")
    p.add_argument("--output", type=Path,
                   default=Path("outputs/spike_pope_grounding.json"))
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    splits = [s.strip() for s in args.splits.split(",")]
    items = load_pope(args.pope_path, splits, args.n)
    logger.info("Loaded %d POPE items across splits %s", len(items), splits)

    sgg = load_pipeline(args.reltr_ckpt, args.gd_model, args.device)

    # SG extraction is ~per-image, but POPE has multiple questions per image.
    # Cache by image filename so we run extract() once per image.
    sg_cache: dict[str, dict] = {}
    records = []
    for i, item in enumerate(items):
        img_name = item["image"]
        question = item["text"]
        label = item["label"]

        if img_name not in sg_cache:
            img_path = args.coco_path / img_name
            if not img_path.exists():
                logger.warning("[%d/%d] missing image: %s", i+1, len(items), img_path)
                continue
            image = Image.open(img_path).convert("RGB")
            # Pre-extract once and store the noun_vocab + raw object list.
            sg = sgg.extract(image)
            sg_cache[img_name] = {
                "noun_vocab": {o.label.lower() for o in sg.objects},
                "n_objects": len(sg.objects),
                "labels": sorted({o.label.lower() for o in sg.objects})[:30],
            }
        cached = sg_cache[img_name]

        targets = parse_targets(question)
        matched = match_targets_to_vocab(targets, cached["noun_vocab"])

        records.append({
            "_split":   item["_split"],
            "image":    img_name,
            "question": question,
            "label":    label,
            "has_match": bool(matched),
            "n_objects": cached["n_objects"],
            "n_targets": len(targets),
            "targets":  targets,
            "matched":  matched,
        })

        if (i + 1) % 50 == 0:
            logger.info("processed %d/%d", i+1, len(items))

    summary = summarise(records)
    out = {
        "n": len(records),
        "n_images": len(sg_cache),
        "summary": summary,
        "records": records[:200],  # cap stored records for compactness
    }
    args.output.write_text(json.dumps(out, indent=2))

    print("\n=== OVERALL ===")
    print(json.dumps(summary["overall"], indent=2))
    print("\n=== BY SPLIT ===")
    print(json.dumps(summary["by_split"], indent=2))
    print(f"\nFull report: {args.output}")


if __name__ == "__main__":
    main()
