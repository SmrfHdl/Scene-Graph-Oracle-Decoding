"""
Main Evaluation — AMBER benchmark (Section 6.2, Benchmark 3).

Multi-dimensional hallucination evaluation:
  - Object existence (yes/no questions about object presence)
  - Attribute hallucination (color, size, material)
  - Relation hallucination (spatial, action)
  - ~1,000 images from A-OKVQA
  - Metrics: Accuracy per dimension, CHAIR-S, CHAIR-I, Coverage

AMBER format (query.json):
  {"id": 1, "image": "amber_1.jpg", "query": "...", "type": "existence"/"attribute"/"relation"}
answer.json:
  {"1": "yes"/"no"/string}

Data required:
  data/amber/query.json
  data/amber/answer.json
  data/amber/image/   (AMBER images from HuggingFace)

Usage:
    python experiments/main_eval/eval_amber.py \\
        --config configs/default.yaml \\
        --amber_path data/amber/
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sgod.utils.output import make_run_dir, save_json

logger = logging.getLogger(__name__)


def eval_answer(pred: str, gt: str, qtype: str) -> bool:
    """Flexible answer matching for AMBER's mixed answer formats."""
    pred = pred.strip().lower().rstrip(".")
    gt   = gt.strip().lower()
    # Yes/no questions
    if gt in ("yes", "no"):
        return pred.startswith(gt)
    # Open-ended: exact match or gt contained in pred
    return gt in pred or pred == gt


def run_llava(model, processor, image, question: str, max_new_tokens: int = 32) -> str:
    import torch
    prompt = f"USER: <image>\n{question}\nASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()


def compute_chair(records: list[dict], pred_key: str, answers: dict) -> dict:
    """Approximate CHAIR-S and CHAIR-I using GT answer as object reference.

    CHAIR-S: fraction of sentences (answers) containing a hallucinated object.
    CHAIR-I: fraction of object mentions that are hallucinated.
    This is a simplified CHAIR; exact CHAIR requires full caption + object list.
    """
    n_sent = 0
    n_hal_sent = 0
    n_obj_total = 0
    n_obj_hal = 0
    for r in records:
        pred = r.get(pred_key, "") or ""
        gt   = answers.get(str(r["id"]), "")
        pred_words = set(pred.lower().split())
        gt_words   = set(gt.lower().split())
        hal_words  = pred_words - gt_words - {"a", "the", "an", "is", "are", "in", "on", "at"}
        n_sent += 1
        n_obj_total += len(pred_words)
        if hal_words:
            n_hal_sent += 1
            n_obj_hal += len(hal_words)
    return {
        "chair_s": round(n_hal_sent / n_sent, 4) if n_sent else 0.0,
        "chair_i": round(n_obj_hal / n_obj_total, 4) if n_obj_total else 0.0,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")

    p = argparse.ArgumentParser()
    p.add_argument("--config",       default="configs/default.yaml")
    p.add_argument("--amber_path",   default="data/amber/")
    p.add_argument("--output",       default=None)
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--baseline_only", action="store_true")
    args = p.parse_args()

    out_dir    = Path(args.output or str(make_run_dir("eval", "amber")))
    amber_path = Path(args.amber_path)

    for fname in ("query.json", "answer.json"):
        if not (amber_path / fname).exists():
            raise FileNotFoundError(
                f"{amber_path / fname} — run download_data.py --benchmarks amber"
            )

    with open(amber_path / "query.json")  as f:
        queries = json.load(f)
    with open(amber_path / "answer.json") as f:
        answers = json.load(f)

    # AMBER image dir can be amber/image/ or amber/images/
    img_dir = amber_path / "image"
    if not img_dir.exists():
        img_dir = amber_path / "images"

    from PIL import Image
    from sgod.utils.model_loader import load_config, load_llava, load_sgod_decoder

    cfg              = load_config(args.config)
    model, processor = load_llava(load_in_4bit=args.load_in_4bit)
    decoder          = None if args.baseline_only else load_sgod_decoder(
        cfg, load_in_4bit=args.load_in_4bit
    )

    records = []
    for i, item in enumerate(queries):
        img_path = img_dir / item["image"]
        if not img_path.exists():
            logger.warning("[%d] missing: %s", i, img_path)
            continue

        image    = Image.open(img_path).convert("RGB")
        question = item["query"]
        gt       = answers.get(str(item["id"]), "")
        qtype    = item.get("type", "unknown")

        pred_base = run_llava(model, processor, image, question)
        pred_sgod = decoder.generate(
            image, f"USER: <image>\n{question}\nASSISTANT:"
        ) if decoder else None

        records.append({
            "id":         item["id"],
            "image":      item["image"],
            "question":   question,
            "gt":         gt,
            "type":       qtype,
            "pred_base":  pred_base,
            "pred_sgod":  pred_sgod,
            "correct_base": eval_answer(pred_base, gt, qtype),
            "correct_sgod": eval_answer(pred_sgod, gt, qtype) if pred_sgod else None,
        })

        if (i + 1) % 100 == 0:
            acc = sum(r["correct_base"] for r in records) / len(records)
            logger.info("[%d/%d] Base Acc=%.3f", i+1, len(queries), acc)

    save_json(records, out_dir / "records.json")

    # Overall
    n        = len(records)
    acc_base = sum(r["correct_base"] for r in records) / n if n else 0.0
    result   = {
        "n": n,
        "baseline": {
            "accuracy": round(acc_base, 4),
            **compute_chair(records, "pred_base", answers),
        },
    }
    if decoder:
        n_s      = sum(1 for r in records if r["correct_sgod"] is not None)
        acc_sgod = sum(r["correct_sgod"] for r in records if r["correct_sgod"] is not None) / n_s
        result["sgod"] = {
            "accuracy":  round(acc_sgod, 4),
            **compute_chair(records, "pred_sgod", answers),
        }
        result["delta_pp"] = round((acc_sgod - acc_base) * 100, 2)

    # Per-type
    by_type: dict[str, list] = defaultdict(list)
    for r in records:
        by_type[r["type"]].append(r)
    result["by_type"] = {}
    for t, recs in by_type.items():
        nt = len(recs)
        result["by_type"][t] = {
            "n": nt,
            "baseline_acc": round(sum(r["correct_base"] for r in recs) / nt, 4),
        }
        if decoder:
            ns = sum(1 for r in recs if r["correct_sgod"] is not None)
            result["by_type"][t]["sgod_acc"] = round(
                sum(r["correct_sgod"] for r in recs if r["correct_sgod"] is not None) / max(ns,1), 4
            )

    save_json(result, out_dir / "amber_results.json")

    print("\n" + "=" * 64)
    print("  AMBER RESULTS")
    print("=" * 64)
    print(f"  N = {n}")
    b = result["baseline"]
    print(f"  Baseline — Acc={b['accuracy']*100:.1f}%  "
          f"CHAIR-S={b['chair_s']*100:.1f}%  CHAIR-I={b['chair_i']*100:.1f}%")
    if "sgod" in result:
        g = result["sgod"]
        print(f"  SGOD     — Acc={g['accuracy']*100:.1f}%  "
              f"CHAIR-S={g['chair_s']*100:.1f}%  CHAIR-I={g['chair_i']*100:.1f}%  "
              f"Δ={result['delta_pp']:+.2f}pp")
    print()
    for t, tm in result["by_type"].items():
        line = f"  {t:<20} n={tm['n']:<4} Base={tm['baseline_acc']*100:.1f}%"
        if "sgod_acc" in tm:
            line += f"  SGOD={tm['sgod_acc']*100:.1f}%"
        print(line)
    print(f"\n  Results saved: {out_dir}")


if __name__ == "__main__":
    main()
