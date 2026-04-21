"""
Main Evaluation — POPE benchmark (Section 6.2, Benchmark 1).

Object hallucination evaluation:
  - adversarial / popular / random splits
  - 9,000 questions, 500 COCO images per split
  - Metric: Accuracy, F1, Precision, Recall

Two modes:
  1. --sgod_only      : run only SGOD (default), compare with baseline internally
  2. --baseline_only  : run only LLaVA baseline (for fast separate timing)

Data required:
  data/pope/coco_pope_adversarial.json
  data/pope/coco_pope_popular.json
  data/pope/coco_pope_random.json
  data/coco/val2014/                (COCO images)

Usage:
    python experiments/main_eval/eval_pope.py \\
        --config configs/default.yaml \\
        --pope_path data/pope/ \\
        --coco_path data/coco/val2014/
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sgod.utils.output import make_run_dir, save_json

logger = logging.getLogger(__name__)

SPLITS = ["adversarial", "popular", "random"]


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_pope_metrics(records: list[dict], pred_key: str) -> dict:
    """Compute Accuracy, F1, Precision, Recall for yes/no POPE questions."""
    tp = fp = tn = fn = 0
    for r in records:
        pred = r[pred_key].strip().lower().rstrip(".")
        gt   = r["label"].strip().lower()
        pred_yes = pred.startswith("yes")
        gt_yes   = gt == "yes"
        if pred_yes and gt_yes:    tp += 1
        elif pred_yes and not gt_yes: fp += 1
        elif not pred_yes and not gt_yes: tn += 1
        else: fn += 1

    n          = tp + fp + tn + fn
    accuracy   = (tp + tn) / n if n else 0.0
    precision  = tp / (tp + fp) if (tp + fp) else 0.0
    recall     = tp / (tp + fn) if (tp + fn) else 0.0
    f1         = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    yes_frac   = (tp + fp) / n if n else 0.0

    return {
        "n": n, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "yes_fraction": round(yes_frac, 4),
    }


def _extract_yes_no(text: str) -> str:
    """Normalise free-text prediction to yes/no."""
    t = text.strip().lower()
    if t.startswith("yes"):
        return "yes"
    if t.startswith("no"):
        return "no"
    # Fallback: look for first occurrence
    if "yes" in t:
        return "yes"
    return "no"


# ── LLaVA helpers ─────────────────────────────────────────────────────────────

def run_llava(model, processor, image, question: str, max_new_tokens: int = 16) -> str:
    import torch
    prompt = f"USER: <image>\n{question}\nAnswer with yes or no.\nASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()


def run_sgod(decoder, image, question: str) -> str:
    return decoder.generate(image, f"USER: <image>\n{question}\nAnswer with yes or no.\nASSISTANT:")


# ── Eval loop ─────────────────────────────────────────────────────────────────

def eval_split(
    split: str,
    pope_path: Path,
    coco_path: Path,
    model,
    processor,
    decoder,
    max_new_tokens: int,
    out_dir: Path,
) -> dict:
    from PIL import Image

    q_file = pope_path / f"coco_pope_{split}.json"
    if not q_file.exists():
        raise FileNotFoundError(f"{q_file} — run download_data.py --benchmarks pope")

    with open(q_file) as f:
        items = json.load(f)

    records = []
    for i, item in enumerate(items):
        img_name  = item["image"]
        question  = item["text"]
        label     = item["label"]
        img_path  = coco_path / img_name

        if not img_path.exists():
            logger.warning("[%d/%d] missing: %s", i+1, len(items), img_path)
            continue

        image = Image.open(img_path).convert("RGB")

        pred_base = run_llava(model, processor, image, question, max_new_tokens)
        pred_sgod = run_sgod(decoder, image, question) if decoder is not None else None

        records.append({
            "image":      img_name,
            "question":   question,
            "label":      label,
            "pred_base":  _extract_yes_no(pred_base),
            "pred_sgod":  _extract_yes_no(pred_sgod) if pred_sgod else None,
            "raw_base":   pred_base,
            "raw_sgod":   pred_sgod,
        })

        if (i + 1) % 100 == 0:
            metrics_b = compute_pope_metrics(records, "pred_base")
            logger.info("[%s] %d/%d  Base F1=%.3f Acc=%.3f",
                        split, i+1, len(items), metrics_b["f1"], metrics_b["accuracy"])

    save_json(records, out_dir / f"records_{split}.json")
    metrics_base = compute_pope_metrics(records, "pred_base")
    result = {"split": split, "n": len(records), "baseline": metrics_base}
    if decoder is not None:
        result["sgod"] = compute_pope_metrics(records, "pred_sgod")
        result["f1_delta"] = round(result["sgod"]["f1"] - metrics_base["f1"], 4)
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")

    p = argparse.ArgumentParser()
    p.add_argument("--config",     default="configs/default.yaml")
    p.add_argument("--pope_path",  default="data/pope/")
    p.add_argument("--coco_path",  default="data/coco/val2014/")
    p.add_argument("--output",     default=None)
    p.add_argument("--splits",     default="adversarial,popular,random")
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--baseline_only", action="store_true",
                   help="Skip SGOD, run baseline only")
    p.add_argument("--max_new_tokens", type=int, default=16)
    args = p.parse_args()

    out_dir  = Path(args.output or str(make_run_dir("eval", "pope")))
    splits   = [s.strip() for s in args.splits.split(",")]

    from sgod.utils.model_loader import load_config, load_llava, load_sgod_decoder

    cfg      = load_config(args.config)
    model, processor = load_llava(load_in_4bit=args.load_in_4bit)
    decoder  = None if args.baseline_only else load_sgod_decoder(
        cfg, load_in_4bit=args.load_in_4bit
    )

    all_results = []
    for split in splits:
        logger.info("Evaluating split: %s", split)
        res = eval_split(
            split,
            pope_path=Path(args.pope_path),
            coco_path=Path(args.coco_path),
            model=model,
            processor=processor,
            decoder=decoder,
            max_new_tokens=args.max_new_tokens,
            out_dir=out_dir,
        )
        all_results.append(res)
        logger.info("Split %s done: %s", split, res)

    save_json(all_results, out_dir / "pope_results.json")

    print("\n" + "=" * 64)
    print("  POPE RESULTS")
    print("=" * 64)
    for res in all_results:
        s = res["split"]
        b = res["baseline"]
        print(f"\n  Split: {s}  (n={res['n']})")
        print(f"    Baseline  — Acc={b['accuracy']*100:.1f}%  F1={b['f1']*100:.1f}%  "
              f"P={b['precision']*100:.1f}%  R={b['recall']*100:.1f}%")
        if "sgod" in res:
            g = res["sgod"]
            print(f"    SGOD      — Acc={g['accuracy']*100:.1f}%  F1={g['f1']*100:.1f}%  "
                  f"P={g['precision']*100:.1f}%  R={g['recall']*100:.1f}%  "
                  f"ΔF1={res['f1_delta']*100:+.1f}pp")
    print(f"\n  Results saved: {out_dir}")


if __name__ == "__main__":
    main()
