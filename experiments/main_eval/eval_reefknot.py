"""
Main Evaluation — Reefknot benchmark (Section 6.2, Benchmark 2).

Relation hallucination evaluation — MAIN PAPER CLAIM.
  - Perceptive relations (on, under, holding, ...)
  - Cognitive relations (helping, threatening, ...)
  - >20,000 samples from Visual Genome
  - Metric: Accuracy (yes/no answer format)

Reefknot format (test.json / val.json):
  [{"image": "VG_...", "question": "...", "answer": "yes"/"no", "type": "perceptive"/"cognitive"}]

Data required:
  data/reefknot/test.json
  data/reefknot/images/           (Visual Genome images)

Usage:
    python experiments/main_eval/eval_reefknot.py \\
        --config configs/default.yaml \\
        --reefknot_path data/reefknot/
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


def compute_metrics(records: list[dict], pred_key: str) -> dict:
    correct = sum(
        r[pred_key].strip().lower().rstrip(".") == r["answer"].strip().lower()
        for r in records
        if r.get(pred_key)
    )
    n = sum(1 for r in records if r.get(pred_key))
    return {"n": n, "accuracy": round(correct / n, 4) if n else 0.0}


def run_llava(model, processor, image, question: str, max_new_tokens: int = 8) -> str:
    import torch
    prompt = f"USER: <image>\n{question}\nAnswer yes or no.\nASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    text = processor.tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip().lower()
    return "yes" if text.startswith("yes") else "no"


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")

    p = argparse.ArgumentParser()
    p.add_argument("--config",          default="configs/default.yaml")
    p.add_argument("--reefknot_path",   default="data/reefknot/")
    p.add_argument("--split",           default="test", choices=["test", "val"])
    p.add_argument("--output",          default=None)
    p.add_argument("--load_in_4bit",    action="store_true")
    p.add_argument("--baseline_only",   action="store_true")
    p.add_argument("--max_samples",     type=int, default=None,
                   help="Limit for quick debugging")
    args = p.parse_args()

    out_dir      = Path(args.output or str(make_run_dir("eval", "reefknot")))
    rk_path      = Path(args.reefknot_path)
    q_file       = rk_path / f"{args.split}.json"
    img_dir      = rk_path / "images"

    if not q_file.exists():
        raise FileNotFoundError(f"{q_file} — run download_data.py --benchmarks reefknot")

    from PIL import Image
    from sgod.utils.model_loader import load_config, load_llava, load_sgod_decoder

    cfg              = load_config(args.config)
    model, processor = load_llava(load_in_4bit=args.load_in_4bit)
    decoder          = None if args.baseline_only else load_sgod_decoder(
        cfg, load_in_4bit=args.load_in_4bit
    )

    with open(q_file) as f:
        items = json.load(f)
    if args.max_samples:
        items = items[:args.max_samples]

    records = []
    for i, item in enumerate(items):
        img_path = img_dir / item["image"]
        if not img_path.exists():
            logger.warning("[%d] missing: %s", i, img_path)
            continue

        image    = Image.open(img_path).convert("RGB")
        question = item["question"]
        answer   = item["answer"]
        rel_type = item.get("type", "unknown")

        pred_base = run_llava(model, processor, image, question)
        pred_sgod = decoder.generate(
            image,
            f"USER: <image>\n{question}\nAnswer yes or no.\nASSISTANT:"
        ) if decoder else None
        if pred_sgod:
            pred_sgod = "yes" if "yes" in pred_sgod.lower() else "no"

        records.append({
            "image":      item["image"],
            "question":   question,
            "answer":     answer,
            "type":       rel_type,
            "pred_base":  pred_base,
            "pred_sgod":  pred_sgod,
        })

        if (i + 1) % 200 == 0:
            m = compute_metrics(records, "pred_base")
            logger.info("[%d/%d] Base Acc=%.3f", i+1, len(items), m["accuracy"])

    save_json(records, out_dir / "records.json")

    metrics_base = compute_metrics(records, "pred_base")
    result: dict = {"split": args.split, "baseline": metrics_base}

    # Per-type breakdown
    by_type: dict[str, list] = defaultdict(list)
    for r in records:
        by_type[r["type"]].append(r)
    result["by_type"] = {
        t: {
            "baseline": compute_metrics(recs, "pred_base"),
            **({"sgod": compute_metrics(recs, "pred_sgod")} if decoder else {}),
        }
        for t, recs in by_type.items()
    }

    if decoder:
        metrics_sgod = compute_metrics(records, "pred_sgod")
        result["sgod"]    = metrics_sgod
        result["delta_pp"] = round(
            (metrics_sgod["accuracy"] - metrics_base["accuracy"]) * 100, 2
        )

    save_json(result, out_dir / "reefknot_results.json")

    print("\n" + "=" * 64)
    print("  REEFKNOT RESULTS (main claim)")
    print("=" * 64)
    print(f"  Split: {args.split}  (n={metrics_base['n']})")
    print(f"  Baseline Accuracy:  {metrics_base['accuracy']*100:.2f}%")
    if decoder:
        g = result["sgod"]
        print(f"  SGOD Accuracy:      {g['accuracy']*100:.2f}%  "
              f"(Δ={result['delta_pp']:+.2f} pp)")
    print()
    for t, tmetrics in result["by_type"].items():
        b = tmetrics["baseline"]
        line = f"  {t:<20} n={b['n']:<5} Base={b['accuracy']*100:.1f}%"
        if "sgod" in tmetrics:
            s = tmetrics["sgod"]
            delta = (s["accuracy"] - b["accuracy"]) * 100
            line += f"  SGOD={s['accuracy']*100:.1f}%  Δ={delta:+.1f}pp"
        print(line)
    print(f"\n  Results saved: {out_dir}")


if __name__ == "__main__":
    main()
