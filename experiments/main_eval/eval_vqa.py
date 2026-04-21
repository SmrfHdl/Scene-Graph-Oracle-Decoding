"""
Main Evaluation — VQAv2 val (Section 6.2, Benchmark 4).

General VQA capability — checks that SGOD doesn't degrade non-hallucination tasks.
  - VQAv2 val: ~214,354 questions, 40,504 COCO images
  - Metric: VQA accuracy (soft match over 10 human answers)
  - We subsample to --max_samples (default 5000) for speed

Data required:
  data/vqav2/v2_Questions_Val_mscoco.json
  data/vqav2/v2_Annotations_Val_mscoco.json
  data/coco/val2017/                (COCO val2017 images)

Usage:
    python experiments/main_eval/eval_vqa.py \\
        --config configs/default.yaml \\
        --vqav2_path data/vqav2/ \\
        --coco_path data/coco/val2017/ \\
        --max_samples 5000
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sgod.utils.output import make_run_dir, save_json

logger = logging.getLogger(__name__)


def vqa_accuracy(pred: str, gt_answers: list[str]) -> float:
    """VQAv2 soft accuracy: min(# humans said pred / 3, 1)."""
    pred = pred.strip().lower().rstrip(".")
    count = sum(1 for a in gt_answers if a.strip().lower() == pred)
    return min(count / 3, 1.0)


def run_llava(model, processor, image, question: str, max_new_tokens: int = 32) -> str:
    import torch
    prompt = f"USER: <image>\n{question}\nAnswer briefly.\nASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")

    p = argparse.ArgumentParser()
    p.add_argument("--config",         default="configs/default.yaml")
    p.add_argument("--vqav2_path",     default="data/vqav2/")
    p.add_argument("--coco_path",      default="data/coco/val2017/")
    p.add_argument("--output",         default=None)
    p.add_argument("--max_samples",    type=int, default=5000)
    p.add_argument("--seed",           type=int, default=42)
    p.add_argument("--load_in_4bit",   action="store_true")
    p.add_argument("--baseline_only",  action="store_true")
    p.add_argument("--max_new_tokens", type=int, default=32)
    args = p.parse_args()

    out_dir  = Path(args.output or str(make_run_dir("eval", "vqa")))
    vqa_path = Path(args.vqav2_path)
    coco_dir = Path(args.coco_path)

    q_file   = vqa_path / "v2_Questions_Val_mscoco.json"
    ann_file = vqa_path / "v2_Annotations_Val_mscoco.json"
    for f in (q_file, ann_file):
        if not f.exists():
            raise FileNotFoundError(f"{f} — run download_data.py --benchmarks vqav2")

    with open(q_file)   as f:
        questions_data = json.load(f)
    with open(ann_file) as f:
        annotations_data = json.load(f)

    ann_lookup: dict[int, list[str]] = {
        a["question_id"]: [ans["answer"] for ans in a["answers"]]
        for a in annotations_data["annotations"]
    }

    questions = questions_data["questions"]
    rng = random.Random(args.seed)
    rng.shuffle(questions)
    questions = questions[:args.max_samples]

    from PIL import Image
    from sgod.utils.model_loader import load_config, load_llava, load_sgod_decoder

    cfg              = load_config(args.config)
    model, processor = load_llava(load_in_4bit=args.load_in_4bit)
    decoder          = None if args.baseline_only else load_sgod_decoder(
        cfg, load_in_4bit=args.load_in_4bit
    )

    records       = []
    acc_base_sum  = 0.0
    acc_sgod_sum  = 0.0

    for i, item in enumerate(questions):
        qid        = item["question_id"]
        img_id     = item["image_id"]
        question   = item["question"]
        gt_answers = ann_lookup.get(qid, [])

        img_path = coco_dir / f"COCO_val2017_{img_id:012d}.jpg"
        if not img_path.exists():
            img_path = coco_dir / f"{img_id:012d}.jpg"
        if not img_path.exists():
            continue

        image     = Image.open(img_path).convert("RGB")
        pred_base = run_llava(model, processor, image, question, args.max_new_tokens)
        pred_sgod = decoder.generate(
            image, f"USER: <image>\n{question}\nAnswer briefly.\nASSISTANT:"
        ) if decoder else None

        acc_b = vqa_accuracy(pred_base, gt_answers)
        acc_s = vqa_accuracy(pred_sgod, gt_answers) if pred_sgod else None

        acc_base_sum += acc_b
        if acc_s is not None:
            acc_sgod_sum += acc_s

        records.append({
            "question_id":    qid,
            "image_id":       img_id,
            "question":       question,
            "gt_answers":     gt_answers[:3],
            "pred_base":      pred_base,
            "pred_sgod":      pred_sgod,
            "vqa_acc_base":   round(acc_b, 4),
            "vqa_acc_sgod":   round(acc_s, 4) if acc_s is not None else None,
        })

        if (i + 1) % 200 == 0:
            n = len(records)
            logger.info("[%d/%d] Base VQA=%.3f", i+1, len(questions), acc_base_sum / n)

    save_json(records, out_dir / "records.json")

    n        = len(records)
    vqa_base = acc_base_sum / n if n else 0.0
    result   = {
        "n":                       n,
        "baseline_vqa_accuracy":   round(vqa_base, 4),
    }
    if decoder:
        n_s      = sum(1 for r in records if r["vqa_acc_sgod"] is not None)
        vqa_sgod = acc_sgod_sum / n_s if n_s else 0.0
        result["sgod_vqa_accuracy"] = round(vqa_sgod, 4)
        result["delta_pp"]          = round((vqa_sgod - vqa_base) * 100, 2)

    save_json(result, out_dir / "vqa_results.json")

    print("\n" + "=" * 64)
    print("  VQAv2 RESULTS")
    print("=" * 64)
    print(f"  N = {n}  (subsampled from val)")
    print(f"  Baseline VQA Accuracy: {vqa_base*100:.2f}%")
    if "sgod_vqa_accuracy" in result:
        print(f"  SGOD VQA Accuracy:     {result['sgod_vqa_accuracy']*100:.2f}%  "
              f"(Δ={result['delta_pp']:+.2f}pp)")
        print("  Note: small negative delta is acceptable — SGOD trades VQA for hallucination")
    print(f"\n  Results saved: {out_dir}")


if __name__ == "__main__":
    main()
