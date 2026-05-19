"""Stage-0 sanity eval on POPE: baseline LLaVA vs DT-SGOD-Stage0.

Question to answer: did Stage-0 distillation produce a checkpoint that is
(a) no worse than baseline (no_harm), and (b) ideally a touch better.

We run both systems on the same POPE questions, decode "yes"/"no", and
compute Accuracy + F1. If DT-SGOD-Stage0 ≈ baseline, distillation was
benign — that's the floor we need before investing in Stage 1.

Usage:
    python scripts/eval_stage0_pope.py \\
        --student-config configs/dt_sgod_llava15.yaml \\
        --ckpt outputs/stage0/dtsgod_stage0.pt \\
        --pope-jsonl data/stage0/calib.jsonl \\
        --n 100 \\
        --out outputs/stage0/eval.json

Note: calib.jsonl drops POPE labels; pass --pope-dir + --coco-root to
re-resolve them from the original POPE files. Or supply --labeled-jsonl
with {image, question, label} rows.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image

from sgod.policies.dt_sgod import DTSGODPolicy
from sgod.runtime import HallucinationDecoder, build_from_config, load_config

log = logging.getLogger("eval_stage0_pope")

POPE_SPLITS = ("random", "popular", "adversarial")


def _load_labeled(pope_dir: Path, coco_root: Path, n: int, seed: int) -> list[dict]:
    """Load labeled POPE examples (image, question, label) with image paths resolved."""
    pool: list[dict] = []
    for split in POPE_SPLITS:
        path = pope_dir / f"coco_pope_{split}.json"
        if not path.exists():
            log.warning("missing %s", path)
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                pool.append({
                    "image_name": rec["image"],
                    "question": rec["text"],
                    "label": rec["label"].strip().lower(),
                    "split": split,
                })
    rng = random.Random(seed)
    rng.shuffle(pool)
    out: list[dict] = []
    for rec in pool:
        if len(out) >= n:
            break
        img_path = coco_root / rec["image_name"]
        if not img_path.exists():
            continue
        out.append({**rec, "image_path": str(img_path)})
    return out


def _extract_yes_no(text: str) -> str:
    t = text.strip().lower().lstrip("assistant:").strip()
    if t.startswith("yes"):
        return "yes"
    if t.startswith("no"):
        return "no"
    return "yes" if "yes" in t.split()[:5] else "no"


def _metrics(records: list[dict], pred_key: str) -> dict:
    tp = fp = tn = fn = 0
    for r in records:
        gt = r["label"]
        pred = r[pred_key]
        if pred == "yes" and gt == "yes":
            tp += 1
        elif pred == "yes" and gt == "no":
            fp += 1
        elif pred == "no" and gt == "no":
            tn += 1
        else:
            fn += 1
    n = tp + fp + tn + fn
    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "n": n, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "yes_frac": round((tp + fp) / n, 4) if n else 0.0,
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--student-config", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True,
                    help="Stage-0 checkpoint (.pt with speaker_adapter + anchor_gate)")
    ap.add_argument("--pope-dir", type=Path, default=Path("data/pope"))
    ap.add_argument("--coco-root", type=Path, default=Path("data/coco/val2014"))
    ap.add_argument("--n", type=int, default=100, help="Number of POPE examples")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("outputs/stage0/eval.json"))
    args = ap.parse_args()

    log.info("Loading %d labeled POPE examples from %s ...", args.n, args.pope_dir)
    examples = _load_labeled(args.pope_dir, args.coco_root, args.n, args.seed)
    if len(examples) < args.n:
        log.warning("only %d/%d examples have resolvable images", len(examples), args.n)
    log.info("Will eval on %d examples", len(examples))

    log.info("Building DT-SGOD student from %s", args.student_config)
    cfg = load_config(args.student_config)
    decoder: HallucinationDecoder = build_from_config(cfg, lazy=False)
    policy = decoder.policy
    assert isinstance(policy, DTSGODPolicy), "expected DTSGODPolicy"

    log.info("Loading Stage-0 checkpoint %s", args.ckpt)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    # Match adapter dtype/device to whatever policy was placed on (orchestrator
    # aligns on first forward; we mirror that here so load_state_dict succeeds).
    sd_speaker = {k: v.to(next(policy.speaker_adapter.parameters()).device) for k, v in ckpt["speaker_adapter"].items()}
    sd_atg = {k: v.to(next(policy.anchor_gate.parameters()).device) for k, v in ckpt["anchor_gate"].items()}
    policy.speaker_adapter.load_state_dict(sd_speaker)
    policy.anchor_gate.load_state_dict(sd_atg)
    log.info("Loaded; gate=%.4f", policy.speaker_adapter.gate.item())

    # Baseline: same backbone, but a no-op policy that returns Δ=0.
    # Cheapest way: temporarily zero out the gate to disable Δ entirely.
    baseline_gate_value = policy.speaker_adapter.gate.item()

    records = []
    for i, ex in enumerate(examples):
        img = Image.open(ex["image_path"]).convert("RGB")
        prompt = f"USER: <image>\n{ex['question']} Answer with yes or no. ASSISTANT:"

        # Baseline: gate = 0 → DT-SGOD ≡ backbone (G1 invariant).
        with torch.no_grad():
            policy.speaker_adapter.gate.data.fill_(0.0)
            ans_base = decoder.generate(image=img, prompt=prompt, question=ex["question"])
            policy.speaker_adapter.gate.data.fill_(baseline_gate_value)
            ans_stage0 = decoder.generate(image=img, prompt=prompt, question=ex["question"])

        records.append({
            **{k: ex[k] for k in ("image_name", "question", "label", "split")},
            "raw_base": ans_base,
            "raw_stage0": ans_stage0,
            "pred_base": _extract_yes_no(ans_base),
            "pred_stage0": _extract_yes_no(ans_stage0),
        })
        if (i + 1) % 20 == 0 or (i + 1) == len(examples):
            m_b = _metrics(records, "pred_base")
            m_s = _metrics(records, "pred_stage0")
            log.info("[%d/%d]  base F1=%.3f acc=%.3f  |  stage0 F1=%.3f acc=%.3f  (ΔF1=%+.3f)",
                     i + 1, len(examples), m_b["f1"], m_b["accuracy"],
                     m_s["f1"], m_s["accuracy"], m_s["f1"] - m_b["f1"])

    base_m = _metrics(records, "pred_base")
    stage0_m = _metrics(records, "pred_stage0")
    summary = {
        "n_examples": len(records),
        "baseline_llava": base_m,
        "dt_sgod_stage0": stage0_m,
        "f1_delta": round(stage0_m["f1"] - base_m["f1"], 4),
        "agree_rate": round(sum(1 for r in records if r["pred_base"] == r["pred_stage0"]) / len(records), 4),
        "gate_at_ckpt": baseline_gate_value,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2)
    log.info("Wrote %s", args.out)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
