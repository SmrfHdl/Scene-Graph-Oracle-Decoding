"""Phase-1 VR-TTS evaluation on Reefknot YESNO.

Runs the decoder at multiple K (number of exploration steps) and reports
accuracy + breakdowns. Used for both the n=100 Week-2 sanity check and the
n=1000 Week-3 go/no-go decision.

Outputs:
  --out <path>    summary JSON (always written)
  --records-out   optional per-example records (large)

Storage tip: for the n=1000 run, write records to a fresh path and DELETE
the records file after analysing the scaling curve — keep only the summary.

Usage:
    .venv/bin/python scripts/eval_vrtts_phase1.py \
        --n 100 --k-list 1,2,3 \
        --out outputs/vrtts/phase1/eval_n100.json

    .venv/bin/python scripts/eval_vrtts_phase1.py \
        --n 1000 --k-list 1,2,3,4,5 \
        --out outputs/vrtts/phase1/eval_n1000.json \
        --records-out outputs/vrtts/phase1/records_n1000.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image

from sgod.backbones.llava15 import LLaVAv15Backbone
from sgod.policies.vrtts import VRTTSDecoder, ConfidenceEstimator
from sgod.policies.vrtts.actions import (
    AddSoMMarks,
    AskSubQuestion,
    ZoomToAttentionRegion,
)
from sgod.policies.vrtts.aggregator import TraceAggregator
from sgod.policies.vrtts.policy import HeuristicExplorationPolicy

log = logging.getLogger("eval_vrtts_phase1")


# ── Data loading ────────────────────────────────────────────────────────────


def _load_examples(jsonl: Path, image_dir: Path, n: int, seed: int) -> list[dict]:
    raw: list[dict] = []
    with open(jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                raw.append(json.loads(line))
    rng = random.Random(seed)
    rng.shuffle(raw)
    out: list[dict] = []
    for r in raw:
        if len(out) >= n:
            break
        img_path = image_dir / f"{r['image_id']}.jpg"
        if not img_path.exists():
            continue
        out.append({
            "image_path": str(img_path),
            "image_id": r["image_id"],
            "question": r["query_prompt"],
            "label": r["label"].strip().lower(),
            "type": r.get("relation_type", "unknown"),
        })
    return out


# ── Minimal GD oracle wrapper (duck-typed for AddSoMMarks) ────────────────


class _GDOracleWrapper:
    """Thin GD wrapper exposing .processor / .model / .device for SoM action."""

    def __init__(self, model_id: str = "IDEA-Research/grounding-dino-base") -> None:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(
            self.device
        )
        self.model.eval()


# ── Metrics ────────────────────────────────────────────────────────────────


def _summary(records: list[dict], pred_key: str) -> dict:
    n = len(records)
    if n == 0:
        return {"n": 0, "accuracy": 0.0}
    correct = sum(1 for r in records if r[pred_key] == r["label"])
    yes_pred = sum(1 for r in records if r[pred_key] == "yes")
    return {
        "n": n,
        "accuracy": round(correct / n, 4),
        "yes_frac": round(yes_pred / n, 4),
    }


def _by_group(records: list[dict], pred_key: str, group_fn) -> dict:
    buckets = defaultdict(list)
    for r in records:
        buckets[group_fn(r)].append(r)
    return {g: _summary(rs, pred_key) for g, rs in buckets.items()}


# ── Bootstrap CI ───────────────────────────────────────────────────────────


def _bootstrap_ci(
    correct_flags: list[int],
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(correct_flags)
    if n == 0:
        return 0.0, 0.0
    samples: list[float] = []
    for _ in range(n_boot):
        boot = [correct_flags[rng.randrange(n)] for _ in range(n)]
        samples.append(sum(boot) / n)
    samples.sort()
    alpha = (1 - ci) / 2
    lo = samples[int(alpha * n_boot)]
    hi = samples[int((1 - alpha) * n_boot)]
    return lo, hi


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model-id", type=str, default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--gd-model-id", type=str,
                    default="IDEA-Research/grounding-dino-base")
    ap.add_argument("--reefknot-jsonl", type=Path,
                    default=Path("data/reefknot/YESNO.jsonl"))
    ap.add_argument("--image-dir", type=Path,
                    default=Path("data/reefknot/images"))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k-list", type=str, default="1,2,3",
                    help="Comma-separated K values to evaluate (K=1 is baseline-equivalent).")
    ap.add_argument("--aggregator", type=str, default="weighted_vote",
                    choices=["last", "weighted_vote"])
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/vrtts/phase1/eval.json"))
    ap.add_argument("--records-out", type=Path, default=None,
                    help="Optional per-record JSONL. Large for n=1000.")
    args = ap.parse_args()

    k_values = sorted(int(x) for x in args.k_list.split(","))
    log.info("K values: %s  n=%d  seed=%d", k_values, args.n, args.seed)

    examples = _load_examples(args.reefknot_jsonl, args.image_dir, args.n, args.seed)
    log.info("Loaded %d examples", len(examples))

    log.info("Loading LLaVA backbone (eager attention)...")
    backbone = LLaVAv15Backbone(
        model_id=args.model_id, dtype="float16", device_map="auto",
        lazy=False, attn_implementation="eager",
    )
    tokenizer = backbone.tokenizer()
    confidence = ConfidenceEstimator(tokenizer)

    log.info("Loading Grounding DINO for SoM action...")
    gd_oracle = _GDOracleWrapper(model_id=args.gd_model_id)

    # Action registry — Phase 1 fixed.
    actions = {
        "zoom": ZoomToAttentionRegion(),
        "som": AddSoMMarks(),
        "subq": AskSubQuestion(),
    }

    # Build one decoder per K (cheap — share backbone + oracle).
    decoders: dict[int, VRTTSDecoder] = {}
    for K in k_values:
        decoders[K] = VRTTSDecoder(
            backbone=backbone,
            actions=actions,
            policy=HeuristicExplorationPolicy(),
            confidence=confidence,
            aggregator=TraceAggregator(mode=args.aggregator),
            oracle=gd_oracle,
            max_steps=K,
            # Disable confidence gating in Phase 1 — measure pure scaling curve.
            # LLaVA's yes/no logit gap is empirically < 0.1, so threshold > 1
            # guarantees the gate never fires.
            confidence_threshold=10.0,
        )

    # Run all examples through all K. Per-record cost dominated by forward
    # passes; share image-load + parse across K.
    records: list[dict] = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    records_fh = None
    if args.records_out is not None:
        args.records_out.parent.mkdir(parents=True, exist_ok=True)
        records_fh = open(args.records_out, "w")

    for i, ex in enumerate(examples):
        img = Image.open(ex["image_path"]).convert("RGB")
        rec = {
            "image_id": ex["image_id"],
            "question": ex["question"],
            "label": ex["label"],
            "type": ex["type"],
        }
        for K in k_values:
            result = decoders[K].run(image=img, question=ex["question"])
            rec[f"pred_K{K}"] = result.answer
            rec[f"steps_K{K}"] = len(result.trace)
            rec[f"conf_K{K}_last"] = round(result.trace[-1].confidence, 4)
            rec[f"actions_K{K}"] = [s.action_applied for s in result.trace]
        records.append(rec)
        if records_fh is not None:
            records_fh.write(json.dumps(rec) + "\n")

        if (i + 1) % 20 == 0 or (i + 1) == len(examples):
            line_parts = [f"[{i+1}/{len(examples)}]"]
            for K in k_values:
                key = f"pred_K{K}"
                acc = sum(1 for r in records if r[key] == r["label"]) / len(records)
                line_parts.append(f"K{K}={acc:.3f}")
            log.info("  ".join(line_parts))

    if records_fh is not None:
        records_fh.close()

    # Summary + bootstrap CI per K.
    summary: dict = {
        "n_examples": len(records),
        "seed": args.seed,
        "k_list": k_values,
        "aggregator": args.aggregator,
        "per_K": {},
    }
    correct_K1 = [int(r["pred_K1"] == r["label"]) for r in records] if 1 in k_values else None
    for K in k_values:
        pkey = f"pred_K{K}"
        correct = [int(r[pkey] == r["label"]) for r in records]
        acc = sum(correct) / len(correct) if correct else 0.0
        lo, hi = _bootstrap_ci(correct)
        block = {
            "accuracy": round(acc, 4),
            "ci95_lo": round(lo, 4),
            "ci95_hi": round(hi, 4),
            "by_type": _by_group(records, pkey, lambda r: r["type"]),
            "by_gt": _by_group(records, pkey, lambda r: f"gt_{r['label']}"),
        }
        if correct_K1 is not None and K != 1:
            diffs = [correct[j] - correct_K1[j] for j in range(len(correct))]
            d_mean = sum(diffs) / len(diffs)
            d_lo, d_hi = _bootstrap_ci(diffs, n_boot=2000, seed=K)
            block["delta_vs_K1"] = {
                "mean": round(d_mean, 4),
                "ci95_lo": round(d_lo, 4),
                "ci95_hi": round(d_hi, 4),
            }
        summary["per_K"][f"K{K}"] = block

    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Wrote summary → %s", args.out)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
