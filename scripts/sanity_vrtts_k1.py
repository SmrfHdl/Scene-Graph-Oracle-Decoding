"""Sanity check: VR-TTS with K=1 (no action applied) must match baseline.

When max_steps=1, the decoder runs one forward, computes (label, conf) from
yes/no logits, and stops — no action ever applied. The aggregator returns
that single label. This should be IDENTICAL to the baseline-argmax pattern
used by existing scripts (eval_reefknot_yesno_oracle_mvp_v4.py).

Mismatch ⇒ a bug in the K=1 path of the orchestrator. We verify on a small
n so failure is obvious before launching n=1000.

Usage:
    .venv/bin/python scripts/sanity_vrtts_k1.py --n 20
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

from sgod.backbones.llava15 import LLaVAv15Backbone
from sgod.policies.vrtts import VRTTSDecoder, ConfidenceEstimator
from sgod.policies.vrtts.actions.zoom import ZoomToAttentionRegion

log = logging.getLogger("sanity_k1")


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
            "question": r["query_prompt"],
            "label": r["label"].strip().lower(),
        })
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", type=str, default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--reefknot-jsonl", type=Path,
                    default=Path("data/reefknot/YESNO.jsonl"))
    ap.add_argument("--image-dir", type=Path,
                    default=Path("data/reefknot/images"))
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    examples = _load_examples(args.reefknot_jsonl, args.image_dir, args.n, args.seed)
    log.info("Sanity n=%d examples", len(examples))

    log.info("Loading LLaVA backbone with attn_implementation='eager'...")
    backbone = LLaVAv15Backbone(
        model_id=args.model_id,
        dtype="float16",
        device_map="auto",
        lazy=False,
        attn_implementation="eager",
    )
    tokenizer = backbone.tokenizer()

    yes_ids = [tokenizer.convert_tokens_to_ids(t) for t in ("Yes", "▁Yes", "yes", "▁yes")]
    no_ids = [tokenizer.convert_tokens_to_ids(t) for t in ("No", "▁No", "no", "▁no")]
    unk = tokenizer.unk_token_id
    yes_ids = [i for i in yes_ids if i is not None and i != unk]
    no_ids = [i for i in no_ids if i is not None and i != unk]

    # K=1 VR-TTS: only the zoom action registered (will never fire — max_steps=1).
    vrtts = VRTTSDecoder(
        backbone=backbone,
        actions={"zoom": ZoomToAttentionRegion()},
        confidence=ConfidenceEstimator(tokenizer),
        max_steps=1,
        confidence_threshold=2.0,  # impossibly high → never gates early
    )

    mismatches = 0
    for i, ex in enumerate(examples):
        img = Image.open(ex["image_path"]).convert("RGB")

        # Baseline path (matches eval_reefknot_yesno_oracle_mvp_v4.py).
        prompt = f"USER: <image>\n{ex['question']} ASSISTANT:"
        inputs = backbone.prepare_inputs(image=img, prompt=prompt)
        with torch.no_grad():
            gen_ids = torch.empty(1, 0, dtype=torch.long)
            _, logits_base = backbone.forward_step(inputs, gen_ids)
        logits_base = logits_base[0].float().cpu()
        yes_b = max(logits_base[i_].item() for i_ in yes_ids)
        no_b = max(logits_base[i_].item() for i_ in no_ids)
        pred_base = "yes" if yes_b > no_b else "no"

        # VR-TTS K=1.
        result = vrtts.run(image=img, question=ex["question"])
        pred_vrtts = result.answer

        match = pred_base == pred_vrtts
        if not match:
            mismatches += 1
        log.info(
            "[%d] base=%s  vrtts_K1=%s  match=%s  | conf=%.3f  steps=%d  | gt=%s",
            i + 1, pred_base, pred_vrtts, match,
            result.trace[-1].confidence, len(result.trace), ex["label"],
        )

    log.info("Summary: %d/%d mismatches", mismatches, len(examples))
    if mismatches > 0:
        log.error("SANITY FAIL — K=1 VR-TTS diverges from baseline. Investigate.")
        return 1
    log.info("SANITY PASS — K=1 VR-TTS matches baseline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
