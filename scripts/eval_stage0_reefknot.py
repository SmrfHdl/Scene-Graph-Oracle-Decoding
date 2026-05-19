"""Stage-0 eval on Reefknot: 3-way baseline / SGOD-v1 / DT-SGOD-Stage0.

Reefknot = relation hallucination benchmark (perceptive + cognitive). This is
the headline benchmark where our RelTR-backed oracle should *actually* help
— POPE saturated at LLaVA-7B F1=0.92, so we pivot to relations.

Reefknot YESNO.jsonl format (one record per line):
    {"image_id": "<VG id>",
     "query_prompt": "Is ... in this photo? Please answer yes or no.",
     "label": "yes|no",
     "relation_type": "perception|cognitive"}

Image filename: data/reefknot/images/<image_id>.jpg (Visual Genome).

Setup (one-time):
    # 1. YESNO.jsonl from GitHub (~2 MB):
    mkdir -p data/reefknot
    curl -L https://raw.githubusercontent.com/JackChen-seu/Reefknot/main/Dataset/YESNO.jsonl \\
        -o data/reefknot/YESNO.jsonl

    # 2. Visual Genome images (~15 GB total, both parts):
    mkdir -p data/reefknot/images
    cd data/reefknot/images
    wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
    wget https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip
    unzip -q images.zip && unzip -q images2.zip
    mv VG_100K/* VG_100K_2/* . && rmdir VG_100K VG_100K_2
    rm images.zip images2.zip
    cd -

Usage:
    .venv/bin/python scripts/eval_stage0_reefknot.py \\
        --student-config configs/dt_sgod_llava15.yaml \\
        --ckpt outputs/stage0/dtsgod_stage0.pt \\
        --teacher-config configs/sgod_v1_llava15.yaml \\
        --reefknot-jsonl data/reefknot/YESNO.jsonl \\
        --image-dir data/reefknot/images \\
        --n 100 --out outputs/stage0/eval_reefknot.json
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

from sgod.policies.dt_sgod import DTSGODPolicy
from sgod.runtime import HallucinationDecoder, build_from_config, load_config

log = logging.getLogger("eval_stage0_reefknot")


def _load_reefknot(rk_jsonl: Path, image_dir: Path, n: int, seed: int) -> list[dict]:
    """Load Reefknot YESNO examples, resolve image paths, sample n."""
    raw: list[dict] = []
    with open(rk_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw.append(json.loads(line))
    rng = random.Random(seed)
    rng.shuffle(raw)
    out: list[dict] = []
    missing = 0
    for r in raw:
        if len(out) >= n:
            break
        img_name = f"{r['image_id']}.jpg"
        img_path = image_dir / img_name
        if not img_path.exists():
            missing += 1
            continue
        out.append({
            "image_name": img_name,
            "question": r["query_prompt"],
            "label": r["label"].strip().lower(),
            "type": r.get("relation_type", "unknown"),
            "image_path": str(img_path),
        })
    if missing:
        log.warning("%d examples skipped (image missing under %s)", missing, image_dir)
    return out


def _extract_yes_no(text: str) -> str:
    t = text.strip().lower().lstrip("assistant:").strip()
    if t.startswith("yes"):
        return "yes"
    if t.startswith("no"):
        return "no"
    return "yes" if "yes" in t.split()[:5] else "no"


def _accuracy(records: list[dict], pred_key: str) -> dict:
    correct = sum(1 for r in records if r.get(pred_key) == r["label"])
    n = len(records)
    yes_pred = sum(1 for r in records if r.get(pred_key) == "yes")
    return {
        "n": n,
        "accuracy": round(correct / n, 4) if n else 0.0,
        "yes_frac": round(yes_pred / n, 4) if n else 0.0,
    }


def _accuracy_by_type(records: list[dict], pred_key: str) -> dict:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        buckets[r["type"]].append(r)
    return {t: _accuracy(rs, pred_key) for t, rs in buckets.items()}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--student-config", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--teacher-config", type=Path, default=None)
    ap.add_argument("--reefknot-jsonl", type=Path, default=Path("data/reefknot/YESNO.jsonl"))
    ap.add_argument("--image-dir", type=Path, default=Path("data/reefknot/images"))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("outputs/stage0/eval_reefknot.json"))
    args = ap.parse_args()

    log.info("Loading %d Reefknot examples from %s ...", args.n, args.reefknot_jsonl)
    examples = _load_reefknot(args.reefknot_jsonl, args.image_dir, args.n, args.seed)
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
    sd_speaker = {k: v.to(next(policy.speaker_adapter.parameters()).device) for k, v in ckpt["speaker_adapter"].items()}
    sd_atg = {k: v.to(next(policy.anchor_gate.parameters()).device) for k, v in ckpt["anchor_gate"].items()}
    policy.speaker_adapter.load_state_dict(sd_speaker)
    policy.anchor_gate.load_state_dict(sd_atg)
    baseline_gate_value = policy.speaker_adapter.gate.item()
    log.info("Loaded; gate=%.4f", baseline_gate_value)

    teacher_decoder = None
    if args.teacher_config is not None:
        log.info("Building SGOD-v1 teacher from %s", args.teacher_config)
        teacher_cfg = load_config(args.teacher_config)
        teacher_decoder = build_from_config(teacher_cfg, lazy=False)

    records = []
    for i, ex in enumerate(examples):
        img = Image.open(ex["image_path"]).convert("RGB")
        # Reefknot's query_prompt already contains "Please answer yes or no."
        prompt = f"USER: <image>\n{ex['question']} ASSISTANT:"

        with torch.no_grad():
            policy.speaker_adapter.gate.data.fill_(0.0)
            ans_base = decoder.generate(image=img, prompt=prompt, question=ex["question"])
            policy.speaker_adapter.gate.data.fill_(baseline_gate_value)
            ans_stage0 = decoder.generate(image=img, prompt=prompt, question=ex["question"])
            ans_teacher = None
            if teacher_decoder is not None:
                ans_teacher = teacher_decoder.generate(image=img, prompt=prompt, question=ex["question"])

        rec = {
            **{k: ex[k] for k in ("image_name", "question", "label", "type")},
            "raw_base": ans_base,
            "raw_stage0": ans_stage0,
            "pred_base": _extract_yes_no(ans_base),
            "pred_stage0": _extract_yes_no(ans_stage0),
        }
        if ans_teacher is not None:
            rec["raw_teacher"] = ans_teacher
            rec["pred_teacher"] = _extract_yes_no(ans_teacher)
        records.append(rec)
        if (i + 1) % 20 == 0 or (i + 1) == len(examples):
            m_b = _accuracy(records, "pred_base")
            m_s = _accuracy(records, "pred_stage0")
            line = (f"[{i+1}/{len(examples)}]  base acc={m_b['accuracy']:.3f}  |  "
                    f"stage0 acc={m_s['accuracy']:.3f} (Δ={m_s['accuracy']-m_b['accuracy']:+.3f})")
            if teacher_decoder is not None:
                m_t = _accuracy(records, "pred_teacher")
                line += f"  |  teacher acc={m_t['accuracy']:.3f} (Δ={m_t['accuracy']-m_b['accuracy']:+.3f})"
            log.info(line)

    base_m = _accuracy(records, "pred_base")
    stage0_m = _accuracy(records, "pred_stage0")
    summary = {
        "n_examples": len(records),
        "baseline_llava": base_m,
        "dt_sgod_stage0": stage0_m,
        "acc_delta_stage0": round(stage0_m["accuracy"] - base_m["accuracy"], 4),
        "agree_rate_stage0": round(
            sum(1 for r in records if r["pred_base"] == r["pred_stage0"]) / len(records), 4
        ),
        "gate_at_ckpt": baseline_gate_value,
        "by_type": {
            "baseline_llava": _accuracy_by_type(records, "pred_base"),
            "dt_sgod_stage0": _accuracy_by_type(records, "pred_stage0"),
        },
    }
    if teacher_decoder is not None:
        teacher_m = _accuracy(records, "pred_teacher")
        summary["sgod_v1_teacher"] = teacher_m
        summary["acc_delta_teacher"] = round(teacher_m["accuracy"] - base_m["accuracy"], 4)
        summary["agree_rate_teacher"] = round(
            sum(1 for r in records if r["pred_base"] == r["pred_teacher"]) / len(records), 4
        )
        summary["by_type"]["sgod_v1_teacher"] = _accuracy_by_type(records, "pred_teacher")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2)
    log.info("Wrote %s", args.out)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
