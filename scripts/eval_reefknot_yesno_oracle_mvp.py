"""MVP: yes/no-aware oracle injection on Reefknot YESNO.

Hypothesis: LLaVA emits exactly 1 token ("Yes" or "No") at step 0 on yes/no
questions. SGOD-v1's anchor-based Δ never fires there because "Yes"/"No"
aren't noun/relation/attr anchors. To intervene at step 0 we need a
different mechanism: look up question entities in the scene graph and bias
the first-token logits directly.

This MVP keeps the rule minimal:
    if (X or Y) NOT in scene-graph objects   → strong bias toward "No"
    if (X and Y) both present                 → weak bias toward "Yes"
    else                                      → no bias (parse failed)

Relation predicates are NOT checked here — that's the next step if the
object-only signal is positive.

Usage:
    .venv/bin/python scripts/eval_reefknot_yesno_oracle_mvp.py \\
        --student-config configs/sgod_v1_llava15.yaml \\
        --reefknot-jsonl data/reefknot/YESNO.jsonl \\
        --image-dir data/reefknot/images \\
        --n 100 \\
        --bias-no 5.0 --bias-yes 1.0 \\
        --out outputs/stage0/eval_reefknot_yesno_mvp.json
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

log = logging.getLogger("yesno_mvp")


# ── Reefknot relation vocabulary (multi-word entries first for greedy match) ──
RELATIONS = sorted([
    # multi-word
    "in front of", "on top of", "next to", "in the middle of", "in this",
    "off the", "outside the", "inside the", "far from",
    "clashing with", "looking at", "watching with",
    "on right of", "on left of", "to the left of", "to the right of",
    # single-word
    "on", "in", "at", "by", "behind", "near", "above", "below",
    "under", "around", "across", "between", "beside", "off", "outside", "inside",
    "with", "to", "topping", "carrying", "holding", "riding", "wearing",
    "watching", "eating", "peeling", "driving", "following", "helping",
    "threatening", "unloading", "down", "from", "of",
], key=len, reverse=True)


def _parse_triplet(q: str) -> tuple[str, str, str] | None:
    """Parse 'Is/Are the X REL Y in this photo?' into (X, REL, Y)."""
    t = q
    for trailing in (" Please answer yes or no.", "Please answer yes or no.",
                     " in this photo?", "in this photo?", "?"):
        t = t.replace(trailing, "")
    t = t.strip()
    for prefix in ("Is the ", "Are the ", "Is a ", "Is ", "Are "):
        if t.startswith(prefix):
            t = t[len(prefix):]
            break

    # Find longest matching relation surrounded by spaces.
    for rel in RELATIONS:
        m = re.search(rf"(?:^|\s){re.escape(rel)}(?:\s|$)", t)
        if m:
            x = t[:m.start()].strip()
            y = t[m.end():].strip()
            if x and y:
                return x, rel, y
    return None


def _normalize_phrase(p: str) -> set[str]:
    """Return set of normalized tokens for object-label matching."""
    p = p.lower().strip()
    p = re.sub(r"\b(the|a|an|some|this|that|these|those)\b", "", p)
    p = re.sub(r"[^\w\s]", " ", p)
    return {w for w in p.split() if len(w) > 2}


def _present_in_scene(phrase: str, sg_object_labels: set[str]) -> bool:
    """Does the phrase share any normalized token with the scene-graph objects?"""
    toks = _normalize_phrase(phrase)
    if not toks:
        return False
    # Match if any token appears in any scene-graph object label.
    for tok in toks:
        for lbl in sg_object_labels:
            if tok in lbl or lbl in tok:
                return True
    return False


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


def _accuracy(records: list[dict], pred_key: str) -> dict:
    correct = sum(1 for r in records if r.get(pred_key) == r["label"])
    n = len(records)
    yes_pred = sum(1 for r in records if r.get(pred_key) == "yes")
    return {
        "n": n,
        "accuracy": round(correct / n, 4) if n else 0.0,
        "yes_frac": round(yes_pred / n, 4) if n else 0.0,
    }


def _accuracy_by_type(records, pred_key) -> dict:
    buckets = defaultdict(list)
    for r in records:
        buckets[r["type"]].append(r)
    return {t: _accuracy(rs, pred_key) for t, rs in buckets.items()}


def _accuracy_by_parse(records, pred_key, parsed_flag) -> dict:
    buckets = {"parsed": [], "unparsed": []}
    for r in records:
        buckets["parsed" if r.get(parsed_flag) else "unparsed"].append(r)
    return {k: _accuracy(rs, pred_key) for k, rs in buckets.items() if rs}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--student-config", type=Path, required=True,
                    help="Use the SGOD v1 config (same backbone + oracle as before)")
    ap.add_argument("--reefknot-jsonl", type=Path, default=Path("data/reefknot/YESNO.jsonl"))
    ap.add_argument("--image-dir", type=Path, default=Path("data/reefknot/images"))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bias-no", type=float, default=5.0,
                    help="Logit boost on 'No' when X or Y is missing from scene graph")
    ap.add_argument("--bias-yes", type=float, default=1.0,
                    help="Logit boost on 'Yes' when X and Y are both present")
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/stage0/eval_reefknot_yesno_mvp.json"))
    args = ap.parse_args()

    log.info("Loading examples...")
    examples = _load_reefknot(args.reefknot_jsonl, args.image_dir, args.n, args.seed)
    log.info("Eval on %d examples", len(examples))

    log.info("Building decoder (lazy=False loads LLaVA + RelTR + GD)...")
    cfg = load_config(args.student_config)
    decoder = build_from_config(cfg, lazy=False)
    backbone = decoder.backbone
    oracle = decoder.oracle

    # We need raw first-token logits — bypass the orchestrator and call the
    # backbone directly. backbone.prepare_inputs + backbone.forward_step gives
    # us what we want without sampling.
    tokenizer = backbone.tokenizer()
    # Locate "Yes" and "No" token IDs. LLaMA tokenizer typically prefixes with ▁ space.
    yes_ids = [tokenizer.convert_tokens_to_ids(t) for t in ("Yes", "▁Yes", "yes", "▁yes")]
    no_ids = [tokenizer.convert_tokens_to_ids(t) for t in ("No", "▁No", "no", "▁no")]
    yes_ids = [i for i in yes_ids if i is not None and i != tokenizer.unk_token_id]
    no_ids = [i for i in no_ids if i is not None and i != tokenizer.unk_token_id]
    log.info("Yes token IDs: %s ; No token IDs: %s", yes_ids, no_ids)

    records = []
    parse_fail = 0
    sg_obj_missing = 0
    for i, ex in enumerate(examples):
        img = Image.open(ex["image_path"]).convert("RGB")
        prompt = f"USER: <image>\n{ex['question']} ASSISTANT:"

        # 1. Scene graph (one Oracle.extract per example).
        evidence = oracle.extract(image=img)
        sg_labels = {o.label.lower() for o in evidence.scene_graph.objects}

        # 2. Parse question.
        triplet = _parse_triplet(ex["question"])
        parsed = triplet is not None
        if parsed:
            x, rel, y = triplet
            x_present = _present_in_scene(x, sg_labels)
            y_present = _present_in_scene(y, sg_labels)
        else:
            parse_fail += 1
            x_present = y_present = True  # no info → no bias

        # 3. Forward LLaVA, get step-0 logits (orchestrator's empty-tensor pattern).
        inputs = backbone.prepare_inputs(image=img, prompt=prompt)
        with torch.no_grad():
            generated_ids = torch.empty(1, 0, dtype=torch.long)
            _hidden, logits = backbone.forward_step(inputs, generated_ids)
        logits = logits[0].float().cpu()  # [V]

        # Compare yes vs no margin BEFORE bias.
        yes_score_base = max(logits[i].item() for i in yes_ids) if yes_ids else float("-inf")
        no_score_base = max(logits[i].item() for i in no_ids) if no_ids else float("-inf")
        margin_base = yes_score_base - no_score_base
        pred_base = "yes" if margin_base > 0 else "no"

        # 4. Apply bias rule.
        bias_applied = 0.0
        bias_target = None
        if parsed and not (x_present and y_present):
            # X or Y missing → push "No"
            for tid in no_ids:
                logits[tid] += args.bias_no
            bias_applied = args.bias_no
            bias_target = "no"
            sg_obj_missing += 1
        elif parsed and x_present and y_present:
            # Both present → mild push "Yes"
            for tid in yes_ids:
                logits[tid] += args.bias_yes
            bias_applied = args.bias_yes
            bias_target = "yes"

        yes_score_mvp = max(logits[i].item() for i in yes_ids) if yes_ids else float("-inf")
        no_score_mvp = max(logits[i].item() for i in no_ids) if no_ids else float("-inf")
        pred_mvp = "yes" if yes_score_mvp > no_score_mvp else "no"

        records.append({
            "image_name": ex["image_name"],
            "question": ex["question"],
            "label": ex["label"],
            "type": ex["type"],
            "parsed": parsed,
            "triplet": triplet,
            "sg_objects_n": len(sg_labels),
            "sg_objects_sample": sorted(sg_labels)[:8],
            "x_present": x_present if parsed else None,
            "y_present": y_present if parsed else None,
            "margin_base": round(margin_base, 3),
            "bias_target": bias_target,
            "bias_applied": bias_applied,
            "pred_base": pred_base,
            "pred_mvp": pred_mvp,
        })

        if (i + 1) % 20 == 0 or (i + 1) == len(examples):
            m_b = _accuracy(records, "pred_base")
            m_m = _accuracy(records, "pred_mvp")
            log.info(
                "[%d/%d]  base acc=%.3f  |  mvp acc=%.3f (Δ=%+.3f)  parse_fail=%d  bias_no_fires=%d",
                i + 1, len(examples), m_b["accuracy"], m_m["accuracy"],
                m_m["accuracy"] - m_b["accuracy"], parse_fail, sg_obj_missing,
            )

    base_m = _accuracy(records, "pred_base")
    mvp_m = _accuracy(records, "pred_mvp")
    summary = {
        "n_examples": len(records),
        "baseline_argmax_yes_no": base_m,
        "mvp_with_object_bias": mvp_m,
        "acc_delta": round(mvp_m["accuracy"] - base_m["accuracy"], 4),
        "parse_fail_count": parse_fail,
        "bias_no_fires": sum(1 for r in records if r["bias_target"] == "no"),
        "bias_yes_fires": sum(1 for r in records if r["bias_target"] == "yes"),
        "by_type": {
            "baseline": _accuracy_by_type(records, "pred_base"),
            "mvp": _accuracy_by_type(records, "pred_mvp"),
        },
        "by_parse": {
            "baseline": _accuracy_by_parse(records, "pred_base", "parsed"),
            "mvp": _accuracy_by_parse(records, "pred_mvp", "parsed"),
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2)
    log.info("Wrote %s", args.out)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
