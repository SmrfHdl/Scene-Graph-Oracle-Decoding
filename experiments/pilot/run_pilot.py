"""
Pilot Experiment (Section 6.1).

Validates that GT scene graph in prompt improves relation hallucination
before committing to full SGOD implementation (2-3 day investment).

3 variants on N GQA images with relation-type questions:
    a. LLaVA baseline (no SG)
    b. LLaVA + GT scene graph as text prefix          ← gate check
    c. LLaVA + RelTR-predicted scene graph prefix     (requires --reltr_ckpt)

Success criterion: variant (b) accuracy > baseline (a) by ≥ 3-5 pp.
If this fails → reconsider approach before building main eval.

GQA data required:
    data/gqa/val_balanced_questions.json   (from questions1.2.zip)
    data/gqa/val_sceneGraphs.json          (from sceneGraphs.zip)
    data/gqa/images/                       (for variant c only)

Usage:
    python experiments/pilot/run_pilot.py \\
        --num_images 200 \\
        --gqa_path data/gqa/ \\
        --output outputs/pilot/run_01/

    # Skip variant C (no RelTR checkpoint):
    python experiments/pilot/run_pilot.py --no_reltr

    # Resume a partial run:
    python experiments/pilot/run_pilot.py --resume outputs/pilot/run_01/results.json

    # 4-bit quantization for single T4 (16 GB):
    python experiments/pilot/run_pilot.py --load_in_4bit
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch

from sgod.sgg.scene_graph import ObjectNode, RelationEdge, SceneGraph
from sgod.utils.output import make_run_dir, save_json

logger = logging.getLogger(__name__)

# ── GQA helpers ───────────────────────────────────────────────────────────────

def load_relation_questions(
    gqa_path: Path,
    n: int,
    seed: int = 42,
) -> list[dict]:
    """Load GQA val questions filtered to semantic type 'relation'.

    Selects one question per unique image (to avoid bias), then samples n images.
    Returns list of dicts with keys: question_id, imageId, question, answer, detailed_type.
    """
    q_file = gqa_path / "val_balanced_questions.json"
    if not q_file.exists():
        raise FileNotFoundError(
            f"{q_file} not found. Run: python scripts/download_data.py --benchmarks gqa"
        )

    logger.info("Loading GQA questions from %s ...", q_file)
    with open(q_file) as f:
        all_q = json.load(f)

    # Filter to relation semantic type ("rel" per GQA 1.2 schema)
    relation_qs = {
        qid: q for qid, q in all_q.items()
        if q.get("types", {}).get("semantic") == "rel"
    }
    logger.info("Relation questions: %d / %d total", len(relation_qs), len(all_q))

    # One question per image
    seen_images: dict[str, dict] = {}
    for qid, q in relation_qs.items():
        img_id = q["imageId"]
        if img_id not in seen_images:
            seen_images[img_id] = {"question_id": qid, **q}

    logger.info("Unique images with relation questions: %d", len(seen_images))

    # Sample n images reproducibly
    rng = random.Random(seed)
    selected = rng.sample(list(seen_images.values()), min(n, len(seen_images)))
    logger.info("Selected %d images for pilot", len(selected))
    return selected


def load_gqa_scene_graphs(gqa_path: Path) -> dict:
    """Load GQA val scene graphs. Returns {image_id: sg_dict}."""
    sg_file = gqa_path / "val_sceneGraphs.json"
    if not sg_file.exists():
        raise FileNotFoundError(f"{sg_file} not found. Run download_data.py --benchmarks gqa")
    logger.info("Loading GQA scene graphs ...")
    with open(sg_file) as f:
        return json.load(f)


def gqa_sg_to_scene_graph(gqa_sg: dict) -> SceneGraph:
    """Convert a GQA scene graph dict for one image → SceneGraph.

    GQA objects have: name, x, y, w, h, attributes[], relations[{name, object}]
    """
    objects_raw = gqa_sg.get("objects", {})

    # Build id→name lookup
    id_to_name = {oid: obj["name"] for oid, obj in objects_raw.items()}

    objects = [
        ObjectNode(
            label=obj["name"],
            confidence=1.0,  # GT: confidence = 1
            bbox=(
                obj.get("x", 0), obj.get("y", 0),
                obj.get("x", 0) + obj.get("w", 1),
                obj.get("y", 0) + obj.get("h", 1),
            ),
        )
        for obj in objects_raw.values()
    ]

    relations = []
    for oid, obj in objects_raw.items():
        subj_name = obj["name"]
        for rel in obj.get("relations", []):
            obj_name = id_to_name.get(rel["object"], rel["object"])
            relations.append(RelationEdge(
                subject=subj_name,
                predicate=rel["name"],
                object=obj_name,
                confidence=1.0,
            ))

    return SceneGraph(objects=objects, relations=relations, attributes=[])


def sg_to_prompt_prefix(sg: SceneGraph) -> str:
    """Compact scene graph text for VLM prompt prefix."""
    return sg.to_prompt_prefix()


# ── LLaVA inference ───────────────────────────────────────────────────────────

def build_llava_prompt(question: str, sg_prefix: str | None = None) -> str:
    """Build the LLaVA-1.5 prompt string.

    LLaVA-1.5 expects: "USER: <image>\\n{text}\\nASSISTANT:"
    """
    text = question
    if sg_prefix:
        text = f"Scene: {sg_prefix}\n{question}"
    return f"USER: <image>\n{text}\nASSISTANT:"


def run_llava(
    model,
    processor,
    image,
    question: str,
    sg_prefix: str | None = None,
    max_new_tokens: int = 32,
) -> str:
    """Run a single LLaVA inference and return the decoded answer."""
    prompt = build_llava_prompt(question, sg_prefix)
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,         # greedy — matches SGODDecoder default
            temperature=1.0,         # ignored when do_sample=False
        )

    # Decode only generated tokens (skip prompt)
    prompt_len = inputs["input_ids"].shape[1]
    gen_ids    = output_ids[0][prompt_len:]
    return processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


# ── Evaluation ────────────────────────────────────────────────────────────────

def eval_exact(pred: str, gt: str) -> bool:
    """GQA-style exact match (lowercased, stripped)."""
    return pred.lower().strip() == gt.lower().strip()


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_pilot(args: argparse.Namespace) -> None:
    from PIL import Image
    from sgod.utils.model_loader import load_llava, load_sgg

    gqa_path = Path(args.gqa_path)
    out_dir  = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.json"

    # Load or resume
    if args.resume and results_path.exists():
        logger.info("Resuming from %s", results_path)
        with open(results_path) as f:
            results = json.load(f)
        done_ids = {r["question_id"] for r in results}
    else:
        results = []
        done_ids = set()

    questions    = load_relation_questions(gqa_path, args.num_images, seed=args.seed)
    gqa_sgs      = load_gqa_scene_graphs(gqa_path)

    # Load LLaVA
    logger.info("Loading LLaVA-1.5-7B ...")
    model, processor = load_llava(
        model_id=args.vlm_model,
        load_in_4bit=args.load_in_4bit,
    )

    # Load RelTR for variant C
    sgg_module = None
    if not args.no_reltr:
        if not Path(args.reltr_ckpt).exists():
            logger.warning(
                "RelTR checkpoint not found at %s. Skipping variant C. "
                "Pass --no_reltr to suppress this warning.", args.reltr_ckpt
            )
        else:
            logger.info("Loading RelTR for variant C ...")
            sgg_module = load_sgg(args.reltr_ckpt)

    # Smoke-test mode: use a single surrogate image for all questions.
    # Accuracy numbers are meaningless but verifies the full inference loop.
    smoke_image = None
    if args.smoke_test_image:
        smoke_path = Path(args.smoke_test_image)
        if not smoke_path.exists():
            raise FileNotFoundError(f"--smoke_test_image not found: {smoke_path}")
        smoke_image = Image.open(smoke_path).convert("RGB")
        logger.info("Smoke-test mode: using %s for all questions", smoke_path)

    img_dir = gqa_path / "images"
    n_total = len(questions)

    for i, item in enumerate(questions):
        qid      = item["question_id"]
        img_id   = item["imageId"]
        question = item["question"]
        answer   = item["answer"]

        if qid in done_ids:
            continue

        if smoke_image is not None:
            image = smoke_image
        else:
            img_path = img_dir / f"{img_id}.jpg"
            if not img_path.exists():
                logger.warning("[%d/%d] Image not found: %s — skipping", i+1, n_total, img_path)
                continue
            image = Image.open(img_path).convert("RGB")

        # Variant A: baseline
        t0      = time.perf_counter()
        pred_a  = run_llava(model, processor, image, question,
                            max_new_tokens=args.max_new_tokens)
        time_a  = time.perf_counter() - t0

        # Variant B: GT scene graph prefix
        gt_sg_dict  = gqa_sgs.get(img_id, {})
        gt_sg       = gqa_sg_to_scene_graph(gt_sg_dict)
        gt_sg_text  = sg_to_prompt_prefix(gt_sg)
        t0          = time.perf_counter()
        pred_b      = run_llava(model, processor, image, question, sg_prefix=gt_sg_text,
                                max_new_tokens=args.max_new_tokens)
        time_b      = time.perf_counter() - t0

        # Variant C: predicted scene graph (optional)
        pred_c  = None
        time_c  = None
        if sgg_module is not None:
            pred_sg     = sgg_module.extract(image)
            pred_sg_text = sg_to_prompt_prefix(pred_sg)
            t0          = time.perf_counter()
            pred_c      = run_llava(model, processor, image, question, sg_prefix=pred_sg_text,
                                    max_new_tokens=args.max_new_tokens)
            time_c      = time.perf_counter() - t0

        record = {
            "question_id": qid,
            "imageId":     img_id,
            "question":    question,
            "gt_answer":   answer,
            "pred_a":      pred_a,
            "pred_b":      pred_b,
            "pred_c":      pred_c,
            "correct_a":   eval_exact(pred_a, answer),
            "correct_b":   eval_exact(pred_b, answer),
            "correct_c":   eval_exact(pred_c, answer) if pred_c is not None else None,
            "detailed_type": item.get("types", {}).get("detailed"),
            "time_a_s":    round(time_a, 3),
            "time_b_s":    round(time_b, 3),
            "time_c_s":    round(time_c, 3) if time_c else None,
            "gt_sg_n_objects":   len(gt_sg.objects),
            "gt_sg_n_relations": len(gt_sg.relations),
        }
        results.append(record)

        # Running accuracy
        n_done = len(results)
        acc_a  = sum(r["correct_a"] for r in results) / n_done
        acc_b  = sum(r["correct_b"] for r in results) / n_done
        acc_c  = (sum(r["correct_c"] for r in results if r["correct_c"] is not None) /
                  max(1, sum(1 for r in results if r["correct_c"] is not None)))
        delta  = acc_b - acc_a
        logger.info(
            "[%d/%d] %s | A=%.1f%% B=%.1f%% (Δ=%+.1f%%)%s",
            n_done, n_total, qid,
            acc_a * 100, acc_b * 100, delta * 100,
            f" C={acc_c*100:.1f}%" if sgg_module else "",
        )

        # Checkpoint after every item
        save_json(results, results_path)

    # Final summary
    n  = len(results)
    if n == 0:
        print("No results collected (are images present in data/gqa/images/?)")
        return

    acc_a = sum(r["correct_a"] for r in results) / n
    acc_b = sum(r["correct_b"] for r in results) / n
    delta = acc_b - acc_a
    threshold_met = delta >= (args.threshold / 100)

    summary = {
        "n":           n,
        "acc_a":       round(acc_a, 4),
        "acc_b":       round(acc_b, 4),
        "delta_pp":    round(delta * 100, 2),
        "threshold_pp": args.threshold,
        "gate_passed": threshold_met,
    }
    if sgg_module:
        n_c     = sum(1 for r in results if r["correct_c"] is not None)
        acc_c   = sum(r["correct_c"] for r in results if r["correct_c"] is not None) / max(1, n_c)
        summary["acc_c"] = round(acc_c, 4)

    save_json(summary, out_dir / "summary.json")
    save_json({"config": vars(args)}, out_dir / "config.json")

    print("\n" + "="*60)
    print("  PILOT EXPERIMENT RESULTS")
    print("="*60)
    print(f"  N questions:    {n}")
    print(f"  Accuracy A (baseline):  {acc_a*100:.1f}%")
    print(f"  Accuracy B (GT SG):     {acc_b*100:.1f}%")
    print(f"  Delta:                  {delta*100:+.1f} pp")
    if sgg_module:
        print(f"  Accuracy C (RelTR SG):  {acc_c*100:.1f}%")
    print(f"\n  Gate threshold:  +{args.threshold:.0f} pp")
    print(f"  Gate PASSED:     {'YES ✓' if threshold_met else 'NO ✗ — reassess direction'}")
    print(f"\n  Results saved to: {out_dir}")
    print("="*60)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SGOD Pilot Experiment (Section 6.1)")
    p.add_argument("--gqa_path",    default="data/gqa/",
                   help="Path to GQA data directory")
    p.add_argument("--num_images",  type=int, default=200,
                   help="Number of images to evaluate")
    p.add_argument("--output",      default=None,
                   help="Output directory (auto-created if omitted)")
    p.add_argument("--vlm_model",   default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--reltr_ckpt",  default="data/checkpoints/reltr/checkpoint0149.pth")
    p.add_argument("--no_reltr",    action="store_true",
                   help="Skip variant C (no RelTR checkpoint)")
    p.add_argument("--load_in_4bit", action="store_true",
                   help="Load LLaVA in 4-bit (for single T4 16 GB)")
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--threshold",   type=float, default=3.0,
                   help="Success threshold in percentage points (default: 3.0)")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--resume",      action="store_true",
                   help="Resume from existing results.json in --output dir")
    p.add_argument("--smoke_test_image", default=None, metavar="PATH",
                   help="Use this image for all questions (smoke test — accuracy meaningless)")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()

    if args.output is None:
        args.output = str(make_run_dir("pilot"))

    run_pilot(args)


if __name__ == "__main__":
    main()
