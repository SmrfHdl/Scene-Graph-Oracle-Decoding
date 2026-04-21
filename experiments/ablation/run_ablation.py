"""
Ablation Study (Section 6.3).

Tests 6 ablation configurations on Reefknot val to isolate which
SGOD components drive the hallucination reduction:

  1. no_clip        — SGG only, no CLIP soft fallback
  2. fixed_lambda   — Fixed λ=0.35, no adaptive context (GenerationContext disabled)
  3. no_anchor      — Oracle applied to ALL tokens, not just anchor positions
  4. noun_only      — Only noun_anchor positions scored (no relation/attr)
  5. gt_sg          — Upper bound: oracle built from GT scene graph
  6. internvl       — Generalization: same SGOD on InternVL2-8B backbone

Run all or specific ablations:
  python experiments/ablation/run_ablation.py --ablation all
  python experiments/ablation/run_ablation.py --ablation no_clip,fixed_lambda

Data required:
  data/reefknot/val.json + data/reefknot/images/
  data/gqa/val_sceneGraphs.json   (for ablation 5: gt_sg)
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sgod.utils.output import make_run_dir, save_json

logger = logging.getLogger(__name__)

ALL_ABLATIONS = ["no_clip", "fixed_lambda", "no_anchor", "noun_only", "gt_sg", "internvl"]


# ── Ablation config builders ──────────────────────────────────────────────────

def build_ablation_config(base_config: dict, ablation: str) -> dict:
    """Return a modified config dict for the given ablation."""
    cfg = copy.deepcopy(base_config)
    if ablation == "no_clip":
        cfg["oracle"]["dual_source"] = False
    elif ablation == "fixed_lambda":
        # Will be handled at decoder construction time
        cfg["_ablation_fixed_lambda"] = 0.35
    elif ablation == "no_anchor":
        cfg["_ablation_no_anchor"] = True
    elif ablation == "noun_only":
        cfg["_ablation_noun_only"] = True
    elif ablation == "gt_sg":
        cfg["_ablation_gt_sg"] = True
    elif ablation == "internvl":
        cfg["decoder"]["vlm_model"] = "OpenGVLab/InternVL2-8B"
    return cfg


# ── Ablated decoder builders ──────────────────────────────────────────────────

def _build_fixed_lambda_decoder(cfg: dict, lam: float, load_in_4bit: bool):
    """Decoder with fixed lambda, bypassing GenerationContext adaptation."""
    from sgod.utils.model_loader import load_clip_factory, load_llava, load_sgg
    from sgod.decoder import SGODDecoder
    from sgod.context import BASE_LAMBDA

    fixed_bl = {k: lam for k in BASE_LAMBDA}
    model, processor = load_llava(load_in_4bit=load_in_4bit)
    sgg = load_sgg(cfg["sgg"]["checkpoint"])
    factory = load_clip_factory(cfg["oracle"]["clip_model"],
                                vocab_cache_path=cfg["oracle"].get("clip_vocab_cache"))
    return SGODDecoder(
        vlm_model=model,
        tokenizer=processor.tokenizer,
        sgg_module=sgg,
        clip_factory=factory,
        base_lambda=fixed_bl,
        top_k=cfg["oracle"]["top_k"],
        min_sg_confidence=cfg["sgg"]["confidence_threshold"],
        max_new_tokens=cfg["decoder"]["max_new_tokens"],
    ), processor


def _build_no_anchor_decoder(cfg: dict, load_in_4bit: bool):
    """Decoder that applies oracle to ALL tokens (patches _score_candidates)."""
    import torch
    from sgod.utils.model_loader import load_llava, load_sgg, load_clip_factory
    from sgod.decoder import SGODDecoder
    from sgod.oracle import VisualOracle

    model, processor = load_llava(load_in_4bit=load_in_4bit)
    sgg = load_sgg(cfg["sgg"]["checkpoint"])
    factory = load_clip_factory(cfg["oracle"]["clip_model"],
                                vocab_cache_path=cfg["oracle"].get("clip_vocab_cache"))

    decoder = SGODDecoder(
        vlm_model=model, tokenizer=processor.tokenizer,
        sgg_module=sgg, clip_factory=factory,
        top_k=cfg["oracle"]["top_k"],
        min_sg_confidence=cfg["sgg"]["confidence_threshold"],
        max_new_tokens=cfg["decoder"]["max_new_tokens"],
    )

    # Monkeypatch: score ALL candidates as noun_anchor regardless of position
    @staticmethod
    def score_all(words, prev_tokens, oracle):
        scores = torch.zeros(len(words))
        for i, word in enumerate(words):
            scores[i] = oracle.score(word, "noun_anchor")
        return scores

    decoder._score_candidates = score_all
    return decoder, processor


def _build_noun_only_decoder(cfg: dict, load_in_4bit: bool):
    """Decoder that only scores noun_anchor positions (ignores relation/attr)."""
    import torch
    from sgod.utils.model_loader import load_llava, load_sgg, load_clip_factory
    from sgod.decoder import SGODDecoder
    from sgod.anchor import detect_anchor

    model, processor = load_llava(load_in_4bit=load_in_4bit)
    sgg = load_sgg(cfg["sgg"]["checkpoint"])
    factory = load_clip_factory(cfg["oracle"]["clip_model"],
                                vocab_cache_path=cfg["oracle"].get("clip_vocab_cache"))

    decoder = SGODDecoder(
        vlm_model=model, tokenizer=processor.tokenizer,
        sgg_module=sgg, clip_factory=factory,
        top_k=cfg["oracle"]["top_k"],
        min_sg_confidence=cfg["sgg"]["confidence_threshold"],
        max_new_tokens=cfg["decoder"]["max_new_tokens"],
    )

    @staticmethod
    def noun_only_score(words, prev_tokens, oracle):
        scores = torch.zeros(len(words))
        for i, word in enumerate(words):
            if detect_anchor(prev_tokens, word) == "noun_anchor":
                scores[i] = oracle.score(word, "noun_anchor")
        return scores

    decoder._score_candidates = noun_only_score
    return decoder, processor


# ── Evaluation (single ablation on Reefknot val) ─────────────────────────────

def eval_ablation(
    ablation: str,
    cfg: dict,
    rk_path: Path,
    out_dir: Path,
    load_in_4bit: bool,
    max_samples: int | None,
    gqa_sgs: dict | None = None,
) -> dict:
    from PIL import Image
    from sgod.utils.model_loader import load_config, load_sgod_decoder

    logger.info("Running ablation: %s", ablation)

    ablation_cfg = build_ablation_config(cfg, ablation)

    if ablation == "fixed_lambda":
        decoder, processor = _build_fixed_lambda_decoder(
            ablation_cfg, lam=ablation_cfg["_ablation_fixed_lambda"], load_in_4bit=load_in_4bit
        )
    elif ablation == "no_anchor":
        decoder, processor = _build_no_anchor_decoder(ablation_cfg, load_in_4bit)
    elif ablation == "noun_only":
        decoder, processor = _build_noun_only_decoder(ablation_cfg, load_in_4bit)
    else:
        from sgod.utils.model_loader import load_llava
        _, processor_base = load_llava(load_in_4bit=load_in_4bit)
        decoder = load_sgod_decoder(ablation_cfg, load_in_4bit=load_in_4bit)
        processor = processor_base

    q_file = rk_path / "val.json"
    img_dir = rk_path / "images"
    with open(q_file) as f:
        items = json.load(f)
    if max_samples:
        items = items[:max_samples]

    correct = 0
    n       = 0

    for i, item in enumerate(items):
        img_path = img_dir / item["image"]
        if not img_path.exists():
            continue

        image    = Image.open(img_path).convert("RGB")
        question = item["question"]
        answer   = item["answer"]

        # Override SGG for gt_sg ablation
        if ablation == "gt_sg" and gqa_sgs is not None:
            img_id = Path(item["image"]).stem
            gt_sg_dict = gqa_sgs.get(img_id, {})
            from experiments.pilot.run_pilot import gqa_sg_to_scene_graph
            from sgod.oracle import VisualOracle, CLIPScorer
            gt_sg = gqa_sg_to_scene_graph(gt_sg_dict)
            # Temporarily override sgg.extract to return GT scene graph
            original_extract = decoder.sgg.extract
            decoder.sgg.extract = lambda _img: gt_sg

        prompt = f"USER: <image>\n{question}\nAnswer yes or no.\nASSISTANT:"
        pred   = decoder.generate(image, prompt)
        pred_norm = "yes" if "yes" in pred.lower() else "no"
        correct += int(pred_norm == answer.strip().lower())
        n += 1

        if ablation == "gt_sg" and gqa_sgs is not None:
            decoder.sgg.extract = original_extract

        if (i + 1) % 100 == 0:
            logger.info("[%s] %d/%d  Acc=%.3f", ablation, i+1, len(items), correct/n)

    accuracy = correct / n if n else 0.0
    result   = {"ablation": ablation, "n": n, "accuracy": round(accuracy, 4)}
    save_json(result, out_dir / f"ablation_{ablation}.json")
    logger.info("[%s] done. Acc=%.4f", ablation, accuracy)
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")

    p = argparse.ArgumentParser()
    p.add_argument("--config",          default="configs/default.yaml")
    p.add_argument("--reefknot_path",   default="data/reefknot/")
    p.add_argument("--gqa_sg_path",     default="data/gqa/val_sceneGraphs.json",
                   help="GQA scene graphs for gt_sg ablation")
    p.add_argument("--ablation",        default="all",
                   help=f"Comma-separated ablations or 'all'. Options: {', '.join(ALL_ABLATIONS)}")
    p.add_argument("--output",          default=None)
    p.add_argument("--load_in_4bit",    action="store_true")
    p.add_argument("--max_samples",     type=int, default=None)
    args = p.parse_args()

    ablations = ALL_ABLATIONS if args.ablation == "all" else [
        a.strip() for a in args.ablation.split(",")
    ]
    invalid = [a for a in ablations if a not in ALL_ABLATIONS]
    if invalid:
        print(f"Unknown ablations: {invalid}. Choose from: {ALL_ABLATIONS}")
        sys.exit(1)

    out_dir  = Path(args.output or str(make_run_dir("ablation", "reefknot")))
    rk_path  = Path(args.reefknot_path)

    if not (rk_path / "val.json").exists():
        raise FileNotFoundError(
            f"{rk_path / 'val.json'} — run download_data.py --benchmarks reefknot"
        )

    from sgod.utils.model_loader import load_config
    cfg = load_config(args.config)

    # Load GT scene graphs for gt_sg ablation
    gqa_sgs = None
    if "gt_sg" in ablations:
        sg_file = Path(args.gqa_sg_path)
        if sg_file.exists():
            with open(sg_file) as f:
                gqa_sgs = json.load(f)
        else:
            logger.warning("GT scene graph file not found: %s — skipping gt_sg ablation", sg_file)
            ablations = [a for a in ablations if a != "gt_sg"]

    all_results = []
    for abl in ablations:
        try:
            res = eval_ablation(
                abl, cfg, rk_path, out_dir,
                load_in_4bit=args.load_in_4bit,
                max_samples=args.max_samples,
                gqa_sgs=gqa_sgs,
            )
            all_results.append(res)
        except Exception as exc:
            logger.error("Ablation %s failed: %s", abl, exc, exc_info=True)
            all_results.append({"ablation": abl, "error": str(exc)})

    save_json(all_results, out_dir / "ablation_summary.json")

    print("\n" + "=" * 64)
    print("  ABLATION RESULTS (Reefknot val)")
    print("=" * 64)
    print(f"  {'Ablation':<20}  {'N':>6}  {'Accuracy':>10}")
    print(f"  {'─'*20}  {'─'*6}  {'─'*10}")
    for r in all_results:
        if "error" in r:
            print(f"  {r['ablation']:<20}  {'ERROR':>6}  {r['error'][:20]:>10}")
        else:
            print(f"  {r['ablation']:<20}  {r['n']:>6}  {r['accuracy']*100:>9.2f}%")
    print(f"\n  Results saved: {out_dir}")


if __name__ == "__main__":
    main()
