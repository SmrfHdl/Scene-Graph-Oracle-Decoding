"""
Main Evaluation — MMHal-Bench (Section 6.2, Benchmark 5).

Open-ended hallucination quality:
  - 96 images × 8 question types = 768 questions
  - Metric: Score 0-6 (GPT-4 evaluator) + HalScore (lower is better)
  - Cost: ~$3-5 GPT-4 API per full run

MMHal-Bench format (mmhal_data.json):
  [{"image_id": "...", "image_src": "url", "question_type": ...,
    "question": "...", "gt_answer": "...", "image_content": [...]}]

Data required:
  data/mmhal_bench/mmhal_data.json
  (Images are fetched from URLs in the JSON on first use, cached to data/mmhal_bench/images/)

Usage:
    python experiments/main_eval/eval_mmhal.py \\
        --config configs/default.yaml \\
        --mmhal_path data/mmhal_bench/ \\
        --gpt4_api_key $OPENAI_API_KEY

    # Skip GPT-4 scoring (get raw predictions only):
    python experiments/main_eval/eval_mmhal.py --no_gpt4
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sgod.utils.output import make_run_dir, save_json

logger = logging.getLogger(__name__)

QUESTION_TYPES = [
    "attribute", "adversarial", "comparison", "counting",
    "relation", "environment", "holistic", "other",
]

# ── GPT-4 scoring ─────────────────────────────────────────────────────────────

EVAL_PROMPT = """You are an expert evaluator for image-question answering.
Given: question, ground truth answer, model response, and image contents.
Score the model response from 0 to 6:
  6: Perfect — factually correct, no hallucinations
  5: Mostly correct, minor errors
  4: Partially correct, some hallucinations
  3: Mixed — correct parts but significant hallucinations
  2: Mostly incorrect or hallucinated
  1: Nearly all hallucinated
  0: Completely wrong / refuses to answer

Respond with ONLY a JSON object: {{"score": <int 0-6>, "reason": "<one sentence>"}}

Question: {question}
Ground truth: {gt_answer}
Image contents: {image_content}
Model response: {model_response}"""


def gpt4_score(client, question: str, gt_answer: str, image_content: list,
               model_response: str) -> dict:
    """Call GPT-4 to score a single response."""
    prompt = EVAL_PROMPT.format(
        question=question,
        gt_answer=gt_answer,
        image_content=", ".join(str(x) for x in image_content),
        model_response=model_response,
    )
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=100,
    )
    text = response.choices[0].message.content.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback: extract score from text
        for word in text.split():
            if word.isdigit() and 0 <= int(word) <= 6:
                return {"score": int(word), "reason": text}
        return {"score": 0, "reason": f"parse error: {text}"}


# ── Image fetching ────────────────────────────────────────────────────────────

def fetch_image(image_src: str, cache_dir: Path, image_id: str):
    """Download image from URL or load from cache. Returns PIL.Image."""
    from PIL import Image

    cache_path = cache_dir / f"{image_id}.jpg"
    if not cache_path.exists():
        try:
            logger.debug("Fetching image: %s", image_src)
            urllib.request.urlretrieve(image_src, cache_path)
        except Exception as exc:
            logger.warning("Could not fetch image %s: %s", image_src, exc)
            return None
    return Image.open(cache_path).convert("RGB")


# ── LLaVA inference ───────────────────────────────────────────────────────────

def run_llava(model, processor, image, question: str, max_new_tokens: int = 128) -> str:
    import torch
    prompt = f"USER: <image>\n{question}\nASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")

    p = argparse.ArgumentParser()
    p.add_argument("--config",          default="configs/default.yaml")
    p.add_argument("--mmhal_path",      default="data/mmhal_bench/")
    p.add_argument("--output",          default=None)
    p.add_argument("--gpt4_api_key",    default=None,
                   help="OpenAI API key for GPT-4 scoring")
    p.add_argument("--no_gpt4",         action="store_true",
                   help="Skip GPT-4 evaluation — save raw predictions only")
    p.add_argument("--load_in_4bit",    action="store_true")
    p.add_argument("--baseline_only",   action="store_true")
    p.add_argument("--max_new_tokens",  type=int, default=128)
    args = p.parse_args()

    out_dir   = Path(args.output or str(make_run_dir("eval", "mmhal")))
    data_path = Path(args.mmhal_path)
    img_cache = data_path / "images"
    img_cache.mkdir(exist_ok=True)

    q_file = data_path / "mmhal_data.json"
    if not q_file.exists():
        raise FileNotFoundError(f"{q_file} — run download_data.py --benchmarks mmhal")

    with open(q_file) as f:
        items = json.load(f)

    from sgod.utils.model_loader import load_config, load_llava, load_sgod_decoder

    cfg              = load_config(args.config)
    model, processor = load_llava(load_in_4bit=args.load_in_4bit)
    decoder          = None if args.baseline_only else load_sgod_decoder(
        cfg, load_in_4bit=args.load_in_4bit
    )

    gpt4_client = None
    if not args.no_gpt4:
        api_key = args.gpt4_api_key or __import__("os").environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.warning("No GPT-4 API key found. Add --no_gpt4 or set OPENAI_API_KEY.")
        else:
            from openai import OpenAI
            gpt4_client = OpenAI(api_key=api_key)

    records = []
    for i, item in enumerate(items):
        image = fetch_image(item.get("image_src", ""), img_cache, item["image_id"])
        if image is None:
            continue

        question  = item["question"]
        gt_answer = item["gt_answer"]
        qtype     = item.get("question_type", "other")

        pred_base = run_llava(model, processor, image, question, args.max_new_tokens)
        pred_sgod = decoder.generate(
            image, f"USER: <image>\n{question}\nASSISTANT:"
        ) if decoder else None

        record: dict = {
            "image_id":    item["image_id"],
            "question":    question,
            "gt_answer":   gt_answer,
            "type":        qtype,
            "pred_base":   pred_base,
            "pred_sgod":   pred_sgod,
        }

        if gpt4_client:
            img_content = item.get("image_content", [])
            record["gpt4_base"] = gpt4_score(
                gpt4_client, question, gt_answer, img_content, pred_base
            )
            if pred_sgod:
                record["gpt4_sgod"] = gpt4_score(
                    gpt4_client, question, gt_answer, img_content, pred_sgod
                )

        records.append(record)
        logger.info("[%d/%d] type=%s", i+1, len(items), qtype)

    save_json(records, out_dir / "records.json")

    # Compute aggregate scores
    result: dict = {"n": len(records)}
    if gpt4_client and any("gpt4_base" in r for r in records):
        scores_base = [r["gpt4_base"]["score"] for r in records if "gpt4_base" in r]
        result["baseline"] = {
            "mean_score": round(sum(scores_base) / len(scores_base), 3),
            "hal_score":  round((6 - sum(scores_base) / len(scores_base)) / 6, 3),
        }
        if decoder:
            scores_sgod = [r["gpt4_sgod"]["score"] for r in records if "gpt4_sgod" in r]
            result["sgod"] = {
                "mean_score": round(sum(scores_sgod) / len(scores_sgod), 3),
                "hal_score":  round((6 - sum(scores_sgod) / len(scores_sgod)) / 6, 3),
            }

    save_json(result, out_dir / "mmhal_results.json")

    print("\n" + "=" * 64)
    print("  MMHAL-BENCH RESULTS")
    print("=" * 64)
    print(f"  N = {len(records)}")
    if "baseline" in result:
        b = result["baseline"]
        print(f"  Baseline — Score={b['mean_score']:.2f}/6  HalScore={b['hal_score']:.3f}")
    if "sgod" in result:
        g = result["sgod"]
        print(f"  SGOD     — Score={g['mean_score']:.2f}/6  HalScore={g['hal_score']:.3f}")
    else:
        print("  (Run with GPT-4 key for scores, or check records.json for raw predictions)")
    print(f"\n  Results saved: {out_dir}")


if __name__ == "__main__":
    main()
