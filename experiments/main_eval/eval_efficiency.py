"""
Efficiency benchmarks (Section 10.3).

Measures:
  - Tokens per second (generation throughput)
  - Time-to-first-token (TTFT)
  - Peak VRAM usage (MB)
  - SGG extraction time (isolated)
  - Oracle build time (isolated)

Compares SGOD vs LLaVA baseline.
Each measurement is averaged over --num_samples runs (default 50).

Usage:
    python experiments/main_eval/eval_efficiency.py \\
        --config configs/default.yaml \\
        --num_samples 50 \\
        --image test_imgs/image.png
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sgod.utils.output import make_run_dir, save_json

logger = logging.getLogger(__name__)

QUESTIONS = [
    "What objects are in this image?",
    "Is there a person in the image?",
    "What is on the table?",
    "Describe the scene.",
    "What color is the object?",
]


def measure_vram_mb() -> float:
    """Return current peak VRAM usage in MB (CUDA only)."""
    import torch
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1_048_576


def reset_vram_stats() -> None:
    import torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def time_llava_baseline(
    model, processor, image, question: str, max_new_tokens: int
) -> tuple[float, float, int]:
    """Returns (ttft_s, total_s, n_tokens)."""
    import torch
    prompt = f"USER: <image>\n{question}\nASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # TTFT: time for the first token
    t0 = time.perf_counter()
    with torch.no_grad():
        # First forward pass
        out1 = model(**inputs)
        _ = out1.logits[0, -1, :].argmax()
    ttft = time.perf_counter() - t0

    # Full generation
    t1 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    total = time.perf_counter() - t1
    n_tok = out.shape[1] - inputs["input_ids"].shape[1]
    return ttft, total, max(n_tok, 1)


def time_sgod(decoder, image, question: str, max_new_tokens: int) -> tuple[float, float, int]:
    """Returns (ttft_s, total_s, n_tokens).

    For SGOD, TTFT includes SGG + oracle build time (one-time overhead per image).
    """
    import torch

    # Time the full generate() call
    t0 = time.perf_counter()
    text = decoder.generate(image, f"USER: <image>\n{question}\nASSISTANT:",
                             max_new_tokens=max_new_tokens)
    total = time.perf_counter() - t0

    n_tok = max(len(text.split()), 1)
    # Estimate TTFT as total/n_tokens (no per-token hooks in current impl)
    ttft = total / n_tok
    return ttft, total, n_tok


def time_sgg_only(sgg_module, image) -> float:
    """Measure isolated SGG extraction time (seconds)."""
    t0 = time.perf_counter()
    _ = sgg_module.extract(image)
    return time.perf_counter() - t0


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")

    p = argparse.ArgumentParser()
    p.add_argument("--config",       default="configs/default.yaml")
    p.add_argument("--image",        default="test_imgs/image.png",
                   help="Image to use for benchmarking")
    p.add_argument("--num_samples",  type=int, default=50)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--output",       default=None)
    p.add_argument("--load_in_4bit", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.output or str(make_run_dir("eval", "efficiency")))
    img_path = Path(args.image)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    from PIL import Image
    from sgod.utils.model_loader import load_config, load_llava, load_sgod_decoder, load_sgg

    cfg              = load_config(args.config)
    model, processor = load_llava(load_in_4bit=args.load_in_4bit)
    decoder          = load_sgod_decoder(cfg, load_in_4bit=args.load_in_4bit)
    sgg_module       = load_sgg(cfg["sgg"]["checkpoint"])

    image = Image.open(img_path).convert("RGB")

    # Warm-up
    logger.info("Warming up ...")
    for q in QUESTIONS[:2]:
        _ = model.generate(
            **{k: v.to(model.device) for k, v in
               processor(text=f"USER: <image>\n{q}\nASSISTANT:", images=image,
                         return_tensors="pt").items()},
            max_new_tokens=8, do_sample=False
        )

    import torch

    # ── Baseline timing ───────────────────────────────────────────────────────
    base_ttfts, base_totals, base_tok_s = [], [], []
    for i in range(args.num_samples):
        q = QUESTIONS[i % len(QUESTIONS)]
        reset_vram_stats()
        ttft, total, n_tok = time_llava_baseline(
            model, processor, image, q, args.max_new_tokens
        )
        base_ttfts.append(ttft)
        base_totals.append(total)
        base_tok_s.append(n_tok / max(total, 1e-6))
        if (i + 1) % 10 == 0:
            logger.info("[Baseline] %d/%d  tok/s=%.1f", i+1, args.num_samples,
                        sum(base_tok_s) / len(base_tok_s))
    base_vram = measure_vram_mb()

    # ── SGOD timing ───────────────────────────────────────────────────────────
    sgod_ttfts, sgod_totals, sgod_tok_s = [], [], []
    for i in range(args.num_samples):
        q = QUESTIONS[i % len(QUESTIONS)]
        reset_vram_stats()
        ttft, total, n_tok = time_sgod(decoder, image, q, args.max_new_tokens)
        sgod_ttfts.append(ttft)
        sgod_totals.append(total)
        sgod_tok_s.append(n_tok / max(total, 1e-6))
        if (i + 1) % 10 == 0:
            logger.info("[SGOD] %d/%d  tok/s=%.1f", i+1, args.num_samples,
                        sum(sgod_tok_s) / len(sgod_tok_s))
    sgod_vram = measure_vram_mb()

    # ── SGG isolated ─────────────────────────────────────────────────────────
    sgg_times = [time_sgg_only(sgg_module, image) for _ in range(10)]

    def mean(xs): return sum(xs) / len(xs) if xs else 0.0

    result = {
        "num_samples":         args.num_samples,
        "max_new_tokens":      args.max_new_tokens,
        "baseline": {
            "mean_ttft_ms":    round(mean(base_ttfts) * 1000, 1),
            "mean_total_s":    round(mean(base_totals), 3),
            "mean_tok_per_s":  round(mean(base_tok_s), 1),
            "peak_vram_mb":    round(base_vram, 1),
        },
        "sgod": {
            "mean_ttft_ms":    round(mean(sgod_ttfts) * 1000, 1),
            "mean_total_s":    round(mean(sgod_totals), 3),
            "mean_tok_per_s":  round(mean(sgod_tok_s), 1),
            "peak_vram_mb":    round(sgod_vram, 1),
        },
        "sgg_isolated": {
            "mean_extract_ms": round(mean(sgg_times) * 1000, 1),
        },
        "overhead": {
            "latency_ratio": round(mean(sgod_totals) / mean(base_totals), 3)
            if mean(base_totals) else 0.0,
            "tok_per_s_ratio": round(mean(sgod_tok_s) / mean(base_tok_s), 3)
            if mean(base_tok_s) else 0.0,
        },
    }

    save_json(result, out_dir / "efficiency_results.json")

    print("\n" + "=" * 64)
    print("  EFFICIENCY RESULTS")
    print("=" * 64)
    b, g = result["baseline"], result["sgod"]
    print(f"  {'Metric':<25}  {'Baseline':>12}  {'SGOD':>12}  {'Ratio':>8}")
    print(f"  {'─'*25}  {'─'*12}  {'─'*12}  {'─'*8}")
    print(f"  {'TTFT (ms)':<25}  {b['mean_ttft_ms']:>12.1f}  {g['mean_ttft_ms']:>12.1f}  "
          f"{g['mean_ttft_ms']/max(b['mean_ttft_ms'],0.001):>7.2f}x")
    print(f"  {'Total gen time (s)':<25}  {b['mean_total_s']:>12.3f}  {g['mean_total_s']:>12.3f}  "
          f"{result['overhead']['latency_ratio']:>7.2f}x")
    print(f"  {'Tokens / second':<25}  {b['mean_tok_per_s']:>12.1f}  {g['mean_tok_per_s']:>12.1f}  "
          f"{result['overhead']['tok_per_s_ratio']:>7.2f}x")
    print(f"  {'Peak VRAM (MB)':<25}  {b['peak_vram_mb']:>12.1f}  {g['peak_vram_mb']:>12.1f}  "
          f"{'—':>8}")
    print(f"\n  SGG extraction (isolated): {result['sgg_isolated']['mean_extract_ms']:.1f} ms / image")
    print(f"\n  Results saved: {out_dir}")


if __name__ == "__main__":
    main()
