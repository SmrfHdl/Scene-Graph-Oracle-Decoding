"""
SGG Module evaluation script.

Runs the SGGModule on sample images and reports:
1. Extracted scene graphs (readable format)
2. Latency statistics (mean, p50, p95, p99)
3. Model metadata (params, memory)

Usage:
    python scripts/eval_sgg.py \
        --checkpoint data/checkpoints/reltr/checkpoint0149.pth \
        --image path/to/image.jpg \
        --benchmark

    # With multiple images:
    python scripts/eval_sgg.py \
        --checkpoint data/checkpoints/reltr/checkpoint0149.pth \
        --image_dir path/to/images/ \
        --top_k 10
"""
import argparse
import glob
import logging
import sys
from pathlib import Path

from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sgod.sgg import SGGModule
from sgod.utils import make_run_dir, save_json


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SGG Module")
    parser.add_argument("--checkpoint", required=True, help="Path to RelTR checkpoint")
    parser.add_argument("--image", type=str, default=None, help="Path to a single image")
    parser.add_argument("--image_dir", type=str, default=None, help="Directory of images")
    parser.add_argument("--device", type=str, default=None, help="Device (auto-detect if not set)")
    parser.add_argument("--confidence", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--top_k", type=int, default=20, help="Max triplets per image")
    parser.add_argument("--benchmark", action="store_true", help="Run latency benchmark")
    parser.add_argument("--n_runs", type=int, default=100, help="Benchmark runs")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Save results to specific JSON path (default: outputs/eval/<run>/results.json)")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    run_dir = Path(args.output_json).parent if args.output_json else make_run_dir("eval", "sgg")
    out_path = Path(args.output_json) if args.output_json else run_dir / "results.json"
    print(f"Output dir: {run_dir}")

    # Collect images
    image_paths = []
    if args.image:
        image_paths.append(args.image)
    if args.image_dir:
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
            image_paths.extend(glob.glob(f"{args.image_dir}/{ext}"))
    if not image_paths:
        # Generate a synthetic test image if no images provided
        logging.info("No images provided, using synthetic test image (640x480)")
        img = Image.new("RGB", (640, 480), color=(128, 100, 80))
        image_paths = ["<synthetic>"]
        images = [img]
    else:
        images = [Image.open(p).convert("RGB") for p in image_paths]

    # Load model
    print(f"\n{'='*60}")
    print(f"SGG Module Evaluation")
    print(f"{'='*60}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device or 'auto'}")
    print(f"Confidence threshold: {args.confidence}")
    print(f"Top-K: {args.top_k}")
    print(f"Images: {len(images)}")
    print(f"{'='*60}\n")

    sgg = SGGModule(
        checkpoint_path=args.checkpoint,
        device=args.device,
        confidence_threshold=args.confidence,
        top_k=args.top_k,
    )

    print(f"Model params: {sgg.num_parameters / 1e6:.1f}M")
    print(f"Model memory: {sgg.memory_mb:.1f}MB")
    print()

    # Extract scene graphs
    results = []
    for i, (path, image) in enumerate(zip(image_paths, images)):
        print(f"{'─'*40}")
        print(f"Image {i+1}/{len(images)}: {path}")
        print(f"  Size: {image.size}")

        sg = sgg.extract(image)
        summary = sg.summary()

        print(f"  Objects: {summary['num_objects']}")
        print(f"  Relations: {summary['num_relations']}")
        print(f"  Avg object conf: {summary['avg_obj_confidence']:.3f}")
        print(f"  Avg relation conf: {summary['avg_rel_confidence']:.3f}")
        print(f"  Oracle active: {summary['oracle_active']}")
        print()
        print(f"  Scene Graph:")
        for line in sg.to_text().split("\n"):
            print(f"    {line}")
        print()

        results.append({
            "image": str(path),
            "summary": summary,
            "scene_graph": {
                "objects": [
                    {"label": o.label, "confidence": o.confidence, "bbox": list(o.bbox)}
                    for o in sg.objects
                ],
                "relations": [
                    {"subject": r.subject, "predicate": r.predicate,
                     "object": r.object, "confidence": r.confidence}
                    for r in sg.relations
                ],
                "attributes": [
                    {"entity": a.entity, "attribute": a.attribute, "confidence": a.confidence}
                    for a in sg.attributes
                ],
            },
            "text": sg.to_text(),
        })

    # Benchmark
    if args.benchmark:
        print(f"\n{'='*60}")
        print("Latency Benchmark")
        print(f"{'='*60}")
        stats = sgg.benchmark(images[0], n_runs=args.n_runs)
        print(f"  Device: {stats['device']}")
        print(f"  Runs: {stats['n_runs']}")
        print(f"  Mean: {stats['mean_ms']:.1f}ms")
        print(f"  P50:  {stats['p50_ms']:.1f}ms")
        print(f"  P95:  {stats['p95_ms']:.1f}ms")
        print(f"  P99:  {stats['p99_ms']:.1f}ms")
        print(f"  Min:  {stats['min_ms']:.1f}ms")
        print(f"  Max:  {stats['max_ms']:.1f}ms")

        for r in results:
            r["benchmark"] = stats

    # Save results
    payload = {
        "config": {
            "checkpoint": args.checkpoint,
            "device": args.device or "auto",
            "confidence_threshold": args.confidence,
            "top_k": args.top_k,
        },
        "model": {
            "params_M": round(sgg.num_parameters / 1e6, 1),
            "memory_mb": round(sgg.memory_mb, 1),
        },
        "results": results,
    }
    if args.benchmark:
        payload["benchmark"] = sgg.benchmark(images[0], n_runs=args.n_runs)

    save_json(payload, out_path)
    print(f"\nResults saved to {out_path}")

    print(f"\n{'='*60}")
    print("Done!")


if __name__ == "__main__":
    main()
