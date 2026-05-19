"""Build a Stage-0 calibration jsonl from POPE question files.

Stage-0 distillation needs (image, question) pairs only — the label is unused
because we are learning to mimic the SGOD-v1 teacher's per-step Δ, not the
ground-truth answer. POPE is a convenient source: short yes/no questions over
COCO val2014, balanced across object-anchor anchor positions.

Output format (matches `distill_sgod_v1.py collect --dataset jsonl:<path>`):

    {"image": "<abs path to .jpg>", "question": "Is there a dog in the image?"}

Usage:

    # 1) Download POPE JSONs (one-time, ~3 MB)
    python scripts/download_data.py --benchmarks pope

    # 2) Manually unzip COCO val2014 to data/coco/val2014/
    #    http://images.cocodataset.org/zips/val2014.zip

    # 3) Build calib set
    python scripts/prepare_stage0_calib.py \\
        --pope-dir data/pope \\
        --coco-root data/coco/val2014 \\
        --out data/stage0/calib.jsonl \\
        --n 500
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

log = logging.getLogger("prepare_stage0_calib")

POPE_SPLITS = ("random", "popular", "adversarial")


def _load_pope_split(pope_dir: Path, split: str) -> list[dict]:
    path = pope_dir / f"coco_pope_{split}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `python scripts/download_data.py --benchmarks pope`"
        )
    items: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rec["_split"] = split
            items.append(rec)
    return items


def _resolve_image(coco_root: Path, image_name: str) -> Path | None:
    p = coco_root / image_name
    return p if p.exists() else None


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pope-dir", type=Path, default=Path("data/pope"),
                    help="Directory containing coco_pope_{random,popular,adversarial}.json")
    ap.add_argument("--coco-root", type=Path, default=Path("data/coco/val2014"),
                    help="Directory containing COCO val2014 JPEGs")
    ap.add_argument("--out", type=Path, default=Path("data/stage0/calib.jsonl"),
                    help="Output jsonl path")
    ap.add_argument("--n", type=int, default=500,
                    help="Number of (image, question) pairs to sample")
    ap.add_argument("--splits", nargs="+", default=list(POPE_SPLITS),
                    choices=POPE_SPLITS,
                    help="Which POPE splits to draw from (default: all 3)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--absolute-paths", action="store_true",
                    help="Write absolute image paths (default: relative to cwd)")
    args = ap.parse_args()

    coco_root = args.coco_root.resolve() if args.absolute_paths else args.coco_root

    # 1. Load + tag all splits.
    pool: list[dict] = []
    for split in args.splits:
        items = _load_pope_split(args.pope_dir, split)
        pool.extend(items)
        log.info("Loaded %d items from split=%s", len(items), split)
    log.info("Total pool: %d items across %d splits", len(pool), len(args.splits))

    # 2. Shuffle deterministically and pick the first N whose image actually exists.
    rng = random.Random(args.seed)
    rng.shuffle(pool)

    chosen: list[dict] = []
    missing = 0
    for rec in pool:
        if len(chosen) >= args.n:
            break
        img_path = _resolve_image(coco_root, rec["image"])
        if img_path is None:
            missing += 1
            continue
        chosen.append({"image": str(img_path), "question": rec["text"]})

    if missing:
        log.warning("%d items skipped (image missing under %s)", missing, coco_root)

    if len(chosen) < args.n:
        log.error("Only resolved %d / %d requested examples — is %s correct?",
                  len(chosen), args.n, coco_root)
        return 1

    # 3. Write jsonl.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for rec in chosen:
            f.write(json.dumps(rec) + "\n")
    log.info("Wrote %d examples to %s", len(chosen), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
