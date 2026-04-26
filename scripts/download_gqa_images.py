"""
Download only the GQA images needed for the pilot experiment.

GQA images are Visual Genome images served from Stanford CDN.
Each image is ~100-300 KB; 200 images ≈ 30-60 MB (vs 20 GB full zip).

Usage:
    python scripts/download_gqa_images.py \
        --gqa_path /dev/shm/gqa/ \
        --num_images 200 \
        --out_dir data/gqa/images/

    # Dry-run (print IDs only, no download):
    python scripts/download_gqa_images.py --dry_run
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

_VG_URL_PATTERNS = [
    "https://cs.stanford.edu/people/rak248/VG_100K/{id}.jpg",
    "https://cs.stanford.edu/people/rak248/VG_100K_2/{id}.jpg",
]


def get_pilot_image_ids(
    gqa_path: Path,
    n: int,
    seed: int = 42,
) -> list[str]:
    """Return the image IDs that run_pilot.py would select."""
    q_file = gqa_path / "val_balanced_questions.json"
    if not q_file.exists():
        raise FileNotFoundError(f"{q_file} not found.")

    with open(q_file) as f:
        all_q = json.load(f)

    relation_qs = {
        qid: q for qid, q in all_q.items()
        if q.get("types", {}).get("semantic") == "rel"
    }
    logger.info("Relation questions: %d", len(relation_qs))

    seen: dict[str, str] = {}
    for qid, q in relation_qs.items():
        img_id = q["imageId"]
        if img_id not in seen:
            seen[img_id] = qid

    rng = random.Random(seed)
    selected = rng.sample(list(seen.keys()), min(n, len(seen)))
    logger.info("Selected %d unique image IDs", len(selected))
    return selected


def download_image(img_id: str, out_dir: Path) -> bool:
    """Try both VG CDN paths; return True if downloaded successfully."""
    out_path = out_dir / f"{img_id}.jpg"
    if out_path.exists():
        return True

    for pattern in _VG_URL_PATTERNS:
        url = pattern.format(id=img_id)
        try:
            urllib.request.urlretrieve(url, out_path)
            return True
        except Exception:
            if out_path.exists():
                out_path.unlink()
    return False


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    p = argparse.ArgumentParser()
    p.add_argument("--gqa_path",    default="/dev/shm/gqa/")
    p.add_argument("--num_images",  type=int, default=200)
    p.add_argument("--out_dir",     default="data/gqa/images/")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--dry_run",     action="store_true",
                   help="Print image IDs without downloading")
    args = p.parse_args()

    gqa_path = Path(args.gqa_path)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_ids = get_pilot_image_ids(gqa_path, args.num_images, seed=args.seed)

    if args.dry_run:
        for img_id in image_ids:
            print(img_id)
        return

    n_ok, n_fail = 0, 0
    for i, img_id in enumerate(image_ids):
        ok = download_image(img_id, out_dir)
        if ok:
            n_ok += 1
        else:
            n_fail += 1
            logger.warning("[%d/%d] FAILED: %s", i + 1, len(image_ids), img_id)

        if (i + 1) % 20 == 0 or (i + 1) == len(image_ids):
            logger.info("[%d/%d] OK=%d FAIL=%d", i + 1, len(image_ids), n_ok, n_fail)

    logger.info("Done. %d downloaded, %d failed → %s", n_ok, n_fail, out_dir)
    if n_fail > 0:
        logger.warning("%d images failed — pilot will skip them.", n_fail)


if __name__ == "__main__":
    main()
