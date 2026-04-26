"""
Download and prepare benchmark datasets.

Auto-downloads (JSON/annotations only):
  - POPE question files  (~50 MB, 3 splits from GitHub)
  - GQA questions + scene graphs  (~600 MB, from Stanford)
  - AMBER questions + answers  (~5 MB, from GitHub)
  - MMHal-Bench questions  (~5 MB, from HuggingFace)
  - Reefknot annotations  (~150 MB, from HuggingFace)
  - VQAv2 questions + annotations  (~40 MB, from S3)

Manual download required (images — too large for auto):
  - COCO val2014 images (~6 GB) — for POPE + VQAv2
  - GQA images (~20 GB) — for pilot + main eval
  See --show-manual for exact instructions.

Usage:
    python scripts/download_data.py --benchmarks pope,gqa,amber
    python scripts/download_data.py --all
    python scripts/download_data.py --show-manual
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"

# ── Download registry ─────────────────────────────────────────────────────────

_POPE_BASE = (
    "https://raw.githubusercontent.com/AoiDragon/POPE"
    "/e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco"
)

_HF_BASE = "https://huggingface.co/datasets"

DATASETS: dict[str, dict] = {
    "pope": {
        "desc": "POPE — object hallucination benchmark (question JSONs, 3 splits)",
        "files": [
            (f"{_POPE_BASE}/coco_pope_adversarial.json",
             DATA_ROOT / "pope" / "coco_pope_adversarial.json",
             "POPE adversarial"),
            (f"{_POPE_BASE}/coco_pope_popular.json",
             DATA_ROOT / "pope" / "coco_pope_popular.json",
             "POPE popular"),
            (f"{_POPE_BASE}/coco_pope_random.json",
             DATA_ROOT / "pope" / "coco_pope_random.json",
             "POPE random"),
        ],
        "manual": (
            "POPE also requires COCO val2014 images (~6 GB):\n"
            "  http://images.cocodataset.org/zips/val2014.zip\n"
            "  → unzip to data/coco/val2014/"
        ),
    },

    "gqa": {
        "desc": "GQA — relation questions + GT scene graphs (Stanford)",
        "files": [
            ("https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip",
             DATA_ROOT / "gqa" / "questions1.2.zip",
             "GQA questions (~400 MB)"),
            ("https://downloads.cs.stanford.edu/nlp/data/gqa/sceneGraphs.zip",
             DATA_ROOT / "gqa" / "sceneGraphs.zip",
             "GQA scene graphs (~200 MB)"),
        ],
        "extract_zips": [
            (DATA_ROOT / "gqa" / "questions1.2.zip",    DATA_ROOT / "gqa"),
            (DATA_ROOT / "gqa" / "sceneGraphs.zip",     DATA_ROOT / "gqa"),
        ],
        "manual": (
            "GQA images (~20 GB) must be downloaded separately:\n"
            "  https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip\n"
            "  → unzip to data/gqa/images/\n"
            "  Required for all pilot variants (A/B/C) — LLaVA needs real images.\n"
            "  Use --smoke_test_image test_imgs/image.png to verify the pipeline first."
        ),
    },

    "amber": {
        "desc": "AMBER — multi-dimensional hallucination benchmark",
        "files": [
            ("https://raw.githubusercontent.com/junyangwang0410/AMBER/master/data/query.json",
             DATA_ROOT / "amber" / "query.json",
             "AMBER queries"),
            ("https://raw.githubusercontent.com/junyangwang0410/AMBER/master/data/answer.json",
             DATA_ROOT / "amber" / "answer.json",
             "AMBER answers"),
        ],
        "manual": (
            "AMBER images (~5 GB) from HuggingFace:\n"
            "  huggingface-cli download AMBER-VG/AMBER --local-dir data/amber/\n"
            "  (or: pip install huggingface_hub && python -c \"from huggingface_hub import snapshot_download; "
            "snapshot_download('AMBER-VG/AMBER', local_dir='data/amber/')\")"
        ),
    },

    "mmhal": {
        "desc": "MMHal-Bench — open-ended hallucination quality (GPT-4 eval)",
        "files": [
            (f"{_HF_BASE}/Shengcao1006/MMHal-Bench/resolve/main/mmhal_data.json",
             DATA_ROOT / "mmhal_bench" / "mmhal_data.json",
             "MMHal-Bench data"),
            (f"{_HF_BASE}/Shengcao1006/MMHal-Bench/resolve/main/response_template.json",
             DATA_ROOT / "mmhal_bench" / "response_template.json",
             "MMHal-Bench template"),
        ],
        "manual": (
            "MMHal-Bench images are embedded as URLs in mmhal_data.json.\n"
            "Run the eval script with --download_images to fetch them on first use."
        ),
    },

    "reefknot": {
        "desc": "Reefknot — relation hallucination benchmark (MAIN CLAIM)",
        "files": [
            (f"{_HF_BASE}/hyintell/ReefKnot/resolve/main/data/test.json",
             DATA_ROOT / "reefknot" / "test.json",
             "Reefknot test set"),
            (f"{_HF_BASE}/hyintell/ReefKnot/resolve/main/data/val.json",
             DATA_ROOT / "reefknot" / "val.json",
             "Reefknot val set"),
        ],
        "manual": (
            "Reefknot images are sourced from Visual Genome.\n"
            "  pip install gdown\n"
            "  python -c \"import gdown; gdown.download_folder("
            "'https://drive.google.com/drive/folders/1gQ-reJ4-Q6DFqTRBMVbSQ6_8P5NGQS73', "
            "output='data/reefknot/images/')\"\n"
            "  (or download VG images from: https://cs.stanford.edu/people/rak248/VG_100K_2/)"
        ),
    },

    "vqav2": {
        "desc": "VQAv2 val — general VQA capability benchmark",
        "files": [
            ("https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip",
             DATA_ROOT / "vqav2" / "v2_Questions_Val_mscoco.zip",
             "VQAv2 val questions"),
            ("https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip",
             DATA_ROOT / "vqav2" / "v2_Annotations_Val_mscoco.zip",
             "VQAv2 val annotations"),
        ],
        "extract_zips": [
            (DATA_ROOT / "vqav2" / "v2_Questions_Val_mscoco.zip",     DATA_ROOT / "vqav2"),
            (DATA_ROOT / "vqav2" / "v2_Annotations_Val_mscoco.zip",   DATA_ROOT / "vqav2"),
        ],
        "manual": (
            "VQAv2 uses COCO val2017 images (~6 GB):\n"
            "  http://images.cocodataset.org/zips/val2017.zip\n"
            "  → unzip to data/coco/val2017/"
        ),
    },
}

# ── Core downloader ───────────────────────────────────────────────────────────

def _progress_hook(desc: str):
    """Returns urllib reporthook that prints a progress bar."""
    try:
        from tqdm import tqdm
        pbar = tqdm(unit="B", unit_scale=True, desc=desc, leave=False)

        def hook(block_num: int, block_size: int, total_size: int) -> None:
            if total_size > 0:
                pbar.total = total_size
            pbar.update(block_size)
            if block_num * block_size >= (total_size or 0):
                pbar.close()

        return hook
    except ImportError:
        downloaded = [0]

        def hook(block_num: int, block_size: int, total_size: int) -> None:
            downloaded[0] += block_size
            mb = downloaded[0] / 1_048_576
            if total_size > 0:
                pct = min(100, downloaded[0] * 100 // total_size)
                print(f"\r  {desc}: {mb:.1f} MB / {total_size/1_048_576:.1f} MB ({pct}%)  ",
                      end="", flush=True)
            else:
                print(f"\r  {desc}: {mb:.1f} MB", end="", flush=True)
            if block_num * block_size >= (total_size or 1):
                print()

        return hook


def download_file(url: str, dest: Path, desc: str, skip_if_exists: bool = True) -> bool:
    """Download url → dest. Returns True if downloaded, False if skipped."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if skip_if_exists and dest.exists() and dest.stat().st_size > 0:
        logger.info("  [skip] %s already exists", dest.name)
        return False

    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        logger.info("  Downloading %s → %s", desc, dest.name)
        urllib.request.urlretrieve(url, tmp, reporthook=_progress_hook(desc))
        tmp.rename(dest)
        logger.info("  Saved %s (%.1f MB)", dest.name, dest.stat().st_size / 1_048_576)
        return True
    except Exception as exc:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(f"Failed to download {desc}: {exc}") from exc


def extract_zip(zip_path: Path, dest_dir: Path, skip_if_done: bool = True) -> None:
    """Extract zip_path into dest_dir."""
    done_marker = dest_dir / f".extracted_{zip_path.stem}"
    if skip_if_done and done_marker.exists():
        logger.info("  [skip] %s already extracted", zip_path.name)
        return
    logger.info("  Extracting %s ...", zip_path.name)
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    done_marker.touch()
    logger.info("  Extracted → %s", dest_dir)


# ── Dataset downloaders ───────────────────────────────────────────────────────

def download_dataset(name: str, force: bool = False) -> None:
    """Download all auto-downloadable files for a single dataset."""
    cfg = DATASETS[name]
    print(f"\n{'─'*60}")
    print(f"  {name.upper()} — {cfg['desc']}")
    print(f"{'─'*60}")

    # Download files
    for url, dest, desc in cfg.get("files", []):
        try:
            download_file(url, Path(dest), desc, skip_if_exists=not force)
        except RuntimeError as exc:
            logger.warning("  WARNING: %s", exc)
            print(f"  [WARN] Could not auto-download: {desc}")
            print(f"         URL: {url}")

    # Extract zips
    for zip_path, dest_dir in cfg.get("extract_zips", []):
        zip_path, dest_dir = Path(zip_path), Path(dest_dir)
        if zip_path.exists():
            try:
                extract_zip(zip_path, dest_dir, skip_if_done=not force)
            except Exception as exc:
                logger.warning("  WARNING: extraction failed: %s", exc)

    # Manual instructions
    if "manual" in cfg:
        print(f"\n  [MANUAL REQUIRED]\n  {cfg['manual']}")

    print(f"\n  Done: {name}")


def show_manual_instructions() -> None:
    print("\n" + "="*70)
    print("  MANUAL DOWNLOAD INSTRUCTIONS")
    print("="*70)
    for name, cfg in DATASETS.items():
        if "manual" in cfg:
            print(f"\n── {name.upper()} ──")
            print(f"  {cfg['manual']}")
    print()


def verify_datasets(names: list[str]) -> None:
    """Check which required files exist and print a status table."""
    print("\n" + "─"*60)
    print("  DATASET STATUS CHECK")
    print("─"*60)
    for name in names:
        cfg = DATASETS[name]
        all_files = [(Path(dest), desc) for _, dest, desc in cfg.get("files", [])]
        statuses = []
        for dest, desc in all_files:
            if dest.exists() and dest.stat().st_size > 0:
                statuses.append(("✓", desc))
            else:
                statuses.append(("✗", desc))
        ok = all(s == "✓" for s, _ in statuses)
        icon = "✓" if ok else "✗"
        print(f"  [{icon}] {name.upper()}")
        for s, desc in statuses:
            print(f"      [{s}] {desc}")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download SGOD benchmark datasets")
    group = p.add_mutually_exclusive_group()
    group.add_argument(
        "--benchmarks", default=None,
        help=f"Comma-separated list of benchmarks to download. Choices: {', '.join(DATASETS)}"
    )
    group.add_argument("--all", action="store_true", help="Download all benchmarks")
    group.add_argument("--show-manual", action="store_true",
                       help="Print manual download instructions and exit")
    group.add_argument("--verify", action="store_true",
                       help="Check which dataset files already exist")
    p.add_argument("--force", action="store_true",
                   help="Re-download even if files already exist")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    if args.show_manual:
        show_manual_instructions()
        return

    if args.verify:
        verify_datasets(list(DATASETS.keys()))
        return

    if args.all:
        targets = list(DATASETS.keys())
    elif args.benchmarks:
        targets = [b.strip() for b in args.benchmarks.split(",")]
        invalid = [t for t in targets if t not in DATASETS]
        if invalid:
            print(f"Unknown benchmarks: {invalid}. Choose from: {list(DATASETS.keys())}")
            sys.exit(1)
    else:
        print("Specify --benchmarks <name,...> or --all. Use --show-manual for image instructions.")
        sys.exit(1)

    print(f"\nDownloading: {targets}")
    errors = []
    for name in targets:
        try:
            download_dataset(name, force=args.force)
        except Exception as exc:
            errors.append((name, exc))
            logger.error("Failed to download %s: %s", name, exc)

    print("\n" + "="*60)
    verify_datasets(targets)

    if errors:
        print(f"[WARN] {len(errors)} dataset(s) had errors:")
        for name, exc in errors:
            print(f"  {name}: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
