"""
M1.1 Spike — Grounding DINO coverage on MMHal-Bench.

Goal: validate the structural-fix hypothesis BEFORE committing to BG-SGOD.

Question: does open-vocabulary detection actually cover the regression-causing
concepts that RelTR's 151-class vocab misses (e.g. kitchen, indoors, weather,
ice rink, carpet, hat, fire truck, ...)?

For each MMHal image we:
  1. Tokenise the GT answer to a set of "answer concepts"
  2. Run Grounding DINO with EXTENDED_VOCAB
  3. Run RelTR-vocab Grounding DINO too (apples-to-apples coverage of the
     same vocab the current SGOD uses)
  4. Compare: detection recall on answer concepts, novel concepts, etc.

Output: a JSON report with per-image and aggregate stats so we can decide
go/no-go on the structural fix.

Usage (server):
    PYTHONPATH=$(pwd) python scripts/spike_grounding_dino.py \
        --mmhal_path data/mmhal_bench/ \
        --n_images 15 \
        --output outputs/spike_grounding_dino.json
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from pathlib import Path

import torch
from PIL import Image

from sgod.sgg.reltr_utils import ENTITY_CLASSES

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


# ── Vocabulary lists ─────────────────────────────────────────────────────

# RelTR-equivalent vocab (drop the "N/A" sentinel at index 0)
RELTR_VOCAB: list[str] = [c for c in ENTITY_CLASSES if c != "N/A"]

# Concepts that appeared in MMHal regressions but are absent from RelTR.
# Built manually from the 11:40 run loss list — these are the answer tokens
# we observed sgod fail to handle: scenes, locations, weather, vehicle types,
# room types, surfaces, etc.
EXTENDED_ADDITIONS: list[str] = [
    # scenes / locations
    "kitchen", "restaurant", "bedroom", "bathroom", "living room", "office",
    "ice rink", "stadium", "court", "tennis court", "basketball court",
    "beach", "park", "garden", "forest", "field", "road", "highway",
    "carnival", "amusement park", "ferris wheel", "carpet", "floor",
    "ceiling", "wall", "stairs", "staircase",
    # indoor / outdoor cues
    "sky", "cloud", "sun", "moon", "star", "rain", "snowflake",
    "lamp post", "street light", "traffic light",
    # vehicles (specific)
    "fire truck", "pickup truck", "yacht", "sailboat", "speedboat",
    "ambulance", "police car", "taxi", "van", "scooter",
    # electronic devices
    "iphone", "ipod", "tablet", "monitor", "tv", "television", "computer",
    "keyboard", "mouse", "remote control", "speaker",
    # household / kitchen
    "microwave", "oven", "stove", "refrigerator", "dishwasher",
    "knife", "spoon", "kettle", "teapot", "wine glass",
    # food / drink
    "cake", "bread", "toast", "cookie", "donut", "sandwich", "burger",
    "salad", "rice", "noodle", "soup", "egg", "chicken", "fish",
    "wine", "coffee", "tea", "juice", "water bottle",
    # furniture
    "sofa", "couch", "armchair", "rug", "mat", "blanket", "curtain",
    "wardrobe", "dresser", "nightstand",
    # apparel additions
    "tie", "scarf", "glove", "watch", "ring", "necklace", "earring",
    "shorts", "skirt", "dress", "sandals",
    # animals additions
    "lion", "tiger", "deer", "rabbit", "fox", "rhino", "hippo",
    "monkey", "panda", "vicuna",
    # body features
    "beard", "moustache", "glasses", "sunglasses",
    # signage / text artefacts
    "sign", "poster", "billboard", "banner", "logo", "label",
    # toys
    "teddy bear", "doll", "ball", "frisbee",
]

# Final extended vocab = RelTR vocab ∪ additions, deduplicated, lowercase.
EXTENDED_VOCAB: list[str] = sorted(
    set(c.lower() for c in RELTR_VOCAB) | set(EXTENDED_ADDITIONS)
)


# ── Tokenisation ─────────────────────────────────────────────────────────

# Words that aren't useful "concept" answers — articles, copulas, etc.
_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "it", "its", "in", "on", "at", "of",
    "and", "or", "but", "with", "without", "for", "to", "from", "by",
    "as", "if", "then", "than", "so", "which", "what", "who", "where",
    "when", "why", "how", "there", "here", "no", "not", "yes",
    "do", "does", "did", "can", "could", "may", "might", "shall", "should",
    "will", "would", "have", "has", "had", "i", "you", "he", "she", "we", "they",
    "his", "her", "their", "our", "my", "your", "image", "photo", "picture",
    "scene", "background", "foreground", "left", "right", "top", "bottom",
    "middle", "center", "side", "front", "back", "appears", "appear",
    "seems", "seem", "shows", "show", "look", "looks", "see", "view",
    "very", "much", "many", "some", "any", "all", "each", "one", "two",
    "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "first", "second", "third", "next", "also", "well",
})


def tokens(text: str) -> set[str]:
    """Lowercase word tokens minus stopwords."""
    return {w for w in re.findall(r"[a-zA-Z]+", text.lower()) if w not in _STOPWORDS}


# ── Model loading ────────────────────────────────────────────────────────

def load_grounding_dino(model_id: str = "IDEA-Research/grounding-dino-base", device: str = "cuda"):
    """Load Grounding DINO via HuggingFace transformers."""
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    logger.info(f"Loading Grounding DINO {model_id} on {device}...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    model.eval()
    return processor, model


@torch.no_grad()
def detect(processor, model, image: Image.Image, vocab: list[str],
           box_threshold: float = 0.25, text_threshold: float = 0.2,
           device: str = "cuda") -> list[tuple[str, float]]:
    """Run Grounding DINO on `image` over `vocab`. Returns list of (label, conf).

    Grounding DINO requires queries joined by ". " and lowercased.
    Long vocabs (>200 tokens) hit the text-encoder limit; we chunk.
    """
    detections: list[tuple[str, float]] = []
    chunk_size = 80  # heuristic — keeps concatenated text under encoder limit

    for chunk_start in range(0, len(vocab), chunk_size):
        chunk = vocab[chunk_start:chunk_start + chunk_size]
        text = ". ".join(chunk) + "."

        inputs = processor(images=image, text=text, return_tensors="pt").to(device)
        outputs = model(**inputs)
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]

        for label, score in zip(results["labels"], results["scores"]):
            label = label.strip().lower()
            if label:
                detections.append((label, float(score)))

    # Dedup by label, keep max score.
    best: dict[str, float] = {}
    for label, score in detections:
        if label not in best or score > best[label]:
            best[label] = score
    return sorted(best.items(), key=lambda x: -x[1])


# ── MMHal loader ─────────────────────────────────────────────────────────

def load_mmhal(path: Path, n: int) -> list[dict]:
    """Load first N MMHal records. Skips entries without a local image file."""
    data_file = path / "response_template.json"
    images_dir = path / "images"
    records = json.loads(data_file.read_text())
    if not isinstance(records, list):
        raise ValueError(f"unexpected MMHal schema in {data_file}")

    out: list[dict] = []
    for r in records:
        img_id = r.get("image_src", "").split("/")[-1]
        candidate = None
        for ext in (".jpg", ".jpeg", ".png", ""):
            p = images_dir / f"{img_id}{ext}" if ext else images_dir / img_id
            if p.exists():
                candidate = p
                break
        if candidate is None:
            continue
        out.append({**r, "_image_path": str(candidate)})
        if len(out) >= n:
            break
    if not out:
        raise FileNotFoundError(
            f"no MMHal images found under {images_dir} — pull/rsync them first")
    return out


# ── Coverage computation ─────────────────────────────────────────────────

def coverage(detected_labels: set[str], answer_concepts: set[str]) -> dict:
    """Per-image coverage of answer concepts by a detection vocab.

    For each answer concept, mark hit if ANY detected label is contained in
    or contains the concept (so "fire truck" detected covers "truck", and
    "truck" detected covers "truck"). This loose matching is intentional —
    we want to know if the bbox is roughly there, not exact label match.
    """
    hits: set[str] = set()
    for concept in answer_concepts:
        for det in detected_labels:
            if concept == det or concept in det.split() or det in concept.split():
                hits.add(concept)
                break
    return {
        "answer_concepts": sorted(answer_concepts),
        "hits": sorted(hits),
        "missed": sorted(answer_concepts - hits),
        "recall": len(hits) / max(len(answer_concepts), 1),
    }


# ── Main ────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mmhal_path", type=Path, default=Path("data/mmhal_bench"))
    p.add_argument("--n_images", type=int, default=15)
    p.add_argument("--output", type=Path, default=Path("outputs/spike_grounding_dino.json"))
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--model_id", type=str, default="IDEA-Research/grounding-dino-base")
    p.add_argument("--box_threshold", type=float, default=0.25)
    p.add_argument("--text_threshold", type=float, default=0.2)
    args = p.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    records = load_mmhal(args.mmhal_path, args.n_images)
    logger.info(f"Loaded {len(records)} MMHal records with images")

    processor, model = load_grounding_dino(args.model_id, args.device)

    per_image = []
    cum = {
        "reltr_recall_sum": 0.0, "ext_recall_sum": 0.0,
        "novel_concepts_total": 0,
        "n": 0,
    }
    novel_counter: Counter[str] = Counter()

    for i, rec in enumerate(records):
        img = Image.open(rec["_image_path"]).convert("RGB")
        gt = rec["gt_answer"]
        question = rec["question"]
        qtype = rec.get("question_type") or rec.get("type", "?")
        ans_concepts = tokens(gt)

        det_reltr = detect(processor, model, img, RELTR_VOCAB,
                           args.box_threshold, args.text_threshold, args.device)
        det_ext = detect(processor, model, img, EXTENDED_VOCAB,
                         args.box_threshold, args.text_threshold, args.device)
        labels_reltr = {label for label, _ in det_reltr}
        labels_ext = {label for label, _ in det_ext}

        cov_reltr = coverage(labels_reltr, ans_concepts)
        cov_ext = coverage(labels_ext, ans_concepts)

        novel = labels_ext - labels_reltr
        for n in novel:
            novel_counter[n] += 1

        per_image.append({
            "image_id": rec.get("image_id"),
            "type": qtype,
            "question": question,
            "gt_answer": gt,
            "answer_concepts": sorted(ans_concepts),
            "n_detected_reltr": len(labels_reltr),
            "n_detected_extended": len(labels_ext),
            "novel_in_extended": sorted(novel),
            "reltr_recall": round(cov_reltr["recall"], 3),
            "extended_recall": round(cov_ext["recall"], 3),
            "missed_by_reltr": cov_reltr["missed"],
            "missed_by_extended": cov_ext["missed"],
        })
        cum["reltr_recall_sum"] += cov_reltr["recall"]
        cum["ext_recall_sum"] += cov_ext["recall"]
        cum["novel_concepts_total"] += len(novel)
        cum["n"] += 1

        logger.info(
            f"[{i+1:>2}/{len(records)}] {qtype:<12} "
            f"reltr={len(labels_reltr):>2} ({cov_reltr['recall']:.2f})  "
            f"ext={len(labels_ext):>2} ({cov_ext['recall']:.2f})  "
            f"+{len(novel)} novel  Q: {question[:60]}"
        )

    n = cum["n"]
    summary = {
        "n_images": n,
        "model_id": args.model_id,
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
        "vocab_sizes": {"reltr": len(RELTR_VOCAB), "extended": len(EXTENDED_VOCAB)},
        "avg_reltr_recall_on_answer_concepts": round(cum["reltr_recall_sum"] / n, 3),
        "avg_extended_recall_on_answer_concepts": round(cum["ext_recall_sum"] / n, 3),
        "avg_novel_concepts_per_image": round(cum["novel_concepts_total"] / n, 2),
        "top_novel_concepts": novel_counter.most_common(20),
    }
    output = {"summary": summary, "per_image": per_image}
    args.output.write_text(json.dumps(output, indent=2))
    logger.info("\n=== SUMMARY ===")
    logger.info(json.dumps(summary, indent=2))
    logger.info(f"\nFull report: {args.output}")


if __name__ == "__main__":
    main()
