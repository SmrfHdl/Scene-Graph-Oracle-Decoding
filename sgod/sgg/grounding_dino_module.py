"""
GroundingDinoModule — open-vocabulary object detection (M1.2 of BG-SGOD).

Wraps HuggingFace's Grounding DINO so we can detect objects against an arbitrary
text vocabulary instead of being locked to RelTR's 151 entity classes.

This module is the "objects" half of the hybrid scene graph: M1.3 fuses these
detections with RelTR's relation triplets to form a SceneGraph that has open-
vocab objects + closed-vocab relations.

Interface:
    gd = GroundingDinoModule(vocab=["dog", "table", "kitchen", ...])
    objects: list[ObjectNode] = gd.detect(image)
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
from PIL import Image

from sgod.sgg.scene_graph import ObjectNode

logger = logging.getLogger(__name__)


# ── Default vocabulary ──────────────────────────────────────────────────

# Concepts beyond RelTR's 151 classes that appear in MMHal/POPE answers
# (locations, scenes, weather, electronics, food, etc.). Keep this list
# focused — Grounding DINO's text encoder has a hard token budget so
# very long vocabs trigger chunking and slower inference.
_DEFAULT_EXTRA: list[str] = [
    # scenes / locations
    "kitchen", "restaurant", "bedroom", "bathroom", "living room", "office",
    "ice rink", "stadium", "tennis court", "beach", "park", "carpet", "floor",
    "ceiling", "wall", "stairs", "ferris wheel",
    # outdoor cues
    "sky", "cloud", "sun", "moon", "rain",
    # vehicles
    "fire truck", "yacht", "sailboat", "ambulance", "police car", "taxi",
    # electronics
    "iphone", "ipod", "tablet", "monitor", "tv", "computer", "keyboard",
    # household
    "microwave", "oven", "stove", "refrigerator", "knife", "spoon",
    # food
    "cake", "bread", "toast", "donut", "sandwich", "rice", "egg", "wine",
    # furniture
    "sofa", "couch", "rug", "wardrobe", "dresser",
    # apparel
    "scarf", "watch", "ring", "shorts", "skirt", "dress", "sandals",
    "glasses", "sunglasses",
    # animals
    "lion", "tiger", "deer", "rabbit", "fox", "rhino", "monkey", "panda",
    # toys / misc
    "teddy bear", "doll", "frisbee", "billboard", "poster",
]


def default_vocabulary() -> list[str]:
    """RelTR-equivalent classes ∪ extra MMHal-coverage concepts (lowercased)."""
    from sgod.sgg.reltr_utils import ENTITY_CLASSES

    base = [c.lower() for c in ENTITY_CLASSES if c != "N/A"]
    return sorted(set(base) | set(_DEFAULT_EXTRA))


class GroundingDinoModule:
    """Open-vocabulary object detector wrapping HuggingFace's Grounding DINO.

    Designed to be called once per image (mirrors SGGModule.extract()).
    Returns a list of ObjectNode the same way SGGModule does, so the rest
    of the pipeline can consume either source uniformly.

    Usage:
        gd = GroundingDinoModule(vocab=default_vocabulary())
        objects = gd.detect(image)
    """

    # Grounding DINO's text encoder caps at ~256 tokens; chunk to stay safe.
    _CHUNK_SIZE = 80

    def __init__(
        self,
        vocab: Optional[list[str]] = None,
        model_id: str = "IDEA-Research/grounding-dino-base",
        device: Optional[str] = None,
        box_threshold: float = 0.25,
        text_threshold: float = 0.2,
        max_objects: int = 30,
    ):
        """
        Args:
            vocab:           Open-vocabulary text queries. Defaults to RelTR ∪ extras.
            model_id:        HuggingFace model id.
            device:          "cuda" / "cpu". Auto-detects.
            box_threshold:   Detection confidence threshold (boxes).
            text_threshold:  Text-alignment threshold.
            max_objects:     Cap detections per image (top-confidence first).
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.max_objects = max_objects
        self.vocab: list[str] = vocab if vocab is not None else default_vocabulary()

        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        logger.info(f"Loading Grounding DINO {model_id} on {device}...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
        self.model.eval()

        # Pre-build chunked text prompts (Grounding DINO expects ". "-joined queries)
        self._chunks: list[str] = [
            ". ".join(self.vocab[i:i + self._CHUNK_SIZE]) + "."
            for i in range(0, len(self.vocab), self._CHUNK_SIZE)
        ]
        # Lowercase vocab as a set for O(1) label-membership tests during fusion split.
        self._vocab_set: set[str] = {v.lower() for v in self.vocab}

    def _split_label(self, label: str) -> list[str]:
        """Split a Grounding DINO label into vocab terms.

        GD's processor sometimes returns a single ``label`` that is the
        concatenation of *adjacent* vocab queries (e.g. our ``". "``-joined
        prompt yields detections labelled ``"road street"`` or
        ``"armchair bench chair"`` when the model attends to a span covering
        multiple terms). We want each underlying vocab term back so that
        downstream target matching (``match_targets_to_vocab``) can hit them
        individually.

        Resolution order:
          1. label is exactly a vocab entry (incl. multi-word like "fire truck")
             → keep as one term.
          2. else, split on whitespace and keep tokens that are vocab entries;
             also try to recover multi-word vocab entries that span 2+ tokens
             (so "fire truck van" yields ["fire truck", "van"]).
          3. fallback: return the original label so we never lose a detection.
        """
        lab = label.strip().lower()
        if not lab:
            return []
        if lab in self._vocab_set:
            return [lab]

        words = lab.split()
        terms: list[str] = []
        i = 0
        # Greedy longest-match against vocab so multi-word entries win over their tokens.
        while i < len(words):
            matched = False
            for span in range(min(len(words) - i, 4), 0, -1):
                cand = " ".join(words[i:i + span])
                if cand in self._vocab_set:
                    terms.append(cand)
                    i += span
                    matched = True
                    break
            if not matched:
                i += 1  # token isn't a vocab entry — skip it
        if terms:
            return terms
        return [lab]  # nothing matched — preserve original

    @torch.no_grad()
    def detect(self, image: Image.Image) -> list[ObjectNode]:
        """Detect objects in `image` over the configured vocabulary.

        Returns:
            List of ObjectNode (label, confidence, bbox), deduplicated by
            label (max-conf retained per label) and capped at max_objects.
        """
        target_size = [image.size[::-1]]  # (H, W) for HF post-processing
        all_dets: list[ObjectNode] = []

        for text in self._chunks:
            inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=target_size,
            )[0]

            for box, score, label in zip(
                results["boxes"], results["scores"], results["labels"]
            ):
                terms = self._split_label(label)
                if not terms:
                    continue
                x1, y1, x2, y2 = (float(v) for v in box.tolist())
                # Fused-label split: emit one ObjectNode per recovered vocab
                # term, all sharing the same bbox/score.
                for term in terms:
                    all_dets.append(ObjectNode(
                        label=term,
                        confidence=float(score),
                        bbox=(x1, y1, x2, y2),
                    ))

        # Dedup by (label, rounded-bbox) — Grounding DINO sometimes returns
        # near-duplicate boxes for the same concept. Keep highest confidence.
        seen: dict[tuple, ObjectNode] = {}
        for obj in all_dets:
            key = (obj.label, round(obj.bbox[0]), round(obj.bbox[1]),
                   round(obj.bbox[2]), round(obj.bbox[3]))
            if key not in seen or obj.confidence > seen[key].confidence:
                seen[key] = obj
        deduped = sorted(seen.values(), key=lambda o: -o.confidence)
        return deduped[:self.max_objects]
