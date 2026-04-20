"""
RelTR utility functions for preprocessing, postprocessing, and model construction.

Bridges between PIL images and the vendored RelTR model.
"""
from __future__ import annotations

import argparse
from typing import Any

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from .reltr import build_model
from .reltr.box_ops import box_cxcywh_to_xyxy

# ── Visual Genome Class Labels ────────────────────────────────────────
# 151 entity classes (index 0 = "N/A" = background)
ENTITY_CLASSES = [
    "N/A", "airplane", "animal", "arm", "bag", "banana", "basket", "beach",
    "bear", "bed", "bench", "bike", "bird", "board", "boat", "book", "boot",
    "bottle", "bowl", "box", "boy", "branch", "building", "bus", "cabinet",
    "cap", "car", "cat", "chair", "child", "clock", "coat", "counter", "cow",
    "cup", "curtain", "desk", "dog", "door", "drawer", "ear", "elephant",
    "engine", "eye", "face", "fence", "finger", "flag", "flower", "food",
    "fork", "fruit", "giraffe", "girl", "glass", "glove", "guy", "hair",
    "hand", "handle", "hat", "head", "helmet", "hill", "horse", "house",
    "jacket", "jean", "kid", "kite", "lady", "lamp", "laptop", "leaf", "leg",
    "letter", "light", "logo", "man", "men", "motorcycle", "mountain", "mouth",
    "neck", "nose", "number", "orange", "pant", "paper", "paw", "people",
    "person", "phone", "pillow", "pizza", "plane", "plant", "plate", "player",
    "pole", "post", "pot", "racket", "railing", "rock", "roof", "room",
    "screen", "seat", "sheep", "shelf", "shirt", "shoe", "short", "sidewalk",
    "sign", "sink", "skateboard", "ski", "skier", "sneaker", "snow", "sock",
    "stand", "street", "surfboard", "table", "tail", "tie", "tile", "tire",
    "toilet", "towel", "tower", "track", "train", "tree", "truck", "trunk",
    "umbrella", "vase", "vegetable", "vehicle", "wave", "wheel", "window",
    "windshield", "wing", "wire", "woman", "zebra",
]

# 51 relation classes (index 0 = "__background__")
RELATION_CLASSES = [
    "__background__", "above", "across", "against", "along", "and", "at",
    "attached to", "behind", "belonging to", "between", "carrying",
    "covered in", "covering", "eating", "flying in", "for", "from",
    "growing on", "hanging from", "has", "holding", "in", "in front of",
    "laying on", "looking at", "lying on", "made of", "mounted on", "near",
    "of", "on", "on back of", "over", "painted on", "parked on", "part of",
    "playing", "riding", "says", "sitting on", "standing on", "to", "under",
    "using", "walking in", "walking on", "watching", "wearing", "wears", "with",
]

# ── Image Preprocessing ──────────────────────────────────────────────

# Standard ImageNet normalization used by RelTR
RELTR_TRANSFORM = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocess a PIL Image for RelTR inference.

    Resizes to 800px on the shorter side, normalizes with ImageNet stats.
    Returns tensor of shape [1, 3, H, W].
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    return RELTR_TRANSFORM(image).unsqueeze(0)


# ── Model Construction ────────────────────────────────────────────────

def get_reltr_default_args() -> argparse.Namespace:
    """Get default args for building a RelTR model (VG dataset)."""
    return argparse.Namespace(
        # Backbone
        backbone="resnet50",
        dilation=False,
        position_embedding="sine",
        lr_backbone=0,  # Frozen backbone for inference
        return_interm_layers=False,
        # Transformer
        enc_layers=6,
        dec_layers=6,
        dim_feedforward=2048,
        hidden_dim=256,
        dropout=0.1,
        nheads=8,
        num_entities=100,
        num_triplets=200,
        pre_norm=False,
        # Loss (not used at inference but needed for build)
        aux_loss=True,
        # Dataset
        dataset="vg",
        # Device
        device="cpu",
    )


def load_reltr_model(
    checkpoint_path: str,
    device: str = "cpu",
) -> torch.nn.Module:
    """Load pre-trained RelTR model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file (e.g., checkpoint0149.pth)
        device: Target device ('cpu', 'cuda', 'cuda:0', etc.)

    Returns:
        RelTR model in eval mode on the specified device
    """
    args = get_reltr_default_args()
    args.device = device

    model = build_model(args)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle both 'model' key and direct state_dict
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model


# ── Output Post-processing ───────────────────────────────────────────

def postprocess_reltr_output(
    outputs: dict[str, torch.Tensor],
    image_size: tuple[int, int],
    confidence_threshold: float = 0.3,
    top_k: int = 20,
) -> dict[str, Any]:
    """Post-process raw RelTR model output into structured predictions.

    Args:
        outputs: Raw model output dict with keys:
            sub_logits, obj_logits, rel_logits,
            sub_boxes, obj_boxes (in [0,1] normalized coords)
        image_size: (width, height) of the original image
        confidence_threshold: Minimum confidence for each component (sub/obj/rel)
        top_k: Maximum number of triplets to return

    Returns:
        Dict with keys: sub_labels, sub_confs, sub_bboxes,
            obj_labels, obj_confs, obj_bboxes, rel_labels, rel_confs
    """
    # Softmax probabilities (exclude background/no-object class at index -1)
    probas_sub = outputs["sub_logits"].softmax(-1)[0, :, :-1]  # [num_triplets, num_classes]
    probas_obj = outputs["obj_logits"].softmax(-1)[0, :, :-1]
    probas_rel = outputs["rel_logits"].softmax(-1)[0, :, :-1]

    # Filter: keep only triplets where all three components pass threshold
    keep = torch.logical_and(
        probas_rel.max(-1).values > confidence_threshold,
        torch.logical_and(
            probas_sub.max(-1).values > confidence_threshold,
            probas_obj.max(-1).values > confidence_threshold,
        ),
    )

    if not keep.any():
        return _empty_result()

    # Get indices of kept triplets, sorted by combined confidence
    keep_indices = torch.nonzero(keep, as_tuple=True)[0]
    combined_scores = (
        probas_sub[keep_indices].max(-1)[0]
        * probas_rel[keep_indices].max(-1)[0]
        * probas_obj[keep_indices].max(-1)[0]
    )
    sorted_indices = torch.argsort(-combined_scores)[:top_k]
    keep_indices = keep_indices[sorted_indices]

    # Rescale bounding boxes from [0,1] to image coordinates
    img_w, img_h = image_size
    scale = torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32,
                         device=outputs["sub_boxes"].device)

    sub_boxes = box_cxcywh_to_xyxy(outputs["sub_boxes"][0, keep_indices]) * scale
    obj_boxes = box_cxcywh_to_xyxy(outputs["obj_boxes"][0, keep_indices]) * scale

    # Convert to labels
    sub_class_ids = probas_sub[keep_indices].argmax(-1)
    obj_class_ids = probas_obj[keep_indices].argmax(-1)
    rel_class_ids = probas_rel[keep_indices].argmax(-1)

    sub_confs = probas_sub[keep_indices].max(-1).values
    obj_confs = probas_obj[keep_indices].max(-1).values
    rel_confs = probas_rel[keep_indices].max(-1).values

    return {
        "sub_labels": [ENTITY_CLASSES[idx] for idx in sub_class_ids.tolist()],
        "sub_confs": sub_confs.tolist(),
        "sub_bboxes": [tuple(b.tolist()) for b in sub_boxes],
        "obj_labels": [ENTITY_CLASSES[idx] for idx in obj_class_ids.tolist()],
        "obj_confs": obj_confs.tolist(),
        "obj_bboxes": [tuple(b.tolist()) for b in obj_boxes],
        "rel_labels": [RELATION_CLASSES[idx] for idx in rel_class_ids.tolist()],
        "rel_confs": rel_confs.tolist(),
    }


def _empty_result() -> dict[str, Any]:
    """Return an empty result dict when no predictions pass the threshold."""
    return {
        "sub_labels": [],
        "sub_confs": [],
        "sub_bboxes": [],
        "obj_labels": [],
        "obj_confs": [],
        "obj_bboxes": [],
        "rel_labels": [],
        "rel_confs": [],
    }
