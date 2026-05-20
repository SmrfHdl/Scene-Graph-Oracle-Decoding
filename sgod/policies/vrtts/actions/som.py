"""AddSoMMarks — Set-of-Mark visual prompting via Grounding DINO.

Procedure:
  1. Parse the question to extract subject phrase X and object phrase Y.
  2. Run Grounding DINO open-vocab queries on X and Y separately to get
     bounding boxes.
  3. Overlay numbered marks ("1" near X, "2" near Y) on a copy of the image
     with high-contrast outline and fill.
  4. Augment the prompt suffix so the LM knows what each mark refers to.

This is the visual analogue of Set-of-Mark (Yang+ 2023) restricted to the
two entities in the relation. We do NOT mark every detected object — that
clutters the image and confuses the LM. Two-mark is the minimum sufficient
for verifying a binary relation.
"""
from __future__ import annotations

import re
from typing import Any

import torch
from PIL import Image, ImageDraw, ImageFont

from sgod.policies.vrtts.actions.base import VisualAction, VisualState


# Reuse the question-parsing logic developed and validated in the spatial-
# verify MVPs. Inline a minimal version to avoid coupling to a script.

_STOPWORDS = {"the", "a", "an", "some", "this", "that", "these", "those", "of"}


def _strip_question(q: str) -> str:
    t = q
    for trailing in (
        " Please answer yes or no.", "Please answer yes or no.",
        " in this photo?", "in this photo?", "?", "."
    ):
        t = t.replace(trailing, "")
    t = t.strip()
    for prefix in ("Is the ", "Are the ", "Is a ", "Is ", "Are "):
        if t.startswith(prefix):
            t = t[len(prefix):]
            break
    return t.strip()


_RELATION_KEYS = [
    "on top of", "in front of", "to the left of", "to the right of",
    "outside the", "inside the", "on left of", "on right of",
    "far from", "next to", "topping", "above", "below", "under",
    "outside", "inside", "around", "between", "behind", "beside",
    "near", "with", "from", "across", "in", "on", "at", "by", "of",
    "to",
]
# Sort longest-first for greedy match.
_RELATION_KEYS_SORTED = sorted(set(_RELATION_KEYS), key=len, reverse=True)


def _parse_xy(question: str) -> tuple[str | None, str | None]:
    """Extract (X, Y) phrases from the question — best-effort."""
    t = _strip_question(question)
    if not t:
        return None, None
    for rel in _RELATION_KEYS_SORTED:
        m = re.search(rf"(?:^|\s){re.escape(rel)}(?:\s|$)", t)
        if m:
            x = t[:m.start()].strip()
            y = t[m.end():].strip()
            return (x or None), (y or None)
    # Fallback: split on first -ing word
    m = re.search(r"\b(\w+ing)\b", t)
    if m:
        return (t[:m.start()].strip() or None), (t[m.end():].strip() or None)
    return t, None


def _strip_articles(p: str) -> str:
    p = p.lower().strip()
    p = re.sub(r"[^\w\s]", " ", p)
    words = [w for w in p.split() if w not in _STOPWORDS and len(w) > 1]
    return " ".join(words) or p


class AddSoMMarks(VisualAction):
    """Detect the two entities in the question and overlay numbered marks.

    Requires `oracle` kwarg in `apply` — an object exposing GD detection.
    For Phase 1 we accept a duck-typed object with `.processor`, `.model`,
    and `.device`. Concretely we plug in a thin wrapper around HF's
    AutoModelForZeroShotObjectDetection (built in eval scripts).

    Args:
        box_threshold:   GD detection confidence threshold.
        text_threshold:  GD text-grounding threshold.
        mark_radius:     pixel radius of the circular mark background.
        mark_font_size:  font size for the numeral.
    """

    name = "som"

    def __init__(
        self,
        box_threshold: float = 0.25,
        text_threshold: float = 0.20,
        mark_radius: int = 20,
        mark_font_size: int = 24,
    ) -> None:
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.mark_radius = mark_radius
        self.mark_font_size = mark_font_size

    def _gd_query(
        self,
        image: Image.Image,
        phrase: str,
        oracle: Any,
    ) -> tuple[float, float] | None:
        """Run GD on a single phrase; return bbox-centre or None."""
        norm = _strip_articles(phrase)
        if not norm:
            return None
        text = f"{norm}."
        inputs = oracle.processor(
            images=image, text=text, return_tensors="pt"
        ).to(oracle.device)
        with torch.no_grad():
            outputs = oracle.model(**inputs)
        results = oracle.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]
        scores = results.get("scores")
        boxes = results.get("boxes")
        if scores is None or len(scores) == 0:
            return None
        best = int(scores.argmax().item())
        x1, y1, x2, y2 = (float(v) for v in boxes[best].tolist())
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def _draw_mark(
        self,
        image: Image.Image,
        centre: tuple[float, float],
        label: str,
    ) -> None:
        draw = ImageDraw.Draw(image)
        cx, cy = centre
        r = self.mark_radius
        # White-outlined red circle for high contrast on most backgrounds.
        draw.ellipse(
            (cx - r, cy - r, cx + r, cy + r),
            fill=(255, 0, 0),
            outline=(255, 255, 255),
            width=3,
        )
        # Text centred — try to load a default font, fall back to bitmap default.
        try:
            font = ImageFont.load_default(size=self.mark_font_size)
        except (TypeError, AttributeError):
            font = ImageFont.load_default()
        # Approximate centring using textbbox if available.
        try:
            tb = draw.textbbox((0, 0), label, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
            draw.text((cx - tw / 2, cy - th / 2), label, fill=(255, 255, 255), font=font)
        except AttributeError:
            draw.text((cx - 6, cy - 8), label, fill=(255, 255, 255), font=font)

    def apply(
        self,
        state: VisualState,
        *,
        attentions: Any | None = None,
        question: str | None = None,
        backbone: Any | None = None,
        oracle: Any | None = None,
    ) -> VisualState:
        if oracle is None:
            raise ValueError("AddSoMMarks requires an oracle (GD processor + model).")
        if question is None:
            raise ValueError("AddSoMMarks requires the question.")

        x_phrase, y_phrase = _parse_xy(question)
        new_state = state.clone()
        annotated = state.image.copy()

        marks: list[tuple[str, str]] = []
        if x_phrase:
            x_centre = self._gd_query(state.image, x_phrase, oracle)
            if x_centre is not None:
                self._draw_mark(annotated, x_centre, "1")
                marks.append(("1", x_phrase))
        if y_phrase:
            y_centre = self._gd_query(state.image, y_phrase, oracle)
            if y_centre is not None:
                self._draw_mark(annotated, y_centre, "2")
                marks.append(("2", y_phrase))

        new_state.image = annotated
        if marks:
            mark_desc = "; ".join(f"mark '{m}' is the {p}" for m, p in marks)
            new_state.prompt_suffix = (
                f" The image has been annotated: {mark_desc}. Use the marks to "
                "decide your answer."
            )
        new_state.metadata["som_marks"] = marks
        return new_state
