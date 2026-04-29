"""
SceneGraph data structure for SGOD.

Defines the structured representation extracted by the SGG Module,
consumed by the Visual Oracle (Component 2).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ObjectNode:
    """A detected object in the scene graph."""
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2) in image coords

    def __repr__(self) -> str:
        return f"{self.label}({self.confidence:.2f})"


@dataclass
class RelationEdge:
    """A relationship between two objects."""
    subject: str
    predicate: str
    object: str
    confidence: float  # Combined confidence = sub_conf * rel_conf * obj_conf
    subject_bbox: tuple[float, float, float, float] = (0, 0, 0, 0)
    object_bbox: tuple[float, float, float, float] = (0, 0, 0, 0)

    def __repr__(self) -> str:
        return f"({self.subject} --{self.predicate}--> {self.object}, {self.confidence:.2f})"


@dataclass
class AttributeNode:
    """An attribute of an entity.

    Note: RelTR does not predict attributes. This is populated by the
    Visual Oracle (Component 2) using CLIP zero-shot classification.
    Kept here for interface completeness.
    """
    entity: str
    attribute: str
    confidence: float

    def __repr__(self) -> str:
        return f"{self.entity}:{self.attribute}({self.confidence:.2f})"


@dataclass
class SceneGraph:
    """Structured scene representation extracted from an image.

    Contains objects, relations, and attributes with confidence scores.
    Produced by SGGModule.extract() and consumed by VisualOracle.
    """
    objects: list[ObjectNode] = field(default_factory=list)
    relations: list[RelationEdge] = field(default_factory=list)
    attributes: list[AttributeNode] = field(default_factory=list)
    image_size: Optional[tuple[int, int]] = None  # (width, height)

    # ── Factory Methods ──────────────────────────────────────────────

    @classmethod
    def from_reltr_output(
        cls,
        sub_labels: list[str],
        sub_confs: list[float],
        sub_bboxes: list[tuple[float, float, float, float]],
        obj_labels: list[str],
        obj_confs: list[float],
        obj_bboxes: list[tuple[float, float, float, float]],
        rel_labels: list[str],
        rel_confs: list[float],
        image_size: Optional[tuple[int, int]] = None,
    ) -> SceneGraph:
        """Build SceneGraph from post-processed RelTR output.

        Args:
            sub_labels: Subject class names for each triplet
            sub_confs: Subject confidences
            sub_bboxes: Subject bounding boxes (x1,y1,x2,y2)
            obj_labels: Object class names for each triplet
            obj_confs: Object confidences
            obj_bboxes: Object bounding boxes (x1,y1,x2,y2)
            rel_labels: Predicate class names
            rel_confs: Predicate confidences
            image_size: (width, height) of the original image
        """
        # Build unique object set from subjects and objects
        seen_objects: dict[str, ObjectNode] = {}
        for label, conf, bbox in zip(sub_labels, sub_confs, sub_bboxes):
            key = f"{label}_{bbox[0]:.0f}_{bbox[1]:.0f}"
            if key not in seen_objects or conf > seen_objects[key].confidence:
                seen_objects[key] = ObjectNode(label=label, confidence=conf, bbox=bbox)
        for label, conf, bbox in zip(obj_labels, obj_confs, obj_bboxes):
            key = f"{label}_{bbox[0]:.0f}_{bbox[1]:.0f}"
            if key not in seen_objects or conf > seen_objects[key].confidence:
                seen_objects[key] = ObjectNode(label=label, confidence=conf, bbox=bbox)

        objects = sorted(seen_objects.values(), key=lambda o: o.confidence, reverse=True)

        # Build relations
        relations = []
        for s_label, s_conf, s_bbox, o_label, o_conf, o_bbox, r_label, r_conf in zip(
            sub_labels, sub_confs, sub_bboxes,
            obj_labels, obj_confs, obj_bboxes,
            rel_labels, rel_confs,
        ):
            combined_conf = s_conf * r_conf * o_conf
            relations.append(RelationEdge(
                subject=s_label,
                predicate=r_label,
                object=o_label,
                confidence=combined_conf,
                subject_bbox=s_bbox,
                object_bbox=o_bbox,
            ))

        relations.sort(key=lambda r: r.confidence, reverse=True)

        return cls(
            objects=objects,
            relations=relations,
            attributes=[],  # RelTR does not predict attributes
            image_size=image_size,
        )

    @classmethod
    def from_hybrid(
        cls,
        gd_objects: list[ObjectNode],
        reltr_relations: list[RelationEdge],
        image_size: Optional[tuple[int, int]] = None,
    ) -> SceneGraph:
        """Build a hybrid SceneGraph: open-vocab objects + RelTR relations.

        Used by BG-SGOD: Grounding DINO supplies the object set (open-vocab,
        bbox-grounded), and RelTR supplies the predicate triplets (closed-vocab
        but well-trained). The two come from independent passes — relation
        subject/object labels stay as RelTR predicted them, since the relation
        scoring path keys on predicates rather than subject identity.

        Args:
            gd_objects:       Open-vocab detections from GroundingDinoModule.
            reltr_relations:  Relation triplets from RelTR (closed predicate vocab).
            image_size:       (width, height) of the original image.
        """
        objects = sorted(gd_objects, key=lambda o: o.confidence, reverse=True)
        relations = sorted(reltr_relations, key=lambda r: r.confidence, reverse=True)
        return cls(
            objects=objects,
            relations=relations,
            attributes=[],
            image_size=image_size,
        )

    # ── Oracle Activation ────────────────────────────────────────────

    def should_activate_oracle(self, min_confidence: float = 0.4) -> bool:
        """Check if scene graph quality is sufficient for oracle use.

        Returns False if:
        - No objects detected
        - Average object confidence below threshold

        Section 4.5: Confidence-based oracle deactivation for abstract/art images.
        """
        if len(self.objects) == 0:
            return False
        avg_conf = sum(o.confidence for o in self.objects) / len(self.objects)
        return avg_conf > min_confidence

    # ── Text Representations ─────────────────────────────────────────

    def to_text(self) -> str:
        """Human-readable text representation for logging and prompt injection."""
        lines = []
        if self.objects:
            obj_strs = [f"{o.label}({o.confidence:.2f})" for o in self.objects]
            lines.append(f"Objects: {', '.join(obj_strs)}")
        if self.relations:
            rel_strs = [
                f"{r.subject} {r.predicate} {r.object} ({r.confidence:.2f})"
                for r in self.relations
            ]
            lines.append(f"Relations: {'; '.join(rel_strs)}")
        if self.attributes:
            attr_strs = [f"{a.entity}:{a.attribute}({a.confidence:.2f})" for a in self.attributes]
            lines.append(f"Attributes: {', '.join(attr_strs)}")
        if not lines:
            return "Empty scene graph"
        return "\n".join(lines)

    def to_prompt_prefix(self) -> str:
        """Compact text for use as VLM prompt prefix (pilot experiment)."""
        parts = []
        if self.objects:
            parts.append("Objects: " + ", ".join(o.label for o in self.objects))
        if self.relations:
            parts.append("Relations: " + "; ".join(
                f"{r.subject} {r.predicate} {r.object}" for r in self.relations))
        return ". ".join(parts)

    # ── Statistics ────────────────────────────────────────────────────

    def summary(self) -> dict:
        """Summary statistics for analysis and logging."""
        obj_confs = [o.confidence for o in self.objects]
        rel_confs = [r.confidence for r in self.relations]
        return {
            "num_objects": len(self.objects),
            "num_relations": len(self.relations),
            "num_attributes": len(self.attributes),
            "avg_obj_confidence": sum(obj_confs) / len(obj_confs) if obj_confs else 0.0,
            "avg_rel_confidence": sum(rel_confs) / len(rel_confs) if rel_confs else 0.0,
            "oracle_active": self.should_activate_oracle(),
            "unique_object_labels": list({o.label for o in self.objects}),
            "unique_predicates": list({r.predicate for r in self.relations}),
        }

    def __len__(self) -> int:
        return len(self.objects) + len(self.relations) + len(self.attributes)

    def __bool__(self) -> bool:
        return len(self.objects) > 0
