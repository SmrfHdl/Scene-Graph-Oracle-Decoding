"""Tests for the framework-level Oracle implementations (sgod.oracles).

CPU-only: underlying RelTR / Grounding DINO modules are mocked via dependency
injection (the wrappers accept a pre-built module to bypass lazy-load).

Coverage:
  - Registry: both wrappers register under "oracle"/<name>.
  - Validation: RelTROracle requires checkpoint_path OR sgg_module.
  - extract() returns OracleEvidence carrying the SceneGraph.
  - GroundingDinoOracle drops relations (object-only by design).
"""
from __future__ import annotations

import pytest

from sgod.core.registry import build, list_registered
from sgod.core.types import OracleEvidence
from sgod.oracles import GroundingDinoOracle, RelTROracle
from sgod.sgg.scene_graph import ObjectNode, RelationEdge, SceneGraph


def _sg_with_objects_and_relations() -> SceneGraph:
    return SceneGraph(
        objects=[
            ObjectNode("dog", 0.9, (0, 0, 1, 1)),
            ObjectNode("mat", 0.8, (1, 1, 2, 2)),
        ],
        relations=[RelationEdge("dog", "on", "mat", 0.8)],
        attributes=[],
        image_size=(640, 480),
    )


# ── Registry ─────────────────────────────────────────────────────────────────

def test_oracles_registered():
    names = list_registered("oracle")
    assert "reltr" in names
    assert "grounding-dino" in names


def test_reltr_oracle_build_via_registry():
    class FakeSGG:
        def extract(self, image):
            return _sg_with_objects_and_relations()

    oracle = build("oracle", "reltr", sgg_module=FakeSGG())
    assert isinstance(oracle, RelTROracle)


def test_grounding_dino_oracle_build_via_registry():
    class FakeGD:
        def detect(self, image):
            return [ObjectNode("teddy bear", 0.7, (0, 0, 1, 1))]

    oracle = build("oracle", "grounding-dino", gd_module=FakeGD())
    assert isinstance(oracle, GroundingDinoOracle)


# ── RelTROracle ──────────────────────────────────────────────────────────────

def test_reltr_oracle_requires_checkpoint_or_module():
    with pytest.raises(ValueError, match="checkpoint_path"):
        RelTROracle()


def test_reltr_oracle_extract_returns_evidence_carrying_scene_graph():
    sg = _sg_with_objects_and_relations()

    class FakeSGG:
        def extract(self, image):
            self.last_image = image
            return sg

    fake = FakeSGG()
    oracle = RelTROracle(sgg_module=fake)
    sentinel = object()
    evidence = oracle.extract(sentinel)

    assert isinstance(evidence, OracleEvidence)
    assert evidence.scene_graph is sg
    assert fake.last_image is sentinel
    # SGG produces relations; RelTROracle preserves them.
    assert len(evidence.scene_graph.relations) == 1


def test_reltr_oracle_vocab_scores_empty_stub():
    class FakeSGG:
        def extract(self, image):
            return _sg_with_objects_and_relations()

    oracle = RelTROracle(sgg_module=FakeSGG())
    evidence = oracle.extract(object())
    # Empty by design — see docstring.
    assert oracle.vocab_scores(tokenizer=None, evidence=evidence) == {}


def test_reltr_oracle_image_meta_ignored():
    """`image_meta` is accepted for interface compatibility but must not error."""
    class FakeSGG:
        def extract(self, image):
            return _sg_with_objects_and_relations()

    oracle = RelTROracle(sgg_module=FakeSGG())
    evidence = oracle.extract(object(), image_meta={"width": 640, "height": 480})
    assert isinstance(evidence, OracleEvidence)


def test_reltr_oracle_does_not_load_module_until_extract():
    """Lazy-load contract: constructing without `sgg_module` and without calling
    extract must not import/load RelTR. We verify by passing a bogus checkpoint
    path that would crash on actual load."""
    oracle = RelTROracle(checkpoint_path="/path/does/not/exist.pth")
    # If construction tried to load, it would raise here.
    assert oracle._sgg is None


# ── GroundingDinoOracle ──────────────────────────────────────────────────────

def test_grounding_dino_oracle_returns_objects_only():
    """GD oracle produces a SceneGraph with relations=[] by design."""
    expected_objs = [
        ObjectNode("teddy bear", 0.85, (0, 0, 1, 1)),
        ObjectNode("kitchen", 0.6, (2, 2, 3, 3)),
    ]

    class FakeGD:
        def detect(self, image):
            return expected_objs

    oracle = GroundingDinoOracle(gd_module=FakeGD())
    evidence = oracle.extract(object())

    assert isinstance(evidence, OracleEvidence)
    sg = evidence.scene_graph
    assert list(sg.objects) == expected_objs
    assert sg.relations == []
    assert sg.attributes == []


def test_grounding_dino_oracle_propagates_image_size_when_available():
    class FakeGD:
        def detect(self, image):
            return []

    class FakeImage:
        size = (224, 224)

    oracle = GroundingDinoOracle(gd_module=FakeGD())
    evidence = oracle.extract(FakeImage())
    assert evidence.scene_graph.image_size == (224, 224)


def test_grounding_dino_oracle_image_size_falls_back_to_none():
    """Plain `object()` has no `.size`; image_size should be None (not crash)."""
    class FakeGD:
        def detect(self, image):
            return []

    oracle = GroundingDinoOracle(gd_module=FakeGD())
    evidence = oracle.extract(object())
    assert evidence.scene_graph.image_size is None


def test_grounding_dino_oracle_does_not_load_until_extract():
    oracle = GroundingDinoOracle(vocab=["dog"])  # no gd_module passed
    assert oracle._gd is None  # lazy
