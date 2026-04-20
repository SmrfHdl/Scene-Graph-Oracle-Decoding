"""
Tests for SGG Module — SceneGraph data structure, reltr_utils, and SGGModule.

Unit tests: no GPU or checkpoint needed.
Integration tests: require GPU + checkpoint (marked with @pytest.mark.integration).
"""
import pytest
import torch
from PIL import Image

from sgod.sgg.scene_graph import SceneGraph, ObjectNode, RelationEdge, AttributeNode
from sgod.sgg.reltr_utils import (
    ENTITY_CLASSES,
    RELATION_CLASSES,
    preprocess_image,
    postprocess_reltr_output,
    get_reltr_default_args,
)


# ════════════════════════════════════════════════════════════════════
# SceneGraph Data Structure Tests
# ════════════════════════════════════════════════════════════════════

class TestSceneGraphCreation:
    """Test SceneGraph creation from raw data."""

    def test_empty_scene_graph(self):
        sg = SceneGraph()
        assert len(sg.objects) == 0
        assert len(sg.relations) == 0
        assert len(sg.attributes) == 0
        assert len(sg) == 0
        assert not sg  # __bool__ returns False for empty
        assert sg.to_text() == "Empty scene graph"

    def test_scene_graph_with_objects(self):
        sg = SceneGraph(
            objects=[
                ObjectNode("dog", 0.92, (10, 20, 100, 200)),
                ObjectNode("cat", 0.76, (150, 30, 250, 180)),
            ]
        )
        assert len(sg.objects) == 2
        assert sg.objects[0].label == "dog"
        assert sg.objects[0].confidence == 0.92
        assert bool(sg) is True

    def test_scene_graph_with_relations(self):
        sg = SceneGraph(
            objects=[
                ObjectNode("dog", 0.92, (10, 20, 100, 200)),
                ObjectNode("mat", 0.88, (5, 150, 200, 300)),
            ],
            relations=[
                RelationEdge("dog", "sitting on", "mat", 0.81,
                             (10, 20, 100, 200), (5, 150, 200, 300)),
            ],
        )
        assert len(sg) == 3  # 2 objects + 1 relation
        assert sg.relations[0].predicate == "sitting on"

    def test_scene_graph_with_attributes(self):
        sg = SceneGraph(
            objects=[ObjectNode("dog", 0.92, (10, 20, 100, 200))],
            attributes=[AttributeNode("dog", "black", 0.90)],
        )
        assert len(sg.attributes) == 1
        assert sg.attributes[0].attribute == "black"


class TestSceneGraphFromReltrOutput:
    """Test the from_reltr_output factory method."""

    def test_basic_factory(self):
        sg = SceneGraph.from_reltr_output(
            sub_labels=["dog", "cat"],
            sub_confs=[0.92, 0.76],
            sub_bboxes=[(10, 20, 100, 200), (150, 30, 250, 180)],
            obj_labels=["mat", "dog"],
            obj_confs=[0.88, 0.92],
            obj_bboxes=[(5, 150, 200, 300), (10, 20, 100, 200)],
            rel_labels=["sitting on", "near"],
            rel_confs=[0.81, 0.65],
            image_size=(640, 480),
        )
        assert len(sg.objects) >= 2  # dog, cat, mat (dog deduplicated)
        assert len(sg.relations) == 2
        assert sg.image_size == (640, 480)
        assert sg.attributes == []  # RelTR doesn't predict attributes

    def test_empty_factory(self):
        sg = SceneGraph.from_reltr_output(
            sub_labels=[], sub_confs=[], sub_bboxes=[],
            obj_labels=[], obj_confs=[], obj_bboxes=[],
            rel_labels=[], rel_confs=[],
        )
        assert len(sg.objects) == 0
        assert len(sg.relations) == 0

    def test_objects_sorted_by_confidence(self):
        sg = SceneGraph.from_reltr_output(
            sub_labels=["cat"],
            sub_confs=[0.5],
            sub_bboxes=[(0, 0, 10, 10)],
            obj_labels=["dog"],
            obj_confs=[0.9],
            obj_bboxes=[(20, 20, 30, 30)],
            rel_labels=["near"],
            rel_confs=[0.7],
        )
        # dog (0.9) should come before cat (0.5)
        assert sg.objects[0].confidence >= sg.objects[1].confidence

    def test_relations_sorted_by_combined_confidence(self):
        sg = SceneGraph.from_reltr_output(
            sub_labels=["cat", "dog"],
            sub_confs=[0.5, 0.9],
            sub_bboxes=[(0, 0, 10, 10), (20, 20, 30, 30)],
            obj_labels=["mat", "table"],
            obj_confs=[0.6, 0.8],
            obj_bboxes=[(40, 40, 50, 50), (60, 60, 70, 70)],
            rel_labels=["on", "under"],
            rel_confs=[0.7, 0.9],
            # Combined: cat-on-mat = 0.5*0.7*0.6 = 0.21
            #           dog-under-table = 0.9*0.9*0.8 = 0.648
        )
        assert sg.relations[0].confidence > sg.relations[1].confidence


class TestShouldActivateOracle:
    """Test confidence-based oracle deactivation (Section 4.5)."""

    def test_empty_graph_deactivates(self):
        sg = SceneGraph()
        assert sg.should_activate_oracle() is False

    def test_low_confidence_deactivates(self):
        sg = SceneGraph(objects=[
            ObjectNode("thing", 0.2, (0, 0, 10, 10)),
            ObjectNode("stuff", 0.3, (20, 20, 30, 30)),
        ])
        # avg = 0.25 < 0.4 threshold
        assert sg.should_activate_oracle() is False

    def test_high_confidence_activates(self):
        sg = SceneGraph(objects=[
            ObjectNode("dog", 0.92, (10, 20, 100, 200)),
            ObjectNode("cat", 0.76, (150, 30, 250, 180)),
        ])
        # avg = 0.84 > 0.4 threshold
        assert sg.should_activate_oracle() is True

    def test_custom_threshold(self):
        sg = SceneGraph(objects=[
            ObjectNode("dog", 0.55, (10, 20, 100, 200)),
        ])
        assert sg.should_activate_oracle(min_confidence=0.5) is True
        assert sg.should_activate_oracle(min_confidence=0.6) is False


class TestSceneGraphText:
    """Test text representations."""

    def test_to_text_with_data(self):
        sg = SceneGraph(
            objects=[ObjectNode("dog", 0.92, (0, 0, 10, 10))],
            relations=[RelationEdge("dog", "on", "mat", 0.5)],
        )
        text = sg.to_text()
        assert "dog" in text
        assert "on" in text
        assert "mat" in text

    def test_to_prompt_prefix(self):
        sg = SceneGraph(
            objects=[ObjectNode("dog", 0.92, (0, 0, 10, 10)),
                     ObjectNode("mat", 0.88, (0, 0, 10, 10))],
            relations=[RelationEdge("dog", "on", "mat", 0.5)],
        )
        prefix = sg.to_prompt_prefix()
        assert "Objects:" in prefix
        assert "Relations:" in prefix

    def test_summary_stats(self):
        sg = SceneGraph(
            objects=[ObjectNode("dog", 0.9, (0, 0, 10, 10)),
                     ObjectNode("cat", 0.8, (0, 0, 10, 10))],
            relations=[RelationEdge("dog", "near", "cat", 0.5)],
        )
        s = sg.summary()
        assert s["num_objects"] == 2
        assert s["num_relations"] == 1
        assert s["num_attributes"] == 0
        assert 0.8 < s["avg_obj_confidence"] < 0.9
        assert s["oracle_active"] is True


# ════════════════════════════════════════════════════════════════════
# Preprocessing Tests
# ════════════════════════════════════════════════════════════════════

class TestPreprocessImage:
    """Test image preprocessing for RelTR."""

    def test_output_shape(self):
        img = Image.new("RGB", (640, 480))
        tensor = preprocess_image(img)
        assert tensor.ndim == 4
        assert tensor.shape[0] == 1  # batch size
        assert tensor.shape[1] == 3  # channels

    def test_resize_to_800(self):
        img = Image.new("RGB", (1920, 1080))
        tensor = preprocess_image(img)
        # Shorter side should be resized to 800
        assert min(tensor.shape[2], tensor.shape[3]) == 800

    def test_handles_grayscale(self):
        img = Image.new("L", (100, 100))  # Grayscale
        tensor = preprocess_image(img)
        assert tensor.shape[1] == 3  # Converted to RGB

    def test_handles_rgba(self):
        img = Image.new("RGBA", (100, 100))
        tensor = preprocess_image(img)
        assert tensor.shape[1] == 3

    def test_normalization_range(self):
        """Verify ImageNet normalization is applied (black image gives negative values)."""
        img = Image.new("RGB", (100, 100), color=(0, 0, 0))
        tensor = preprocess_image(img)
        # Black pixel after ImageNet normalization: (0 - mean) / std < 0
        assert tensor.min() < 0


# ════════════════════════════════════════════════════════════════════
# Post-processing Tests
# ════════════════════════════════════════════════════════════════════

class TestPostprocessReltrOutput:
    """Test conversion from raw model tensors to structured predictions."""

    def _make_mock_output(self, n_triplets=200, n_entity_classes=151, n_rel_classes=51):
        """Create mock RelTR output tensors."""
        return {
            "sub_logits": torch.randn(1, n_triplets, n_entity_classes + 1),
            "obj_logits": torch.randn(1, n_triplets, n_entity_classes + 1),
            "rel_logits": torch.randn(1, n_triplets, n_rel_classes + 1),
            "sub_boxes": torch.rand(1, n_triplets, 4).clamp(0.1, 0.9),
            "obj_boxes": torch.rand(1, n_triplets, 4).clamp(0.1, 0.9),
        }

    def test_basic_postprocess(self):
        outputs = self._make_mock_output()
        # Set some high-confidence predictions
        outputs["sub_logits"][0, 0, 37] = 10.0  # "dog"
        outputs["obj_logits"][0, 0, 127] = 10.0  # "table"
        outputs["rel_logits"][0, 0, 31] = 10.0  # "on"

        result = postprocess_reltr_output(outputs, image_size=(640, 480))
        # Should return lists (may or may not have results depending on threshold)
        assert isinstance(result["sub_labels"], list)
        assert isinstance(result["rel_labels"], list)
        assert len(result["sub_labels"]) == len(result["rel_labels"])

    def test_empty_when_low_confidence(self):
        outputs = self._make_mock_output()
        # All logits very negative → softmax gives uniform low probs
        for key in ["sub_logits", "obj_logits", "rel_logits"]:
            outputs[key] = torch.zeros_like(outputs[key])
        result = postprocess_reltr_output(
            outputs, image_size=(640, 480), confidence_threshold=0.9
        )
        assert len(result["sub_labels"]) == 0

    def test_bbox_rescaling(self):
        outputs = self._make_mock_output()
        # Force a high-confidence prediction
        outputs["sub_logits"][0, 0, :] = -100
        outputs["sub_logits"][0, 0, 37] = 100  # dog
        outputs["obj_logits"][0, 0, :] = -100
        outputs["obj_logits"][0, 0, 127] = 100  # table
        outputs["rel_logits"][0, 0, :] = -100
        outputs["rel_logits"][0, 0, 31] = 100  # on
        outputs["sub_boxes"][0, 0] = torch.tensor([0.5, 0.5, 0.2, 0.2])  # center

        result = postprocess_reltr_output(outputs, image_size=(640, 480))
        if result["sub_bboxes"]:
            bbox = result["sub_bboxes"][0]
            # Bbox should be in image coordinates (not [0,1])
            assert any(v > 1 for v in bbox), f"Bbox should be rescaled: {bbox}"

    def test_top_k_limit(self):
        outputs = self._make_mock_output()
        # Make all predictions high confidence
        for key in ["sub_logits", "obj_logits", "rel_logits"]:
            outputs[key][0, :, 1] = 100.0  # All predict class 1 with high conf

        result = postprocess_reltr_output(
            outputs, image_size=(640, 480), confidence_threshold=0.01, top_k=5
        )
        assert len(result["sub_labels"]) <= 5


# ════════════════════════════════════════════════════════════════════
# Constants Tests
# ════════════════════════════════════════════════════════════════════

class TestConstants:
    """Verify class label constants match RelTR expectations."""

    def test_entity_classes_count(self):
        assert len(ENTITY_CLASSES) == 151

    def test_entity_classes_starts_with_na(self):
        assert ENTITY_CLASSES[0] == "N/A"

    def test_relation_classes_count(self):
        assert len(RELATION_CLASSES) == 51

    def test_relation_classes_starts_with_background(self):
        assert RELATION_CLASSES[0] == "__background__"

    def test_common_entities_present(self):
        for entity in ["dog", "cat", "person", "car", "table", "chair"]:
            assert entity in ENTITY_CLASSES, f"{entity} not in ENTITY_CLASSES"

    def test_common_relations_present(self):
        for rel in ["on", "near", "under", "behind", "holding", "wearing"]:
            assert rel in RELATION_CLASSES, f"{rel} not in RELATION_CLASSES"

    def test_default_args(self):
        args = get_reltr_default_args()
        assert args.backbone == "resnet50"
        assert args.hidden_dim == 256
        assert args.num_entities == 100
        assert args.num_triplets == 200


# ════════════════════════════════════════════════════════════════════
# Integration Tests (require GPU + checkpoint)
# ════════════════════════════════════════════════════════════════════

CHECKPOINT_PATH = "data/checkpoints/reltr/checkpoint0149.pth"


@pytest.mark.integration
class TestSGGModuleIntegration:
    """End-to-end tests. Run with: pytest -m integration"""

    @pytest.fixture(scope="class")
    def sgg(self):
        from sgod.sgg import SGGModule
        return SGGModule(CHECKPOINT_PATH)

    @pytest.fixture
    def sample_image(self):
        """Create a simple test image."""
        return Image.new("RGB", (640, 480), color=(128, 100, 80))

    def test_extract_returns_scene_graph(self, sgg, sample_image):
        sg = sgg.extract(sample_image)
        assert isinstance(sg, SceneGraph)
        assert sg.image_size == (640, 480)

    def test_extract_deterministic(self, sgg, sample_image):
        sg1 = sgg.extract(sample_image)
        sg2 = sgg.extract(sample_image)
        assert len(sg1.objects) == len(sg2.objects)
        assert len(sg1.relations) == len(sg2.relations)

    def test_extract_batch(self, sgg, sample_image):
        results = sgg.extract_batch([sample_image, sample_image])
        assert len(results) == 2
        assert all(isinstance(sg, SceneGraph) for sg in results)

    def test_benchmark_latency(self, sgg, sample_image):
        stats = sgg.benchmark(sample_image, n_runs=10, warmup=2)
        assert stats["mean_ms"] < 5000  # <100ms on GPU; CPU ~1000ms is acceptable
        assert stats["n_runs"] == 10

    def test_model_metadata(self, sgg):
        assert sgg.num_parameters > 0
        assert sgg.memory_mb > 0
