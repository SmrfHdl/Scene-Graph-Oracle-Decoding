"""
SGGModule — Scene Graph Generation wrapper.

Component 1 of the SGOD framework (Section 3.2).
Wraps RelTR to extract scene graphs from images.
Runs once per image before generation begins.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import torch
from PIL import Image

from .reltr_utils import (
    load_reltr_model,
    preprocess_image,
    postprocess_reltr_output,
)
from .scene_graph import SceneGraph

logger = logging.getLogger(__name__)


class SGGModule:
    """Scene Graph Generation module using RelTR.

    Extracts structured scene representation from images.
    Designed for one-time extraction before VLM generation.

    Usage:
        sgg = SGGModule("data/checkpoints/reltr/checkpoint0149.pth")
        scene_graph = sgg.extract(image)
        print(scene_graph.to_text())
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        confidence_threshold: float = 0.3,
        top_k: int = 20,
    ):
        """Initialize SGGModule.

        Args:
            checkpoint_path: Path to RelTR checkpoint file
            device: Target device. Auto-detects CUDA if available.
            confidence_threshold: Min confidence for each triplet component
            top_k: Max number of triplets to extract per image
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.top_k = top_k

        logger.info(f"Loading RelTR from {checkpoint_path} on {device}")
        self.model = load_reltr_model(checkpoint_path, device=device)
        logger.info(
            f"RelTR loaded: {sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M params"
        )

    @torch.no_grad()
    def extract(self, image: Image.Image) -> SceneGraph:
        """Extract scene graph from a single image.

        Args:
            image: PIL Image (any mode, will be converted to RGB)

        Returns:
            SceneGraph with objects, relations, and metadata
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        image_size = image.size  # (width, height)

        # Preprocess and run model
        img_tensor = preprocess_image(image).to(self.device)
        outputs = self.model(img_tensor)

        # Post-process
        result = postprocess_reltr_output(
            outputs,
            image_size=image_size,
            confidence_threshold=self.confidence_threshold,
            top_k=self.top_k,
        )

        # Build SceneGraph
        scene_graph = SceneGraph.from_reltr_output(
            sub_labels=result["sub_labels"],
            sub_confs=result["sub_confs"],
            sub_bboxes=result["sub_bboxes"],
            obj_labels=result["obj_labels"],
            obj_confs=result["obj_confs"],
            obj_bboxes=result["obj_bboxes"],
            rel_labels=result["rel_labels"],
            rel_confs=result["rel_confs"],
            image_size=image_size,
        )

        logger.debug(
            f"Extracted: {len(scene_graph.objects)} objects, "
            f"{len(scene_graph.relations)} relations"
        )
        return scene_graph

    @torch.no_grad()
    def extract_batch(self, images: list[Image.Image]) -> list[SceneGraph]:
        """Extract scene graphs from multiple images.

        Currently processes sequentially. Batch processing can be added
        if this becomes a bottleneck during evaluation.
        """
        return [self.extract(img) for img in images]

    def benchmark(self, image: Image.Image, n_runs: int = 100, warmup: int = 5) -> dict:
        """Measure extraction latency.

        Args:
            image: Test image
            n_runs: Number of timed runs
            warmup: Number of warmup runs (not timed)

        Returns:
            Dict with mean, p50, p95, p99 latencies in milliseconds
        """
        # Warmup
        for _ in range(warmup):
            self.extract(image)

        # Synchronize CUDA before timing
        if "cuda" in self.device:
            torch.cuda.synchronize()

        times = []
        for _ in range(n_runs):
            if "cuda" in self.device:
                torch.cuda.synchronize()
            start = time.perf_counter()

            self.extract(image)

            if "cuda" in self.device:
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000  # ms
            times.append(elapsed)

        times.sort()
        n = len(times)
        return {
            "mean_ms": sum(times) / n,
            "p50_ms": times[n // 2],
            "p95_ms": times[int(n * 0.95)],
            "p99_ms": times[int(n * 0.99)],
            "min_ms": times[0],
            "max_ms": times[-1],
            "n_runs": n,
            "device": self.device,
        }

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    @property
    def memory_mb(self) -> float:
        """Approximate model memory in MB."""
        return sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)
