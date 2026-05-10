"""
Visual Oracle — Component 2 (Section 3.3).
Dual-source oracle combining SGG (structured) + CLIP (distributional).
"""
from sgod.oracle.bbox_clip_scorer import BboxClipScorer
from sgod.oracle.clip_scorer import CLIPScorer
from sgod.oracle.visual_oracle import VisualOracle

__all__ = ["BboxClipScorer", "CLIPScorer", "VisualOracle"]
