"""VR-TTS — Visual Reasoning via Test-Time Scaling.

A primitive for action-augmented iterative VLM decoding. At each step the
decoder evaluates the current visual state, gates on confidence, and applies
a visual action (zoom, mark, sub-question) to update the state. The trace
of (state, answer, confidence) tuples is aggregated into a final answer.

Public surface:
- `VRTTSDecoder`       — orchestrator binding backbone + actions + policy.
- `VisualAction`       — abstract base for actions.
- `ZoomToAttentionRegion`, `AddSoMMarks`, `AskSubQuestion` — Phase-1 actions.
- `ConfidenceEstimator` — yes/no logit-gap confidence.
- `TraceAggregator`    — weighted vote over the action trace.

Phase-1 scope: heuristic policy, 3 actions, Reefknot YESNO evaluation.
Phase 2+: learned exploration policy, more actions, multi-benchmark.
"""
from sgod.policies.vrtts.actions.base import VisualAction, VisualState
from sgod.policies.vrtts.confidence import ConfidenceEstimator
from sgod.policies.vrtts.decoder import VRTTSDecoder

__all__ = [
    "VRTTSDecoder",
    "VisualAction",
    "VisualState",
    "ConfidenceEstimator",
]
