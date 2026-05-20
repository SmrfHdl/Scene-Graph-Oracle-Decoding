"""VR-TTS visual actions.

Each action takes a `VisualState` (image + accumulated context) and returns
a new state representing a refinement of the visual evidence.
"""
from sgod.policies.vrtts.actions.base import VisualAction, VisualState
from sgod.policies.vrtts.actions.som import AddSoMMarks
from sgod.policies.vrtts.actions.subquestion import AskSubQuestion
from sgod.policies.vrtts.actions.zoom import ZoomToAttentionRegion

__all__ = [
    "VisualAction",
    "VisualState",
    "ZoomToAttentionRegion",
    "AddSoMMarks",
    "AskSubQuestion",
]
