"""Aggregate the step-wise (label, confidence) trace into a final answer."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TraceStep:
    """One step of the VR-TTS trace."""

    step: int
    action_applied: str | None  # the action that produced this step's state (None for step 0)
    label: str                  # "yes" or "no"
    confidence: float           # in [0, 1]


class TraceAggregator:
    """Aggregate a trace of TraceStep into a single yes/no answer.

    Two modes:
      - "last": return the label of the final step (when confidence gate fired
                or K was reached).
      - "weighted_vote": sum confidence per label and return the higher.
    """

    def __init__(self, mode: str = "weighted_vote") -> None:
        if mode not in ("last", "weighted_vote"):
            raise ValueError(f"Unknown aggregator mode: {mode}")
        self.mode = mode

    def aggregate(self, trace: list[TraceStep]) -> str:
        if not trace:
            raise ValueError("Empty trace — must have at least one step.")
        if self.mode == "last":
            return trace[-1].label
        yes = sum(s.confidence for s in trace if s.label == "yes")
        no = sum(s.confidence for s in trace if s.label == "no")
        return "yes" if yes >= no else "no"
