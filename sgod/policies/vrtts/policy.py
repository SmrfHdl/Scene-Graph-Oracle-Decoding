"""Heuristic exploration policy for Phase 1.

A fixed action cycle: zoom → mark → sub-question. Phase 2+ will replace this
with a learned policy (small MLP over attention + trace features).
"""
from __future__ import annotations


class HeuristicExplorationPolicy:
    """Cycles through a fixed list of action names.

    `select(step_idx)` returns the action name to apply next, or None if the
    policy has no more actions to suggest (orchestrator should stop).
    """

    def __init__(self, action_sequence: list[str] | None = None) -> None:
        self.action_sequence = action_sequence or ["zoom", "som", "subq"]

    def select(self, step_idx: int) -> str | None:
        """Return the action name for the (step_idx)-th exploration step.

        step_idx=0 is the initial state (no action applied yet). The orchestrator
        consults the policy starting from step_idx=1.
        """
        i = step_idx - 1
        if i < 0 or i >= len(self.action_sequence):
            return None
        return self.action_sequence[i]
