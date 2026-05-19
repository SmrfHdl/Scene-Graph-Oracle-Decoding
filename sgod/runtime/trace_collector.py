"""TraceCollector — record per-step (state, Δ) tuples during generation.

Used to capture Stage-0 distillation traces: run the orchestrator with the
SGOD v1 teacher policy, attach a `TraceCollector` as `step_hook`, and after
generation `collector.traces` holds the per-step training signal the DT-SGOD
student adapter consumes.

Tensors are detached and moved to CPU before being recorded so the trace
buffer never holds onto autograd graphs or GPU memory.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from sgod.core.types import GenerationState, OracleEvidence


@dataclass
class StepTrace:
    """One decode step's training-relevant state.

    Snapshot taken right after the teacher policy computed `delta_teacher`,
    before sampling. Reproduces the exact inputs a student policy would see
    if it ran on the same step.
    """
    step: int
    hidden_states: torch.Tensor          # [B, d] — backbone output, CPU
    lm_logits: torch.Tensor              # [B, V] — backbone output, CPU
    delta_teacher: torch.Tensor          # [B, V] — what the teacher would inject
    evidence: OracleEvidence             # SceneGraph + extras (lightweight pickle)
    rule_anchor_type: str | None = None  # rule-based anchor label, if available


@dataclass
class TraceCollector:
    """Callable matching the orchestrator's `step_hook` signature.

    Usage:
        collector = TraceCollector()
        decoder = HallucinationDecoder(..., step_hook=collector)
        decoder.generate(image, prompt)
        traces: list[StepTrace] = collector.traces
    """
    traces: list[StepTrace] = field(default_factory=list)

    def __call__(self, state: GenerationState, delta: torch.Tensor) -> None:
        # Detach + CPU so the buffer doesn't pin GPU memory or carry grads.
        self.traces.append(StepTrace(
            step=state.step,
            hidden_states=state.hidden_states.detach().to("cpu"),
            lm_logits=state.lm_logits.detach().to("cpu"),
            delta_teacher=delta.detach().to("cpu"),
            evidence=state.evidence,
            rule_anchor_type=state.rule_anchor_type,
        ))

    def clear(self) -> None:
        self.traces.clear()

    def save(self, path: str | Any) -> None:
        """Persist traces as a torch checkpoint for offline training.

        Stores StepTrace as plain dicts so loading doesn't require the
        StepTrace class to be importable on the consumer side — useful for
        cross-environment Stage-0 setups.
        """
        torch.save([_as_dict(t) for t in self.traces], path)

    @classmethod
    def load(cls, path: str | Any) -> list[StepTrace]:
        """Inverse of `save`: reconstruct StepTrace instances from disk."""
        raw = torch.load(path, weights_only=False)
        return [_from_dict(d) for d in raw]


# ── internals ────────────────────────────────────────────────────────────────

def _as_dict(t: StepTrace) -> dict:
    return {
        "step": t.step,
        "hidden_states": t.hidden_states,
        "lm_logits": t.lm_logits,
        "delta_teacher": t.delta_teacher,
        "evidence": t.evidence,
        "rule_anchor_type": t.rule_anchor_type,
    }


def _from_dict(d: dict) -> StepTrace:
    return StepTrace(**d)


__all__ = ["StepTrace", "TraceCollector"]
