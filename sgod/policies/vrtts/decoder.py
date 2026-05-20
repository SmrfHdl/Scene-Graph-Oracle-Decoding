"""VRTTSDecoder — orchestrator for action-augmented iterative VLM decoding.

Pipeline per (image, question):

    state_0 = (image, prompt)
    for t = 0 .. K-1:
        attentions_t (if needed by next action),
        logits_t = backbone.forward(state_t)
        label_t, conf_t = confidence(logits_t)
        trace.append(step=t, action=None_or_prior_action,
                     label=label_t, conf=conf_t)

        if conf_t >= τ:                       # gating: enough evidence
            break

        action_name = policy.select(step_idx=t+1)
        if action_name is None: break         # policy exhausted
        action = actions[action_name]
        state_t+1 = action.apply(state_t, attentions=...)

    answer = aggregator.aggregate(trace)

For Phase 1 the action set is fixed (zoom, som, subq) and the policy is a
fixed cycle. Phase 2+ swaps in a learned policy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from PIL import Image

from sgod.policies.vrtts.actions.base import VisualAction, VisualState
from sgod.policies.vrtts.aggregator import TraceAggregator, TraceStep
from sgod.policies.vrtts.confidence import ConfidenceEstimator
from sgod.policies.vrtts.policy import HeuristicExplorationPolicy


@dataclass
class VRTTSResult:
    """Output of a VRTTSDecoder.run() call."""

    answer: str
    trace: list[TraceStep]
    final_state: VisualState
    metadata: dict[str, Any] = field(default_factory=dict)


class VRTTSDecoder:
    """Action-augmented iterative decoder for yes/no VQA.

    Args:
        backbone:        VLM backbone with `prepare_inputs` and
                         `forward_first_step_with_attentions` (used when an
                         action needs attention). For zoom we always need
                         attention; for som/subq we don't, but using the
                         same call site keeps things simple.
        actions:         dict mapping action-name → VisualAction.
        policy:          ExplorationPolicy with `select(step_idx) → name | None`.
        confidence:      ConfidenceEstimator.
        aggregator:      TraceAggregator.
        oracle:          optional external oracle (used by SoM marking).
        max_steps:       maximum K iterations (initial + K-1 explorations).
        confidence_threshold: τ. If a step's confidence ≥ τ, stop exploring.
        prompt_template: format string with {question} placeholder for the
                         main question. Default matches existing scripts.
    """

    def __init__(
        self,
        backbone: Any,
        actions: dict[str, VisualAction],
        policy: HeuristicExplorationPolicy | None = None,
        confidence: ConfidenceEstimator | None = None,
        aggregator: TraceAggregator | None = None,
        oracle: Any = None,
        max_steps: int = 4,
        confidence_threshold: float = 0.5,
        prompt_template: str = "USER: <image>\n{question} ASSISTANT:",
    ) -> None:
        self.backbone = backbone
        self.actions = actions
        self.policy = policy or HeuristicExplorationPolicy()
        self.confidence = confidence or ConfidenceEstimator(backbone.tokenizer())
        self.aggregator = aggregator or TraceAggregator()
        self.oracle = oracle
        self.max_steps = max_steps
        self.confidence_threshold = confidence_threshold
        self.prompt_template = prompt_template

    def _build_prompt(self, question: str, suffix: str = "") -> str:
        """Compose the main-question prompt with optional action-injected suffix."""
        base = self.prompt_template.format(question=question)
        if not suffix:
            return base
        # Insert suffix BEFORE "ASSISTANT:" so the LM sees it as context.
        marker = "ASSISTANT:"
        idx = base.rfind(marker)
        if idx < 0:
            return base + " " + suffix
        return base[:idx] + suffix.strip() + " " + base[idx:]

    def _forward(
        self,
        state: VisualState,
        question: str,
        *,
        emit_attentions: bool,
    ) -> tuple[torch.Tensor, dict[str, Any] | None]:
        """Run a single first-step forward. Returns (logits, attn_bundle | None)."""
        prompt = self._build_prompt(question, state.prompt_suffix)
        inputs = self.backbone.prepare_inputs(image=state.image, prompt=prompt)
        if emit_attentions:
            _, logits, attns, meta = self.backbone.forward_first_step_with_attentions(inputs)
            bundle = {
                "layers": attns,
                "image_token_positions": meta["image_token_positions"],
            }
            return logits, bundle
        # Cheaper path: regular forward_step (still does the full pass on first call).
        # We synthesize an empty generated_ids — orchestrator pattern from existing scripts.
        gen_ids = torch.empty(1, 0, dtype=torch.long)
        _, logits = self.backbone.forward_step(inputs, gen_ids)
        return logits, None

    def run(
        self,
        image: Image.Image,
        question: str,
    ) -> VRTTSResult:
        """Run K-step VR-TTS on a single (image, question) pair."""
        state = VisualState(image=image.copy(), original_image=image)
        trace: list[TraceStep] = []
        last_action_name: str | None = None
        # Decide upfront whether to emit attentions on the very first forward.
        # We need them iff the FIRST scheduled action requires them. For Phase
        # 1 the first action is always zoom (which needs them), so emit.
        need_attn_next = True

        for step_idx in range(self.max_steps):
            logits, attn_bundle = self._forward(
                state, question, emit_attentions=need_attn_next
            )
            label, conf = self.confidence.score(logits)
            trace.append(
                TraceStep(
                    step=step_idx,
                    action_applied=last_action_name,
                    label=label,
                    confidence=conf,
                )
            )

            # Gate: if confident enough, stop.
            if conf >= self.confidence_threshold:
                break

            # Otherwise consult the policy for the next action.
            next_action_name = self.policy.select(step_idx=step_idx + 1)
            if next_action_name is None:
                break
            action = self.actions.get(next_action_name)
            if action is None:
                raise KeyError(
                    f"Policy requested action {next_action_name!r} but no such "
                    f"action is registered. Available: {list(self.actions)}"
                )

            # Apply action.
            kwargs: dict[str, Any] = {
                "question": question,
                "backbone": self.backbone,
                "oracle": self.oracle,
            }
            if action.name == "zoom":
                kwargs["attentions"] = attn_bundle
            state = action.apply(state, **kwargs)
            last_action_name = next_action_name
            # Decide whether the NEXT step needs attention based on the action
            # that follows. For Phase 1 we conservatively emit attentions on
            # every step (cheap relative to full forward; simplifies logic).
            need_attn_next = True

        answer = self.aggregator.aggregate(trace)
        return VRTTSResult(
            answer=answer,
            trace=trace,
            final_state=state,
            metadata={"num_steps": len(trace)},
        )
