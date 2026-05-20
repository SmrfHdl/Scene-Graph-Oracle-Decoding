"""AskSubQuestion — sub-question chain-of-thought style visual context.

Procedure:
  1. Parse the question → extract X and Y phrases.
  2. Generate up to 2 sub-questions about X and Y individually.
  3. For each, run the backbone short-greedy-decode (≤20 tokens) to get a
     natural-language answer.
  4. Append the (sub-q, sub-a) pairs to the prompt suffix so the main
     forward sees them as in-context evidence.

This is the CoT-style analogue of test-time visual scaling: the LM
articulates intermediate visual facts before committing to the relation.
"""
from __future__ import annotations

from typing import Any

import torch

from sgod.policies.vrtts.actions.base import VisualAction, VisualState
from sgod.policies.vrtts.actions.som import _parse_xy


def _greedy_short(
    backbone: Any,
    image: Any,
    prompt: str,
    max_new_tokens: int = 20,
) -> str:
    """Greedy decode at most `max_new_tokens` tokens; stop at EOS."""
    inputs = backbone.prepare_inputs(image=image, prompt=prompt)
    gen_ids = torch.empty(1, 0, dtype=torch.long)
    eos = backbone.tokenizer().eos_token_id
    out_ids: list[int] = []
    for _ in range(max_new_tokens):
        _, logits = backbone.forward_step(inputs, gen_ids)
        next_id = int(logits[0].argmax().item())
        if next_id == eos:
            break
        out_ids.append(next_id)
        gen_ids = torch.cat(
            [gen_ids, torch.tensor([[next_id]], dtype=torch.long)], dim=1
        )
    text = backbone.tokenizer().decode(out_ids, skip_special_tokens=True)
    return text.strip()


class AskSubQuestion(VisualAction):
    """Generate visual sub-question evidence prior to the main question.

    Args:
        max_subq:        how many sub-questions to ask (1 or 2).
        max_new_tokens:  cap per sub-answer.
        templates:       list of "{X}" / "{Y}" formatted question templates.
    """

    name = "subq"

    def __init__(
        self,
        max_subq: int = 2,
        max_new_tokens: int = 20,
        templates: tuple[str, ...] = (
            "Where is the {X} in this image?",
            "Where is the {Y} in this image?",
        ),
    ) -> None:
        if not (1 <= max_subq <= 4):
            raise ValueError("max_subq must be in [1, 4]")
        self.max_subq = max_subq
        self.max_new_tokens = max_new_tokens
        self.templates = templates

    def _build_subqs(self, x: str | None, y: str | None) -> list[str]:
        subqs: list[str] = []
        for tpl in self.templates:
            if "{X}" in tpl and not x:
                continue
            if "{Y}" in tpl and not y:
                continue
            subqs.append(tpl.format(X=x or "", Y=y or ""))
            if len(subqs) >= self.max_subq:
                break
        return subqs

    def apply(
        self,
        state: VisualState,
        *,
        attentions: Any | None = None,
        question: str | None = None,
        backbone: Any | None = None,
        oracle: Any | None = None,
    ) -> VisualState:
        if backbone is None:
            raise ValueError("AskSubQuestion requires `backbone` to generate sub-answers.")
        if question is None:
            raise ValueError("AskSubQuestion requires the main question.")

        x_phrase, y_phrase = _parse_xy(question)
        subqs = self._build_subqs(x_phrase, y_phrase)
        if not subqs:
            return state  # nothing to ask

        new_state = state.clone()
        qa_pairs: list[tuple[str, str]] = list(state.sub_qa_trace)
        for sub_q in subqs:
            sub_prompt = f"USER: <image>\n{sub_q} ASSISTANT:"
            answer = _greedy_short(
                backbone, state.image, sub_prompt, max_new_tokens=self.max_new_tokens
            )
            qa_pairs.append((sub_q, answer))

        new_state.sub_qa_trace = qa_pairs
        # Build a prompt suffix that the orchestrator will see on the next
        # main-question forward.
        ctx_lines = " ".join(f"Q: {q} A: {a}." for q, a in qa_pairs)
        new_state.prompt_suffix = (
            (state.prompt_suffix + " " if state.prompt_suffix else "")
            + f"Context: {ctx_lines}"
        )
        new_state.metadata["subq_trace"] = qa_pairs
        return new_state
