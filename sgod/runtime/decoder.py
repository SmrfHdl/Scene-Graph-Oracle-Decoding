"""HallucinationDecoder — orchestrator that drives Backbone + Oracle + Policy.

This is the seam where the framework's three pluggable layers come together:

    Oracle.extract(image)            → evidence  (one-time per image)
    Policy.init_state(evidence)      → policy_state
    Backbone.prepare_inputs(image, p)→ inputs   (KV cache lives in here)

    loop:
        Backbone.forward_step        → (h_t, lm_logits)
        Policy.adjust_logits         → Δ
        sample(lm_logits + Δ)        → next_id
        Policy.update_state          → policy_state'

The orchestrator is intentionally minimal — anything model-specific (image
preprocessing, KV-cache layout, anchor detection, slot recurrence) is owned
by one of the three ABCs.

Per-image policies that need access to the raw image or the user's question
(e.g., SGODv1Policy builds a CLIPScorer and a GenerationContext) read those
from `OracleEvidence.extra`. The orchestrator stashes them there before
calling `policy.init_state`.
"""
from __future__ import annotations

from typing import Any, Callable

import torch

from sgod.core.interfaces import Backbone, Oracle, Policy
from sgod.core.types import GenerationState


class HallucinationDecoder:
    """Generic generation loop wrapping any (Backbone, Oracle, Policy) triple.

    Args:
        backbone:        Wraps the VLM; provides per-step (h_t, lm_logits).
        oracle:          Image → structured belief (SceneGraph).
        policy:          Decoding-time logit adjustment.
        max_new_tokens:  Default cap on generated tokens.
        temperature:     Sampling temperature; 0.0 = greedy argmax.

    Single-batch (B=1) at this stage. Batched generation is on the Week-2
    backlog when we wire up offline evaluation pipelines.
    """

    def __init__(
        self,
        backbone: Backbone,
        oracle: Oracle,
        policy: Policy,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        hidden_buffer_size: int = 8,
        step_hook: Callable[[GenerationState, torch.Tensor], None] | None = None,
    ) -> None:
        self.backbone = backbone
        self.oracle = oracle
        self.policy = policy
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        # K_t: window of recent backbone hidden states the orchestrator
        # publishes to the policy each step (via GenerationState.hidden_buffer).
        # DT-SGOD reads it to build the slow-module prefix summary. Set to 0
        # to disable buffering entirely (saves memory; policies that need it
        # then fall back to the current step's hidden).
        self.hidden_buffer_size = hidden_buffer_size
        # Called after the policy emits Δ but BEFORE sampling. Receives
        # `(state, delta)` so it can record per-step training traces, log
        # gate-firing stats, etc. The hook must not mutate the state — the
        # orchestrator's invariants depend on the policy fully owning the
        # state. Used by `sgod.runtime.trace_collector.TraceCollector`.
        self.step_hook = step_hook

    # ── public API ────────────────────────────────────────────────────────

    def generate(
        self,
        image: Any,
        prompt: str,
        question: str | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Generate text for an (image, prompt) pair.

        Args:
            image:         Whatever the backbone & oracle accept (typically a PIL.Image).
            prompt:        The full prompt the backbone consumes — may include
                           chat-template tokens (e.g. "USER: <image>\\n...\\nASSISTANT:").
            question:      The raw user question, used by oracle/policy for
                           question-type detection. Defaults to `prompt`.
            max_new_tokens, temperature: Per-call overrides.

        Returns:
            Decoded answer string (prompt tokens stripped).
        """
        max_new_tokens = max_new_tokens if max_new_tokens is not None else self.max_new_tokens
        temperature = temperature if temperature is not None else self.temperature
        question = question if question is not None else prompt

        # 1) One-time per-image evidence.
        evidence = self.oracle.extract(image)
        # Stash the raw inputs so question/image-aware policies (SGODv1) can build their state.
        evidence.extra.setdefault("image", image)
        evidence.extra.setdefault("question", question)
        evidence.extra.setdefault("prompt", prompt)

        # 2) Policy init.
        policy_state = self.policy.init_state(evidence)

        # 3) Backbone prep (also seeds the KV cache slot).
        inputs = self.backbone.prepare_inputs(image, prompt)
        tokenizer = self.backbone.tokenizer()
        eos_id = getattr(tokenizer, "eos_token_id", None)

        # 4) Decode loop.
        generated_ids = torch.empty(1, 0, dtype=torch.long)
        prompt_ids = self._extract_prompt_ids(inputs)
        # Rolling K_t buffer of detached hidden states (oldest first, newest last).
        # Detach so the buffer never holds onto autograd graphs across steps.
        hidden_buffer: list[torch.Tensor] | None = (
            [] if self.hidden_buffer_size > 0 else None
        )

        for step in range(max_new_tokens):
            hidden, lm_logits = self.backbone.forward_step(inputs, generated_ids)
            if hidden_buffer is not None:
                hidden_buffer.append(hidden.detach())
                if len(hidden_buffer) > self.hidden_buffer_size:
                    hidden_buffer.pop(0)

            state = GenerationState(
                prompt_ids=prompt_ids,
                generated_ids=generated_ids,
                hidden_states=hidden,
                lm_logits=lm_logits,
                evidence=evidence,
                step=step,
                policy_state=policy_state,
                hidden_buffer=hidden_buffer,
            )

            delta = self.policy.adjust_logits(state)
            if self.step_hook is not None:
                self.step_hook(state, delta)
            adjusted = lm_logits + delta

            if temperature == 0.0:
                next_id = int(torch.argmax(adjusted[0]).item())
            else:
                probs = torch.softmax(adjusted[0] / temperature, dim=-1)
                next_id = int(torch.multinomial(probs, num_samples=1).item())

            generated_ids = torch.cat(
                [generated_ids,
                 torch.tensor([[next_id]], dtype=generated_ids.dtype, device=generated_ids.device)],
                dim=-1,
            )
            policy_state = self.policy.update_state(state, sampled_token_id=next_id)

            if eos_id is not None and next_id == eos_id:
                break

        return tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=True)

    # ── internals ─────────────────────────────────────────────────────────

    @staticmethod
    def _extract_prompt_ids(inputs: dict[str, Any]) -> torch.Tensor:
        """Best-effort retrieval of prompt token ids from a backbone's `inputs`.

        Used only for diagnostic GenerationState.prompt_ids. Returns an empty
        [1, 0] tensor when the backbone shape isn't recognized — policies
        currently don't read this field.
        """
        proc = inputs.get("proc_out")
        if proc is not None:
            try:
                ids = proc["input_ids"]
                if isinstance(ids, torch.Tensor):
                    return ids
            except (KeyError, TypeError):
                pass
        return torch.empty(1, 0, dtype=torch.long)
