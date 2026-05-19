"""Tests for the orchestrator step_hook + TraceCollector.

CPU-only. Drives the orchestrator with fake (Backbone, Oracle, Policy) and
verifies traces capture exactly the right per-step (state, Δ) snapshot.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from sgod.core.interfaces import Backbone, Oracle, Policy
from sgod.core.types import OracleEvidence, SceneGraph
from sgod.runtime import HallucinationDecoder, TraceCollector
from sgod.runtime.trace_collector import StepTrace


D = 4
V = 8


class _FakeTok:
    eos_token_id = 0
    def decode(self, ids, skip_special_tokens=False):
        return " ".join(str(int(i)) for i in ids if not (skip_special_tokens and int(i) == 0))
    def batch_decode(self, ids):
        return [str(int(i)) for i in ids]


class _FakeBackbone(Backbone):
    @property
    def hidden_dim(self): return D
    @property
    def vocab_size(self): return V
    @property
    def lora_target_modules(self): return []
    @property
    def device(self): return "cpu"
    def tokenizer(self): return _FakeTok()
    def prepare_inputs(self, image, prompt): return {"past_key_values": None}
    def forward_step(self, inputs, generated_ids):
        step = generated_ids.shape[1]
        # Deterministic: hidden = full-step, logits with arg max at token 1.
        hidden = torch.full((1, D), float(step))
        logits = torch.zeros(1, V)
        logits[0, 1] = 100.0
        inputs["past_key_values"] = (inputs.get("past_key_values") or 0) + 1
        return hidden, logits


class _FakeOracle(Oracle):
    def extract(self, image, image_meta=None) -> OracleEvidence:
        return OracleEvidence(scene_graph=SceneGraph())
    def vocab_scores(self, tokenizer, evidence): return {}


class _NonzeroPolicy(Policy):
    """Emits a Δ that depends on the step — gives the collector something to record."""

    def init_state(self, evidence): return {}

    def adjust_logits(self, state):
        d = torch.zeros_like(state.lm_logits)
        d[0, 3] = float(state.step + 1)  # step 0 → 1, step 1 → 2, ...
        return d

    def update_state(self, state, sampled_token_id):
        return state.policy_state


def _runner(steps: int, collector: TraceCollector | None = None) -> HallucinationDecoder:
    return HallucinationDecoder(
        backbone=_FakeBackbone(),
        oracle=_FakeOracle(),
        policy=_NonzeroPolicy(),
        max_new_tokens=steps,
        step_hook=collector,
    )


# ── basic capture ────────────────────────────────────────────────────────────

def test_trace_collector_records_one_entry_per_step():
    collector = TraceCollector()
    dec = _runner(steps=3, collector=collector)
    dec.generate(object(), "prompt")
    assert len(collector.traces) == 3
    assert all(isinstance(t, StepTrace) for t in collector.traces)


def test_trace_step_indices_are_consecutive():
    collector = TraceCollector()
    _runner(steps=4, collector=collector).generate(object(), "prompt")
    assert [t.step for t in collector.traces] == [0, 1, 2, 3]


def test_trace_captures_hidden_and_logits():
    collector = TraceCollector()
    _runner(steps=2, collector=collector).generate(object(), "prompt")
    t0, t1 = collector.traces
    # Backbone emits hidden full-of-step at each step.
    assert torch.equal(t0.hidden_states, torch.full((1, D), 0.0))
    assert torch.equal(t1.hidden_states, torch.full((1, D), 1.0))
    assert t0.lm_logits.shape == (1, V)


def test_trace_captures_teacher_delta():
    collector = TraceCollector()
    _runner(steps=3, collector=collector).generate(object(), "prompt")
    for step, t in enumerate(collector.traces):
        # Δ[0, 3] = step + 1; everything else zero.
        assert t.delta_teacher[0, 3].item() == step + 1
        assert t.delta_teacher.abs().sum().item() == step + 1


def test_trace_tensors_detached_and_on_cpu():
    collector = TraceCollector()
    _runner(steps=2, collector=collector).generate(object(), "prompt")
    for t in collector.traces:
        assert not t.hidden_states.requires_grad
        assert not t.delta_teacher.requires_grad
        assert t.hidden_states.device.type == "cpu"
        assert t.delta_teacher.device.type == "cpu"


# ── hook contract ────────────────────────────────────────────────────────────

def test_no_step_hook_means_no_overhead():
    """When step_hook is None, generation must run unaffected."""
    dec = _runner(steps=3, collector=None)
    out = dec.generate(object(), "prompt")
    # Just check it produced something and didn't crash.
    assert isinstance(out, str)


def test_step_hook_called_before_sampling():
    """The hook receives (state, delta). At call time, state.step matches the iteration."""
    seen: list[int] = []
    def hook(state, delta):
        seen.append(state.step)
    dec = HallucinationDecoder(
        backbone=_FakeBackbone(),
        oracle=_FakeOracle(),
        policy=_NonzeroPolicy(),
        max_new_tokens=3,
        step_hook=hook,
    )
    dec.generate(object(), "prompt")
    assert seen == [0, 1, 2]


# ── save / load round trip ───────────────────────────────────────────────────

def test_save_and_load_round_trip():
    collector = TraceCollector()
    _runner(steps=3, collector=collector).generate(object(), "prompt")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "traces.pt"
        collector.save(path)
        loaded = TraceCollector.load(path)

    assert len(loaded) == 3
    for orig, back in zip(collector.traces, loaded):
        assert orig.step == back.step
        assert torch.equal(orig.hidden_states, back.hidden_states)
        assert torch.equal(orig.delta_teacher, back.delta_teacher)


def test_clear_empties_buffer():
    collector = TraceCollector()
    _runner(steps=2, collector=collector).generate(object(), "prompt")
    assert len(collector.traces) == 2
    collector.clear()
    assert collector.traces == []
