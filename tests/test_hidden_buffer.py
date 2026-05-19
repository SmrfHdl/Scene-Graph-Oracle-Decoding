"""Tests for the K_t hidden-state rolling buffer.

Two contracts:
  - Orchestrator: maintains a sliding window of the last K_t backbone hidden
    states (detached, oldest first) and publishes it in `GenerationState.hidden_buffer`.
  - DTSGODPolicy: when the buffer is supplied, the slow module's prefix_summary
    is the mean over the buffer (not just the current hidden state). G1 invariant
    must still hold (γ=0 ⇒ Δ=0) regardless of buffer content.
"""
from __future__ import annotations

import torch

from sgod.core.interfaces import Backbone, Oracle
from sgod.core.types import GenerationState, OracleEvidence, SceneGraph
from sgod.policies import DTSGODPolicy
from sgod.policies.dt_sgod.config import DTSGODConfig
from sgod.runtime import HallucinationDecoder


D = 8
V = 16


class _AlwaysFire(torch.nn.Module):
    """Stub ATG that always returns fire_prob=1.0 for every batch element."""

    def forward(self, h, h_slow_pooled, lm_entropy):
        return torch.ones(h.shape[0])


def _force_atg_fire(policy):
    """Replace the policy's anchor_gate with an always-fire stub.

    nn.Module assignment checks the type, so we must use a Module subclass
    rather than a lambda.
    """
    policy.anchor_gate = _AlwaysFire()


# ── Shared fakes ─────────────────────────────────────────────────────────────

class FakeTokenizer:
    eos_token_id = 0

    def decode(self, ids, skip_special_tokens: bool = False):
        return " ".join(str(int(i)) for i in ids if not (skip_special_tokens and int(i) == 0))

    def batch_decode(self, ids):
        return [str(int(i)) for i in ids]


class FakeBackbone(Backbone):
    """Emits a deterministic, monotonically increasing hidden state.

    Step t produces hidden[0, :] = t. This lets us read back the buffer
    contents from the orchestrator and verify the sliding-window invariant.
    """

    @property
    def hidden_dim(self): return D
    @property
    def vocab_size(self): return V
    @property
    def lora_target_modules(self): return []
    @property
    def device(self): return "cpu"

    def tokenizer(self): return FakeTokenizer()

    def prepare_inputs(self, image, prompt):
        return {"past_key_values": None}

    def forward_step(self, inputs, generated_ids):
        step = generated_ids.shape[1]
        hidden = torch.full((1, D), float(step))
        logits = torch.zeros(1, V)
        # Force a non-EOS token (1) every step.
        logits[0, 1] = 100.0
        inputs["past_key_values"] = (inputs.get("past_key_values") or 0) + 1
        return hidden, logits


class FakeOracle(Oracle):
    def extract(self, image, image_meta=None) -> OracleEvidence:
        return OracleEvidence(scene_graph=SceneGraph())
    def vocab_scores(self, tokenizer, evidence):
        return {}


class _BufferSpyPolicy:
    """Minimal policy that records the buffer it sees at every step."""

    def __init__(self):
        self.seen_buffers: list[list | None] = []

    def init_state(self, evidence):
        return {}

    def adjust_logits(self, state: GenerationState):
        # Snapshot the buffer contents (shallow copy so later mutations don't affect us).
        if state.hidden_buffer is None:
            self.seen_buffers.append(None)
        else:
            self.seen_buffers.append([h.clone() for h in state.hidden_buffer])
        return torch.zeros_like(state.lm_logits)

    def update_state(self, state, sampled_token_id):
        return state.policy_state


# ── Orchestrator buffer maintenance ──────────────────────────────────────────

def test_orchestrator_passes_buffer_to_policy():
    backbone = FakeBackbone()
    oracle = FakeOracle()
    spy = _BufferSpyPolicy()
    dec = HallucinationDecoder(backbone, oracle, spy, hidden_buffer_size=4, max_new_tokens=3)
    dec.generate(object(), "prompt")
    # 3 steps → 3 snapshots. Each snapshot includes hidden state for steps 0..t.
    assert len(spy.seen_buffers) == 3
    # Snapshot at step 0 contains exactly [h_0]; step 1 → [h_0, h_1]; step 2 → [h_0, h_1, h_2].
    assert [len(b) for b in spy.seen_buffers] == [1, 2, 3]
    # Hidden at step t is the tensor full of value t.
    for t, snap in enumerate(spy.seen_buffers):
        for k, h in enumerate(snap):
            assert torch.equal(h, torch.full((1, D), float(k))), \
                f"snap@step={t}, pos={k} should equal full-{k}, got {h}"


def test_orchestrator_buffer_window_caps_at_K():
    backbone = FakeBackbone()
    oracle = FakeOracle()
    spy = _BufferSpyPolicy()
    K = 3
    dec = HallucinationDecoder(backbone, oracle, spy, hidden_buffer_size=K, max_new_tokens=6)
    dec.generate(object(), "prompt")
    # After 6 steps with K=3, the last buffer should hold the last K hidden states
    # corresponding to steps 3, 4, 5.
    last = spy.seen_buffers[-1]
    assert len(last) == K
    expected_values = [3.0, 4.0, 5.0]
    for h, v in zip(last, expected_values):
        assert torch.equal(h, torch.full((1, D), v))


def test_orchestrator_buffer_disabled_when_size_zero():
    backbone = FakeBackbone()
    oracle = FakeOracle()
    spy = _BufferSpyPolicy()
    dec = HallucinationDecoder(backbone, oracle, spy, hidden_buffer_size=0, max_new_tokens=2)
    dec.generate(object(), "prompt")
    assert all(b is None for b in spy.seen_buffers)


def test_buffer_tensors_are_detached():
    """Hidden buffer entries must not carry autograd graph across steps."""
    backbone = FakeBackbone()
    oracle = FakeOracle()
    spy = _BufferSpyPolicy()
    dec = HallucinationDecoder(backbone, oracle, spy, hidden_buffer_size=4, max_new_tokens=2)
    dec.generate(object(), "prompt")
    for snap in spy.seen_buffers:
        for h in snap:
            assert not h.requires_grad


# ── DTSGODPolicy consumes buffer mean as prefix_summary ──────────────────────

def test_dtsgod_uses_buffer_mean_when_provided(monkeypatch):
    """When state.hidden_buffer is set, GP.forward must receive its mean as prefix_summary."""
    policy = DTSGODPolicy(hidden_dim=D, vocab_size=V)
    policy.eval()
    # Force the ATG to always fire so the slow path runs.
    _force_atg_fire(policy)

    captured_prefix: list[torch.Tensor] = []
    original_forward = policy.grounding_planner.forward

    def spy_forward(h_slow_prev, prefix_summary, evidence):
        captured_prefix.append(prefix_summary.detach().clone())
        return original_forward(h_slow_prev, prefix_summary, evidence)

    monkeypatch.setattr(policy.grounding_planner, "forward", spy_forward)

    evidence = OracleEvidence(scene_graph=SceneGraph(objects=[
        # Need at least one object for GP to do non-trivial work; otherwise
        # GP.forward returns h_slow_prev unchanged (the empty-SG no-op).
    ]))
    # Add a dummy object via direct dataclass — keep test self-contained.
    from sgod.core.types import ObjectNode
    evidence = OracleEvidence(scene_graph=SceneGraph(
        objects=[ObjectNode("dog", 0.9, (0, 0, 1, 1))],
        relations=[], attributes=[], image_size=(640, 480),
    ))
    h_curr = torch.full((1, D), 10.0)
    buffer = [torch.full((1, D), 2.0), torch.full((1, D), 4.0), torch.full((1, D), 6.0)]
    # Mean over buffer = 4.0, not h_curr (10.0).
    state = GenerationState(
        prompt_ids=torch.zeros(1, 1, dtype=torch.long),
        generated_ids=torch.zeros(1, 1, dtype=torch.long),
        hidden_states=h_curr,
        lm_logits=torch.zeros(1, V),
        evidence=evidence,
        step=0,
        policy_state=policy.init_state(evidence),
        hidden_buffer=buffer,
    )
    policy.adjust_logits(state)
    assert len(captured_prefix) == 1
    assert torch.allclose(captured_prefix[0], torch.full((1, D), 4.0))


def test_dtsgod_falls_back_to_current_hidden_without_buffer(monkeypatch):
    policy = DTSGODPolicy(hidden_dim=D, vocab_size=V)
    policy.eval()
    _force_atg_fire(policy)

    captured: list[torch.Tensor] = []
    original = policy.grounding_planner.forward

    def spy(h_slow_prev, prefix_summary, evidence):
        captured.append(prefix_summary.detach().clone())
        return original(h_slow_prev, prefix_summary, evidence)

    monkeypatch.setattr(policy.grounding_planner, "forward", spy)

    from sgod.core.types import ObjectNode
    evidence = OracleEvidence(scene_graph=SceneGraph(
        objects=[ObjectNode("dog", 0.9, (0, 0, 1, 1))],
        relations=[], attributes=[], image_size=(640, 480),
    ))
    h_curr = torch.full((1, D), 7.0)
    state = GenerationState(
        prompt_ids=torch.zeros(1, 1, dtype=torch.long),
        generated_ids=torch.zeros(1, 1, dtype=torch.long),
        hidden_states=h_curr,
        lm_logits=torch.zeros(1, V),
        evidence=evidence,
        step=0,
        policy_state=policy.init_state(evidence),
        hidden_buffer=None,
    )
    policy.adjust_logits(state)
    assert torch.equal(captured[0], h_curr), "fallback to h_t when no buffer"


def test_g1_invariant_holds_with_buffer():
    """γ=0 ⇒ Δ=0 even when GP runs with a non-trivial prefix_summary from the buffer."""
    policy = DTSGODPolicy(hidden_dim=D, vocab_size=V, config=DTSGODConfig())
    policy.eval()

    from sgod.core.types import ObjectNode
    evidence = OracleEvidence(scene_graph=SceneGraph(
        objects=[ObjectNode("dog", 0.9, (0, 0, 1, 1))],
        relations=[], attributes=[], image_size=(640, 480),
    ))
    # Force ATG fire and provide a real buffer.
    _force_atg_fire(policy)
    buffer = [torch.randn(1, D) for _ in range(5)]
    state = GenerationState(
        prompt_ids=torch.zeros(1, 1, dtype=torch.long),
        generated_ids=torch.zeros(1, 1, dtype=torch.long),
        hidden_states=torch.randn(1, D),
        lm_logits=torch.randn(1, V),
        evidence=evidence,
        step=0,
        policy_state=policy.init_state(evidence),
        hidden_buffer=buffer,
    )
    delta = policy.adjust_logits(state)
    assert torch.all(delta == 0), f"max |Δ| = {delta.abs().max().item()}"
