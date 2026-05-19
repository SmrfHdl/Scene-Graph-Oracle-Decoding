"""Unit tests for LLaVAv15Backbone.forward_step.

CPU-only: the underlying HF model is mocked. We verify:
  - First call passes full prompt + pixel_values, captures past_key_values.
  - Subsequent calls send only the last token + cached past_key_values.
  - Returned (hidden, logits) match the slice contract: [B, d] / [B, V].
  - Cache mutation: `inputs["past_key_values"]` is updated in place.

The real-model smoke tests live behind `@pytest.mark.integration`. They
default to fp16 (~14 GB VRAM, fits comfortably on a 24 GB 4090). Set
`SMOKE_4BIT=1` to opt into 4-bit quantization (needs `bitsandbytes`).
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

from sgod.backbones.llava15 import LLaVAv15Backbone


def _smoke_load_in_4bit() -> bool:
    """Whether integration smokes should use 4-bit quantization.

    Read at test time, not import time, so flipping the env var doesn't
    require a reimport.
    """
    return os.environ.get("SMOKE_4BIT", "0") == "1"


D = 16   # tiny mock hidden_dim
V = 32   # tiny mock vocab_size


class _MockModel:
    """Stand-in for HF LlavaForConditionalGeneration.

    Records call args so the test can verify what the backbone sent in.
    Returns a SimpleNamespace shaped like HF's CausalLMOutputWithPast +
    `hidden_states` (a tuple-of-layers).
    """

    def __init__(self):
        self.calls: list[dict] = []
        # One "parameter" so `next(model.parameters()).device` works.
        self._param = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        return iter([self._param])

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        # Determine sequence length from input_ids.
        seq_len = kwargs["input_ids"].shape[1]
        B = kwargs["input_ids"].shape[0]
        hidden = torch.randn(B, seq_len, D)
        logits = torch.randn(B, seq_len, V)
        # New cache is just a counter so we can verify mutation in place.
        new_cache = (kwargs.get("past_key_values") or 0) + 1
        return SimpleNamespace(
            logits=logits,
            past_key_values=new_cache,
            hidden_states=(torch.randn(B, seq_len, D), hidden),  # 2 layers; last is real
        )


class _MockProcessor:
    class _Tok:
        eos_token_id = 2

    tokenizer = _Tok()

    def __call__(self, images, text, return_tensors="pt"):
        # Return a dict-like with input_ids + pixel_values + attention_mask.
        return {
            "input_ids": torch.tensor([[1, 5, 6, 7]]),       # [1, 4] prompt
            "pixel_values": torch.zeros(1, 3, 16, 16),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
        }


def _backbone_with_mock() -> tuple[LLaVAv15Backbone, _MockModel, _MockProcessor]:
    b = LLaVAv15Backbone(
        hidden_dim_override=D,
        vocab_size_override=V,
        lazy=True,
    )
    model = _MockModel()
    proc = _MockProcessor()
    # Bypass HF load by injecting mocks directly.
    b._model = model
    b._processor = proc
    return b, model, proc


# ── happy path ───────────────────────────────────────────────────────────────

def test_forward_step_first_call_uses_full_prompt_and_pixel_values():
    backbone, model, _ = _backbone_with_mock()
    inputs = backbone.prepare_inputs(image=object(), prompt="describe")
    assert inputs["past_key_values"] is None

    gen_ids = torch.empty(1, 0, dtype=torch.long)  # nothing sampled yet
    hidden, logits = backbone.forward_step(inputs, gen_ids)

    assert len(model.calls) == 1
    call = model.calls[0]
    assert call["input_ids"].shape == (1, 4), "first call should use full prompt"
    assert call["pixel_values"] is not None
    assert "past_key_values" not in call, "first call must not pass past_key_values"
    assert call["use_cache"] is True
    assert call["output_hidden_states"] is True
    assert hidden.shape == (1, D)
    assert logits.shape == (1, V)
    # Cache must be mutated in place.
    assert inputs["past_key_values"] == 1


def test_forward_step_subsequent_call_passes_only_last_token_and_cache():
    backbone, model, _ = _backbone_with_mock()
    inputs = backbone.prepare_inputs(image=object(), prompt="describe")

    # First step.
    backbone.forward_step(inputs, torch.empty(1, 0, dtype=torch.long))
    cache_after_first = inputs["past_key_values"]
    assert cache_after_first == 1

    # Second step: orchestrator has appended one new token id.
    gen_ids = torch.tensor([[42]])
    hidden, logits = backbone.forward_step(inputs, gen_ids)

    call = model.calls[1]
    assert call["input_ids"].shape == (1, 1), "incremental call sends 1 token"
    assert call["input_ids"][0, 0].item() == 42
    assert call["past_key_values"] == cache_after_first
    assert "pixel_values" not in call, "no pixel_values on incremental steps"
    assert hidden.shape == (1, D)
    assert logits.shape == (1, V)
    assert inputs["past_key_values"] == 2  # bumped


def test_forward_step_returns_last_token_slice_not_full_sequence():
    """hidden/logits must be the NEW token's only — shape [B, D] / [B, V]."""
    backbone, model, _ = _backbone_with_mock()
    inputs = backbone.prepare_inputs(image=object(), prompt="x y z w v")  # 4-token prompt
    hidden, logits = backbone.forward_step(inputs, torch.empty(1, 0, dtype=torch.long))
    # Model returned [B, 4, D] / [B, 4, V]; backbone must slice last position.
    assert hidden.dim() == 2 and hidden.shape == (1, D)
    assert logits.dim() == 2 and logits.shape == (1, V)


def test_prepare_inputs_initializes_empty_cache():
    backbone, _, _ = _backbone_with_mock()
    inputs = backbone.prepare_inputs(image=object(), prompt="x")
    assert "past_key_values" in inputs
    assert inputs["past_key_values"] is None


# ── integration (real LLaVA on GPU — opt-in) ─────────────────────────────────

@pytest.mark.integration
def test_forward_step_real_llava_smoke():
    """Load LLaVA-1.5-7B and run 3 incremental decode steps.

    Skipped by default; run with `pytest -m integration`. Requires GPU.
    """
    from PIL import Image

    pytest.importorskip("transformers")

    backbone = LLaVAv15Backbone(
        load_in_4bit=_smoke_load_in_4bit(),
        lazy=False,
    )
    img = Image.new("RGB", (336, 336), color=(127, 127, 127))
    prompt = "USER: <image>\nDescribe the image. ASSISTANT:"
    inputs = backbone.prepare_inputs(img, prompt)

    gen = torch.empty(1, 0, dtype=torch.long)
    for step in range(3):
        hidden, logits = backbone.forward_step(inputs, gen)
        assert hidden.shape == (1, backbone.hidden_dim)
        assert logits.shape == (1, backbone.vocab_size)
        next_id = int(torch.argmax(logits, dim=-1).item())
        gen = torch.cat([gen, torch.tensor([[next_id]])], dim=-1)


@pytest.mark.integration
def test_dt_sgod_at_init_matches_baseline_on_real_llava():
    """G1 invariant end-to-end: DT-SGOD@init wrapping real LLaVA = LLaVA-base.

    With γ=0 and out_proj=0, the policy's Δ ≡ 0, so wrapping the backbone in
    a HallucinationDecoder driven by DTSGODPolicy must produce *identical*
    tokens to driving the backbone alone. Single byte difference = bug.

    This is the canonical Gate G1 verification on real hardware.
    """
    from PIL import Image

    from sgod.core.interfaces import Oracle
    from sgod.core.types import OracleEvidence, SceneGraph
    from sgod.policies import DTSGODPolicy
    from sgod.runtime import HallucinationDecoder

    pytest.importorskip("transformers")

    class _EmptyOracle(Oracle):
        def extract(self, image, image_meta=None):
            return OracleEvidence(scene_graph=SceneGraph())
        def vocab_scores(self, tokenizer, evidence): return {}

    backbone = LLaVAv15Backbone(load_in_4bit=_smoke_load_in_4bit(), lazy=False)
    img = Image.new("RGB", (336, 336), color=(127, 127, 127))
    prompt = "USER: <image>\nDescribe the image. ASSISTANT:"
    n_tokens = 10

    # ── Path A: backbone alone, greedy argmax loop.
    inputs_a = backbone.prepare_inputs(img, prompt)
    gen_a = torch.empty(1, 0, dtype=torch.long)
    for _ in range(n_tokens):
        _, logits = backbone.forward_step(inputs_a, gen_a)
        next_id = int(torch.argmax(logits, dim=-1).item())
        gen_a = torch.cat([gen_a, torch.tensor([[next_id]])], dim=-1)
        if next_id == backbone.tokenizer().eos_token_id:
            break
    tokens_a = gen_a[0].tolist()

    # ── Path B: same backbone via HallucinationDecoder + DTSGODPolicy@init.
    policy = DTSGODPolicy(
        hidden_dim=backbone.hidden_dim, vocab_size=backbone.vocab_size,
    )
    policy.eval()
    assert policy.is_identity_at_init(), "policy must be identity at init"

    decoder = HallucinationDecoder(
        backbone=backbone, oracle=_EmptyOracle(), policy=policy,
        max_new_tokens=n_tokens, temperature=0.0, hidden_buffer_size=0,
    )
    # Decode via the orchestrator. We compare the produced token sequence,
    # not just the decoded string, to catch any single-token divergence.
    tokens_b_str = decoder.generate(image=img, prompt=prompt, question="Describe the image.")
    # Drive the loop manually mirroring HallucinationDecoder so we have
    # the token IDs in hand for a head-to-head ID-level comparison with Path A.
    inputs_b = backbone.prepare_inputs(img, prompt)
    evidence = _EmptyOracle().extract(img)
    evidence.extra["image"] = img
    evidence.extra["question"] = "Describe the image."
    policy_state = policy.init_state(evidence)
    gen_b = torch.empty(1, 0, dtype=torch.long)
    from sgod.core.types import GenerationState
    for step in range(n_tokens):
        hidden, lm_logits = backbone.forward_step(inputs_b, gen_b)
        state = GenerationState(
            prompt_ids=torch.empty(1, 0, dtype=torch.long),
            generated_ids=gen_b,
            hidden_states=hidden,
            lm_logits=lm_logits,
            evidence=evidence,
            step=step,
            policy_state=policy_state,
        )
        delta = policy.adjust_logits(state)
        assert torch.all(delta == 0), \
            f"step {step}: Δ must be exactly zero at init; got max |Δ|={delta.abs().max().item()}"
        next_id = int(torch.argmax((lm_logits + delta)[0]).item())
        gen_b = torch.cat([gen_b, torch.tensor([[next_id]])], dim=-1)
        policy_state = policy.update_state(state, sampled_token_id=next_id)
        if next_id == backbone.tokenizer().eos_token_id:
            break
    tokens_b = gen_b[0].tolist()

    assert tokens_a == tokens_b, (
        f"DT-SGOD@init must produce identical tokens to baseline.\n"
        f"  baseline: {tokens_a}\n"
        f"  dt-sgod:  {tokens_b}\n"
        f"  first diff at index "
        f"{next((i for i, (a, b) in enumerate(zip(tokens_a, tokens_b)) if a != b), 'n/a')}"
    )
    # The string-level result from the orchestrator should also match the
    # baseline (sanity that the orchestrator path doesn't strip anything weird).
    tokens_a_str = backbone.tokenizer().decode(tokens_a, skip_special_tokens=True)
    assert tokens_a_str == tokens_b_str, (
        f"orchestrator decode differs from baseline decode:\n"
        f"  baseline: {tokens_a_str!r}\n"
        f"  orch:     {tokens_b_str!r}"
    )


@pytest.mark.integration
def test_orchestrator_end_to_end_with_real_llava():
    """Orchestrator produces a non-empty answer when driving real LLaVA.

    Lighter than the G1 invariant test — just verifies the framework's
    generate() path returns plausible output on a real model. Useful as a
    quick smoke when iterating on the orchestrator without re-running the
    full G1 comparison.
    """
    from PIL import Image

    from sgod.core.interfaces import Oracle
    from sgod.core.types import OracleEvidence, SceneGraph
    from sgod.policies import DTSGODPolicy
    from sgod.runtime import HallucinationDecoder

    pytest.importorskip("transformers")

    class _EmptyOracle(Oracle):
        def extract(self, image, image_meta=None):
            return OracleEvidence(scene_graph=SceneGraph())
        def vocab_scores(self, tokenizer, evidence): return {}

    backbone = LLaVAv15Backbone(load_in_4bit=_smoke_load_in_4bit(), lazy=False)
    img = Image.new("RGB", (336, 336), color=(127, 127, 127))
    policy = DTSGODPolicy(hidden_dim=backbone.hidden_dim, vocab_size=backbone.vocab_size)
    policy.eval()
    decoder = HallucinationDecoder(
        backbone=backbone, oracle=_EmptyOracle(), policy=policy,
        max_new_tokens=20, temperature=0.0, hidden_buffer_size=4,
    )
    answer = decoder.generate(
        image=img,
        prompt="USER: <image>\nWhat colour is the image? ASSISTANT:",
        question="What colour is the image?",
    )
    assert isinstance(answer, str)
    assert len(answer.strip()) > 0, "model produced empty answer"
