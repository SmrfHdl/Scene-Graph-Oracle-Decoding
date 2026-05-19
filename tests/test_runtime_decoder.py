"""Tests for HallucinationDecoder — the Backbone+Oracle+Policy orchestrator.

Exercises the full generate loop with CPU-only fakes:
  - EOS stop, max_new_tokens stop.
  - Backbone receives correct incremental gen_ids (proxy for KV use).
  - DT-SGOD@init wrapping a backbone is bit-identical to that backbone alone
    (G1 invariant carries through the orchestrator).
  - SGOD v1 wired via the orchestrator boosts SG nouns at anchor positions.
"""
from __future__ import annotations

import torch

from sgod.core.interfaces import Backbone, Oracle
from sgod.core.types import OracleEvidence
from sgod.policies import DTSGODPolicy, SGODv1Policy
from sgod.runtime import HallucinationDecoder
from sgod.sgg.scene_graph import ObjectNode, RelationEdge, SceneGraph


# ── shared fakes ─────────────────────────────────────────────────────────────

VOCAB = {
    "<eos>": 0,
    "the": 1, "dog": 2, "is": 3, "on": 4, "mat": 5,
    "red": 6, "no": 7, "a": 8, "cat": 9, "?": 10,
}
ID2WORD = {v: k for k, v in VOCAB.items()}
EOS_ID = VOCAB["<eos>"]
V = len(VOCAB) + 1
D = 8


class FakeTokenizer:
    eos_token_id = EOS_ID

    def encode(self, text: str) -> list[int]:
        return [VOCAB.get(w, 99) for w in text.lower().split()]

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        if skip_special_tokens:
            ids = [i for i in ids if i != EOS_ID]
        return " ".join(ID2WORD.get(int(i), "?") for i in ids)

    def batch_decode(self, ids) -> list[str]:
        return [ID2WORD.get(int(i), "?") for i in ids]


class FakeBackbone(Backbone):
    """Emits a deterministic token sequence; records forward_step args.

    `token_sequence` is the IDs that argmax should land on at each step.
    Hidden state is a deterministic function of step so we can detect when
    the orchestrator forwards the right thing.
    """

    def __init__(self, token_sequence: list[int]) -> None:
        self.token_sequence = token_sequence
        self.calls: list[dict] = []

    # interface
    @property
    def hidden_dim(self) -> int: return D
    @property
    def vocab_size(self) -> int: return V
    @property
    def lora_target_modules(self) -> list[str]: return []
    @property
    def device(self) -> str: return "cpu"

    def tokenizer(self): return FakeTokenizer()

    def prepare_inputs(self, image, prompt):
        return {"prompt": prompt, "image": image, "past_key_values": None}

    def forward_step(self, inputs, generated_ids):
        step = generated_ids.shape[1]
        self.calls.append({
            "step": step,
            "gen_ids_len": int(generated_ids.shape[1]),
            "cache_state": inputs.get("past_key_values"),
        })
        # Drive logits so argmax returns token_sequence[step].
        target = self.token_sequence[min(step, len(self.token_sequence) - 1)]
        logits = torch.full((1, V), -10.0)
        logits[0, target] = 100.0
        hidden = torch.full((1, D), float(step))
        # Mutate KV cache slot like the real backbone does.
        inputs["past_key_values"] = (inputs.get("past_key_values") or 0) + 1
        return hidden, logits


class FakeSGOracle(Oracle):
    def __init__(self, sg: SceneGraph) -> None:
        self._sg = sg

    def extract(self, image, image_meta=None) -> OracleEvidence:
        return OracleEvidence(scene_graph=self._sg)

    def vocab_scores(self, tokenizer, evidence) -> dict[int, float]:
        return {}


class FakeCLIPScorer:
    def score_single(self, w: str) -> float:
        return 0.5


def _good_sg() -> SceneGraph:
    return SceneGraph(
        objects=[ObjectNode("dog", 0.9, (0, 0, 1, 1)),
                 ObjectNode("mat", 0.9, (0, 0, 1, 1))],
        relations=[RelationEdge("dog", "on", "mat", 0.9)],
        attributes=[],
    )


# ── EOS / max_new_tokens stops ───────────────────────────────────────────────

def test_generate_stops_at_eos():
    backbone = FakeBackbone([VOCAB["dog"], EOS_ID, VOCAB["cat"]])
    oracle = FakeSGOracle(SceneGraph())  # empty → use_oracle False for SGOD v1
    policy = SGODv1Policy(tokenizer=FakeTokenizer(), clip_factory=lambda _i: FakeCLIPScorer())
    dec = HallucinationDecoder(backbone, oracle, policy)
    out = dec.generate(object(), "Is there a dog?", max_new_tokens=10)
    assert out.strip() == "dog"
    # Two forward calls: one to produce "dog", one to produce EOS.
    assert len(backbone.calls) == 2


def test_generate_stops_at_max_new_tokens():
    backbone = FakeBackbone([VOCAB["dog"]] * 5)
    oracle = FakeSGOracle(SceneGraph())
    policy = SGODv1Policy(tokenizer=FakeTokenizer(), clip_factory=lambda _i: FakeCLIPScorer())
    dec = HallucinationDecoder(backbone, oracle, policy, max_new_tokens=3)
    out = dec.generate(object(), "What is this?")
    assert out.strip().split() == ["dog", "dog", "dog"]


# ── Incremental KV usage (proxy: cache slot mutated, gen_ids grows) ─────────

def test_backbone_receives_growing_generated_ids():
    backbone = FakeBackbone([VOCAB["a"], VOCAB["dog"], EOS_ID])
    oracle = FakeSGOracle(SceneGraph())
    policy = SGODv1Policy(tokenizer=FakeTokenizer(), clip_factory=lambda _i: FakeCLIPScorer())
    dec = HallucinationDecoder(backbone, oracle, policy)
    dec.generate(object(), "Is there a dog?", max_new_tokens=10)
    # Each call's gen_ids length should equal step index (0,1,2,...).
    seen_lens = [c["gen_ids_len"] for c in backbone.calls]
    assert seen_lens == list(range(len(seen_lens)))


def test_backbone_cache_slot_mutated_in_place():
    backbone = FakeBackbone([VOCAB["dog"], EOS_ID])
    oracle = FakeSGOracle(SceneGraph())
    policy = SGODv1Policy(tokenizer=FakeTokenizer(), clip_factory=lambda _i: FakeCLIPScorer())
    dec = HallucinationDecoder(backbone, oracle, policy)
    dec.generate(object(), "x", max_new_tokens=5)
    # First call sees no cache (None); subsequent calls see the bumped int.
    assert backbone.calls[0]["cache_state"] is None
    assert backbone.calls[1]["cache_state"] == 1


# ── DT-SGOD@init invariant carries through the orchestrator ─────────────────

def test_dt_sgod_at_init_identical_to_baseline_argmax():
    """With γ=0, orchestrator output must match plain argmax of backbone logits."""
    seq = [VOCAB["the"], VOCAB["dog"], EOS_ID]
    backbone = FakeBackbone(seq)
    oracle = FakeSGOracle(_good_sg())
    policy = DTSGODPolicy(hidden_dim=D, vocab_size=V)
    policy.eval()

    dec = HallucinationDecoder(backbone, oracle, policy)
    out = dec.generate(object(), "Describe.", max_new_tokens=10)
    assert out.strip() == "the dog"


# ── SGOD v1 wired via orchestrator: anchor boost flips a near-tied logit ─────

def test_sgod_v1_orchestrated_anchor_boost():
    """At noun_anchor (prev='a'), SG-in-vocab 'dog' must beat near-tied 'cat'."""

    class BiasedBackbone(FakeBackbone):
        def forward_step(self, inputs, generated_ids):
            step = generated_ids.shape[1]
            self.calls.append({"step": step,
                               "gen_ids_len": int(generated_ids.shape[1]),
                               "cache_state": inputs.get("past_key_values")})
            logits = torch.full((1, V), -100.0)
            if step == 0:
                logits[0, VOCAB["a"]] = 100.0     # step 0 emits "a"
            else:
                # step 1+: cat narrowly beats dog → without oracle, "cat" wins.
                logits[0, VOCAB["cat"]] = 9.1
                logits[0, VOCAB["dog"]] = 9.0
                logits[0, EOS_ID] = -50.0
            inputs["past_key_values"] = (inputs.get("past_key_values") or 0) + 1
            return torch.zeros(1, D), logits

    sg = SceneGraph(
        objects=[ObjectNode("dog", 0.99, (0, 0, 1, 1))],
        relations=[], attributes=[],
    )
    backbone = BiasedBackbone([])  # token_sequence unused
    oracle = FakeSGOracle(sg)
    policy = SGODv1Policy(
        tokenizer=FakeTokenizer(),
        clip_factory=lambda _i: FakeCLIPScorer(),
        top_k=V,
    )
    dec = HallucinationDecoder(backbone, oracle, policy)
    out = dec.generate(object(), "Is there a ?", max_new_tokens=3)
    # After "a", oracle boost on "dog" (in SG, conf 0.99) should beat "cat".
    assert "dog" in out
