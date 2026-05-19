"""End-to-end smoke tests with REAL models.

Unlike `test_backbone_llava.py` (which uses an empty oracle to isolate
the G1 invariant), this file exercises the full SGOD pipeline:

    LLaVA-1.5-7B    backbone (frozen)
        ↓
    RelTR           oracle  (Visual Genome SGG)
        ↓
    SGODv1Policy    Δ-injection via oracle scores
        ↓
    HallucinationDecoder runtime

Run on a real image and confirm:
  1. The oracle extracts a non-empty SceneGraph.
  2. SGODv1Policy injects a Δ at >0 anchor positions (oracle is doing work).
  3. The decoder produces a non-empty answer.
  4. DT-SGOD@init wrapping the same Oracle still preserves Δ=0 (G1 holds
     even when evidence is non-trivial — the policy's γ=0 wins).

Skip cleanly if checkpoints / images are missing; this is the integration
suite, not the unit suite.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

# ── Resource discovery ───────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RELTR_CKPT = _REPO_ROOT / "data" / "checkpoints" / "reltr" / "checkpoint0149.pth"
_CLIP_CACHE_CANDIDATES = [
    _REPO_ROOT / "data" / "checkpoints" / "clip" / "clip_vocab_cache.pt",
    _REPO_ROOT / "data" / "checkpoints" / "clip" / "clip_vocab_cache_reltr.pt",
]
_TEST_IMG = _REPO_ROOT / "test_imgs" / "image.png"


def _find_clip_cache() -> Path | None:
    for p in _CLIP_CACHE_CANDIDATES:
        if p.exists():
            return p
    return None


def _require_resources():
    """Skip helper. Returns (img_path, clip_cache_or_none)."""
    if not _RELTR_CKPT.exists():
        pytest.skip(f"RelTR checkpoint not found at {_RELTR_CKPT}")
    if not _TEST_IMG.exists():
        pytest.skip(f"Test image not found at {_TEST_IMG}")
    return _TEST_IMG, _find_clip_cache()


def _smoke_load_in_4bit() -> bool:
    return os.environ.get("SMOKE_4BIT", "0") == "1"


# ── Real-pipeline tests ──────────────────────────────────────────────────────

@pytest.mark.integration
def test_real_pipeline_sgod_v1_generates_grounded_answer(capsys):
    """SGOD v1 driven through the new framework on a real LLaVA + RelTR + CLIP stack.

    Promises:
      - Oracle extracts at least one object from the test image.
      - The Δ-injection path fires at ≥1 anchor token during generation.
      - The decoder produces a non-empty answer.
    """
    from PIL import Image

    from sgod.backbones.llava15 import LLaVAv15Backbone
    from sgod.oracles import RelTROracle
    from sgod.policies import SGODv1Policy
    from sgod.runtime import HallucinationDecoder
    from sgod.utils.model_loader import load_clip_factory

    pytest.importorskip("transformers")
    pytest.importorskip("open_clip")
    img_path, clip_cache = _require_resources()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = LLaVAv15Backbone(load_in_4bit=_smoke_load_in_4bit(), lazy=False)
    oracle = RelTROracle(
        checkpoint_path=str(_RELTR_CKPT),
        device=device,
        confidence_threshold=0.3,
        top_k=20,
    )

    # Pre-test: oracle must produce a non-empty SceneGraph on a real photo.
    img = Image.open(img_path).convert("RGB")
    evidence = oracle.extract(img)
    sg = evidence.scene_graph
    assert len(sg.objects) > 0, (
        f"RelTR must detect ≥1 object in {img_path}; got 0. "
        f"Check checkpoint compatibility / image content."
    )
    obj_labels = sorted({o.label.lower() for o in sg.objects})

    # CLIP factory; pass cache only if found.
    clip_factory = load_clip_factory(
        model_name="ViT-L-14",
        vocab_cache_path=str(clip_cache) if clip_cache else None,
        device=device,
    )
    policy = SGODv1Policy(
        tokenizer=backbone.tokenizer(),
        clip_factory=clip_factory,
        top_k=50,
    )

    # Capture Δ stats via step_hook — verify oracle actually injects somewhere.
    anchor_fires = 0
    max_abs_delta = 0.0

    def _hook(_state, delta):
        nonlocal anchor_fires, max_abs_delta
        nz = float(delta.abs().sum().item())
        if nz > 0:
            anchor_fires += 1
            max_abs_delta = max(max_abs_delta, float(delta.abs().max().item()))

    decoder = HallucinationDecoder(
        backbone=backbone, oracle=oracle, policy=policy,
        max_new_tokens=40, temperature=0.0, hidden_buffer_size=0,
        step_hook=_hook,
    )

    question = "What is in the image?"
    prompt = f"USER: <image>\n{question} ASSISTANT:"
    answer = decoder.generate(img, prompt, question=question)

    # Always print for diagnostic value when running with -s.
    with capsys.disabled():
        print(f"\n[smoke] image:        {img_path.name}")
        print(f"[smoke] sg.objects:   {obj_labels}")
        print(f"[smoke] sg.relations: {[f'{r.subject}-{r.predicate}-{r.object}' for r in sg.relations[:5]]}")
        print(f"[smoke] answer:       {answer!r}")
        print(f"[smoke] anchor_fires: {anchor_fires}, max |Δ|: {max_abs_delta:.4g}")

    assert isinstance(answer, str)
    assert len(answer.strip()) > 0, "decoder produced empty answer"
    assert anchor_fires > 0, (
        f"SGOD v1 oracle never fired across {decoder.max_new_tokens} tokens. "
        f"Pipeline is bypassing the oracle — verify anchor detector + question type."
    )


@pytest.mark.integration
def test_real_pipeline_dt_sgod_at_init_preserves_g1_with_real_oracle():
    """G1 still holds when the oracle is non-trivial (real RelTR scene graph).

    The previous LLaVA G1 test used an empty oracle — there's a theoretical
    chance the policy interacts with rich evidence in some unexpected way.
    This test plugs in a real RelTR oracle and re-verifies Δ ≡ 0 at init.
    """
    from PIL import Image

    from sgod.backbones.llava15 import LLaVAv15Backbone
    from sgod.oracles import RelTROracle
    from sgod.policies import DTSGODPolicy
    from sgod.runtime import HallucinationDecoder

    pytest.importorskip("transformers")
    img_path, _ = _require_resources()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = LLaVAv15Backbone(load_in_4bit=_smoke_load_in_4bit(), lazy=False)
    oracle = RelTROracle(
        checkpoint_path=str(_RELTR_CKPT),
        device=device,
        confidence_threshold=0.3,
        top_k=20,
    )
    policy = DTSGODPolicy(hidden_dim=backbone.hidden_dim, vocab_size=backbone.vocab_size)
    policy.eval()
    assert policy.is_identity_at_init()

    max_abs_deltas: list[float] = []
    def _hook(_state, delta):
        max_abs_deltas.append(float(delta.abs().max().item()))

    decoder = HallucinationDecoder(
        backbone=backbone, oracle=oracle, policy=policy,
        max_new_tokens=10, temperature=0.0,
        hidden_buffer_size=4,  # exercise the buffer path
        step_hook=_hook,
    )
    img = Image.open(img_path).convert("RGB")
    decoder.generate(img, "USER: <image>\nDescribe the image. ASSISTANT:",
                     question="Describe the image.")

    assert max_abs_deltas, "step_hook must fire at least once"
    assert all(d == 0.0 for d in max_abs_deltas), (
        f"Δ must be exactly 0 at init even with a non-trivial SceneGraph;\n"
        f"got max |Δ| = {max(max_abs_deltas):.3e} (sample: {max_abs_deltas[:5]}...)"
    )
