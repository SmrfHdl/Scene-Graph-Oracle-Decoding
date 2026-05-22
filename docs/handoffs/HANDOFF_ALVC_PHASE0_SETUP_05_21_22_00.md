# Handoff: ALVC Phase 0 Setup Complete — Direction Committed, Ready to Implement

**Created:** 2026-05-21 ~22:00 local
**Branch:** `feat/alvc-phase0` (5 commits, pushed to origin)
**Session Duration:** ~6h (continued from VR-TTS failed reassessment)

---

## Summary

After extensive brainstorm, pivoted from VR-TTS (failed at n=1000) to **ALVC (Adaptive-Length Vision Connector)** as the Q1 paper direction — a Shape 2 architecture primitive at the VLM connector level (BLIP-2 reference). Backbone switched from TinyLLaVA (transformers compat issues) to Phi-3.5-Vision-Instruct, with forward pass verified end-to-end on uet. Skeleton + sanity script + tests committed. Ready to implement HaltingHead + ALVCConnector for week 1 of Phase 0.

---

## Work Completed

### Changes Made

- [x] Brainstormed 10+ candidate directions; rejected lego-shape pitches; converged on ALVC after taste calibration (Shape 2 architecture primitive, BLIP-2 reference)
- [x] Dispatched 2 literature/theory agents in parallel — verified whitespace clean (zero papers with query-conditioned K + PonderNet halting + rate-distortion in VLM connector)
- [x] **Critical math correction:** original theorem `K* ≤ I/log(vocab)` flagged as vacuous (vision tokens are continuous embeddings, not vocab indices). Corrected to `K ≥ ⌈I(X;Y|Q) / C⌉` where C = per-token continuous channel capacity in nats
- [x] Branch `feat/alvc-phase0` created from `feat/vrtts-phase1` (user explicitly requested not to base off main)
- [x] Folder skeleton `sgod/policies/alvc/` with 4 modules (connector, halting, training, eval) + package docstring with foundational claim
- [x] Sanity script `scripts/sanity_alvc_phi35v.py` with 3 modes: `--import-only` (CPU), `--no-forward` (load only), full (GPU)
- [x] 4 skeleton tests passing locally
- [x] Backbone pivot: TinyLLaVA → Phi-3.5-Vision (compat issue with transformers 5.x `_supports_sdpa` removal)
- [x] Discovered empirically that Phi-3.5-Vision projector receives **4096-dim** input (4 CLIP layers concatenated), not 1024 as `image_dim_out` suggests
- [x] Resolved uet env chain: torch downgrade to 2.5/2.6+cu118 (py3.13 + CUDA 12.0 driver compat), transformers downgrade to 4.45-4.49 (for `from_legacy_cache`), forward pass uses `use_cache=False` to skip broken Phi3V cache API path
- [x] All 5 commits pushed to `origin/feat/alvc-phase0`
- [x] Memory updated: new `project_alvc_direction.md`, `project_vrtts_direction.md` marked SUPERSEDED, MEMORY.md index updated

### Key Decisions

| Decision | Rationale | Alternatives Considered |
| --- | --- | --- |
| Pivot from VR-TTS to ALVC | VR-TTS zoom + MVP v4 both collapsed at n=1000 (structural — training-free can't add information). User wants Shape 2 (architecture primitive) like BLIP-2's Q-Former, not lego. | Continue VR-TTS, V-SAE, causal VLM, functional VLM, vision state machine |
| Shape 2 + BLIP-2 reference | User explicit taste: new computational block at vision-language interface. Not formalism-only (Shape 1), not problem-reframe (Shape 4), not new representation (Shape 6). | Shape 1/4/5/6 |
| ALVC over Vision State Machine (VSM) | Modern Hopfield = attention (Ramsauer 2020) weakens VSM foundational claim; ALVC math (info bottleneck + rate-distortion) is cleaner | VSM, DEQ-VC |
| Corrected theorem to use channel capacity C, not log(vocab) | Vision projector outputs continuous embeddings — vocab denominator is wrong dimension. Agent flagged original as vacuous. | Keep original (would fail review) |
| Lead paper with **query-conditional K** ablation | Matryoshka M3 already claims "variable K per image". Query-conditional is the cleanest single differentiator no prior paper has. | Lead with adaptive-length only (collapses into Matryoshka territory) |
| Phi-3.5-Vision over TinyLLaVA | TinyLLaVA's custom modeling code uses removed transformers API (`_supports_sdpa`); MS-maintained Phi-3.5-Vision more compat | Switch to LLaVA-1.5-7B, monkey-patch TinyLLaVA, SmolVLM |
| `use_cache=False` for sanity / Phase 0 | Phi3V cache API drift across `from_legacy_cache` and `get_usable_length` — death-by-thousand-patches. Cache only needed for generation, not Phase 0 training. | Patch DynamicCache, downgrade transformers further |
| Phase 0 G5 as critical week-1 gate | Without proving K depends on QUERY (not just image), paper claim collapses into Matryoshka territory. Test early. | Test G5 only at end of Phase 0 |

---

## Files Affected

### Created (all committed on `feat/alvc-phase0`)

- `sgod/policies/alvc/__init__.py` — package surface + foundational claim docstring + corrected theorem
- `sgod/policies/alvc/connector.py` — `ALVCConnector` + `ALVCConfig` skeleton. Defaults: n_visual_patches=144, d_v=4096, d_lm=3072, K_max=144 (Phi-3.5-Vision dims, empirically verified)
- `sgod/policies/alvc/halting.py` — `HaltingHead` + `HaltingConfig` (PonderNet-style, lambda_p=0.1, beta=0.01, query_conditioned=True by default)
- `sgod/policies/alvc/training.py` — rate-distortion loss placeholder
- `sgod/policies/alvc/eval.py` — G1-G5 gates documented (G5 = critical query-conditional test)
- `scripts/sanity_alvc_phi35v.py` — 3-mode sanity check
- `tests/test_alvc_skeleton.py` — 4 skeleton tests

### Memory (outside repo, in `~/.claude/projects/.../memory/`)

- `project_alvc_direction.md` (NEW) — direction, theorem, gates, backbone, uet env setup details
- `project_vrtts_direction.md` — marked **SUPERSEDED 2026-05-21**, original content preserved
- `MEMORY.md` — index updated with new entries

### Reference (not modified, read for context)

- `docs/handoffs/HANDOFF_VRTTS_FAILED_REASSESS_05_20_18_30.md` — prior session context
- `sgod/backbones/llava15.py`, `sgod/policies/vrtts/*` — existing patterns to mirror, kept isolated
- `pyproject.toml` — confirmed deps OK

---

## Technical Context

### Architecture / Design Notes

**ALVC primitive (foundational claim):**

> Existing VLM connectors (linear projector, Q-Former, Perceiver IO) output fixed-length token sequence regardless of image complexity or query. This is information-theoretically wasteful. ALVC replaces them with a learned policy π(K | image, query) producing variable K ∈ [K_min, K_max].

**Corrected theorem** (to prove during Phase 1-2):

```
K(x, q) ≥ ⌈I(X; Y | Q) / C⌉
```

where C = per-token continuous channel capacity ≈ d · log(2πeσ²)/2 (nats), estimated empirically per-layer. This is a Shannon-style rate-distortion converse; ALVC is the achievability via PonderNet-trained halting.

**Phi-3.5-Vision integration target:**

`model.model.vision_embed_tokens.img_projection` is the 2-layer MLP:
- `Linear(4096 → 3072) + GELU + Linear(3072 → 3072)`
- ~21M params
- This is what ALVC replaces

**Phase 0 architecture (sequence-level halting, num_crops=1):**

```
Phi-3.5-Vision features [B, 144, 4096]
        │
        ▼
┌──────────────────────────────────────┐
│  HaltingHead (PonderNet-style)       │  ← inputs: vision features + query embedding (d_q=3072)
│  outputs: distribution p(K)           │
└──────────────────────────────────────┘
        │
        ▼
Top-K selection (differentiable; design TBD — Gumbel-softmax or expected-loss formulation)
        │
        ▼
Linear projection [K, 4096] → [K, 3072] → LM token space
```

### Dependencies

- No new deps. Uses existing torch/transformers/PIL infrastructure.
- uet env: torch 2.5.x/2.6.x +cu118, transformers ≥4.45 <5
- No flash-attn (eager only) — fine for Phase 0; install later for training speedup

### Configuration Changes

- None at pyproject level. ALVC dims hard-coded in `ALVCConfig` and `HaltingConfig` defaults.
- Phase 0 fixes `num_crops=1` in processor (note: experimentally pixel_values still shows 2 crops — investigate when implementing halting)

---

## Things to Know

### Gotchas & Pitfalls

- **Phi-3.5-Vision modeling code uses legacy transformers cache API** (`DynamicCache.from_legacy_cache`, `get_usable_length`). Set `use_cache=False` for training and Phase 0 forward; only worry about cache for generation/eval.
- **`image_dim_out` ≠ projector input dim.** Phi-3.5-Vision config reports `image_dim_out=1024` but the projector receives 4096 (4 CLIP layers concatenated). **Verify empirically** when integrating — don't trust config names.
- **Processor `num_crops=1` arg is ignored** — `pixel_values` still has 2 crops by default. Need to investigate to fix K_max calculation.
- **uet CUDA driver only supports CUDA 12.0.x.** Don't use torch + cu121/cu124 wheels; stick with cu118.
- **Matryoshka M3 (arxiv 2405.17430)** is the closest threat — must explicit-contrast in related work. Lead paper with query-conditioned ablation, not "variable K" framing.
- **Halting collapse failure modes** (per agent review): PonderNet under VLM autoregressive loss has not been validated; collapse to K=1 or K=max is real risk. Need regularization carefully.

### Assumptions Made

- ALVC adapter trains in 1-2 GPU-weeks on consumer GPU (uet 4090). To be validated in Phase 0.
- Query embedding for halting head = LM hidden state after processing query prompt (before image tokens). Specific extraction point TBD.
- Sequence-level halting (not per-patch) for Phase 0. Per-patch is future generalization.

### Known Issues

- `num_crops=1` not respected by processor — investigate when integrating halting head
- No integration tests for ALVC yet (only skeleton unit tests)
- Cache API drift for `model.generate()` — will need fix before eval phase

---

## Current State

### What's Working

- Phi-3.5-Vision forward pass end-to-end on uet (`output.logits` shape `(1, 330, 32064)`)
- ALVC folder skeleton, configs sized to Phi-3.5-Vision dims (d_v=4096, d_lm=3072, K_max=144)
- 4 skeleton tests pass locally
- `--import-only` sanity works on CPU-only local laptop
- Branch `feat/alvc-phase0` clean and pushed to origin

### What's Not Working / Unknown

- HaltingHead, ALVCConnector, training loop — all skeletons with `NotImplementedError`
- Corrected theorem unwritten formally (only in memory)
- G5 eval protocol not designed; no paired (image, query) dataset yet
- `model.generate()` blocked by cache API drift (defer to eval phase)

### Tests

- [x] Unit tests: 4 passed (`pytest tests/test_alvc_skeleton.py -v`)
- [ ] Integration tests: not written
- [x] Manual end-to-end sanity: Phi-3.5-Vision forward pass works on uet

---

## Next Steps

### Immediate (Start Here)

1. **Implement `HaltingHead`** in `sgod/policies/alvc/halting.py` — PonderNet-style sequence-level halting:
   - Input: pooled vision features [B, D_v] + pooled query embedding [B, D_q]
   - Output: halting probabilities lambda_n for n in [k_min, k_max], halting distribution p_n = lambda_n * prod_{i<n} (1 - lambda_i)
   - Make sure shapes match (d_v=4096, d_q=3072 confirmed empirically)
   - Add unit tests verifying p_n sums to 1, gradient flows correctly, geometric prior KL works

2. **Implement `ALVCConnector`** in `sgod/policies/alvc/connector.py`:
   - Wraps HaltingHead + top-K selection + linear projection (4096→3072)
   - Top-K selection: start with differentiable Gumbel-softmax or expected-loss across N values. Decide between approaches in implementation.
   - Mirror `img_projection` interface so it can drop-in replace Phi-3.5-Vision's projector

3. **Write theorem derivation doc** at `docs/alvc_theorem.md`:
   - State assumptions explicitly (Gaussian channel? other?)
   - Derive `K ≥ ⌈I(X;Y|Q) / C⌉` step by step
   - Show channel capacity estimator (empirical or analytic)
   - Identify when bound is tight vs loose

4. **Design G5 eval protocol** in `sgod/policies/alvc/eval.py`:
   - GQA-Balanced is candidate (multiple questions per image)
   - Need paired (image, query_simple, query_complex) triples
   - Metric: median |ΔK| > 0.15 * K_max with bootstrap CI95 excluding 0

### Subsequent

- Phase 0 week 1: G5 test (gate decision)
- Phase 0 week 2: G1-G4 + ablations + Phase 1 design doc
- Investigate `num_crops=1` not respected (affects K_max)
- Decide on per-token halting vs sequence-level for paper (currently sequence-level)

### Blocked On

- None now. Setup complete.

---

## Related Resources

### Documentation

- Prior handoff: `docs/handoffs/HANDOFF_VRTTS_FAILED_REASSESS_05_20_18_30.md` (context for pivot decision)
- Memory: `project_alvc_direction.md`, `research_breakthrough_patterns.md`, `project_framework_ambition.md`

### Commands to Run

```bash
# Local CPU sanity (verify imports + skeleton tests)
.venv/bin/python -m pytest tests/test_alvc_skeleton.py -v
.venv/bin/python scripts/sanity_alvc_phi35v.py --import-only

# uet GPU full forward sanity (already verified)
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/sanity_alvc_phi35v.py

# When implementing HaltingHead, test with:
.venv/bin/python -m pytest tests/test_alvc_skeleton.py tests/test_alvc_halting.py -v
```

### Search Queries

- `grep -rn "ALVCConnector\|HaltingHead\|HaltingConfig" sgod/` — ALVC module usages
- `grep -n "img_projection" /home/uet/.cache/huggingface/modules/transformers_modules/microsoft/Phi*/modeling_phi3_v.py` — find Phi-3.5-Vision's projector definition
- `grep -rn "from_legacy_cache\|get_usable_length" .venv/` — track cache API issues

### Literature References

- Matryoshka M3 (arxiv 2405.17430) — closest threat, contrast in related work
- LLaVA-PruMerge (2403.15388) — adjacent, image-only K
- PonderNet (Banino 2021, 2107.05407) — halting mechanism
- ACT (Graves 2016, 1603.08983) — original adaptive computation

---

## Open Questions

- [ ] How to extract query embedding for halting head? Use LM's first-layer hidden state of last query token, or pooled embedding across query tokens?
- [ ] Differentiable top-K selection: Gumbel-softmax (high variance) vs expected-loss across all K values (compute-heavy)?
- [ ] How to estimate continuous channel capacity C in practice? Per-layer Gaussian assumption, or empirical entropy estimator?
- [ ] G5 dataset: GQA-Balanced is candidate. Need to confirm multiple-questions-per-image structure + filter for varying complexity.
- [ ] When to introduce per-patch halting (Phase 2+)? Or commit fully to sequence-level for paper?
- [ ] Should we publish negative scaling study (VR-TTS + MVP v4 n=1000 collapse) as standalone short paper or fold into ALVC paper as motivation?

---

## Session Notes

- This session had 3 major direction shifts: VR-TTS reassessment → Oracle Attention pitch (rejected as lego) → 5+ candidate brainstorm → Shape calibration → ALVC commit. User's taste = Shape 2 architecture primitive at BLIP-2/Q-Former level.
- **Critical insight from agent review:** original theorem `K* ≤ I/log(vocab)` was wrong (vocab dim is for discrete tokens; vision projector outputs continuous embeddings). Corrected to `K ≥ ⌈I/C⌉` with continuous channel capacity. If this hadn't been caught, paper would have been rejected on math.
- **The query-conditional ablation is THE single differentiator** vs prior work (Matryoshka M3, LLaVA-PruMerge). Without it, the paper collapses into known territory. Make this the lead empirical claim.
- uet env compatibility took 4 iterations to resolve (TinyLLaVA → Phi-3.5-Vision → torch downgrade → transformers downgrade → use_cache=False). Document this thoroughly in memory so future sessions don't repeat.
- User explicitly aspires to "Attention Is All You Need" scale impact. Tempered to "Mamba scale" as realistic 3-6 month target. ALVC fits this realistically.
- Phi-3.5-Vision's `image_dim_out=1024` reported in config but actual projector input is 4096 (4 CLIP layers stacked). This is the kind of detail that breaks shape assertions silently — always verify empirically.

---

_This handoff was generated after Phase 0 setup completion. Resume by reading the "Immediate Next Steps" section and starting with HaltingHead implementation._
