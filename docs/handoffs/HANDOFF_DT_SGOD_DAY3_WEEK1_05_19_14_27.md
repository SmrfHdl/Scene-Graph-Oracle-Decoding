# Handoff: DT-SGOD Day 3 + Week 1 Wrap + Pipeline Verified

**Created:** 2026-05-19 14:27
**Branch:** `feat/dt-sgod`
**Latest commit:** `9281ade` (fix CLIP cache mismatch in load_clip_factory)
**Session type:** Feature implementation + integration debugging on real GPU machine
**Test count:** 449 unit pass + 5 integration pass + 5 skipped (no resources locally)

---

## Summary

Picked up from [HANDOFF_DT_SGOD_DAY1_DAY2_05_18_16_53.md](./HANDOFF_DT_SGOD_DAY1_DAY2_05_18_16_53.md) at Day 3. Shipped:

1. **Day 3** (recommended order): SGODv1Policy refactor → LLaVAv15Backbone.forward_step → HallucinationDecoder orchestrator. All verified on real LLaVA-1.5-7B (Gate G1 confirmed on hardware).
2. **Week 1 wrap**: Oracle wrappers (RelTROracle, GroundingDinoOracle), K_t hidden buffer, pluggable CLIP label embedder, config-driven builder, Stage 0 distillation skeleton with traces.
3. **Bug fixes from real-pipeline smoke**: orchestrator dtype/device auto-alignment, CLIP vocab-cache dim mismatch guard (in both CLIPScorer and load_clip_factory).

---

## What's verified on real hardware (uet machine, 2× RTX 4090 48GB)

| Test | Status | Notes |
|---|---|---|
| `test_forward_step_real_llava_smoke` | ✅ PASS | LLaVA-1.5-7B fp16, KV cache, 3 incremental steps |
| `test_dt_sgod_at_init_matches_baseline_on_real_llava` | ✅ PASS | G1: DT-SGOD@init = LLaVA-base bit-identical |
| `test_orchestrator_end_to_end_with_real_llava` | ✅ PASS | Orchestrator generate() with real LLaVA |
| `test_real_pipeline_sgod_v1_generates_grounded_answer` | ✅ PASS | LLaVA + RelTR + CLIP + SGOD v1, 4 anchor fires, max \|Δ\|=0.005 |
| `test_real_pipeline_dt_sgod_at_init_preserves_g1_with_real_oracle` | ✅ PASS | G1 holds with non-trivial scene graph evidence |

**Concrete sample output:**
```
[smoke] sg.objects:   ['chair', 'table']
[smoke] sg.relations: ['chair-at-table', ...]
[smoke] answer:       'The image features a living room with a dining table, chairs, and a television.'
[smoke] anchor_fires: 4, max |Δ|: 0.005089
```

---

## Code shipped (commits since `dec2169` Day 2)

| Commit | Description |
|---|---|
| `52c30e6` (user) | Day 3 work + Week 1 wrap bundled (config files, distillation scripts, orchestrator) — user manually committed mid-session |
| `e1e305d` (user) | pyproject.toml setuptools.packages.find fix (multi-top-level-package error) |
| `3286f46` | runtime: orchestrator policy dtype/device auto-alignment (fixes c10::Half != float crash) |
| `33b53f8` | tests: real-pipeline smoke + production config validation |
| `4930633` | clip: skip mismatched vocab cache in CLIPScorer.__init__ |
| `9281ade` | clip: same guard in load_clip_factory (the actual codepath the smoke test hit) |

---

## Files affected

### Created (this session)
- `sgod/oracles/__init__.py`, `sgod/oracles/reltr_oracle.py`, `sgod/oracles/grounding_dino_oracle.py`
- `sgod/runtime/__init__.py`, `sgod/runtime/decoder.py`, `sgod/runtime/builder.py`, `sgod/runtime/trace_collector.py`
- `sgod/training/__init__.py`, `sgod/training/distill_stage0.py`
- `sgod/policies/sgod_v1.py`, `sgod/policies/dt_sgod/label_embedders.py`
- `configs/dt_sgod_llava15.yaml`, `configs/sgod_v1_llava15.yaml`
- `scripts/run_dtsgod.py`, `scripts/distill_sgod_v1.py`
- `tests/test_sgod_v1_policy.py`, `tests/test_backbone_llava.py`, `tests/test_runtime_decoder.py`,
  `tests/test_oracle_wrappers.py`, `tests/test_hidden_buffer.py`, `tests/test_label_embedders.py`,
  `tests/test_builder.py`, `tests/test_trace_collector.py`, `tests/test_distill_stage0.py`,
  `tests/test_pipeline_smoke.py`, `tests/test_clip_cache_mismatch.py`

### Modified
- `sgod/backbones/llava15.py` — `forward_step` KV-cache wiring
- `sgod/core/types.py` — `GenerationState.hidden_buffer` optional field
- `sgod/policies/dt_sgod/{policy,grounding_planner}.py` — K_t buffer consumption + pluggable embedder
- `sgod/policies/__init__.py`, `sgod/policies/dt_sgod/__init__.py` — exports
- `sgod/oracle/clip_scorer.py`, `sgod/utils/model_loader.py` — cache dim guard
- `tests/conftest.py` — shared framework fakes (`test-fake-{backbone,oracle,policy}`)
- `pyproject.toml` — setuptools packages.find

---

## Architecture status

### Three-layer abstraction (fully realized)
```
Backbone (Module)     Oracle (CPU/GPU)          Policy (nn.Module)
────────────────      ──────────────────         ────────────────────
LLaVAv15Backbone      RelTROracle               DTSGODPolicy
                      GroundingDinoOracle       SGODv1Policy
                      (3-layer pluggable, registered via @register decorator)
                              ↓
              HallucinationDecoder orchestrator
              (KV cache + dtype align + K_t buffer + step_hook)
```

### G1 invariant (proven 3 ways)
1. CPU mocked: 5 unit tests in `test_dt_skeleton.py`
2. CPU real Δ comparison: `test_dt_sgod_with_real_gp_preserves_zero_invariant`
3. Real LLaVA: `test_dt_sgod_at_init_matches_baseline_on_real_llava` + variant with non-empty SceneGraph

### Stage 0 distillation pipeline (skeleton, untested on data)
```
Phase 1: SGOD v1 teacher → HallucinationDecoder + step_hook = TraceCollector
         → list[StepTrace] (hidden, lm_logits, δ_teacher, evidence, anchor flag)
         → torch.save → traces.pt

Phase 2: load traces → DTSGODPolicy student → prepare_for_stage0(γ=0.1)
         → for each trace: distill_step (MSE on δ + BCE on ATG)
         → checkpoint
```
Entry point: `scripts/distill_sgod_v1.py collect | train`.

---

## Things to know

### Gotchas
- **Double-zero bootstrap problem**: γ=0 + out_proj=0 is a flat point — no gradient flows. `prepare_for_stage0(gate_init=0.1)` warms γ off zero so out_proj has a non-zero gradient on step 1.
- **CLIP cache dim mismatch is silent until matmul**: cache from old ViT-B-32 (512-dim) + new ViT-L-14 model (768-dim) was the bug behind the uet smoke failure. Fix lives in 2 places (CLIPScorer + load_clip_factory) because `_preloaded` bypasses the inner guard.
- **dtype alignment is lazy in the orchestrator**: policy stays fp32 until first `backbone.forward_step()` returns, then `_align_policy_to_backbone` casts in place. Don't construct policy with explicit `.to(...)` — let the orchestrator do it.
- **Hidden buffer is per-call, not persistent**: each `decoder.generate()` resets the buffer. K_t is `hidden_buffer_size` in constructor.

### Assumptions
- LLaVA-1.5-7B hidden_dim=4096, vocab_size=32064 — hardcoded constants in `LLaVAv15Backbone`. Override via constructor params for tiny test models.
- RelTROracle.extract is one-shot per image (oracle is independent of VLM — `design_independence_principle.md`).
- `OracleEvidence.extra` carries per-call data: `{"image", "question", "prompt"}`. SGODv1Policy reads these in init_state; DT-SGOD ignores.

### Known issues / tech debt
- **Yes/no first-token injection** from legacy SGODDecoder NOT ported to SGODv1Policy. Documented in docstring.
- **GAT in GroundingPlanner** still node-only (no edge encoding). Week 2 enhancement.
- **`SGODv1Policy` is single-batch (B=1)** at scoring — iterates over batch dim. Fine for current pipeline; batched eval would need rework.
- **CLIP text embedder for GP**: works with mocks, real CLIP load probed lazily. Real production run untested.
- **3 pre-existing ruff errors in vendored RelTR / sgg/reltr_utils.py** — not in our scope. Out of scope.

---

## Current state

### What's working
- Full framework Backbone/Oracle/Policy with registries
- Config-driven build (`scripts/run_dtsgod.py --config configs/dt_sgod_llava15.yaml ...`)
- Real LLaVA-1.5-7B inference through orchestrator
- Real RelTR SGG → SceneGraph → SGODv1Policy Δ-injection
- DT-SGOD@init wrapping any backbone is bit-identical (G1 verified)
- Stage 0 distillation step (verified loss decreases on synthetic data)

### What's NOT working / pending
- **`uet` machine torch CUDA mismatch**: torch 2.11 built for newer CUDA than driver 570.124.04 (CUDA 12.8) supports. Pipeline runs CPU-only on uet, ~10× slower than GPU. **Stage 0 blocked here.**
- Stage 0 actual training run on a dataset (~1K calibration examples)
- Stage 0 evaluation: A/B/C on POPE/MMHal/AMBER between baseline / SGOD v1 / DT-SGOD-Stage0

---

## Next steps

### Immediate (start here)

**1. Resolve uet CUDA driver / torch mismatch** *(user's machine, not Claude's task)*
   ```bash
   # On uet, reinstall torch with CU 12.8 binaries
   .venv/bin/pip install --force-reinstall torch torchvision \
       --index-url https://download.pytorch.org/whl/cu128
   # Verify
   .venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```
   Without this, Stage 0 take ~14h on CPU vs ~1.5h on GPU.

**2. Stage 0 dataset prep**
   - Recommended: 500-1000 examples from POPE (existing in `data/pope/`).
   - Format: jsonl with `{"image": "<abs path>", "question": "..."}`.
   - Script template: `scripts/distill_sgod_v1.py collect --dataset jsonl:...`

**3. Run Stage 0**
   ```bash
   python scripts/distill_sgod_v1.py collect \
       --teacher-config configs/sgod_v1_llava15.yaml \
       --dataset jsonl:data/stage0/calib.jsonl \
       --out-traces outputs/stage0/traces.pt --limit 500

   python scripts/distill_sgod_v1.py train \
       --student-config configs/dt_sgod_llava15.yaml \
       --traces outputs/stage0/traces.pt \
       --out-ckpt outputs/stage0/dtsgod_stage0.pt \
       --epochs 3 --lr 1e-4
   ```

### Subsequent (Week 2 plan)
- Load Stage 0 checkpoint into DTSGODPolicy at inference time (no helper yet — add `policy.load_state_dict(torch.load(ckpt)["speaker_adapter"], strict=False)`).
- Eval Stage 0 student on POPE / MMHal: compare baseline LLaVA vs SGOD v1 vs DT-SGOD-Stage0.
- If results positive, advance to Stage 1: DPO on Δ-flipped preference pairs.
- Add GAT to GroundingPlanner (currently node-only slot attention).

### Blocked on
- uet CUDA driver/torch mismatch — user task.
- Dataset preparation — user's strategic choice (POPE vs COCO vs GQA).

---

## Related resources

### Documentation
- `docs/dt_sgod_proposal.md` — architecture
- `docs/dt_sgod_implementation_plan.md` — 18-week plan, decision gates G1-G8
- `docs/handoffs/HANDOFF_DT_SGOD_DAY1_DAY2_05_18_16_53.md` — previous handoff

### Commands
```bash
# Full test suite (CPU)
.venv/bin/python -m pytest tests/ --override-ini="addopts=" -q
# Should print: 449 passed, 10 skipped

# Run integration tests (needs GPU + checkpoints + LLaVA cache)
.venv/bin/pytest tests/test_backbone_llava.py tests/test_pipeline_smoke.py \
    -m integration -v -s --override-ini="addopts="

# Lint new code
.venv/bin/ruff check sgod/ tests/test_*.py scripts/

# Single example inference (after Stage 0)
python scripts/run_dtsgod.py \
    --config configs/dt_sgod_llava15.yaml \
    --image /path/to/img.jpg \
    --question "Is there a dog?"

# Git state
git log --oneline -10
git diff --stat main..feat/dt-sgod
```

### Search queries
```bash
grep -rn "register(\"" sgod/                  # all framework registrations
grep -rn "@pytest.mark.integration" tests/    # GPU-required tests
grep -rn "TODO\|FIXME\|XXX" sgod/             # outstanding work markers
```

---

## Open questions

- [ ] **Stage 0 dataset choice**: POPE (3K, binary GT) vs GQA-val (12K, free-form) vs COCO captions (118K, only captions, no questions). Recommended POPE first — small, fast, has clear evaluation.
- [ ] **Stage 0 trainable scope**: just SpeakerAdapter + ATG (recommended in `stage0_trainable_params`) or also include GP? Trade-off: GP freezes faster convergence vs GP-trains better quality.
- [ ] **Are 2× 4090 48GB cards consumer or workstation?** The 48GB is unusual — possibly modded or special editions. Constraint memory file says "1-2 consumer GPU" — uet exceeds this comfortably.
- [ ] **CUDA driver vs torch version**: should we pin torch in pyproject.toml to a CUDA-compatible version, or document the install steps?

---

## Session notes

- User speaks Vietnamese; respond in Vietnamese.
- User on a different machine (`uet@ubuntu`) than where Claude runs locally — pull/push workflow via origin/feat/dt-sgod. User already does `uv sync --extra dev --extra train` workflow.
- User committed mid-session at `52c30e6` and `e1e305d` — expect this pattern.
- The `_align_policy_to_backbone` was the first real bug surfaced by GPU run. Easy fix but conceptually important: the orchestrator owns the dtype contract.
- CLIP vocab cache mismatch bug was real and would affect production. Fix landed in two places.
- 5 integration tests confirm pipeline. Stage 0 is the next big unlock.

---

*Generated 2026-05-19 14:27 at the end of Day 3 + Week 1 wrap. Resume by reading `MEMORY.md` index (especially `project_dt_sgod_week1_progress.md`) and then this handoff.*
