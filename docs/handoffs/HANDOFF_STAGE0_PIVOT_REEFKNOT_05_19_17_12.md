# Handoff: Stage 0 Ran End-to-End → POPE Saturated → Pivoting to Reefknot

**Created:** 2026-05-19 17:12
**Branch:** `feat/dt-sgod`
**Previous handoff:** [HANDOFF_DT_SGOD_DAY3_WEEK1_05_19_14_27.md](./HANDOFF_DT_SGOD_DAY3_WEEK1_05_19_14_27.md)
**Session type:** Stage 0 distillation execution + eval + benchmark pivot

---

## Summary

Continued from Day 3 handoff. Shipped: GPU CUDA fix on uet → end-to-end Stage 0 pipeline (collect 11616 traces, train 3 epochs, eval on POPE). Two scientific findings: **(1) Stage 0 distillation produces a degenerate student (gate γ collapsed 0.10 → 0.005) due to sparse-MSE loss imbalance**, and **(2) POPE is empirically saturated for LLaVA-7B** — both baseline and SGOD-v1 teacher hit F1=0.918, so distillation has no signal to learn from. Pivoting to Reefknot (relation hallucination, the actual MAIN CLAIM benchmark). User currently searching uet for pre-existing Reefknot data before downloading.

---

## Work Completed

### Changes Made

- [x] Fixed uet CUDA torch reinstall via `uv pip` (not `.venv/bin/pip` — uv doesn't seed pip)
- [x] Verified 2 integration smoke tests pass on real GPU (28s, vs ~5min on CPU)
- [x] Created `scripts/prepare_stage0_calib.py` — POPE → calib.jsonl converter
- [x] Created `scripts/inspect_stage0_traces.py` — sanity-check traces.pt with 4-metric verdict
- [x] Created `scripts/eval_stage0_pope.py` — 3-way eval (baseline / SGOD-v1 / DT-SGOD-Stage0)
- [x] Created `scripts/eval_stage0_reefknot.py` — same for Reefknot YESNO.jsonl
- [x] Fixed `sys.path` injection in `scripts/distill_sgod_v1.py` and `scripts/run_dtsgod.py`
- [x] Fixed empty registry bug in `sgod/runtime/builder.py` — `_ensure_components_registered()` lazy-imports concrete subpackages
- [x] Fixed dtype/device mismatch in `sgod/training/distill_stage0.py` `distill_step` — traces cast to policy dtype/device
- [x] Added GPU placement + per-step progress logging in `scripts/distill_sgod_v1.py` train phase
- [x] Ran Stage 0 collect: **11616 traces, 500 examples, 11.5% effective anchor rate**
- [x] Ran Stage 0 train (twice — w_delta=1.0 then 1e7): both produced degenerate γ→0
- [x] Ran POPE 3-way eval on 100 examples: **all 3 systems F1=0.918, 100% agree**
- [x] Updated 2 memory files: `project_dt_sgod_week1_progress.md`, `research_vlm_halluc_sota.md`

### Key Decisions

| Decision | Rationale | Alternatives Considered |
|---|---|---|
| Use `uv pip` not `.venv/bin/pip` on uet | uv doesn't seed pip in venv by default | `python -m ensurepip`; rebuild venv from scratch |
| `_ensure_components_registered()` lazy-import inside `build_from_config` | Triggers `@register` decorators only when needed — preserves fast unit-test path | Eager import in `sgod/runtime/__init__.py` (would pull torch always) |
| Cast traces to policy dtype/device in `distill_step` | Local fix, robust to mixed-source traces (fp16 LLaVA → fp32 student) | Cast at TraceCollector save time (less flexible) |
| Inject project root via `sys.path` in scripts | Matches `experiments/main_eval/eval_pope.py` pattern; works regardless of editable install state | `uv pip install -e .` (cleaner long-term but requires user action) |
| Try `w_delta=1e7` to fix sparse loss imbalance | Quickest test; balanced w·L_delta vs w·L_atg numerically | Sparse MSE on non-zero positions (cleaner but bigger code change) |
| Eval BEFORE fixing sparse loss | Need to know if teacher even beats baseline before investing in distillation tuning | Fix loss first (would have wasted effort given POPE saturation) |
| Pivot to Reefknot over AMBER/MMHal | Relation hallucination = our oracle's strength; marked "MAIN CLAIM" in codebase | AMBER (multi-dim, generative — harder to eval); MMHal (LLM-as-judge — slow) |
| Use GitHub mirror not HF for Reefknot data | HF `hyintell/ReefKnot` is gated (401 Unauthorized); GitHub `JackChen-seu/Reefknot` is public | huggingface-cli login + accept terms |

---

## Files Affected

### Created (this session)

- `scripts/prepare_stage0_calib.py` — POPE JSONLs → calib.jsonl with `{image, question}` rows
- `scripts/inspect_stage0_traces.py` — Volume/anchor/Δ/scene-graph sanity check
- `scripts/eval_stage0_pope.py` — 3-way POPE eval with optional `--teacher-config`
- `scripts/eval_stage0_reefknot.py` — 3-way Reefknot YESNO eval, per-type breakdown
- `outputs/stage0/calib.jsonl` (on uet) — 500 POPE examples
- `outputs/stage0/traces.pt` (on uet) — 11616 SGOD-v1 teacher traces
- `outputs/stage0/dtsgod_stage0.pt` (on uet) — degenerate student ckpt (γ=0.005)
- `outputs/stage0/eval.json` (on uet) — POPE 3-way results
- `docs/handoffs/HANDOFF_STAGE0_PIVOT_REEFKNOT_05_19_17_12.md` — this file

### Modified

- `scripts/distill_sgod_v1.py` — added `sys.path.insert(...)`, `policy.to("cuda")` after `prepare_for_stage0`, per-step progress log every `len/20` steps
- `scripts/run_dtsgod.py` — added `sys.path.insert(...)`
- `sgod/runtime/builder.py` — added `_ensure_components_registered()` lazy-import; called at top of `build_from_config`
- `sgod/training/distill_stage0.py` — `distill_step` now casts `h`, `lm_logits`, `delta_teacher` to `policy.speaker_adapter.parameters().dtype/device` before forward (fixes fp16/fp32 mismatch)

### Read (Reference)

- `sgod/runtime/decoder.py:130-176` — to find where `rule_anchor_type` should be set (orchestrator doesn't currently)
- `sgod/policies/sgod_v1.py` — to confirm SGOD v1 has internal anchor detection, doesn't write to `state.rule_anchor_type`
- `sgod/training/distill_stage0.py:14` — to confirm ATG loss uses `indicator(δ_teacher non-zero)`, NOT `rule_anchor_type` (wiring bug is cosmetic)
- `sgod/policies/dt_sgod/speaker_adapter.py:46-54` — `out_proj = zeros`, `gate = nn.Parameter(speaker_gate_init)` semantics
- `experiments/main_eval/eval_pope.py` — POPE JSONL format reference
- `experiments/main_eval/eval_reefknot.py` — original Reefknot format (stale: actual format differs)
- `scripts/download_data.py` — POPE/Reefknot download URLs

---

## Technical Context

### Key Architectural Observation

`HallucinationDecoder` orchestrator at [decoder.py:144-153](sgod/runtime/decoder.py#L144) builds `GenerationState` **without** ever calling `detect_anchor()` to set `state.rule_anchor_type`. SGOD v1's Δ-fire decision happens *internally* (top-K candidate scoring). So traces always show `rule_anchor_type=None` even when teacher Δ ≠ 0. This is a cosmetic bug — `distill_step` ATG loss uses `indicator(δ_teacher non-zero)`, not the field.

### Stage 0 Loss Failure Mode (root cause)

```
Teacher Δ shape: [B, V=32064], non-zero only at top-K=50 positions on 11.5% of steps
Student Δ from SpeakerAdapter: dense over full vocab
F.mse_loss(student, teacher) averages over all 32064 × steps positions
→ Optimum: student outputs 0 everywhere (99.84% positions × 88.5% steps want 0)
→ gate γ collapses to ~0 (minimizes "noise" on non-anchor positions)
```

L_delta plateaus at ~3.5e-9 because (teacher_max ≈ 0.017)² / 32064 ≈ 1e-9. Bumping `w_delta` 1→1e7 just pushes harder toward zero (γ collapsed 0.10→0.005 instead of 0.10→0.13).

**Proper fix not yet implemented:** masked MSE on `(teacher_δ != 0).float()` positions + smaller weight on background. Deferred until we know if there's a learnable signal at all (requires Reefknot eval).

### POPE Saturation (empirical)

| System | F1 | Acc | Yes-frac | Agree w/ baseline |
|---|---|---|---|---|
| LLaVA-7B baseline | 0.9184 | 0.92 | 0.48 | (ref) |
| SGOD-v1 teacher | 0.9184 | 0.92 | 0.48 | 100% |
| DT-SGOD-Stage0 | 0.9184 | 0.92 | 0.48 | 100% |

Per-split (n=100): random 90.3%, popular 90.9%, adversarial 94.4% (adversarial highest — likely few FPs). 5 FN + 3 FP errors are object-presence ambiguities, not relation issues — exactly where our SG oracle has nothing to add.

### Dependencies (new on uet)

- `gdown` (only if going Google Drive route for VG; alternative `wget` from Stanford works)
- torch cu128 (already installed via `uv pip install --reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128`)

---

## Things to Know

### Gotchas

- **`.venv/bin/pip` doesn't exist on uet** — uv doesn't seed pip. Use `uv pip` or `.venv/bin/python -m pip` after `ensurepip`.
- **`build_from_config` failed silently before fix** — registry was empty because nothing imported concrete `sgod.backbones/oracles/policies`. Smoke tests passed because they import classes directly.
- **Stage 0 train was running on CPU silently** — `_phase_train` never moved policy to GPU; first run took ~10× longer than expected, no per-step log meant looked stuck.
- **Trace dtype mismatch** — LLaVA outputs fp16, default policy is fp32. Inference orchestrator handles this via `_align_policy_to_backbone`; train path bypasses orchestrator, needed separate fix.
- **HF `hyintell/ReefKnot` is gated** (401) — use GitHub `JackChen-seu/Reefknot/Dataset/YESNO.jsonl` instead (public, no login).
- **Reefknot format docs in `experiments/main_eval/eval_reefknot.py` are stale** — actual schema is `image_id` (no extension) + `query_prompt` + `label` + `relation_type` ("perception"|"cognitive"), NOT `image` + `question` + `answer` + `type`.

### Assumptions

- 2× RTX 4090 48GB on uet (workstation, 96GB total VRAM)
- POPE images sit at `data/coco/val2014/` on uet (user downloaded earlier in session)
- User is reachable on `uet@ubuntu` via git push/pull on `feat/dt-sgod`

### Known Issues / Tech Debt

- `state.rule_anchor_type` wiring bug in orchestrator (cosmetic; doesn't affect training)
- Sparse-MSE loss formulation in `distill_step` produces degenerate student
- `QuickGELU mismatch` warning from open_clip (cosmetic; no quality impact)
- No `--load-ckpt` flag on `scripts/run_dtsgod.py` yet — only eval scripts load Stage 0 ckpts

---

## Current State

### What's Working

- Full Stage 0 pipeline executes end-to-end: prepare → collect → inspect → train → eval (no crashes)
- 453 unit tests pass + 5 integration tests pass on real GPU
- G1 invariant verified at SCALE on 100 POPE examples (gate=0 → DT-SGOD bit-identical to LLaVA)
- 3-way eval scripts ready for both POPE and Reefknot

### What's Not Working

- Stage 0 student is degenerate (γ=0.005, effectively no-op vs baseline) — **structural loss issue, not bug**
- POPE saturated — no signal to learn from; SGOD-v1 teacher F1 = baseline F1
- Reefknot eval blocked on data: HF gated, user searching uet for pre-existing copy

### Tests

- [x] Unit tests: 453 pass, 10 skipped (0 regression after all fixes)
- [x] Integration tests: 5 pass on real GPU (uet)
- [x] Manual end-to-end: collect/train/eval pipeline confirmed working
- [ ] Stage 0 eval on Reefknot: pending data acquisition

---

## Next Steps

### Immediate (Start Here)

**1. Resolve Reefknot data on uet** (user is mid-search)
   - User asked for command to search whole machine + trash. Last message gave 6+ find commands.
   - If NOT found: download from public sources:
     ```bash
     mkdir -p data/reefknot
     curl -L https://raw.githubusercontent.com/JackChen-seu/Reefknot/main/Dataset/YESNO.jsonl \
         -o data/reefknot/YESNO.jsonl
     # Verify: wc -l data/reefknot/YESNO.jsonl   → 9740
     mkdir -p data/reefknot/images
     cd data/reefknot/images
     wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
     wget https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip
     unzip -q images.zip && unzip -q images2.zip
     mv VG_100K/* VG_100K_2/* . && rmdir VG_100K VG_100K_2
     rm *.zip
     ```
   - If FOUND elsewhere: `ln -s <path> data/reefknot/images` and `ln -s <jsonl> data/reefknot/YESNO.jsonl`

**2. Run Reefknot 3-way eval**
   ```bash
   .venv/bin/python scripts/eval_stage0_reefknot.py \
       --student-config configs/dt_sgod_llava15.yaml \
       --ckpt outputs/stage0/dtsgod_stage0.pt \
       --teacher-config configs/sgod_v1_llava15.yaml \
       --reefknot-jsonl data/reefknot/YESNO.jsonl \
       --image-dir data/reefknot/images \
       --n 100 --out outputs/stage0/eval_reefknot.json
   ```

**3. Decision tree based on `acc_delta_teacher`**
   - `> +0.05`: 🎯 **Signal exists.** Fix sparse-MSE loss + re-train Stage 0 + scale to 500/full test
   - `+0.02..+0.05`: Try stronger oracle (lower CLIP threshold 0.3→0.1, raise λ), then DPO
   - `-0.02..+0.02`: Saturated even on relations — pivot benchmark to MMHal or scale to LLaVA-13B
   - `< -0.02`: Oracle injecting noise — debug RelTR vocabulary vs Reefknot relations mismatch
   - Especially check `by_type` breakdown: perception vs cognitive (RelTR has 51 mostly-perceptive predicates → expect perception Δ > cognitive Δ)

### Subsequent

- **If signal found**: implement masked MSE in `distill_step`:
  ```python
  mask = (delta_teacher.abs() > 0).float()  # [B, V]
  loss_fire = ((delta_student - delta_teacher) ** 2 * mask).sum() / mask.sum().clamp(min=1)
  loss_silent = ((delta_student * (1 - mask)) ** 2).mean()
  loss_delta = loss_fire + 0.01 * loss_silent
  ```
- Fix `state.rule_anchor_type` wiring in orchestrator (cosmetic, do alongside other decoder.py work)
- Add `--load-ckpt` to `scripts/run_dtsgod.py` for single-image inference with Stage 0 weights
- Stage 1: DPO on Δ-flipped preference pairs (only after Stage 0 produces non-degenerate student)

### Blocked On

- Reefknot data availability on uet (user's current task)

---

## Related Resources

### Documentation

- `docs/dt_sgod_proposal.md` — architecture
- `docs/dt_sgod_implementation_plan.md` — 18-week plan, gates G1-G8
- [`HANDOFF_DT_SGOD_DAY3_WEEK1_05_19_14_27.md`](./HANDOFF_DT_SGOD_DAY3_WEEK1_05_19_14_27.md) — previous handoff

### Commands to Run

```bash
# Full test suite (CPU laptop or uet)
.venv/bin/python -m pytest tests/ --override-ini="addopts=" -q
# Should print: 453 passed, 10 skipped

# Inspect Stage 0 traces (after collect)
.venv/bin/python scripts/inspect_stage0_traces.py outputs/stage0/traces.pt

# Stage 0 train (3 epochs ≈ 4 min on 4090)
.venv/bin/python scripts/distill_sgod_v1.py train \
    --student-config configs/dt_sgod_llava15.yaml \
    --traces outputs/stage0/traces.pt \
    --out-ckpt outputs/stage0/dtsgod_stage0.pt \
    --epochs 3 --lr 1e-4

# POPE 3-way eval (n=100 ≈ 15 min on 4090)
.venv/bin/python scripts/eval_stage0_pope.py \
    --student-config configs/dt_sgod_llava15.yaml \
    --ckpt outputs/stage0/dtsgod_stage0.pt \
    --teacher-config configs/sgod_v1_llava15.yaml \
    --n 100

# Git state
git log --oneline -10
git diff --stat main..feat/dt-sgod
```

### Search Queries

```bash
grep -rn "_ensure_components_registered" sgod/         # the auto-registration fix
grep -rn "rule_anchor_type" sgod/                      # cosmetic wiring bug location
grep -rn "loss_delta\|w_delta" sgod/training/          # sparse loss problem area
```

---

## Open Questions

- [ ] **Reefknot result** — does the teacher F1 beat baseline by ≥+0.02 on relation questions? Single biggest unknown blocking everything else.
- [ ] **Masked MSE design** — should silent positions (where teacher Δ=0) get zero weight or small weight (e.g. 0.01)? Zero is cleaner but might let student inject noise on inference; small weight balances both.
- [ ] **Pivot if Reefknot also saturates** — try LLaVA-13B (more VRAM headroom) or MMHal (LLM-as-judge, harder)?
- [ ] **Stage 0 batch size** — currently 1 trace at a time. Worth batching for 3-5× speedup if we're going to re-train many times?

---

## Session Notes

- User on uet (`uet@ubuntu`) with 2× RTX 4090 48GB. CUDA driver issues from previous handoff RESOLVED this session via `uv pip install --reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128`.
- Workflow pattern: user runs heavy GPU jobs on uet, Claude runs lint/test/dev on laptop. Coordination via `git pull/push` on `feat/dt-sgod`.
- User responds in Vietnamese — keep replies in Vietnamese.
- User accepts pragmatic "ship-then-fix" approach: ran eval to confirm POPE saturation BEFORE investing in sparse-loss fix. Was the right call.
- 3 separate bugs surfaced + fixed today in `distill_sgod_v1.py` / `distill_stage0.py` / `builder.py` — common theme: orchestrator does dtype/device alignment lazily, but `_phase_train` bypasses orchestrator and missed it. Worth a follow-up to centralize "policy ready to use" logic.
- Memory `research_vlm_halluc_sota.md` had predicted POPE saturation; this session empirically confirmed. Memory updated with empirical evidence.

---

_This handoff was generated at context window capacity. Start a new session and use this document + `MEMORY.md` index as initial context. The single most important next action is acquiring Reefknot data + running the 3-way eval — every downstream decision depends on `acc_delta_teacher`._
