# Handoff: DT-SGOD Architecture Pivot + Day 1-2 Implementation

**Created:** 2026-05-18 16:53
**Branch:** `feat/dt-sgod`
**Session Type:** Architecture redesign + initial implementation

---

## Summary

This session pivoted SGOD from a "lego" rule-based decoder to **DT-SGOD** (Dual-Timescale Scene-Graph Decoder), an HRM-inspired breakthrough architecture targeting Q2 journal publication. After thorough research synthesis, the user chose DT-SGOD over two alternative candidates (Mamba-shaped Selective-SG, LLaDA-shaped Co-DiSG). Implementation Day 1 + Day 2 are committed; Day 3 is staged. The user also expressed framework ambition (SGLang-style deployable lib), shaping the code architecture to be backbone-agnostic from day 1.

---

## Work Completed

### Documents written
- [x] `docs/dt_sgod_proposal.md` — full architecture proposal (~5000 words, 12 sections)
- [x] `docs/dt_sgod_implementation_plan.md` — 18-week plan in 4 phases with 8 go/no-go gates

### Code shipped (2 commits on `feat/dt-sgod`)
- [x] `a548ed3` Day 1: Core interfaces (`sgod/core/`), LLaVA backbone wrapper, DT-SGOD policy skeleton with γ=0 invariant verified
- [x] `dec2169` Day 2: Full GroundingPlanner forward (node featurization + slot attention + GRU) + wired into DTSGODPolicy

### Settings/permissions
- [x] Enabled `WebFetch(domain:arxiv.org)` in `.claude/settings.local.json` (user trimmed broader perms to arxiv-only)

### Memory (persisted across sessions)
- [x] `~/.claude/projects/.../memory/` — 8 memory files capturing user profile, project state, compute constraints, DT-SGOD direction, breakthrough patterns, VLM-halluc SOTA, independence principle, framework ambition

### Key Decisions

| Decision | Rationale | Alternatives Considered |
|---|---|---|
| DT-SGOD (dual-timescale) as breakthrough architecture | HRM-level novelty, cognitive scaffold, recovers SGOD v1 as ablation, fits 1×4090 compute | Selective-SG (Mamba-shaped, simpler), Co-DiSG (LLaDA×HRM, high compute risk) |
| 3-layer pluggable abstraction from Day 1 (Backbone/Oracle/Policy) | Framework ambition — saves 2-week refactor in Phase 3; ~1 day up-front cost | Hardcode LLaVA, refactor later |
| Grounding DINO over RelTR for future backbones | Open-vocab, errors uncorrelated with VLM | Qwen self-grounding (rejected: circular) |
| Hand-roll GAT (defer to Week 2) — Day 2 uses node-only slot attention | Avoid torch_geometric heavy dep; node-only is sufficient to prove the slot+GRU primitive | torch_geometric, full GAT immediately |
| md5-seeded label embeddings as Day 2 placeholder | Deterministic, stable, swappable for CLIP text features without API change | CLIP text features now (premature integration), random per-call (non-deterministic) |
| Defer LLaVA `forward_step` wiring to Day 3 | Requires GPU + real model load — risk of HF API rabbit-hole; current invariants prove the skeleton works without it | Wire it Day 1, deal with HF quirks immediately |

---

## Files Affected

### Created (Day 1)
- `sgod/core/__init__.py` — exports Backbone/Oracle/Policy + registry + types
- `sgod/core/interfaces.py` — three ABCs, locked signatures for Phase 1
- `sgod/core/registry.py` — `@register("kind", "name")` decorator + `build()` + `list_registered()`
- `sgod/core/types.py` — `GenerationState`, `OracleEvidence` dataclasses; `SceneGraph` re-exported
- `sgod/backbones/__init__.py`
- `sgod/backbones/llava15.py` — `LLaVAv15Backbone` lazy-load wrapper; `forward_step` raises `NotImplementedError` (Day 3 task)
- `sgod/policies/__init__.py`
- `sgod/policies/dt_sgod/__init__.py`
- `sgod/policies/dt_sgod/config.py` — `DTSGODConfig` dataclass, all hyperparameters
- `sgod/policies/dt_sgod/grounding_planner.py` — Day 2 full impl
- `sgod/policies/dt_sgod/speaker_adapter.py` — full impl, γ=0 + out_proj=0 init
- `sgod/policies/dt_sgod/anchor_gate.py` — 2-layer MLP, randomly init
- `sgod/policies/dt_sgod/policy.py` — `DTSGODPolicy` orchestrator
- `tests/test_interfaces.py` — 13 tests (registry, ABCs, re-exports)
- `tests/test_dt_skeleton.py` — 5 tests (gate=0 invariant)
- `tests/test_dt_grounding_planner.py` — 10 tests (GP behaviour)

### Modified (Day 2)
- `sgod/policies/dt_sgod/grounding_planner.py` — from stub to full GAT-less slot attention + GRU
- `sgod/policies/dt_sgod/policy.py` — `adjust_logits` now calls GP on ATG fire, per-batch mask

### Read (Reference)
- `sgod/sgg/scene_graph.py` — SceneGraph dataclass definition
- `sgod/utils/model_loader.py` — existing LLaVA loading utility (wrapped by Day 1 backbone)
- `sgod/decoder/sgod_decoder.py` — SGOD v1 decoder (target for Day 3 SGODv1Policy refactor)
- `pyproject.toml` — deps
- `pytest.ini` — markers, addopts

---

## Technical Context

### Architecture (DT-SGOD)
- **Slow module (Grounding Planner):** GAT-less for Day 2 (node-only slot attention), GRU update per slot, 8 slots × `hidden_dim`. Empty SG = no-op.
- **Fast module (Speaker Adapter):** Cross-attention adapter, Flamingo-style zero-init gate γ. At init: γ=0 AND `out_proj.weight=0` (double zero-defense). Per-token.
- **Anchor-Trigger Gate (ATG):** 2-layer MLP on `[h_t, h_slow_pooled, lm_entropy]`. Replaces rule-based anchor detector. Day 2 uses hard threshold > 0.5 (no STE yet).

### Invariant
**Gate G1 (proven by tests):** With γ=0 and `out_proj=0`, `DTSGODPolicy.adjust_logits` returns *exactly* zero, so DT-SGOD@init wrapping any backbone is bit-identical to that backbone alone. Test `test_dt_sgod_with_real_gp_preserves_zero_invariant` even forces ATG fire and confirms.

### Dependencies (no new deps added Day 1-2)
- `torch`, `peft` (LoRA), `transformers` — all already in `pyproject.toml`
- Planned: hand-roll GAT (no `torch_geometric`); `trl` for DPO at Stage 1

### Configuration
- `DTSGODConfig` dataclass (`sgod/policies/dt_sgod/config.py`) — all hyperparameters in one place
- `configs/dt_sgod_llava15.yaml` — not yet created (Day 3+)

---

## Things to Know

### Gotchas
- `hash()` is randomized per-process for strings; `_featurize_label` uses `hashlib.md5` for true determinism across runs
- `torch.no_grad()` is on `_featurize_label` and `_build_node_features` — embeddings are non-learnable features (label embed is a placeholder for CLIP text features in Week 2)
- The `forward_step` of `LLaVAv15Backbone` raises `NotImplementedError` deliberately — smoke tests bypass it by constructing `GenerationState` directly
- The user committed `docs/` themselves (commit `a6ee9a9 update docs`) — when working with this user, expect them to commit their own doc edits

### Assumptions
- LLaVA-1.5-7B hidden_dim=4096, vocab_size=32064, LoRA targets `[q,k,v,o]_proj` — hardcoded in `llava15.py`
- Image size always available in SceneGraph; fallback (1.0, 1.0) is treated as already-normalized
- 1 × 4090 (24 GB) is the design compute target; 2 × 4090 halves wall time

### Known issues / tech debt
- `forward_step` not yet wired (Day 3)
- Prefix summarizer is Day-2 stand-in (current `h_t` instead of K_t-token mean) — Week 1 fixes via orchestrator
- GAT is currently a node-only attention (no edge encoding) — Week 2 enhancement
- SGOD v1 has not yet been refactored as `SGODv1Policy` — needed for Stage 0 distillation teacher

---

## Current State

### What's Working
- **Framework abstractions:** Backbone/Oracle/Policy ABCs locked, registry functional
- **DT-SGOD skeleton:** All three modules (GP, SpeakerAdapter, ATG) instantiate and forward
- **Gate G1 invariant:** Δ=0 at init, even with real GP firing — proven by tests
- **GroundingPlanner forward:** Real slot attention + GRU update, gradient flow to all params, slot diversity preserved
- **Tests:** 364 passed + 5 skipped (integration). +10 new vs Day 1, zero regressions.

### What's Not Working
- `LLaVAv15Backbone.forward_step` — `NotImplementedError`. Needed for any end-to-end LLaVA test (Day 3)
- `SGODv1Policy` — not yet refactored from `SGODDecoder` (Day 3)
- Stage 0 distillation pipeline — not yet started (Week 2)
- Real CLIP text embedding for labels — placeholder is md5-seeded random (Week 2)

### Tests
- Unit tests: **364 passed + 5 skipped** (CPU only). Existing skipped are GPU-marked integration tests.
- Integration tests: not run this session (GPU not available in env)
- Manual testing: smoke-test via `pytest tests/test_dt_*.py` confirms γ=0 invariant

---

## Next Steps

### Immediate (Day 3 — Start Here)

**Recommended order** (least risk first):

1. **Refactor `SGOD v1` → `SGODv1Policy(Policy)`** (CPU-OK, self-contained):
   - New file `sgod/policies/sgod_v1.py`
   - Wrap existing `SGODDecoder` logic in `Policy` interface (`init_state`, `adjust_logits`, `update_state`)
   - Register as `@register("policy", "sgod-v1")`
   - Add `tests/test_sgod_v1_policy.py` proving it produces same logits as legacy `SGODDecoder` on a mock generation
   - **Why first:** No GPU needed. Needed for Stage 0 distillation. Risk-free.

2. **Wire `LLaVAv15Backbone.forward_step`** (needs GPU, riskier):
   - Use HF `model(input_ids, ..., past_key_values=cache, output_hidden_states=True, use_cache=True)`
   - Return `(last_hidden_state[:, -1, :], logits[:, -1, :])`
   - Maintain KV cache in `inputs` dict (mutate in place)
   - Smoke test: load LLaVA in 4-bit on small image, generate 5 tokens, verify shape and that DTSGODPolicy still returns Δ=0 (γ=0 invariant)
   - **Risk mitigation:** Use `transformers` version pinned in `pyproject.toml`; if HF API breaks, fall back to writing a generation loop using existing `model_loader.load_sgod_decoder` reference patterns

3. **Build orchestrator skeleton** (`sgod/runtime/decoder.py`):
   - Class that drives Backbone + Policy in a generation loop
   - Tracks last K_t hidden states for prefix summary (replaces Day 2 stand-in)
   - Public API: `HallucinationDecoder.generate(image, prompt) → text`

### Subsequent (Week 1 wrap)
- Implement `RelTROracle` and `GroundingDinoOracle` as proper `Oracle` subclasses (wrap existing modules)
- Replace md5 label embeddings with CLIP text features (uses existing `clip_vocab_cache.pt`)
- Build Stage 0 distillation script (`scripts/distill_sgod_v1.py`)

### Blocked On
- Nothing currently blocks Day 3. Day 3 step 2 needs GPU but that's expected — local env may have a GPU (LLaVA-1.5-7B already runs in this repo for existing tests).

---

## Related Resources

### Documentation
- `docs/dt_sgod_proposal.md` — architecture & theoretical scaffold
- `docs/dt_sgod_implementation_plan.md` — 18-week plan, decision gates G1-G8, pivot paths
- `docs/sgod_proposal.md` — SGOD v1 (predecessor)

### Commands

```bash
# Run new DT-SGOD tests (CPU-only, fast)
.venv/bin/python -m pytest tests/test_dt_*.py tests/test_interfaces.py -v --override-ini="addopts="

# Full suite (zero regression check)
.venv/bin/python -m pytest tests/ --override-ini="addopts=" -q

# Lint new code
.venv/bin/ruff check sgod/core/ sgod/backbones/ sgod/policies/ tests/test_dt_*.py tests/test_interfaces.py

# See current branch state
git log --oneline -5
git diff --stat main..feat/dt-sgod
```

### Search Queries (if needed)
- `grep -rn "DTSGODPolicy" sgod/ tests/` — every reference to the new policy
- `grep -rn "Backbone\|Oracle\|Policy" sgod/core/` — interface usage
- `grep -rn "load_llava\|load_sgod_decoder" sgod/` — for Day 3 LLaVA wiring reference

---

## Open Questions

- [ ] **Day 3 priority order:** SGODv1Policy refactor first vs. LLaVA `forward_step` first? *(User to confirm; recommended SGODv1Policy first for safety.)*
- [ ] **GAT or GAT-less for paper:** Day 2 ships node-only slot attention; do we ship the paper without full GAT, or upgrade in Week 2? Plan says Week 2. *(Defer decision; ablation will show whether GAT is needed.)*
- [ ] **Stage 0 distillation data scale:** 100K samples planned; verify storage/time tradeoff once initial trace pipeline works.
- [ ] **CLIP vocab cache compatibility with new GP node embed dim (128):** Will need to either match dim or add projection. *(Trivial — projection layer absorbs any dim mismatch.)*

---

## Session Notes

- User speaks Vietnamese; respond in Vietnamese.
- User is sharp on architecture critique — "lego rule-based" rejection of SGOD v1 was correct, drove this whole pivot. Don't soften or cheerlead.
- User is a researcher targeting Q2 journal AND wants the work to ship as a deployable framework. Both ambitions must be served simultaneously — hence 3-layer abstraction from Day 1.
- The memory system at `~/.claude/projects/-home-truongpv-code-journals-scene-graph-oracle-decoding/memory/` is comprehensive (8 files); read `MEMORY.md` index first on new session.
- Verified arxiv papers used to extract breakthrough recipe: Mamba (2312.00752), MoD (2404.02258), LLaDA (2502.09992), HRM (2506.21734). Only `WebFetch(domain:arxiv.org)` is in the allow list — broader Web tools (other domains, WebSearch) are not allowed in this env.
- `git status` at session start showed `feat/dev-may` clean; the user manually committed docs as `a6ee9a9` mid-session, then `feat/dt-sgod` was branched off (commits `a548ed3` Day 1 + `dec2169` Day 2).
- 3 unrelated pre-existing ruff errors in `tests/test_decoder.py` — not Day 1-2 scope, do not touch unless asked.

---

*This handoff was generated at the user's request after Day 2 completion. Start the next session by reading `MEMORY.md`, then `docs/dt_sgod_implementation_plan.md` §3 (Week 1 detail) before picking up Day 3.*
