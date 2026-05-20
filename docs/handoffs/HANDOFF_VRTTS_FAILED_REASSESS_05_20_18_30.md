# Handoff: VR-TTS Zoom Failed at Scale — Direction Reassessment Mid-Pivot

**Created:** 2026-05-20 ~18:30
**Branch:** `feat/vrtts-phase1`
**Previous handoff:** [HANDOFF_REEFKNOT_SPATIAL_VERIFY_05_20_14_42.md](./HANDOFF_REEFKNOT_SPATIAL_VERIFY_05_20_14_42.md)
**Session type:** MVP v4 ship → n=1000 collapse → VR-TTS pivot → zoom fail → broader brainstorm → honest tier reassessment

---

## Summary

Built MVP v4 (top-edge + bulk + idiom blacklist) for spatial verification — n=100 looked promising (+0.020) but **n=1000 collapsed to −0.013**. Pivoted to VR-TTS (Visual Reasoning via Test-Time Scaling) on new branch `feat/vrtts-phase1`, implemented zoom + SoM + sub-question actions with full framework abstractions. Same pattern repeated: VR-TTS zoom n=100 = +0.020, n=1000 = −0.006. After literature survey + user pushback, honestly reassessed: **VEV (multi-oracle voting) at surface = ensemble + Bayesian voting = Q2 not Q1**. Currently awaiting user decision on 3 paths: safe Q2 (VEV), risky Q1 (one of info-theoretic bound / V-SAE / Test-Time Training), or sequential (VEV first + V-SAE derisk parallel).

---

## Work Completed

### Changes Made

- [x] **MVP v4** (`scripts/eval_reefknot_yesno_oracle_mvp_v4.py`) — top-edge + bulk fallback for above/below; idiom blacklist for "up"/"down" particles
- [x] **n=100 v4 eval** → Δ=+0.020 (1 helped via idiom fix, 0 harmed vs v3)
- [x] **n=1000 v4 eval** → Δ=**−0.013** (CI95 [−0.019, +0.008]) — collapse confirmed
- [x] **Failure-mode taxonomy** (47 harmed cases) → 5 modes: bbox-coincident (11), bias=yes FP (12), GD missing (8), X≈Y size (3), other (13)
- [x] **Counterfactual simulation**: even perfect abstention on all 4 failure modes → exactly baseline (Δ=0). No latent signal recoverable.
- [x] **Created branch `feat/vrtts-phase1`** from `feat/dt-sgod`
- [x] **VR-TTS Week 1 skeleton**: `VRTTSDecoder`, `VisualAction`, `ZoomToAttentionRegion`, `ConfidenceEstimator`, `TraceAggregator`, `HeuristicExplorationPolicy`
- [x] **LLaVA backbone extension**: `forward_first_step_with_attentions()` + `attn_implementation="eager"` option (SDPA can't emit attentions)
- [x] **Bug fix**: HF processor pre-expands `<image>` placeholder to 576 visual tokens — corrected position-finding logic
- [x] **Sanity check** (n=20): K=1 byte-perfect matches baseline ✅
- [x] **VR-TTS Week 2**: `AddSoMMarks` + `AskSubQuestion` actions
- [x] **Phase 1 n=100 eval** K=1,2,3: Δ(K=2 vs K=1) = +0.020 CI95[Δ] crossing 0
- [x] **Phase 1 n=1000 eval** K=1,2: Δ = **−0.006** CI95[Δ] [−0.019, +0.008] — VR-TTS zoom collapse
- [x] **2 literature agents dispatched**: (1) Reefknot SOTA + steering whitespace, (2) VLM test-time scaling 2024-26
- [x] **8-frame test-time scaling taxonomy** documented
- [x] **Honest tier reassessment**: VEV at surface = lego + voting = Q2, not Q1
- [x] 480 unit tests pass (27 new VR-TTS tests, no regressions)

### Key Decisions

| Decision | Rationale | Alternatives Considered |
|---|---|---|
| Branch from `feat/dt-sgod` not `main` | Inherits LLaVABackbone + oracles + framework | Branch from main (reimplement infra) |
| `forward_first_step_with_attentions()` as separate method, not Backbone interface | Only VR-TTS needs attentions; don't pollute abstract Backbone | Add to base interface |
| `attn_implementation="eager"` only for VR-TTS callers | Default sdpa keeps existing tests fast; eager only when needed | Force eager globally |
| Disable confidence gating (threshold=10) in Phase 1 | LLaVA yes/no logit gap empirically < 0.1; pure scaling measurement | Tune threshold |
| Scale only K=1,2 to n=1000 (not K=1..5) | K=3 added nothing over K=2 at n=100; saved 5x compute | Run full K=1..5 |
| Honest tier reassessment after VEV critique | User correctly identified VEV = ensemble + voting | Continue claiming Q1 |
| Save memory `project_vrtts_direction.md` | Significant project pivot worth persisting | Wait until phase 1 confirmed |

---

## Files Affected

### Created (this session)

- `scripts/eval_reefknot_yesno_oracle_mvp_v4.py` — top-edge + bulk + idiom fixes (commit `6dcf52f`)
- `sgod/policies/vrtts/__init__.py` — VR-TTS package surface
- `sgod/policies/vrtts/decoder.py` — `VRTTSDecoder` orchestrator with prompt_suffix injection
- `sgod/policies/vrtts/confidence.py` — yes/no logit gap normalized confidence
- `sgod/policies/vrtts/aggregator.py` — `TraceAggregator` (last + weighted_vote modes)
- `sgod/policies/vrtts/policy.py` — `HeuristicExplorationPolicy` (zoom → som → subq cycle)
- `sgod/policies/vrtts/actions/__init__.py`
- `sgod/policies/vrtts/actions/base.py` — `VisualAction` ABC + `VisualState` dataclass
- `sgod/policies/vrtts/actions/zoom.py` — attention grid → top-k bbox → crop with min-frac + padding
- `sgod/policies/vrtts/actions/som.py` — `AddSoMMarks` via GD per-phrase queries + PIL overlay
- `sgod/policies/vrtts/actions/subquestion.py` — `AskSubQuestion` via short greedy decode + context append
- `tests/test_vrtts_unit.py` — 27 unit tests (no GPU)
- `scripts/sanity_vrtts_k1.py` — K=1 vs baseline equivalence check
- `scripts/eval_vrtts_phase1.py` — n=N eval with bootstrap-2000 CI per K
- `docs/handoffs/HANDOFF_VRTTS_FAILED_REASSESS_05_20_18_30.md` (this file)

### Modified (this session)

- `sgod/backbones/llava15.py` — added `forward_first_step_with_attentions()` + `attn_implementation` ctor arg
- `sgod/utils/model_loader.py` — added `attn_implementation` kwarg to `load_llava()`
- `docs/paper_section_3_method.md` — §3.4 with v4 numbers + §3.5 new limitations (commit `f3c1426`)

### Memory updated (`~/.claude/projects/.../memory/`)

- `finding_reefknot_yesno_structural.md` — v4 numbers, resolved/remaining limitations
- `project_vrtts_direction.md` (NEW) — pivot direction, Phase 1 roadmap, V-SAE fallback
- `MEMORY.md` — index entries for both

### Output files (gitignored, on uet)

- `outputs/stage0/eval_reefknot_yesno_mvp_v4.json` (n=100, Δ=+0.020)
- `outputs/stage0/eval_reefknot_yesno_mvp_v4_n1000.json` (n=1000, Δ=−0.013)
- `outputs/vrtts/phase1/sanity_k1.log` (20/20 match)
- `outputs/vrtts/phase1/eval_n100.json` (K=1,2,3; Δ=+0.020)
- `outputs/vrtts/phase1/eval_n1000_K1K2.json` (K=1,2; Δ=−0.006)

### Read (Reference)

- `sgod/backbones/llava15.py:90-200` — KV-cache pattern, hidden state extraction
- `sgod/core/interfaces.py:21-86` — Backbone abstract contract
- `sgod/sgg/grounding_dino_module.py:170-216` — GD detect() signature for SoM reuse
- `outputs/stage0/eval_reefknot_yesno_mvp_v3.json` — n=100 v3 baseline for v4 diff analysis

---

## Technical Context

### Critical empirical finding (paper-worthy independent of next direction)

Two independent training-free methods (MVP v4 spatial-verify + VR-TTS zoom) **both showed +0.020 at n=100 and collapsed at n=1000** (−0.013 and −0.006 respectively). Both followed identical pattern: GT=yes accuracy drops more than GT=no gains. The deeper cause is structural — **neither method introduces new information to the system**:

- MVP v4: GD bbox detects same pixels LLaVA already saw → no new info
- VR-TTS zoom: crops pixels LLaVA already saw → strictly less info (information *loss*)
- Both methods rearrange logits but cannot manufacture information that doesn't exist

LLaVA-1.5-7B baseline on Reefknot spatial subset = **0.504 (chance)** at n=1000 → model literally does not have the answer. Training-free interventions on same information substrate hit a structural ceiling.

This is itself a **publishable finding** (rigorous n=1000 scaling study of training-free interventions).

### Architecture decisions (VR-TTS framework)

- VR-TTS as `Policy` peer to `dt_sgod/` and `sgod_v1.py` in `sgod/policies/`
- `VisualAction` ABC with `apply(state, *, attentions, question, backbone, oracle) -> VisualState` — optional kwargs let each action pull what it needs without expanding base interface
- `VRTTSDecoder.run()` loop: forward → confidence → if not gated, policy.select() → action.apply() → repeat
- `state.prompt_suffix` is the channel through which actions communicate with the next forward
- Backbone attention extraction lives ONLY on `LLaVAv15Backbone` (sibling method, not base interface) — VEV/MOT-V style backbone-independence preserved for other oracles

### Dependencies (already present)

- `stream-unzip`, `httpx` (from prior session for VG download)
- All VR-TTS code uses only existing deps (`torch`, `transformers`, `PIL`)

---

## Things to Know

### Critical user feedback (drives next decision)

User's final critique correctly identified:

1. **"VEV simplified = lego + voting"** — multi-oracle Bayesian aggregation is ensemble theory (Hoeting+ 1999) + sensor fusion + tool-augmented LMs. Not Q1-novel.
2. **Backbone independence (yes for Qwen-VL swap)** — but this is *because* VEV is black-box ensemble; it doesn't leverage VLM-specific structure, which weakens claims to mechanistic novelty.

User has consistently wanted **HRM/Mamba-level novelty** (see `memory/research_breakthrough_patterns.md`, `memory/user_profile.md`). VEV as currently framed does not meet this bar.

### Literature whitespace verified (2 agents)

- **Reefknot benchmark essentially uncovered** by 12 surveyed clusters — only the original paper used it
- **Cluster 4 (external tools at test-time)** most underexplored for relations — SpaceTools (2512.04069) and SpatialRGPT (2406.01584) are training-time
- **Predicate-perturbation contrastive decoding** untried (VCD perturbs IMAGE, not predicate)
- **Activation steering × relation hallucination** completely empty (6 recent steering papers all test POPE/MME)

### Gotchas

- **HF LLaVA processor pre-expands `<image>` placeholder** to 576 image_token_id tokens BEFORE forward — image-token positions found via `[i for i,t in enumerate(ids_row) if t == image_token_id]`, expect exactly 576
- **SDPA attention silently returns None** when `output_attentions=True` — must load with `attn_implementation="eager"` (~2-3× slower than sdpa)
- **LLaVA yes/no logit gap empirically < 0.1** — confidence threshold > 1.0 effectively disables gating
- **n=100 first seed=42 split is ~4.5pp easier than population** — both v3, v4, and VR-TTS show +0.020 at n=100 that becomes ~0 at n=1000. Don't trust n=100 signals.
- **eager attention runtime**: ~5s per LLaVA forward on 4090; n=1000 × K=2 ≈ 2-3h
- **Zoom REMOVES context** — fundamental disanalogy with LM CoT scaling which ADDS reasoning tokens

### Assumptions

- Reefknot YESNO.jsonl on uet at `data/reefknot/YESNO.jsonl` + images at `data/reefknot/images/`
- LLaVA-1.5-7B + Grounding DINO base + (planned) DepthAnything available on uet
- Seed=42 split is reproducible across all eval scripts

### Known issues / Tech debt

- VR-TTS zoom action provably fails at scale — should NOT promote to framework as-is
- `eval_vrtts_phase1.py` runs all K through same decoder instances (correct, but slow); could parallelize across GPUs
- No integration test for VR-TTS (only unit tests + manual sanity script)
- `docs/paper_section_3_method.md` claims v4 spatial verifier is current SOTA — needs revision now that v4 collapsed (frame as "ablation")

---

## Current State

### What's Working

- VR-TTS framework abstractions clean (`VRTTSDecoder`, `VisualAction`, `ConfidenceEstimator`, `TraceAggregator`, `HeuristicExplorationPolicy`)
- 3 actions implemented (zoom, som, subq) with unit tests
- LLaVA backbone attention extraction + eager-mode option
- Sanity check passes (K=1 = baseline byte-perfect)
- 480 unit tests pass

### What's Not Working

- **VR-TTS zoom does not provide signal at n=1000** (Δ=−0.006, CI [−0.019, +0.008])
- SoM action did not add over zoom at n=100 K=3 (untested at scale)
- SubQ action untested at scale (only n=100 K=3 with confounded zoom+som)
- The "test-time scaling primitive" framing claimed too much novelty — user correctly identified as ensemble + voting

### Tests

- [x] Unit tests: 480 pass (27 new VR-TTS), 10 deselected integration
- [ ] Integration tests: not run this session
- [x] Manual end-to-end: sanity K=1 (20/20), Phase 1 n=100 + n=1000

---

## Next Steps

### Immediate (Start Here) — AWAITING USER DECISION

User must choose path before proceeding. Last assistant message (verbatim) summarized:

**Path 1 — Safe (VEV as Q2)**: Implement 6-oracle ablation + Bayesian aggregator on Reefknot. ~4-6 weeks. High chance Q2 ship.

**Path 2 — Risky (Q1 swing)**:
- **X**: Information-theoretic lower bound on training-free interventions (theoretical paper)
- **Y**: V-SAE (Visual Sparse Autoencoders) on LLaVA encoder — mechanistic interpretability (agreed fallback per memory)
- **Z**: Test-Time Training with self-distillation — gradients at inference

**Path 3 — Sequential**: 6 weeks Path 1 + 2 weeks Path Y derisk in parallel.

**Resume action**: Read user's response. If Path 1 → implement 6-oracle ablation matrix (counterfactual + mirror-flip + depth-overlay + GD-bbox-MVPv4-reused + CLIP-alignment + LLaVA-logit). If Path 2/3 → derisk Y (V-SAE) first by checking if LLaVA visual encoder has linear concept structure for spatial relations.

### Subsequent (after path chosen)

- **For VEV (any path)**: implement individual oracles, train Bayesian aggregator MLP on 200-sample calibration split, n=300 dev eval per oracle (ablation), then n=1000 if signal
- **For V-SAE**: train SAE on LLaVA's projector output (CLIP-ViT-L/14 24×24 patches → 4096-d), discover features, probe for spatial concepts
- **Update paper draft** `docs/paper_section_3_method.md` to reflect v4 collapse (current text claims v4 is SOTA)
- **Commit current branch + push** before pivoting to V-SAE if that's the choice

### Blocked On

- **User decision on path** (most critical — affects next 4 weeks-6 months of work)
- Possibly: disk space on uet (~9GB free; SAE training may need cleanup)

---

## Related Resources

### Documentation

- `docs/paper_section_3_method.md` — paper draft (CURRENTLY OUT OF DATE re: v4 collapse)
- `docs/handoffs/HANDOFF_REEFKNOT_SPATIAL_VERIFY_05_20_14_42.md` — previous handoff (start of this session)
- Memory: `finding_reefknot_yesno_structural.md`, `project_vrtts_direction.md`

### Commands to Run

```bash
# To resume: read user's path choice from latest message, then:

# IF PATH 1 (VEV ablation):
# Implement 6 oracles as new sgod/policies/vrtts/oracles/*.py
# Build new script scripts/eval_vrtts_oracles.py for ablation
# (none exist yet — Week-1 of new sub-plan)

# IF PATH 2/3 (V-SAE derisk):
# New module sgod/policies/vsae/ — SAE training + intervention
# (none exists yet — Week-1 of new sub-plan)

# Sanity check current state (no-GPU)
.venv/bin/python -m pytest tests/test_vrtts_unit.py -q --override-ini="addopts="

# On uet to verify VR-TTS still operates
.venv/bin/python scripts/sanity_vrtts_k1.py --n 5
```

### Search Queries

- `grep -rn "VRTTSDecoder\|VisualAction" sgod/` — VR-TTS framework usages
- `grep -rn "forward_first_step_with_attentions" sgod/` — attention extraction call sites
- `grep -rn "attn_implementation" sgod/` — eager-mode threading

---

## Open Questions

- [ ] **Which path (1/2/3)?** This blocks all next steps.
- [ ] If Path 2 — which of X/Y/Z? (V-SAE is agreed fallback per memory)
- [ ] Is paper draft Section 3 (claiming spatial-verify is contribution) salvageable as "negative ablation" framing, or rewrite entirely?
- [ ] If VEV chosen, does paper title shift to "Multi-Oracle Test-Time Verification" or keep VR-TTS framing?
- [ ] Should we publish the negative scaling study (v4 + zoom both collapse) as a standalone short paper? It's a real, publishable finding.

---

## Session Notes

- This session had **3 major direction shifts**: spatial-verify v4 ship → VR-TTS pivot → VEV reframe → honest tier reassessment. Each shift driven by data (n=1000 collapse) or user pushback ("only zoom tested", "VEV = lego + voting").
- The user is consistently push for genuine novelty (HRM/Mamba-level per memory). I over-claimed twice (VR-TTS as primitive, then VEV as primitive). User correctly identified both inflations. Last response was honest tier reassessment.
- The **n=1000 scaling collapse is itself a meaningful paper-worthy result** — most training-free intervention papers in literature report only n<500. Our two failed methods (v4 + zoom) plausibly mirror unreported failures elsewhere.
- The user is Vietnamese — keep replies in Vietnamese.
- **Two literature agents dispatched and completed**: results saved in agent task outputs (not in this repo; references in `memory/finding_reefknot_yesno_structural.md` and conversation history).
- Branch `feat/vrtts-phase1` is clean (3 commits on top of `feat/dt-sgod`); push to origin already done. Safe to abandon VR-TTS branch if path 2 chosen, or merge to `feat/dt-sgod` if path 1.
- No commit needed for this handoff — output files all gitignored, all code already committed.

---

*This handoff was generated near context window capacity. To resume: open this file, read the "Immediate Next Steps — AWAITING USER DECISION" section, check the user's latest reply to determine which of Path 1/2/3 was chosen, then proceed accordingly. The structural finding (n=1000 collapse of v4 + zoom) is the most important durable insight — preserve it regardless of next direction.*
