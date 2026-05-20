# Handoff: Reefknot Yes/No Structural Finding + Spatial-Verify MVP Working

**Created:** 2026-05-20 14:42
**Branch:** `feat/dt-sgod`
**Previous handoff:** [HANDOFF_STAGE0_PIVOT_REEFKNOT_05_19_17_12.md](./HANDOFF_STAGE0_PIVOT_REEFKNOT_05_19_17_12.md)
**Session type:** Reefknot data acquisition + 3 MVP iterations + paper Section 3 draft

---

## Summary

Picked up from Day 3 pivot handoff (POPE confirmed saturated → pivoted to Reefknot). Downloaded Reefknot YESNO + 15GB Visual Genome via stream-unzip on a disk-constrained machine. Ran 3 MVP iterations of yes/no logit-bias injection, discovering the **core structural finding**: anchor-based oracles cannot touch yes/no QA because LLaVA emits exactly 1 token at step 0 and "Yes"/"No" are never noun/relation/attr anchors. **MVP v3 spatial bounding-box verification works**: +4.0% on n=25 spatial-parseable subset, +6.0% on GT=no hard cases, **0% degradation** on cognitive/non-spatial subsets (conservative abstention). Wrote paper Section 3 (Method) draft capturing the mechanism. The narrative for the paper has crystallised — anchor-based methods can't intervene on yes/no QA; **spatial verification can**.

---

## Work Completed

### Changes Made

- [x] Wrote `scripts/stream_unzip_vg.py` — stream Visual Genome zips via HTTP+`stream_unzip` (no sudo, no intermediate zip storage; peak disk = extracted size only)
- [x] Resolved disk space (`uv cache clean` freed 14.8 GB → 23 GB free, enough for VG)
- [x] Downloaded 108,249 VG images to `data/reefknot/images/` (uet)
- [x] Downloaded Reefknot YESNO.jsonl (9740 questions, balanced 50/50) from GitHub mirror
- [x] Ran 3-way Reefknot eval (baseline / SGOD-v1 teacher / DT-SGOD-Stage0) on 100 examples → **all 3 byte-identical, Acc=0.670 each**
- [x] Diagnosed root cause: LLaVA emits 1-token answer at step 0; anchor oracle never fires on yes/no
- [x] **MVP v1** (`eval_reefknot_yesno_oracle_mvp.py`) — object presence via RelTROracle SG, ΔAcc = **−0.03** (synonym + vocab issues)
- [x] **MVP v2** (`eval_reefknot_yesno_oracle_mvp_v2.py`) — per-question Grounding DINO open-vocab query, ΔAcc = **−0.11** (GD detects both → bias_yes fires 95/100, GT=no tanks)
- [x] **MVP v3** (`eval_reefknot_yesno_oracle_mvp_v3.py`) — spatial bbox geometric verification + asymmetric conservative bias, ΔAcc = **+0.010** overall, **+4.0% on spatial subset**, **+6.0% on GT=no**
- [x] Wrote `docs/paper_section_3_method.md` — full Section 3 (Method) draft with equations, verifier table, decision tree, empirical results
- [x] Updated 2 memory files: `finding_reefknot_yesno_structural.md` (new), `MEMORY.md` (index)

### Key Decisions

| Decision | Rationale | Alternatives Considered |
|---|---|---|
| GitHub mirror for Reefknot data | HF `hyintell/ReefKnot` is gated (401); GitHub `JackChen-seu/Reefknot/Dataset/YESNO.jsonl` is public | huggingface-cli login + accept terms |
| Stream-unzip via Python (no sudo) | bsdtar not available, `sudo apt install` denied. `stream-unzip + httpx` is pure-Python, lazy peak disk | wget → unzip (needs 28 GB peak, only had 16 GB free) |
| `uv cache clean` (14.8 GB) before HF cache cleanup | Pure download cache, fully rebuildable, no risk to shared models | Removing Qwen-7B etc. (uncertain shared use) |
| Three MVP iterations before paper draft | Each MVP isolated one failure mode (synonym, vocab gap, relation grounding) → clearest paper narrative | Single ambitious script (would have conflated multiple issues) |
| **Spatial vs cognitive split** in v3 | Spatial relations admit bounding-box geometry; cognitive require visual semantics. Conservative abstention on cognitive preserves baseline | Try CLIP-similarity verifier on all (would add cost + noise; defer to next iteration) |
| **Asymmetric β_no=5, β_yes=2** | Spatial mismatch = strong "No" evidence; both present = necessary but not sufficient for "Yes" | Symmetric ±5 (would tank GT=yes; v2 showed this) |
| Per-question GD query (not pre-computed SG) | Open-vocab phrase queries handle synonyms (bike↔bicycle) + OOV objects (powder, pepperoni). RelTR 151 vocab insufficient | Pre-computed SG with synonym matching (wouldn't solve OOV) |
| Paper draft before scaling experiments | Capture mechanism + narrative while fresh; numbers can be filled in once n=1000 run completes | Wait for n=1000 first (risks losing structural insight) |

---

## Files Affected

### Created (this session)

- `scripts/stream_unzip_vg.py` — stream-download + extract Visual Genome zips (no intermediate file)
- `scripts/eval_reefknot_yesno_oracle_mvp.py` — MVP v1, RelTR-based object presence (degraded)
- `scripts/eval_reefknot_yesno_oracle_mvp_v2.py` — MVP v2, per-question GD object presence (degraded)
- `scripts/eval_reefknot_yesno_oracle_mvp_v3.py` — MVP v3, **spatial bbox verification + asymmetric bias (WORKS)**
- `docs/paper_section_3_method.md` — **Section 3 (Method) paper draft**, ~1900 words with equations
- `data/reefknot/YESNO.jsonl` (uet) — 9740 Reefknot yes/no questions
- `data/reefknot/images/*.jpg` (uet) — 108,249 Visual Genome images (~14 GB)
- `outputs/stage0/eval_reefknot.json` (uet) — 3-way Reefknot baseline run (n=100, all identical)
- `outputs/stage0/eval_reefknot_yesno_mvp.json` (uet) — MVP v1 results
- `outputs/stage0/eval_reefknot_yesno_mvp_v2.json` (uet) — MVP v2 results
- `outputs/stage0/eval_reefknot_yesno_mvp_v3.json` (uet) — **MVP v3 results (current state-of-art)**
- `docs/handoffs/HANDOFF_REEFKNOT_SPATIAL_VERIFY_05_20_14_42.md` — this file

### Memory updated

- `memory/finding_reefknot_yesno_structural.md` (new) — captures the anchor-coverage gap + MVP v3 numbers
- `memory/MEMORY.md` — added index entry

### Read (Reference)

- `sgod/backbones/llava15.py:90-135` — verified `tokenizer()` is method, `prepare_inputs/forward_step` API for raw step-0 logits
- `sgod/runtime/decoder.py:117-131` — `generated_ids = torch.empty(1, 0, dtype=torch.long)` pattern for first step
- `sgod/oracles/reltr_oracle.py:73` — `extract(image, image_meta)` signature (no `prompt`/`question` kwargs)
- `sgod/sgg/grounding_dino_module.py:170-216` — `detect()` flow, processor + model API
- `experiments/main_eval/eval_reefknot.py` — POPE/Reefknot question format (note: actual schema differs from docs in this file)

### Uncommitted

- `docs/paper_section_3_method.md` — paper draft, untracked (intentionally not committed yet; review-pending)

Most recent commits this session: `aca19da`, `ba66f3d`, `92487e0`, `c32a194`, `ee85627` — all `feat(scripts)` MVP iterations and stream-unzip.

---

## Technical Context

### Core finding (the paper contribution)

LLaVA-1.5-7B emits exactly 1 token ("Yes" / "No") at step 0 on Reefknot yes/no questions, then EOS. Anchor-based oracles (SGOD v1) require the current candidate token to be in noun/relation/attr vocabulary; "Yes"/"No" aren't, so `detect_anchor → neutral` and Δ ≡ 0. **The oracle has zero effect on yes/no QA.**

Reefknot's hard cases are *not* object hallucinations — both X and Y are usually present in the image; the asked **relation** is wrong (cow inside field, not outside). Object-presence-only intervention pushes everything to "yes" and tanks GT=no.

**Spatial bbox verification** (MVP v3) bridges the gap: parse triplet → GD query both phrases → for spatial R, verify geometry (`cy_X < cy_Y` for "above" etc.) → asymmetric bias to "Yes"/"No" token IDs. Conservative abstention on cognitive R (riding/watching) preserves baseline.

### MVP v3 mechanism (per question)

```
1. Parse "Is X REL Y in this photo?" → (X, R, Y)
2. GD query X and Y with text="phrase." → (present, score, bbox) each
3. If R ∈ SPATIAL ({above, below, in, outside, near, far, ...}):
     if both bboxes: verify geometry → match → bias_yes; mismatch → bias_no
     else: bias_no (missing operand)
   elif R ∈ COGNITIVE (verbs):
     if both X,Y missing: bias_no
     else: abstain
4. Inject bias on yes_ids/no_ids of step-0 logits; argmax over {yes, no}
```

### Empirical numbers (n=100, MVP v3)

| Subset | Baseline | MVP v3 | Δ |
|---|---|---|---|
| Overall | 0.670 | 0.680 | +0.010 |
| Perception type (n=49) | 0.755 | **0.776** | **+0.020** |
| Spatial-parseable (n=25) | 0.560 | **0.600** | **+0.040** |
| GT=no (n=50) | 0.600 | **0.660** | **+0.060** |
| GT=yes (n=50) | 0.740 | 0.700 | −0.040 |
| Cognitive (n=51) | 0.588 | 0.588 | 0.000 (untouched) |
| Non-spatial (n=75) | 0.707 | 0.707 | 0.000 (untouched) |

Bias fires on 25/100 (high-precision low-recall by design). 4 helped − 3 harmed = +1 example = +1% global.

### Dependencies added (uet only)

- `stream-unzip` — pure-Python streaming zip decoder
- `httpx` — async-capable HTTP client (already an HF dep, but explicit)
- Both installed via `uv pip install` (no sudo, no system-wide change)

### Disk situation (uet)

- 915 GB drive, was at 99% (16 GB free) — shared multi-user `/home/uet/` 
- `uv cache clean` freed 14.8 GB → 23 GB free
- After VG download: ~9 GB free (tight but stable)
- Other large dirs (Qwen-7B 15 GB, etc.) untouched — likely shared-user models

---

## Things to Know

### Gotchas

- **No sudo on uet** — `apt install`, `apt-get` blocked. Use pure-Python pip workarounds.
- **No pip in `.venv` on uet** — uv doesn't seed pip; use `uv pip install` not `.venv/bin/pip`.
- **HF dataset `hyintell/ReefKnot` is gated** (401) — use GitHub mirror.
- **`backbone.tokenizer()` is a method, not a property** — must be called.
- **`RelTROracle.extract()` signature is `(image, image_meta=None)`** — does NOT accept `prompt`/`question` (those are policy-state concerns).
- **First-step `generated_ids` must be `torch.empty(1, 0, dtype=torch.long)`** — orchestrator's pattern, not `inputs.get("input_ids")` (which is None).
- **GD detects too eagerly** — yields ~95% precision on object presence questions. Object-only bias rules will catastrophically over-predict "yes".
- **Reefknot YESNO format docs in `experiments/main_eval/eval_reefknot.py` are stale** — actual fields are `image_id`/`query_prompt`/`label`/`relation_type` (perception/cognitive), not `image`/`question`/`answer`/`type`.
- **Reefknot subject/object phrases use synonyms vs RelTR 151 vocab** — "bicycle" vs "bike", "lady" vs "woman", "slope" vs "hill". Don't use RelTR for query lookup; use GD open-vocab per phrase.

### Assumptions

- Stage 0 distillation checkpoint (`outputs/stage0/dtsgod_stage0.pt` on uet) is degenerate (γ=0.005) — confirmed by prior handoff. The current intervention path (MVP v3) does NOT depend on this checkpoint.
- LLaVA-1.5-7B fp16 on uet GPU (one of 2× RTX 4090 48 GB)
- All MVPs answer yes/no by argmax over {Yes, ▁Yes, yes, ▁yes} vs {No, ▁No, no, ▁no} token IDs at step 0
- Reefknot test set seed=42 100-sample split is fixed across MVPs for fair comparison

### Known Issues / Tech Debt

- **bbox-center heuristic too coarse** for vertical relations when boxes coincide (e.g., "powder topping bread" → cy_X = cy_Y → match=False → wrong "no"). Fix: use top-edge for above/below predicates.
- **"up"/"down" parsed as spatial in idioms** ("dried up apple"). Fix: blacklist verb+up patterns.
- **GD recall misses → spurious bias_no**. Fix: reduce bias amount when only 1 object missing.
- **No mechanism for cognitive relations** — 51% of test set untouched. Planned: CLIP-similarity score over `b_X ∪ b_Y` crop vs "X R Y" phrase.
- **n=100 is small for statistical significance** on the n=25 spatial subset (+4% = 1 example). Need n=1000 for paper.

---

## Current State

### What's Working

- Reefknot data fully available on uet (`data/reefknot/{YESNO.jsonl, images/}`)
- 3 MVP scripts run end-to-end with proper logging and metric breakdowns
- **MVP v3 delivers positive signal** on the targeted subset (spatial + GT=no)
- Section 3 (Method) paper draft written with equations + verifier table + decision tree
- Memory captures the structural insight (`finding_reefknot_yesno_structural.md`)

### What's Not Working

- Cognitive relations (51% of test set) untouched — current mechanism has no path to verify "X riding Y"
- 3 GT=yes cases harmed by bbox-center artefacts (powder/bread, dried/apple, cabinet/stove)
- n=100 sample too small for confidence intervals; spatial subset n=25 directional only

### Tests

- [ ] Unit tests: not run this session (last session: 453 pass, no changes to `sgod/` source code)
- [ ] Integration tests: not run this session (last session: 5 pass on real GPU)
- [x] Manual end-to-end: MVP v1, v2, v3 all ran cleanly on uet GPU
- [ ] Significance test (n=1000): pending

---

## Next Steps

### Immediate (Start Here)

**1. Scale MVP v3 to n=1000** — confirm the +4% spatial signal isn't noise
   ```bash
   cd ~/truong_vlm/sgod-dt/Scene-Graph-Oracle-Decoding
   .venv/bin/python scripts/eval_reefknot_yesno_oracle_mvp_v3.py \
       --reefknot-jsonl data/reefknot/YESNO.jsonl \
       --image-dir data/reefknot/images \
       --n 1000 \
       --out outputs/stage0/eval_reefknot_yesno_mvp_v3_n1000.json \
       2>&1 | tee outputs/stage0/eval_reefknot_yesno_mvp_v3_n1000.log
   ```
   Expected runtime: ~2-3h on 4090. Fill in `Δ = +X.XX, p < Y.YY` in `docs/paper_section_3_method.md` §3.4.

**2. Implement two quick fixes** (predicted +1-2% additional)
   - Switch above/below verifier from centre comparison to top-edge: replace `cy_X < cy_Y` with `(b_X[1] + b_X[3])/2 < (b_Y[1] + b_Y[3])/2` → with top-edge `b_X[1] < b_Y[1]`
   - Blacklist idiomatic "up"/"down" patterns:
     ```python
     IDIOM_VERBS_UP = {"dried", "stood", "stand", "wake", "wakes", "woken", "look", "grew"}
     # If word before "up" matches → not spatial
     ```

**3. Re-run n=100 with fixes** to verify +1-2% improvement before scaling

**4. Polish paper draft** — fill in pending numbers, cross-check claims against actual logs

### Subsequent

- **Implement cognitive verifier** — CLIP similarity score over `b_X ∪ b_Y` crop vs phrase "$X$ $R$ $Y$"; expect +2-3% on cognitive subset
- **Eval on Reefknot Multichoice.jsonl + VQA.jsonl** — same images, different question formats, would broaden paper scope
- **Frame MVP v3 as proper framework Policy** (currently standalone scripts):
  - Create `SpatialVerifyPolicy` in `sgod/policies/spatial_verify.py`
  - Register `@register("policy", "spatial-verify")`
  - Reuse `HallucinationDecoder` orchestrator's step-0 hook (orchestrator currently iterates all steps; need first-step-only mode)
- **Write paper §1 Intro, §2 Background, §4 Experiments, §5 Results**

### Blocked On

- Statistical significance run (n=1000) — user task on uet
- Disk space for additional artefacts (only 9 GB free on uet currently; may need to clean Qwen-7B or similar if storing more checkpoints)

---

## Related Resources

### Documentation

- `docs/paper_section_3_method.md` — **the main artefact from this session** — paper Section 3 draft
- `docs/dt_sgod_proposal.md` — original architecture proposal (now superseded for yes/no QA)
- `docs/handoffs/HANDOFF_STAGE0_PIVOT_REEFKNOT_05_19_17_12.md` — previous handoff (Stage 0 failure + pivot decision)
- Memory: `finding_reefknot_yesno_structural.md` — the core insight + numbers, indexed in MEMORY.md

### Commands to Run

```bash
# Quick re-verify v3 mechanism still works on n=100
.venv/bin/python scripts/eval_reefknot_yesno_oracle_mvp_v3.py \
    --reefknot-jsonl data/reefknot/YESNO.jsonl \
    --image-dir data/reefknot/images \
    --n 100 --out outputs/stage0/eval_reefknot_yesno_mvp_v3_repro.json

# Scale to n=1000 (3h on 4090)
# See "Immediate Next Steps" above

# Tweak bias amounts (grid search)
for bn in 3 5 7; do for by in 1 2 3; do
    .venv/bin/python scripts/eval_reefknot_yesno_oracle_mvp_v3.py \
        --n 200 --bias-no $bn --bias-yes $by \
        --out outputs/stage0/grid_bn${bn}_by${by}.json
done; done

# Unit tests sanity
.venv/bin/python -m pytest tests/ --override-ini="addopts=" -q
```

### Search Queries

```bash
grep -rn "SPATIAL_RELATIONS\|_verify_spatial" scripts/    # spatial predicate table
grep -rn "bias_target\|bias_reason" scripts/              # bias-firing diagnostics
grep -rn "_query_gd\|stream_unzip" scripts/                # new utility functions
grep -rn "rule_anchor_type" sgod/                         # legacy wiring bug (cosmetic, not affecting v3)
```

---

## Open Questions

- [ ] **Does the spatial +4% hold at n=1000?** Critical for §3.4 paper claim. Single biggest unknown.
- [ ] **Can CLIP-similarity verify cognitive relations?** 51% of Reefknot is cognitive; would unlock the rest of the benchmark.
- [ ] **Top-edge vs centre for above/below**: how much of the GT=yes regression (3 cases) is due to centre coincidence? Need ablation.
- [ ] **Should MVP v3 become a proper `Policy` in the framework now, or wait until n=1000 confirms signal?** Currently it's a standalone script.
- [ ] **What threshold for "abstain"?** Currently 25/100 fire. Could be tuned: higher fire-rate may help recall but risk GT=yes harm.

---

## Session Notes

- Session focused on data acquisition (Reefknot + VG via stream-unzip), then 3 fast MVP iterations, then paper draft. Tight feedback loop: ~15 minutes per MVP run on uet, ~5-10 min code-iteration on laptop.
- The structural finding (anchor oracle can't touch yes/no QA) was identified by inspecting raw outputs after MVP v0 — all 3 systems produced **byte-identical 1-token answers**. This is the most important diagnostic moment of the session.
- MVP v1 → v2 → v3 isolation strategy paid off: each version pinned exactly one failure mode, making the paper narrative crystal clear.
- User chose to write paper draft NOW (vs scaling to n=1000) — correct call, captures the mechanism while context is fresh; numbers can be filled in.
- User responds in Vietnamese — keep replies in Vietnamese.
- 5 commits made this session by user (manual mid-session, expected pattern):
  `aca19da`, `ba66f3d`, `92487e0`, `c32a194`, `ee85627`. `docs/paper_section_3_method.md` still untracked.
- The disk-space saga (1h+) was a side quest but produced a reusable script (`stream_unzip_vg.py`) for future VG/COCO-scale downloads.

---

*This handoff was generated near context window capacity. To resume: open `docs/paper_section_3_method.md` to see the current paper narrative, then run `n=1000` scaling experiment as priority 1. The mechanism is in `scripts/eval_reefknot_yesno_oracle_mvp_v3.py`. Empirical numbers in `outputs/stage0/eval_reefknot_yesno_mvp_v3.json`. Memory index has the high-level pointer.*
