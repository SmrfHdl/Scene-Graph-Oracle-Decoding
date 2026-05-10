# Handoff: BG-SGOD Direction Test + Yes/No Anchor Mechanism

**Created:** 2026-05-08
**Branch:** feat/dev-apr (clean — all work committed in 588b8ea)
**Session topic:** Diagnosed why soft-injection is a no-op under greedy LLaVA, then built and shipped a yes/no-specific injection path for POPE

---

## Summary

Diagnosed a fundamental issue: under greedy decoding, the `logit + λ × oracle_score` mechanism is too weak (~0.045 logits) to flip LLaVA's argmax (~1-5 logit gaps), so all prior MMHal "wins" with temp=0.7 were sampling noise. Pivoted to POPE for direction-testing the structural pipeline, found a clean **8× direction signal** (P(match|yes)=0.41 vs P(match|no)=0.05). Built a **yes/no anchor mechanism** that bypasses the soft-additive path entirely and injects directly onto yes/no token logits on the first generated token — currently shipped at `yesno_lambda=2.0`, **not yet eval'd on server**.

---

## Work Completed

### Changes Made (all in commit 588b8ea + later edits — verify via git log)

- [x] M1.5a: `BboxClipScorer` class — bbox-conditioned CLIP with crop/text caching, per-crop calibration (16 unit tests)
- [x] M1.5b: `VisualOracle` integration — `bbox_scorer` + `question` kwargs, `_resolve_target_bboxes()`, route OOV attr through bbox path (8 unit tests)
- [x] M1.5c: `SGODDecoder.enable_bbox_oracle` flag, per-flip diagnostic (`flips`, `n_flips`, `n_flip_anchor_attempts`)
- [x] Hybrid SGG: `SGGModule` accepts optional `gd_module` for open-vocab objects + RelTR relations
- [x] `bbox_score_multiplier` direction-test knob (set 5.0 in config)
- [x] **Question parser rewrite** (NP-walk replaces adjective-skip, modal verbs added to non-nouns, "person" un-blacklisted, head length ≥2)
- [x] **Yes/no anchor mechanism** (`is_yesno_question` + `_collect_yesno_ids` + first-token direct injection)
- [x] `temperature: 0.0` in default.yaml — eliminates sampling noise from base/sgod comparison
- [x] POPE grounding spike script `scripts/spike_pope_grounding.py` (no LLaVA, just SG + parser)

### Key Decisions

| Decision | Rationale | Alternatives Considered |
|---|---|---|
| Pivot from MMHal to POPE for direction test | MMHal token-overlap is too coarse; POPE binary metric is sensitive | Stay on MMHal; switch to GPT-4 scoring (needs API) |
| Bypass anchor detector for yes/no | "Yes"/"No" tokens are neutral — soft-additive path never fires; need direct injection | Add `yesno_anchor` type to detector (more invasive) |
| Asymmetric injection magnitude | When `has_match=True` (P(yes\|match)=0.89, strong); `has_match=False` (P(no\|no_match)=0.95 strong but P(yes\|no_match)=0.39 noisy) | Symmetric only-when-match; only-when-no-match |
| `yesno_lambda=2.0` (=2 logits) | LLaVA logit gaps on POPE typically 1-3; 2.0 flips moderate-confidence cases without overriding strong cases | 1.0 (likely no flips), 5.0 (overrides too aggressively) |
| Parser: walk full NP, take last word as head | Multi-word ("dining table") was emitting first non-adjective ("dining"); spike showed this is the dominant POPE recall miss | Try multi-target emission (full NP + head) — more complex |
| Remove "person" from generic referents | POPE has many "Is there a person?" questions; multi-target oracle handles MMHal "person riding X" via max-pool anyway | Context-aware filtering |

---

## Files Affected

### Created
- `sgod/oracle/bbox_clip_scorer.py` — `BboxClipScorer` class, ~250 LoC
- `scripts/spike_pope_grounding.py` — standalone POPE grounding test (no LLaVA)
- `tests/test_bbox_clip_scorer.py` — 16 tests
- `tests/test_grounding_dino_split.py` — 9 tests for fused-label split

### Modified
- `sgod/oracle/visual_oracle.py` — `bbox_scorer` + `question` + `bbox_score_multiplier` kwargs; `_resolve_target_bboxes()`; `has_bbox_target` property
- `sgod/oracle/clip_scorer.py` — stash `_preprocess`, add `to_preloaded()` for sharing weights with sibling scorers
- `sgod/oracle/question_parser.py` — full NP-walk parser; `_YESNO_AUX`; `is_yesno_question()`
- `sgod/sgg/sgg_module.py` — optional `gd_module` for hybrid mode (`SceneGraph.from_hybrid`)
- `sgod/sgg/grounding_dino_module.py` — fused-label split via `_split_label()`; HF API fix (`box_threshold` → `threshold`)
- `sgod/decoder/sgod_decoder.py` — `enable_bbox_oracle`, `bbox_score_multiplier`, `flip_log_cap`, `yesno_lambda`, `_collect_yesno_ids()`, first-token yesno injection block
- `sgod/utils/model_loader.py` — read `enable_bbox_oracle`, `bbox_pad_ratio`, `bbox_score_multiplier`, `yesno_lambda`, `use_grounding_dino`, GD thresholds from config
- `configs/default.yaml` — `temperature: 0.0`, `enable_bbox_oracle: true`, `bbox_score_multiplier: 5.0`, `yesno_lambda: 2.0`, `use_grounding_dino: true`
- `tests/test_oracle.py` — +8 bbox-oracle tests + multiplier test + `has_bbox_target` test
- `tests/test_question_parser.py` — +24 tests (multi-word NP, person, tv, modal verbs, yesno detector)

---

## Technical Context

### Architecture / Diagnostic Flow

```
Pipeline (per generate call):
  image → SGGModule.extract()
    ├─ RelTR forward → relations
    ├─ GroundingDinoModule.detect() → open-vocab ObjectNodes
    └─ SceneGraph.from_hybrid(gd_objects, reltr_relations)

  clip = clip_factory(image)          # full-image CLIP
  bbox_scorer = BboxClipScorer(image, _preloaded=clip.to_preloaded())
  oracle = VisualOracle(sg, clip, bbox_scorer=bbox_scorer, question=question,
                        bbox_score_multiplier=5.0)
    ├─ parse_targets(question) → list of head nouns
    ├─ match_targets_to_vocab(targets, gd_object_labels) → matched
    └─ _target_bboxes = bboxes of matched objects (max conf per label)

Decoder loop (greedy):
  do_yesno = (yesno_lambda > 0) AND use_oracle AND is_yesno_question(q)
  for step in range(max_new_tokens):
      logits = vlm(input_ids, pixel_values)[..., -1, :]
      if do_yesno and step == 0:
          if oracle.has_bbox_target:
              logits[yes_ids] += λ_yesno; logits[no_ids] -= λ_yesno
          else:
              logits[yes_ids] -= λ_yesno; logits[no_ids] += λ_yesno
          # log yesno_top1_before/after to last_stats
      # Existing soft-additive path on anchor tokens (mostly no-op for POPE)
      next_id = argmax(logits)
```

### Direction signal (POPE spike v2, parser fixed)
```
Overall (n=600):  Acc=67.3%, Prec=89.3%, Recall=40.9%, F1=56.1%
P(match|yes)=0.41   P(match|no)=0.05    →  ratio = 8×
By split: adversarial=popular=0.40, random=0.42 (consistent)
```

### Configuration Changes

`configs/default.yaml` — current state:
```yaml
sgg:
  use_grounding_dino: true
  grounding_dino_box_threshold: 0.25
  grounding_dino_text_threshold: 0.2
oracle:
  enable_bbox_oracle: true
  bbox_pad_ratio: 0.15
  bbox_score_multiplier: 5.0
  yesno_lambda: 2.0
decoder:
  temperature: 0.0   # greedy — DO NOT set higher; sampling noise destroys A/B
```

---

## Things to Know

### Critical Gotchas

1. **temp=0.0 is NOT optional** — under temp=0.7, oracle-off records still differ from base due to sampling, polluting comparison. All "wins" pre-2026-05-01 were noise.
2. **Soft-injection is structurally too weak under greedy.** Don't waste server time tuning λ_context above 0.5 — math doesn't work. The yesno path bypasses this; for MMHal-style questions, would need redesign.
3. **POPE prompt format matters**: `"Answer with yes or no.\nASSISTANT:"` — `is_yesno_question()` detects via "yes or no" substring + first-word check.
4. **Bbox path is narrow**: only fires on attr_anchor with resolved target. ~2% of MMHal tokens. Don't expect MMHal F1 wins until injection mechanism redesigned.
5. **Disk space on server was full** during this session — user fixed by clearing caches. Watch for re-occurrence.

### Assumptions

- LLaVA logit gap top-1 to top-2 is typically 1-3 (no measurement, derived from flip rate at λ=5 being 0/1137).
- `yesno_lambda=2.0` is a guess. May need to tune up if 0 flips, down if false-positive flips dominate.
- POPE's "person" detection is GD-recall-limited; ~50% miss rate even when present.

### Known Issues

- GD detection misses on common COCO classes: backpack, bicycle, motorcycle, snowboard, vase, cup, bottle, knife, fork, spoon, remote, tv, toothbrush, mouse. (Lower `box_threshold` to 0.20 if recall too low after eval.)
- Question parser still has gerund issues for MMHal — "the man wearing a hat" emits "wearing". Not blocking POPE.

---

## Current State

### What's Working
- Full BG-SGOD pipeline: hybrid SG, bbox-conditioned CLIP, question parser, oracle integration ✓
- POPE grounding signal: 8× direction ratio, F1=56% standalone ✓
- Yes/no mechanism: code complete, 336 unit tests pass, smoke imports OK ✓
- Per-flip diagnostic logs `yesno_fired`, `yesno_pushed`, `yesno_top1_before/after` ✓

### What's Not Working (or Untested)
- **POPE eval not yet run with yesno mechanism** — this is the blocking experiment
- MMHal: oracle is no-op under greedy. Pending mechanism redesign (out of scope for current session)
- Direction signal partial (recall 41%, not 60%) — bottleneck is GD recall on common COCO classes

### Tests
- [x] Unit tests: **336 passed, 5 skipped** (`pytest tests/ --override-ini="addopts="`)
- [ ] Integration tests: skipped by default (need GPU + checkpoints)
- [ ] Manual POPE eval: NOT YET RUN — this is next step

---

## Next Steps

### Immediate (Start Here)

1. **Run POPE eval on server** (uncommitted task — may need disk space check first):
   ```bash
   df -h /            # ensure ≥10 GB free
   PYTHONPATH=$(pwd) CUDA_VISIBLE_DEVICES=1 $SPIN \
       experiments/main_eval/eval_pope.py \
       --config configs/default.yaml \
       --pope_path data/pope/ \
       --coco_path data/coco/val2014/
   ```
2. **Analyze POPE results** — specifically:
   - `pope_results.json` for ΔF1 per split (adversarial / popular / random)
   - `records_*.json` for per-record `sgod_stats.yesno_fired`, `yesno_top1_before/after`
   - Compute: flip rate (% records where before≠after), direction success (% flips where after matches GT label)
3. **Decision branch on results:**
   - If ΔF1 ≥ +2pp on adversarial → BG-SGOD direction validated, paper viable. Tune `yesno_lambda` and lower GD `box_threshold` for further gains.
   - If ΔF1 ≈ 0 with low flip rate → bump `yesno_lambda` to 3.0 or 4.0, re-run.
   - If ΔF1 negative → direction wrong; check `yesno_pushed` distribution. Maybe asymmetric: only push when `has_match=True`.

### Subsequent
- Lower `grounding_dino_box_threshold` 0.25 → 0.20 to recover GD recall
- Multi-target emission in parser (emit both "dining table" AND "table") for tighter vocab matches
- Once POPE works: revisit MMHal injection mechanism redesign (path A from prior plan: bypass context-λ for trusted bbox path)

### Blocked On
- Server disk space — user has 3.6 GB free, may fail LLaVA download. See "Gotchas" #5.

---

## Related Resources

### Documentation
- Prior handoffs: `docs/handoffs/HANDOFF_MMHAL_ORACLE_RECALIBRATION_04_28_19_30.md`, `HANDOFF_POPE_DEBUGGED_MMHAL_NEXT_04_28_15_00.md`
- POPE spike v1: `outputs/spike_pope_grounding.json` (recall 0.33)
- POPE spike v2 (parser fixed): `outputs/spike_pope_grounding_v2.json` (recall 0.41, **8× direction ratio**)
- Most recent MMHal eval (multiplier=5, 1 flip total): `outputs/eval/2026-05-08_01-08028_mmhal/`

### Commands to Run

```bash
# Full test suite (no GPU):
python -m pytest tests/ --override-ini="addopts=" -q

# Re-run POPE grounding spike (no LLaVA):
PYTHONPATH=$(pwd) CUDA_VISIBLE_DEVICES=1 $SPIN scripts/spike_pope_grounding.py \
    --pope_path data/pope/ --coco_path data/coco/val2014/ --n 200 \
    --output outputs/spike_pope_grounding_v3.json

# POPE eval with yesno mechanism (BLOCKING NEXT STEP):
PYTHONPATH=$(pwd) CUDA_VISIBLE_DEVICES=1 $SPIN \
    experiments/main_eval/eval_pope.py \
    --config configs/default.yaml \
    --pope_path data/pope/ \
    --coco_path data/coco/val2014/
```

### Search Queries
- `grep -rn "yesno_lambda\|is_yesno_question" sgod/` — find yes/no mechanism code paths
- `grep -rn "has_bbox_target\|_target_bboxes" sgod/` — find bbox grounding flow
- `grep -rn "bbox_score_multiplier" sgod/ configs/` — find multiplier wiring

---

## Open Questions

- [ ] What is LLaVA's actual logit gap distribution on POPE? (Empirical — could log via diagnostic for first ~50 records to verify `yesno_lambda=2.0` is right magnitude.)
- [ ] Does `yesno_lambda=2.0` flip too aggressively when `has_match=False` but GT=yes (recall miss)? Direction analysis post-eval will tell.
- [ ] If POPE eval shows clear win, do we revisit MMHal mechanism redesign or focus on POPE/AMBER for the paper?

---

## Session Notes

This session was the inflection point of the project: realized the soft-additive injection mechanism was structurally insufficient (math + 0/1137 flips at multiplier=5 confirmed), then pivoted to a methodology that actually measures whether the structural pipeline (hybrid SG + bbox grounding + question parser) has the right direction. The 8× direction ratio on POPE is the strongest positive signal we've had — it justifies the whole BG-SGOD direction even though MMHal numbers haven't shown it.

Key insight worth preserving: **token-overlap on free-form generation is too noisy/coarse to measure decode-time interventions; binary tasks (POPE) or proper hallucination scoring (GPT-4) should be the eval bedrock.**

The yesno mechanism is the first injection that has any chance of producing measurable LLaVA changes in this codebase. Treat its first eval as the make-or-break experiment for paper viability.

---

_This handoff was generated at context window capacity. Start a new session and use this document as your initial context._
