# Handoff: Components 4 & 5 implemented — ready for pilot experiment

**Created:** 2026-04-21 01:50
**Branch:** `feat/dev-apr`
**Session Duration:** ~single session

---

## Summary

Implemented Component 4 (GenerationContext tracker) and Component 5 (SGODDecoder) from their stub files. All five SGOD components (SGG, Oracle, Anchor, Context, Decoder) are now complete and unit-tested — **249 tests pass, 5 integration tests skipped**. The entire pipeline runs end-to-end in test scenarios using mocks. Next milestone is the pilot experiment on 200 GQA images, which requires real LLaVA-1.5-7B and a GPU.

---

## Work Completed

### Changes Made

- [x] Implemented `GenerationContext` class (Component 4) — negation depth + question-type + adaptive lambda
- [x] Implemented `SGODDecoder` class (Component 5) — end-to-end integration of all 5 components
- [x] Implemented `decode_top_k` + `apply_oracle_scores` helpers in `token_utils.py`
- [x] Wrote 56 unit tests for context tracker (`tests/test_context.py`)
- [x] Wrote 19 unit tests for decoder with full fake VLM/tokenizer/SGG/CLIP mocks (`tests/test_decoder.py`)
- [x] Created `scripts/demo_context.py` — 6-step token walkthrough of lambda adaptation
- [x] Created `scripts/demo_decoder.py` — pipeline visualization with synthetic or real SG
- [x] Fixed earlier bug carried over from Component 3: rule ordering (multi-word prep completion must precede single-token spatial-prep check)

### Key Decisions

| Decision | Rationale | Alternatives Considered |
|---|---|---|
| **Per-candidate anchor detection in decoder** — each top-K word gets its own `detect_anchor(prev, word)` call, scored with its own anchor type | A single position can have top-K candidates of different types: "on" is relation_anchor, "red" is attr_anchor, "dog" is noun_anchor. Scoring each with its own type is semantically correct. | Proposal suggested single `detect_anchor(prev, "")` per position then `batch_score(top_k, anchor)` — but that makes rules 5/6 (current-token-based) impossible to fire. |
| **Greedy decoding (argmax) as default** | Deterministic — makes tests reproducible and demo output predictable. | Proposal uses `torch.multinomial` with temperature=0.7. Sampling can be added later via a `temperature` parameter. |
| **`clip_factory: Callable[[image], CLIPScorer]` in `__init__`** | CLIPScorer needs the image, which isn't known at decoder construction time. Factory pattern defers construction until `generate()` is called. | Pass `CLIPScorer` class + kwargs; caller constructs scorer themselves. Factory is cleaner. |
| **In-place logit mutation in `apply_oracle_scores`** | Saves a tensor allocation per token step; hot path. | Return a new tensor — cleaner but slower at 256 tokens × vocab_size. |
| **Oracle injects only into top-K logit positions**, not full vocab | Risk 3 in proposal: scoring full 32K vocab is a 640s/token bottleneck. Top-K (default 50) keeps overhead ~1ms. | Full-vocab scoring with cached CLIP embeddings — faster but not yet wired up. |
| **Test mocks use `FakeVLM`/`FakeTokenizer`/`FakeSGGModule`/`FakeCLIPScorer`** | Unit tests must run in CI without GPU/checkpoints. Mocks give full control over logit outputs. | `unittest.mock.patch` — less explicit, harder to reason about VLM step-by-step behavior. |

---

## Files Affected

### Created

- `scripts/demo_context.py` — 6-step CLI walkthrough of question type detection, negation tracking, lambda recovery
- `scripts/demo_decoder.py` — pipeline visualization; supports `--image` for real RelTR+CLIP or synthetic SG
- `docs/handoffs/HANDOFF_COMPONENTS_4_5_04_21_01_50.md` — this file

### Modified

- `sgod/context/tracker.py` — implemented `GenerationContext`, `NEGATION_TOKENS`, `QUESTION_TYPE_SIGNALS`, `BASE_LAMBDA`
- `sgod/context/__init__.py` — exports from tracker
- `sgod/decoder/sgod_decoder.py` — implemented `SGODDecoder` class
  - `generate(image, question, max_new_tokens)` — Phase 1 oracle build + Phase 2 autoregressive loop
  - `_score_candidates()` — per-word anchor detection + oracle scoring
- `sgod/decoder/token_utils.py` — implemented `decode_top_k`, `apply_oracle_scores`
- `sgod/decoder/__init__.py` — exports from decoder + token_utils
- `tests/test_context.py` — 56 tests (vocab, question types, negation, lambda)
- `tests/test_decoder.py` — 19 tests (token_utils, EOS, oracle activation, per-candidate scoring)
- `sgod/anchor/detector.py`, `sgod/anchor/vocabularies.py`, `tests/test_anchor.py` — minor changes carried from prior session (already documented in earlier handoff)

### Read (Reference)

- `docs/sgod_proposal.md` — Section 3.5 (GenerationContext), 3.6 (SGODDecoder), Risk 2/3 (tokenization, top-K)
- `sgod/oracle/visual_oracle.py` — confirming `VisualOracle.score()` / `should_activate_oracle()` API
- `sgod/sgg/scene_graph.py` — confirming `SceneGraph` dataclass fields

---

## Technical Context

### Architecture Notes

- **Data flow per token step** (see `SGODDecoder.generate`):
  1. `vlm(input_ids)` → `logits[vocab_size]`
  2. `ctx.get_lambda()` — return 0.0 if hypothetical question or oracle deactivated
  3. If `lam != 0.0`: `decode_top_k(logits, tokenizer, k=50)` → candidate words
  4. `_score_candidates(words, prev_tokens, oracle)` — per-word anchor detection
  5. `apply_oracle_scores(logits, top_k_ids, oracle_scores, lam)` — in-place injection
  6. `argmax(logits)` → next token ID
  7. `ctx.update(decoded_token)` advances negation state
  8. Stop if `next_id == tokenizer.eos_token_id`
- **Oracle activation short-circuit**: if `SceneGraph.should_activate_oracle()` returns False (avg obj confidence < 0.4 or empty SG), entire Phase 2 runs without oracle injection — fallback to vanilla LM.
- **Rule ordering fix (carried from C3)**: in `detect_anchor`, multi-word prep completion check (→ relation_anchor) must run before single-token spatial-prep-as-prev check (→ noun_anchor) so "across from" fires correctly after `"across"` was added to `SPATIAL_PREPS`.

### Dependencies

No new dependencies added. Existing: `torch`, `transformers`, `open_clip_torch`, `PIL`, `pytest`.

### Configuration

`configs/default.yaml` already contains C4/C5 settings (`base_lambda`, `negation_decay`, `max_negation_depth`, `oracle_lambda`, `max_new_tokens`). These are **not yet wired into `SGODDecoder.__init__`** — decoder currently uses hard-coded defaults from `tracker.py` constants. Config loading is a TODO when scaling to experiments.

---

## Things to Know

### Gotchas & Pitfalls

- **Question type priority matters**: `QUESTION_TYPE_SIGNALS` is iterated in insertion order, first match wins. Questions like "How many chairs are there?" fire **existential** (because `"are there"` matches) not descriptive. Tests were written then fixed to reflect this. Document this clearly if updating signals.
- **Negation decay uses quarter-steps**: depth decrements by 0.25 per non-negation token, so 4 neutral tokens fully reset depth. This means "No, the dog is not red" still triggers deep negation around "not" but "No ... [5 tokens later] ... zebra" won't.
- **`FakeTokenizer` in tests is single-token-per-word**. Real LLaVA uses BPE and may split "zebra" into ["ze", "bra"]. Decoder's per-word `detect_anchor` assumes whole words. **Risk 2 in proposal (BPE subword handling) is NOT yet implemented** — works for tokens that decode to single words, may under-fire for subword-split words.
- **Oracle boost is small relative to logit gaps**. Test `test_oracle_applied_when_prev_is_determiner` needed logit gap = 0.1 for oracle (boost ~0.49) to flip winner. In production with temperature sampling, this matters less because smaller gaps are already ambiguous.
- **`ctx.update(token)` is called with the raw decoded token string**. LLaVA BPE tokens may have leading-space markers (`▁dog`) that don't match `NEGATION_TOKENS`. Need to strip markers when integrating real tokenizer.

### Assumptions Made

- `tokenizer.eos_token_id` is an int (HF-compatible).
- `vlm(input_ids=...).logits` is `[batch, seq, vocab]` (standard HF causal LM).
- `SceneGraph.attributes` may be empty (RelTR doesn't predict them). Oracle handles this; no crashes.
- All tests run on CPU. No GPU-specific code added.

### Known Issues

- BPE subword handling (Risk 2) not implemented — treat as followup when integrating LLaVA.
- Full-vocab oracle scoring (Risk 1 mitigation with precomputed CLIP embeddings) not yet used in decoder — `decode_top_k` covers the top-K path instead. When full-vocab is needed, `apply_oracle_scores` accepts any tensor of IDs.
- Sampling (temperature, top-p) not implemented — greedy only.
- Config file not yet loaded by decoder.

---

## Current State

### What's Working

- **Component 1 (SGG)**: done (prior session). RelTR extraction works on real images.
- **Component 2 (Visual Oracle)**: done (prior session). `should_activate_oracle()` proxy added.
- **Component 3 (Anchor Detector)**: done (prior session). 104 tests pass including all RelTR-vocab spatial preps.
- **Component 4 (GenerationContext)**: done this session. 56 tests.
- **Component 5 (SGODDecoder)**: done this session. 19 tests, mock-based integration.
- **Demo scripts**: `demo_oracle.py`, `demo_anchor.py`, `demo_context.py`, `demo_decoder.py` all run.

### What's Not Working

- No real LLaVA integration tested yet — `SGODDecoder.generate()` hasn't been called with a real transformers VLM. Interface assumed from HF conventions, not verified.
- All experiment scripts (`experiments/pilot/*`, `experiments/main_eval/*`, `experiments/ablation/*`) are empty stubs.
- `scripts/precompute_clip_vocab.py` is a stub — must be filled to cache LLaVA-vocab CLIP embeddings for fast full-vocab scoring.

### Tests

- [x] Unit tests: **249 passed, 5 skipped** (`pytest tests/`)
  - `test_anchor.py`: 104
  - `test_context.py`: 56
  - `test_decoder.py`: 19
  - `test_oracle.py`: 39
  - `test_scene_graph.py`, `test_sgg.py`, etc: remainder
- [ ] Integration tests: 5 marked `@pytest.mark.integration`, require RelTR checkpoint + GPU, not run here
- [x] Manual testing: `demo_context.py` and `demo_decoder.py` both produce expected output

---

## Next Steps

### Immediate (Start Here)

1. **Implement `scripts/precompute_clip_vocab.py`** (1 day, GPU needed). Load real CLIP ViT-L/14, encode entire LLaVA-1.5-7B tokenizer vocab → save to `data/checkpoints/clip/clip_vocab_cache.pt`. This is blocking for any experiment.

2. **Implement pilot experiment** `experiments/pilot/run_pilot.py` + `analyze_pilot.py` (2-3 days). Per proposal Section 6.1:
   - Load LLaVA-1.5-7B + 200 GQA images with GT scene graphs
   - Three variants: (a) LLaVA baseline, (b) LLaVA + GT SG in prompt prefix, (c) LLaVA + predicted SG in prompt prefix
   - Success threshold: (b) > (a) by ≥ 3-5% on GQA relation-type questions
   - **If threshold fails → reconsider entire direction before building more.**

3. **Manually verify `SGODDecoder.generate()` with real LLaVA** on one image before running at scale. HF interface conventions may differ from assumed.

### Subsequent

- Implement main eval scripts in this order: `eval_pope.py` → `eval_reefknot.py` (main claim) → `eval_amber.py` → `eval_mmhal.py` → `eval_vqa.py` → `eval_efficiency.py`
- Implement ablation in `experiments/ablation/run_ablation.py` (6 configs per Section 6.3)
- Handle BPE subword anchor detection (Risk 2) when real LLaVA output shows gaps
- Wire `configs/default.yaml` into `SGODDecoder` constructor
- Add temperature/sampling support to decoder

### Blocked On

- **GPU access for real integration**: pilot + all main evaluations need a GPU with ≥16GB VRAM for LLaVA-1.5-7B.
- **Dataset downloads**: GQA, POPE, Reefknot, AMBER must be downloaded to `data/`. See `scripts/download_data.py` (also stub).

---

## Related Resources

### Documentation

- `docs/sgod_proposal.md` — full architecture spec; Sections 3.5, 3.6, 4.x, 5.x most relevant now
- `docs/handoffs/HANDOFF_COMPONENT2_ORACLE_04_20_16_01.md` — prior handoff (C1–C2 completion)

### Commands to Run

```bash
# Re-run full suite (should pass 249)
.venv/bin/pytest tests/ -q

# Run demos
.venv/bin/python scripts/demo_context.py
.venv/bin/python scripts/demo_decoder.py
.venv/bin/python scripts/demo_decoder.py --image test_imgs/image.png   # requires RelTR ckpt

# Commit current work
git add sgod/ tests/ scripts/demo_*.py docs/handoffs/
git commit -m "implement components 4 and 5"
```

### Search Queries

- `rg "GenerationContext" sgod/` — find all usages
- `rg "SGODDecoder" sgod/ experiments/` — confirm no accidental usages in experiment stubs
- `rg "TODO|FIXME" sgod/decoder/` — find deferred work
- `rg "@pytest.mark.integration" tests/` — locate GPU-required tests

---

## Open Questions

- [ ] Does real LLaVA-1.5-7B's `.logits` tensor shape match assumption `[batch, seq, vocab]`? (Likely yes, but must verify on real run.)
- [ ] Should decoder support batched generation (multiple images at once)? Current impl is single-image per `generate()` call. Deferring unless pilot shows latency issue.
- [ ] Is `clip_vocab_cache.pt` keyed by LLaVA's or CLIP's tokenizer? Proposal implies LLaVA vocab — confirm when implementing `precompute_clip_vocab.py`.
- [ ] For the pilot, should baseline variant use the *same* greedy decoding we wrote, or LLaVA's default sampling? Recommend same greedy for apples-to-apples.

---

## Session Notes

The rule-ordering bug in `detect_anchor` (caught during C3 work) is now covered by test `test_across_from_completion_yields_relation_anchor`. Future edits to `detect_anchor` must preserve: multi-word completion checks (→ relation_anchor) **before** single-token spatial-prep-as-prev checks (→ noun_anchor).

The demo_decoder.py shows "WRONG" winners in most rows — this is a deliberate synthetic setup (base_logit=5.0 < competing_logit=5.5) to make oracle deltas visible. With the fake CLIP scorer returning flat 0.5, boosts are small. Real CLIP + real SG will show stronger effects; use `--image test_imgs/image.png` for that.

All three demo_*.py scripts use the same `_PROJECT_ROOT / sys.path.insert` pattern to handle being launched from anywhere. Keep this if adding new scripts.

---

_This handoff was generated at a checkpoint. Start the next session with: pilot experiment implementation._
