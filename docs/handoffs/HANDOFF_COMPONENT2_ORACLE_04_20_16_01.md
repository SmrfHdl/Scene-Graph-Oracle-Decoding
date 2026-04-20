# Handoff: SGOD Component 2 — Visual Oracle Implementation

**Created:** 2026-04-20 16:01
**Branch:** master (no commits yet — repo is uncommitted)
**Session Duration:** continuation from prior Component 1 session + Component 2 implementation

---

## Summary

Component 2 (Visual Oracle, Section 3.3 of `docs/sgod_proposal.md`) is now implemented end-to-end: `CLIPScorer` wraps open_clip for image/text scoring with optional pre-computed vocab cache, and `VisualOracle` implements dual-source (SGG confidence + CLIP fallback) scoring for noun/relation/attribute anchor types. 34 new unit tests pass (65 total, no regression). Pre-compute script verified working on CPU producing a 200-word × 512-dim cache. Component 3 (Anchor Detector) is the next target.

---

## Work Completed

### Changes Made

- [x] Implemented `sgod/oracle/clip_scorer.py` — `CLIPScorer` class with image encoding, word scoring, vocab cache support
- [x] Implemented `sgod/oracle/visual_oracle.py` — `VisualOracle` class with dual-source scoring (3 anchor types)
- [x] Updated `sgod/oracle/__init__.py` to export `CLIPScorer`, `VisualOracle`
- [x] Wrote `tests/test_oracle.py` — 34 unit tests covering formulas, ranges, edge cases (no CLIP/GPU needed)
- [x] Added `open-clip-torch` to core dependencies in `pyproject.toml`
- [x] Added `pytest-json-report` to dev dependencies + `pytest.ini` addopts → `outputs/test/last_run.json`
- [x] Rewrote `scripts/precompute_clip_vocab.py` — supports `--reltr_only` (fast) and `--vlm_tokenizer` (full LLaVA vocab)
- [x] Updated `README.md` with Component 2 section (scoring table, test cmd, pre-compute cmd, programmatic example)
- [x] Verified pre-compute script end-to-end: produced `data/checkpoints/clip/clip_vocab_cache_reltr.pt` (200 words × 512 dim)

### Key Decisions

| Decision | Rationale | Alternatives Considered |
|----------|-----------|------------------------|
| Use `open_clip` (`open-clip-torch`) | Actively maintained; supports multiple CLIP variants; easy API | OpenAI `clip` (less maintained); `transformers` CLIPModel (heavier) |
| Defer `import open_clip` inside `CLIPScorer.__init__` | Unit tests can mock `CLIPScorer` without loading the real library | Top-level import (would force CLIP load in all tests) |
| `MockCLIPScorer` stub in `test_oracle.py` (not pytest-mock) | Keeps dependency minimal; has only 2 methods to fake | unittest.mock.MagicMock (more fragile for tensor returns) |
| Default CLIP model `ViT-B-32` | 10× faster than ViT-L-14 for dev; fine for fallback signal | ViT-L-14 default (too slow for default path) |
| Vocab cache format: `{embeddings: Tensor, words: list[str]}` | Allows fast dict lookup + matrix multiply at inference | Separate files (more I/O); tokenizer-coupled format (less portable) |
| `pytest-json-report` alongside JUnit XML | User feedback: XML is hard to read. JSON summary is grep-able | HTML report (requires browser); plain text (less structured) |
| Strip whitespace + lowercase in `_score_*` | Robust to BPE leading spaces, casing variation | Strict match (too brittle for real decoder inputs) |

---

## Files Affected

### Created

- `sgod/oracle/clip_scorer.py` — `CLIPScorer` class: encode image once, `score_words()`, `score_single()`, `encode_words()`, optional vocab cache
- `sgod/oracle/visual_oracle.py` — `VisualOracle` class: `score()`, `batch_score()`, private `_score_noun/_score_relation/_score_attribute()`; module-level `_build_conf_map()`
- `tests/test_oracle.py` — 34 unit tests + `MockCLIPScorer` stub
- `docs/handoffs/HANDOFF_COMPONENT2_ORACLE_04_20_16_01.md` — this file
- `data/checkpoints/clip/clip_vocab_cache_reltr.pt` — 200-word cache (200×512 tensor)

### Modified

- `sgod/oracle/__init__.py` — now exports `CLIPScorer, VisualOracle`
- `scripts/precompute_clip_vocab.py` — full implementation (was stub docstring)
- `pyproject.toml` — added `open-clip-torch` to core deps, `pytest-json-report` to dev deps
- `pytest.ini` — `addopts` now also writes `outputs/test/last_run.json`
- `README.md` — new "Component 2 — Visual Oracle" section; updated Checkpoints needed

### Read (Reference)

- `docs/sgod_proposal.md` — Section 3.3 scoring formulas
- `.claude/CLAUDE.md` — project architecture overview
- `sgod/sgg/scene_graph.py` — `SceneGraph`, `ObjectNode`, `RelationEdge`, `AttributeNode` interfaces
- `sgod/sgg/reltr_utils.py` — `ENTITY_CLASSES`, `RELATION_CLASSES` for `--reltr_only` mode

---

## Technical Context

### Architecture/Design Notes

**Dual-source scoring formula (Section 3.3):**

| Anchor type | Token in SG | Token NOT in SG |
|-------------|-------------|-----------------|
| noun | `w_sg·sg_conf + w_clip·clip` | `w_sg·(-0.3) + w_clip·(clip-0.5)` |
| relation | `sg_conf` | `-0.2` |
| attr | `sg_conf` | `clip - 0.5` |

where `w_sg = sg_conf`, `w_clip = 1 - sg_conf`. CLIP scores are normalized to `[0, 1]` via `(cos+1)/2`. All outputs in `[-1, 1]`.

**Cache format in `clip_vocab_cache.pt`:**
```python
{"embeddings": torch.Tensor,  # [V, D], L2-normalized
 "words":      list[str]}     # V entries, index-aligned with embeddings
```
`CLIPScorer` builds `self._vocab_index: dict[str, int]` on load for O(1) lookup.

### Dependencies

Added:
- `open-clip-torch` (core) — pulled in `ftfy`, `timm`, `wcwidth` transitively
- `pytest-json-report` (dev) — pulled in `pytest-metadata`

### Configuration Changes

- `pytest.ini` addopts: `--junitxml=outputs/test/last_run.xml --json-report --json-report-file=outputs/test/last_run.json --json-report-indent=2`

---

## Things to Know

### Gotchas & Pitfalls

- `open_clip` emits a `QuickGELU mismatch` warning when using `ViT-B-32/openai`. It's cosmetic (different activation used than in original OpenAI training) and doesn't break scoring. Don't "fix" it.
- `torch.load(..., weights_only=True)` is required by modern PyTorch for cache loading; file only contains tensors + strings which is safe.
- `MockCLIPScorer._scores` dict defaults unknown words to `0.3`. Tests relying on known behavior must explicitly set entries.
- CLIP tokenizer silently truncates long strings — not an issue for single words but worth knowing if someone scores phrases later.
- `sg_conf=0` + in-vocab fallback edge case: `_score_noun` returns `0*0 + 1*clip_score`. Could be any value in `[0, 1]`. Technically SG said "yes but zero confidence" — current behavior trusts CLIP, which is correct.

### Assumptions Made

- Scene graph labels are English single words (matches RelTR's VG vocabulary).
- All CLIP/SGG inputs are lowercased before comparison. Original casing only retained for display.
- `AttributeNode` list in `SceneGraph` will eventually be populated by external means (CLIP zero-shot over attribute vocab) — the oracle already handles the case of non-empty attribute lists.
- VLM decoder will pass already-decoded string tokens to `VisualOracle.score()`, not raw token IDs. Token-ID handling is Component 5's job.

### Known Issues

- `precompute_clip_vocab.py --vlm_tokenizer` path was NOT executed in this session (requires ~500MB tokenizer download). Code path is straightforward but should be smoke-tested before first real usage.
- `CLIPScorer` loads the entire CLIP model even if only `encode_words` is needed for pre-computation — minor inefficiency, not worth fixing.

---

## Current State

### What's Working

- Component 1 (SGG Module): 31 unit tests + 5 integration tests (skipped without `-m integration`) — all pass. Real RelTR inference verified in prior session (`outputs/eval/2026-04-20_15-21-14_sgg/results.json`).
- Component 2 (Visual Oracle): 34 unit tests pass. Pre-compute script verified on CPU.
- Output infrastructure: `outputs/{test,eval,infer,train}/` with timestamped subdirs. JUnit XML + JSON reports both written.

### What's Not Working

- None. No known blockers.

### Tests

- [x] Unit tests: 65 passed, 5 skipped (integration) — `pytest tests/`
- [x] Integration tests: 5 skipped (all for SGG, require RelTR checkpoint; Component 2 has no integration tests yet — could add one that exercises real `open_clip` on a synthetic PIL image)
- [x] Manual testing: `precompute_clip_vocab.py --reltr_only` verified; `CLIPScorer` vocab cache load verified with 3 sample words

---

## Next Steps

### Immediate (Start Here)

1. **Implement Component 3 (Anchor Detector) — `sgod/anchor/`** per Section 3.4 of proposal:
   - `sgod/anchor/vocabularies.py` — `DETERMINERS`, `SPATIAL_PREPS`, `COLOR_ATTRS`, `SIZE_ATTRS` sets (all in proposal)
   - `sgod/anchor/detector.py` — `detect_anchor(prev_tokens: list[str], current_token: str) -> str` returning one of `noun_anchor | relation_anchor | attr_anchor | neutral`
   - `sgod/anchor/__init__.py` — export `detect_anchor`
   - `tests/test_anchor.py` — currently a stub; should test all 4 patterns + edge cases (empty prev_tokens, multi-word preps note)
   - Pure stateless functions; zero runtime dependencies beyond Python stdlib

2. **Consider adding an optional Component 2 integration test** that loads real `open_clip` ViT-B-32 with a synthetic image to verify `CLIPScorer.score_words()` returns values in `[0, 1]` for arbitrary words. Mark with `@pytest.mark.integration`.

3. **Smoke-test the full LLaVA path in `precompute_clip_vocab.py`** once LLaVA model is downloaded for Component 5 work.

### Subsequent

- Component 4 (GenerationContext) — `sgod/context/tracker.py` (Section 3.5). Rule-based negation tracking + question-type detection. Straightforward.
- Component 5 (SGODDecoder) — `sgod/decoder/sgod_decoder.py` (Section 3.6). Wires everything together. Needs LLaVA loaded. Biggest component.
- Pilot experiment (Section 6.1): 200 GQA images, compare baseline vs GT-SG-in-prompt vs predicted-SG-in-prompt.

### Blocked On

- Nothing currently.

---

## Related Resources

### Documentation

- `docs/sgod_proposal.md` Section 3.3 — Visual Oracle spec (just implemented)
- `docs/sgod_proposal.md` Section 3.4 — Anchor Detector spec (next)
- `docs/sgod_proposal.md` Section 5, Risk 1 — Why vocab pre-compute matters (640s → 1ms)
- `README.md` — Updated usage for Components 1 & 2

### Commands to Run

```bash
# Verify environment before resuming
pytest tests/                                        # should show 65 passed, 5 skipped

# Component 2 verification
pytest tests/test_oracle.py -v

# Re-run vocab pre-compute if needed
python scripts/precompute_clip_vocab.py --reltr_only \
    --output data/checkpoints/clip/clip_vocab_cache_reltr.pt

# Quick oracle smoke-check with real CLIP
python -c "
from PIL import Image
from sgod.sgg import SGGModule
from sgod.oracle import CLIPScorer, VisualOracle

sgg = SGGModule('data/checkpoints/reltr/checkpoint0149.pth')
img = Image.open('path/to/any.jpg')
sg = sgg.extract(img)
clip = CLIPScorer(img, vocab_cache_path='data/checkpoints/clip/clip_vocab_cache_reltr.pt')
oracle = VisualOracle(sg, clip)
print('dog  ->', oracle.score('dog', 'noun_anchor'))
print('on   ->', oracle.score('on', 'relation_anchor'))
print('black->', oracle.score('black', 'attr_anchor'))
"
```

### Search Queries

- `grep -rn "detect_anchor" sgod/ tests/` — finds anchor-related references (currently none outside test stub)
- `grep -rn "VisualOracle\|CLIPScorer" sgod/ tests/` — finds Component 2 integration points
- Proposal Section 3.4 for anchor detection rules

---

## Open Questions

- [ ] Should `detect_anchor` take the decoder's *BPE token ID* or the *decoded string*? Proposal shows string form, but real integration in Component 5 may need both. Recommend string for now; add BPE→string mapping utility in `sgod/decoder/token_utils.py` when Component 5 lands.
- [ ] Do we want an `AttributeExtractor` sub-component that fills `SceneGraph.attributes` using CLIP zero-shot over `COLOR_ATTRS ∪ SIZE_ATTRS`? Proposal implies yes (Component 2 "populates attributes") but the current `VisualOracle` handles the case of empty attribute lists gracefully without it. Defer until needed.
- [ ] Should vocab cache entries match exactly (current behavior) or also support `strip()/lower()` normalization on cache keys? Currently only exact match triggers fast path. Probably fine since the scorer already lowercases/strips before lookup — just need to ensure pre-compute also lowercases.

---

## Session Notes

- The prior session completed Component 1 (SGG Module). This session's Component 2 was implemented cleanly against the already-working `SceneGraph` dataclass.
- `CLIPScorer` API surface was intentionally kept minimal (3 public methods) to make mocking in tests trivial.
- Vocab cache choice: `ViT-B-32` gives 512-dim embeddings; `ViT-L-14` would be 768-dim. Format is model-agnostic as long as `CLIPScorer.model_name` matches at inference.
- The proposal's `_score_relation` uses `-0.2` penalty while `_score_attribute` uses CLIP-based fallback. Kept asymmetric as specified — rationale in proposal is that relation vocab (51 classes) is closed-world while attribute vocab is open-world.
- Git repo has no commits yet (`fatal: your current branch 'master' does not have any commits yet`). Everything is untracked. When the user is ready, initial commit should probably cover all of `sgod/`, `tests/`, `scripts/`, `configs/`, `docs/`, `README.md`, `pyproject.toml`, `pytest.ini`, `.gitignore`.

---

_This handoff was generated at context window capacity. Start a new session and use this document as your initial context._
