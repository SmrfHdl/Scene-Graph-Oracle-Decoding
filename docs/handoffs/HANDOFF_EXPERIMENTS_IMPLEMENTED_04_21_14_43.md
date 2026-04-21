# Handoff: All Experiment Scripts Implemented — Ready for GPU Runs

**Created:** 2026-04-21 14:43
**Branch:** `feat/dev-apr`
**Session Duration:** ~single session (continues from HANDOFF_COMPONENTS_4_5_04_21_01_50.md)

---

## Summary

Implemented all remaining non-GPU code for the SGOD paper: dataset downloader, model loaders, pilot experiment (3-variant GQA gate), 6 main eval scripts (POPE, Reefknot, AMBER, MMHal-Bench, VQAv2, efficiency), ablation study (6 configs), plus three polish items from the previous handoff — BPE subword handling, config wiring into `SGODDecoder`, and temperature/top-p sampling. **249 tests pass, 5 skipped — no regressions.** Next milestone is running the pilot on GPU to validate the gate threshold (variant B > baseline by ≥3 pp) before committing to full main eval.

---

## Work Completed

### Changes Made

- [x] `scripts/download_data.py` — auto-downloads POPE/GQA/AMBER/MMHal/Reefknot/VQAv2 (JSON/annotations); manual instructions for images
- [x] `sgod/utils/model_loader.py` — `load_llava()`, `load_sgg()`, `load_clip_factory()`, `load_sgod_decoder()`; supports 4-bit (bitsandbytes) + `device_map="auto"`
- [x] BPE subword fix — `_clean_token()` helper in decoder + strip `▁` in `decode_top_k`
- [x] Config wiring — `SGODDecoder.__init__` accepts `base_lambda`, `negation_decay`, `max_negation_depth`, `max_new_tokens`, `temperature`; passed through to `GenerationContext`
- [x] Temperature sampling — `generate()` uses `torch.multinomial` when `temperature > 0`, else `argmax`
- [x] `experiments/pilot/run_pilot.py` — 3 variants × N GQA images; per-record checkpointing; resume support; `--no_reltr` fallback
- [x] `experiments/pilot/analyze_pilot.py` — Wilson CI, paired t-test, per-type breakdown, matplotlib plots (optional)
- [x] `experiments/main_eval/eval_pope.py` — Acc/F1/P/R over 3 splits
- [x] `experiments/main_eval/eval_reefknot.py` — main claim benchmark, per-type breakdown (perceptive/cognitive)
- [x] `experiments/main_eval/eval_amber.py` — accuracy + approximate CHAIR-S/CHAIR-I
- [x] `experiments/main_eval/eval_mmhal.py` — optional GPT-4 scoring; raw predictions always saved
- [x] `experiments/main_eval/eval_vqa.py` — VQAv2 soft accuracy (subsampled 5k default)
- [x] `experiments/main_eval/eval_efficiency.py` — TTFT / tok-per-s / peak VRAM / SGG-isolated timing
- [x] `experiments/ablation/run_ablation.py` — 6 configs: no_clip / fixed_lambda / no_anchor / noun_only / gt_sg / internvl

### Key Decisions

| Decision | Rationale | Alternatives Considered |
|---|---|---|
| **Keep auto-download strictly to annotations** — images require manual download | COCO/GQA/VG images are 6-20 GB each; auto-downloading them would blow up CI and most users already have them locally | Attempt `huggingface_hub.snapshot_download` — decided instructions are cleaner |
| **4-bit quantization as opt-in flag (`--load_in_4bit`)** rather than default | fp16 on 3090 / 2×T4 works fine; 4-bit is only needed for single 16 GB T4. Default fp16 avoids bitsandbytes build surprises | Always use 4-bit — slower & accuracy drop; never — breaks single T4 |
| **Per-record checkpointing in pilot** (save after every image) | 200-image run at ~10s/image = 30+ min; resume-from-crash essential. Cost = 200 tiny JSON writes | Batch-save every N — risk losing partial progress on OOM/crash |
| **Ablations mutate `_score_candidates` via monkeypatch** (no_anchor, noun_only) | Keeps `SGODDecoder` clean — no ablation-specific flags leak into production code | Add `ablation_mode` param to `__init__` — adds dead code in the main path |
| **GPT-4 scoring in MMHal is optional** (`--no_gpt4`) | Eval costs $3-5 per run; users should get raw predictions for free and score selectively | Always GPT-4 — forces API key, blocks offline runs |
| **BPE fix = `_clean_token(s).lstrip("▁").strip()`** applied at 2 call sites | Minimal surface: only `generate()` and `decode_top_k` need it. `.strip()` alone misses U+2581 (sentencepiece marker) | Global tokenizer wrapper — overkill; 2 fixes cover both producer paths |
| **`GenerationContext` stores config as instance attrs**, not module globals | Allows per-decoder config without process-wide state leak; needed for ablations | Keep constants, pass dict — but then ablations would need thread-safe mutation |

---

## Files Affected

### Created

- `experiments/pilot/run_pilot.py` — 3-variant pilot (LLaVA baseline / GT SG prefix / RelTR SG prefix)
- `experiments/pilot/analyze_pilot.py` — statistical analysis + plots
- `experiments/main_eval/eval_pope.py` — POPE evaluation
- `experiments/main_eval/eval_reefknot.py` — Reefknot (main claim)
- `experiments/main_eval/eval_amber.py` — AMBER multi-dim eval
- `experiments/main_eval/eval_mmhal.py` — MMHal-Bench w/ optional GPT-4
- `experiments/main_eval/eval_vqa.py` — VQAv2 regression check
- `experiments/main_eval/eval_efficiency.py` — latency/throughput benchmarks
- `experiments/ablation/run_ablation.py` — 6 ablations on Reefknot val
- `scripts/download_data.py` — dataset downloader
- `docs/handoffs/HANDOFF_EXPERIMENTS_IMPLEMENTED_04_21_14_43.md` — this file

### Modified

- `sgod/decoder/sgod_decoder.py` — `SGODDecoder.__init__` gained 5 config params; `generate()` now supports temperature sampling; `_clean_token()` helper strips `▁`/whitespace before `ctx.update()` and `prev_tokens.append()`
- `sgod/decoder/token_utils.py` — `decode_top_k` now strips `▁` from `batch_decode` output
- `sgod/context/tracker.py` — `GenerationContext.__init__` accepts `base_lambda`, `negation_decay`, `max_negation_depth`; uses instance attrs in `update()` / `get_lambda()`
- `sgod/utils/model_loader.py` — full implementation (was stub)

### Read (Reference)

- `docs/handoffs/HANDOFF_COMPONENTS_4_5_04_21_01_50.md` — prior handoff (gate expectations)
- `configs/default.yaml` — config schema driving `load_sgod_decoder()`
- `sgod/sgg/scene_graph.py` — `SceneGraph`, `ObjectNode`, `RelationEdge` dataclasses (needed for GQA SG conversion)
- `sgod/oracle/visual_oracle.py` — confirm `.score()` / `.should_activate_oracle()` API

---

## Technical Context

### Architecture/Design Notes

- **Pilot flow**: Load GQA val_balanced_questions → filter `types.semantic == "relation"` → unique-image sample → for each image run 3 LLaVA calls (vanilla / +GT SG prefix / +RelTR SG prefix) → GQA-style exact match → save per-record JSON after each item.
- **GQA SG → SceneGraph conversion** lives in `experiments/pilot/run_pilot.py::gqa_sg_to_scene_graph()` because the structure of GQA scene graphs (nested `{oid: {name, x, y, relations[{object, name}]}}`) is pilot-specific; don't promote to `sgod/sgg/` unless reused.
- **Ablation 5 (`gt_sg`)** works by monkeypatching `decoder.sgg.extract = lambda _img: gt_sg` per-image then restoring. This bypasses RelTR entirely while reusing the full decode path.
- **Ablation `no_anchor` / `noun_only`** replace `decoder._score_candidates` with a new static method. Cleaner than adding flags to `SGODDecoder`.

### Dependencies

No new hard dependencies. Optional at runtime:
- `bitsandbytes` — only if `--load_in_4bit` used
- `matplotlib` — only if `analyze_pilot.py --plot`
- `tqdm` — progress bars in downloader (fallback to plain prints)
- `openai` — only if MMHal GPT-4 scoring enabled
- `huggingface_hub` / `gdown` — mentioned in manual download instructions, not imported

### Configuration Changes

None. `configs/default.yaml` is unchanged — `load_sgod_decoder()` reads it as-is. New `SGODDecoder` params all have backward-compatible defaults so existing `demo_decoder.py` and tests keep working.

---

## Things to Know

### Gotchas & Pitfalls

- **GQA images are 20 GB manual download** — pilot variant A + B work without images only if images exist. The script hard-fails if images are missing. The `data/gqa/images/` directory must be populated before running pilot.
- **COCO image filename formats**: POPE uses `COCO_val2014_{id:012d}.jpg`, VQAv2 can be either that or `{id:012d}.jpg` (val2017) — `eval_vqa.py` tries both.
- **LLaVA prompt template**: All eval scripts use `USER: <image>\n{question}\nASSISTANT:`. If LLaVA-1.5-7B's processor expects a different template, answers will be gibberish. Verify on one image first (see "Immediate Next Steps").
- **`load_sgod_decoder()` loads LLaVA + RelTR + CLIP all at once** — that's ~16 GB VRAM. For 16 GB T4, pass `--load_in_4bit`.
- **`eval_mmhal.py` fetches images from URLs in `mmhal_data.json`** on first use and caches to `data/mmhal_bench/images/`. First run is slow due to network I/O, subsequent runs hit the cache.
- **Ablation `gt_sg` depends on GQA image IDs matching Reefknot image names**. If Reefknot uses VG IDs and GQA uses different IDs, the lookup will silently skip all items. Verify before running.

### Assumptions Made

- HuggingFace `LlavaForConditionalGeneration(...).logits` has shape `[batch, seq, vocab]` — standard HF contract, but **not yet empirically verified on real LLaVA** (same unverified assumption as prior handoff).
- GQA `val_balanced_questions.json` entries have a `types.semantic` field — true per GQA 1.2 spec.
- Reefknot `test.json` / `val.json` have keys `image`, `question`, `answer`, `type` — based on HuggingFace dataset description; exact format should be confirmed on first download.
- AMBER query.json entries have `id`, `image`, `query`, `type` — and `answer.json` is a dict keyed by string IDs.

### Known Issues

- `_norm_cdf()` in `analyze_pilot.py` uses a 5-term polynomial approximation — fine for n≥30 but gives rough p-values for small pilots.
- CHAIR-S / CHAIR-I in `eval_amber.py` are **approximate** (token-level set-diff vs GT answer). The paper's CHAIR requires a full object vocabulary + caption parsing — revisit if main eval shows the approximation misleads.
- `eval_efficiency.py::time_sgod` estimates TTFT as `total / n_tokens` because `SGODDecoder.generate()` doesn't expose per-token hooks. True TTFT would require instrumenting the generation loop.

---

## Current State

### What's Working

- All unit tests pass: **249 passed, 5 skipped** (`pytest tests/`)
- `SGODDecoder` now reads full config from YAML via `load_sgod_decoder()`
- BPE markers no longer leak into anchor detection or negation tracking
- Temperature sampling available as `decoder.generate(image, q, temperature=0.7)`
- All experiment scripts have complete, runnable implementations

### What's Not Working

- **No GPU runs yet** — none of the new scripts have been executed against real LLaVA / RelTR / CLIP. All code is written but unverified at runtime.
- Dataset images (COCO, GQA, VG) need manual download — the pilot cannot start until `data/gqa/images/` is populated.

### Tests

- [x] Unit tests: **249 passed, 5 skipped** — no regressions from BPE/config/sampling changes
- [ ] Integration tests: 5 marked `@pytest.mark.integration`, require GPU — not run here
- [ ] Manual testing (GPU): **not yet performed** — first GPU run is the immediate next step

---

## Next Steps

### Immediate (Start Here)

1. **[GPU] Sanity check SGODDecoder with real LLaVA on one image** before the pilot. Verify `.logits` shape, BPE marker handling, and that `generate()` returns coherent text. Use `scripts/demo_decoder.py --image test_imgs/image.png` as the starting point; if it outputs garbage, debug before running pilot.

2. **[GPU] Download datasets + GQA images**:
   ```bash
   python scripts/download_data.py --all
   python scripts/download_data.py --show-manual   # print image download steps
   # Then manually: wget downloads.cs.stanford.edu/nlp/data/gqa/images.zip → unzip to data/gqa/images/
   ```

3. **[GPU] Run pilot experiment (gate check)**:
   ```bash
   python experiments/pilot/run_pilot.py --num_images 200
   python experiments/pilot/analyze_pilot.py --results outputs/pilot/<timestamp>/results.json --plot
   ```
   Gate: variant B accuracy ≥ variant A + 3 pp. **If gate fails → stop and reconsider direction per Section 6.1 of proposal.**

### Subsequent

- If gate passes: run main eval in this order — `eval_pope.py` → `eval_reefknot.py` (main claim) → `eval_amber.py` → `eval_efficiency.py` → `eval_mmhal.py` (needs OPENAI_API_KEY) → `eval_vqa.py` (regression check)
- Run ablation study: `python experiments/ablation/run_ablation.py --ablation all`
- Revisit approximate CHAIR if AMBER numbers look off
- Add true per-token TTFT hook in `SGODDecoder.generate()` for `eval_efficiency.py`

### Blocked On

- **GPU access** — everything from step 1 onward. Recommended hardware from prior discussion: RTX 3090 (24 GB, single-card, no quantization needed). Fallback: 2×T4 with `device_map="auto"`, or 1×T4 with `--load_in_4bit`.
- **GQA image download** (20 GB) — blocks pilot variants A/B/C. Variant C also blocked on RelTR checkpoint at `data/checkpoints/reltr/checkpoint0149.pth`.
- **CLIP vocab cache** — pilot doesn't need it (uses prompt prefix, not oracle); main eval scripts require it via `load_sgod_decoder()` → `load_clip_factory()`. Already precomputable via `scripts/precompute_clip_vocab.py` (implemented in prior session).

---

## Related Resources

### Documentation

- `docs/handoffs/HANDOFF_COMPONENTS_4_5_04_21_01_50.md` — prior handoff (component implementation)
- `docs/sgod_proposal.md` — Sections 6.1 (pilot), 6.2 (main eval), 6.3 (ablation), Risk 2/3
- `configs/default.yaml` — canonical config consumed by `load_sgod_decoder()`

### Commands to Run

```bash
# Re-verify tests
.venv/bin/pytest tests/ -q                              # 249 passed

# Check download status
python scripts/download_data.py --verify

# Pilot gate
python experiments/pilot/run_pilot.py --num_images 200 --load_in_4bit  # for 1×T4
python experiments/pilot/analyze_pilot.py --results outputs/pilot/<ts>/results.json --plot

# Main eval (example: POPE)
python experiments/main_eval/eval_pope.py --config configs/default.yaml

# Ablation
python experiments/ablation/run_ablation.py --ablation all --max_samples 500

# Commit current uncommitted work
git add scripts/ sgod/ experiments/ docs/handoffs/
git commit -m "implement pilot + main eval + ablation scripts"
```

### Search Queries

- `rg "load_sgod_decoder" experiments/` — all entry points that load the full stack
- `rg "_ablation_" sgod/ experiments/` — ablation-specific flags
- `rg "TODO|FIXME" experiments/` — deferred work in scripts
- `rg "_clean_token" sgod/` — BPE fix call sites

---

## Open Questions

- [ ] Does real LLaVA-1.5-7B processor tokenizer's `batch_decode([id])` return `"▁dog"` or `" dog"` for individual IDs? `_clean_token` handles both — but worth confirming which path actually fires.
- [ ] Reefknot dataset schema — the auto-downloader fetches `test.json`/`val.json` from HuggingFace. If schema differs from assumed `{image, question, answer, type}`, `eval_reefknot.py` needs adjustment.
- [ ] For ablation `internvl`: does InternVL2-8B's HF interface match LLaVA (same `model(**inputs).logits` contract)? If not, `SGODDecoder` may need a separate adapter.
- [ ] Should pilot variant B use the oracle path (SGOD full pipeline with GT SG injected) or the prompt-prefix path (current implementation)? Current = prompt prefix — cleaner gate test but doesn't exercise SGOD code. Consider adding a 4th variant if gate passes.

---

## Session Notes

All code was written without GPU execution, so the next session's first job is a smoke test. The `scripts/demo_decoder.py --image test_imgs/image.png` command (from prior handoff) is the fastest way to find integration bugs before the pilot.

Uncommitted state: 20 files modified + 3 new demo scripts + 1 prior handoff untracked. Current changes span decoder core + full experiments layer — recommend a single commit "implement pilot + main eval + ablation scripts" rather than splitting, since the changes are coherent (decoder ergonomics improvements enable the experiment scripts).

`model_loader.py::load_sgod_decoder` is the single entry point for the full stack — every eval script calls it. If the LLaVA interface changes (version bump, processor API changes), that's the one function to update.

---

_This handoff was generated mid-session before GPU runs began. Start next session with the LLaVA sanity check in "Immediate Next Steps" step 1._
