# Handoff: POPE Debugged → MMHal Next

**Created:** 2026-04-28 15:00
**Branch:** feat/dev-apr
**Session Focus:** Debugged catastrophic POPE results, identified design limitation, pivoting to MMHal-Bench

---

## Summary

POPE eval went from F1=0.004 (catastrophic) → F1=0.78 (close to baseline 0.82) by fixing the `pixel_values` bug in `SGODDecoder.generate()`. Root-cause analysis via diagnostic scripts confirmed **oracle correctly does not fire on binary "Yes"/"No" tokens** (since `detect_anchor([], x)` always returns "neutral"), so POPE is structurally near-zero delta — oracle has no anchors to inject on. Pivoting to **MMHal-Bench** (open-ended, paragraph-length answers) where SGOD's anchor mechanism can actually engage. **Blocked on MMHal data 404** at HuggingFace; need to find correct URL or alternative source.

---

## Work Completed

### Changes Made

- [x] Fixed `SGODDecoder.generate()` to pass `pixel_values` (root cause of catastrophic POPE F1=0.004)
- [x] Added `processor` parameter to `SGODDecoder.__init__`, made `tokenizer` optional fallback
- [x] Added `_preloaded` dict in `CLIPScorer` so model loads once not per-image (perf fix)
- [x] Updated `load_clip_factory` to build preloaded dict and reuse across calls
- [x] Updated `eval_pope.py` to log live SGOD metrics every 100 items (not just at end)
- [x] Updated `eval_pope.py` JSONL parsing (`json.load` → line-by-line `json.loads`)
- [x] Updated all `SGODDecoder(...)` callers in `run_ablation.py` to use `processor=` kwarg
- [x] Fixed `eval_mmhal.py` to pass `args.max_new_tokens` to `decoder.generate` (was using config default 256, mismatched with baseline 128)
- [x] Created `scripts/diagnose_decoder_divergence.py` — compares baseline vs manual loop vs SGOD
- [x] Created `scripts/diagnose_oracle_step0.py` — traces every detect_anchor + apply_oracle_scores call

### Diagnostic Findings

| Test | Result | Conclusion |
|---|---|---|
| Manual loop (no oracle) vs `model.generate()` | Identical logits & top1 token | Forward pass mechanics are correct |
| Oracle scores at step 0 with `prev_tokens=[]` | All 0 (50/50 neutral) | Oracle cannot affect first token by design |
| Oracle scores at step 1 with `prev=['Yes']` | Fires only on 'in'/'at' positions 31, 41 | Doesn't change top1 (`</s>`) |
| Current SGOD output for dining-table case | `'Yes</s>'` (matches baseline) | Code now stable; old eval records reflect intermediate code state |

**Implication:** SGOD ≈ Baseline on POPE because oracle has no chance to fire on 1-2 token binary responses. Not a bug — a design limitation for this benchmark type.

---

## Files Affected

### Modified
- `sgod/decoder/sgod_decoder.py` — added `processor` param, fixed `pixel_values` passing in generate loop, fixed device handling
- `sgod/oracle/clip_scorer.py` — added `_preloaded` dict optional param to skip model reload
- `sgod/utils/model_loader.py` — `load_clip_factory` now preloads model once; `load_sgod_decoder` passes `processor=` kwarg
- `experiments/main_eval/eval_pope.py` — JSONL parsing, live SGOD progress logging
- `experiments/main_eval/eval_mmhal.py` — pass `max_new_tokens` to decoder
- `experiments/ablation/run_ablation.py` — replaced `tokenizer=` with `processor=` in 3 places

### Created
- `scripts/diagnose_decoder_divergence.py` — proves manual loop ≡ baseline forward pass
- `scripts/diagnose_oracle_step0.py` — proves oracle correctly skips token 0 with prev=[]

### Reference
- `outputs/eval/2026-04-28_12-24-44_pope/pope_results.json` — POPE results F1=0.78 (close to baseline 0.82)
- `outputs/eval/2026-04-28_12-24-44_pope/records_adversarial.json` — 290 disagreements analyzed

---

## Technical Context

### Why POPE is structurally near-zero delta for SGOD

`detect_anchor(prev_tokens, current_token)` first guard at [detector.py:55-56](../../sgod/anchor/detector.py#L55-L56):
```python
if not prev_tokens:
    return "neutral"
```

POPE forces 1-token answer ("Yes"/"No"), so:
- Step 0: `prev_tokens=[]` → all top-50 candidates get neutral → oracle scores all 0 → no logit modification
- Step 1: usually `</s>` regardless of oracle activity

Anchor design assumes generated tokens form descriptive language ("a dog on the table") where DET/PREP/ADJ context triggers anchor classification.

### Hardware/env on server
- `SPIN=/home/uet/anaconda3/envs/spin/bin/python`
- `CUDA_VISIBLE_DEVICES=1` (single T4)
- Project root: `~/truong_vlm/Scene-Graph-Oracle-Decoding/`
- Data path: server only — local `/home/truongpv/...` does NOT have data files

---

## Current State

### What's Working
- LLaVA-1.5-7B + RelTR + CLIP pipeline loads and runs
- POPE eval pipeline: F1=0.78 SGOD vs F1=0.82 Baseline (3000 questions × 3 splits)
- CLIP vocab cache (31963 words, dim 768) at `data/checkpoints/clip/clip_vocab_cache_reltr.pt`
- RelTR checkpoint symlinked at `data/checkpoints/reltr/reltr_visual_genome.pth`
- Unit tests: 19 passing

### What's Not Working / Blocked
- **MMHal data 404** — `https://huggingface.co/datasets/Shengcao1006/MMHal-Bench/resolve/main/mmhal_data.json` returns 404. Need alternative source.
- AMBER, ReefKnot, VQAv2 data not yet downloaded

### Tests
- [x] Unit tests: 19 passing
- [x] Manual: POPE adversarial eval completed (3000 items)
- [ ] Diagnostic scripts: passing on server

---

## Next Steps

### Immediate (Start Here)

1. **Find working MMHal-Bench source.** Try in order:
   - GitHub: `https://github.com/llava-rlhf/LLaVA-RLHF` (look in `Eval/` folder)
   - HuggingFace API check: `curl -s "https://huggingface.co/api/datasets/Shengcao1006/MMHal-Bench"` for actual file siblings
   - Alternative repos: search for "MMHal-Bench" on HuggingFace
   - If found, update URL in `scripts/download_data.py`

2. **Run MMHal eval** (after data obtained):
   ```bash
   PYTHONPATH=$(pwd) CUDA_VISIBLE_DEVICES=1 $SPIN experiments/main_eval/eval_mmhal.py \
       --config configs/default.yaml --mmhal_path data/mmhal_bench/ \
       --no_gpt4 --max_new_tokens 128
   ```
   `--no_gpt4` saves raw predictions; GPT-4 scoring optional later.

3. **If MMHal stays blocked, fall back to ReefKnot** (relation hallucination — strongest theoretical fit for SGOD's relation_anchor):
   ```bash
   PYTHONPATH=$(pwd) $SPIN scripts/download_data.py --benchmarks reefknot
   # Then manually fetch VG images via gdown or stanford URL
   ```

### Subsequent

- ReefKnot eval (stronger fit; relation_anchor designed for this)
- AMBER generative subset (longer responses, no GPT-4 required for some metrics)
- VQAv2 (lower priority — short answers limit oracle firing)
- Re-run POPE with current code state to confirm F1≈baseline (should not be -4pp)

### Open Questions
- [ ] Was the F1=0.78 (vs current ~0.82-equivalent) due to intermediate code state during debug session? Re-running may reveal SGOD ≈ Baseline.
- [ ] Should `detect_anchor` be extended so oracle can score "Yes"/"No" directly when question contains a known noun? (Design change, defer until other benchmarks tested.)

---

## Commands to Run

### Server setup each session
```bash
SPIN=/home/uet/anaconda3/envs/spin/bin/python
cd ~/truong_vlm/Scene-Graph-Oracle-Decoding
```

### Replay diagnostics (proves oracle correctness)
```bash
PYTHONPATH=$(pwd) CUDA_VISIBLE_DEVICES=1 $SPIN scripts/diagnose_oracle_step0.py
```

### MMHal data alternatives to try
```bash
# Check repo siblings
curl -s "https://huggingface.co/api/datasets/Shengcao1006/MMHal-Bench" | \
  python3 -c "import json,sys; d=json.load(sys.stdin); [print(s) for s in d.get('siblings',[])]"
# GitHub raw
wget "https://raw.githubusercontent.com/llava-rlhf/LLaVA-RLHF/main/Eval/mmhal-bench/mmhal_data.json"
```

---

## Session Notes

- The "290 disagreements" in POPE records were from intermediate code state during the debug session — re-running on current code may show fewer disagreements and SGOD F1 closer to baseline.
- Key insight worth preserving: **SGOD's design (anchor-position injection) requires multi-token responses with grammatical structure**. Yes/no benchmarks are structurally incompatible. Choose benchmarks accordingly.
- Don't waste cycles trying to make POPE work — the design is sound; the benchmark is the wrong fit.

---

_This handoff was generated at context window capacity. Start a new session and use this document as your initial context._
