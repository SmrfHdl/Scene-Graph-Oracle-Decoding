# Handoff: Pilot Experiment Passed — Ready for Main Eval

**Created:** 2026-04-27 00:10  
**Branch:** `feat/dev-apr`  
**Session Duration:** ~3 hours (CUDA debugging + pilot run)

---

## Summary

The 200-image pilot experiment completed successfully: Variant B (GT scene graph oracle) outperforms Variant A (baseline LLaVA) by **+31.5 pp** (A=15.5%, B=47.0%), far exceeding the +3 pp gate threshold. All code changes have been committed. README has been updated with a Quick Start section (not yet committed). The next task is running the main evaluation benchmarks.

---

## Work Completed

### Changes Made (committed in last 2 commits)

- [x] `sgod/anchor/detector.py` — Added Rule 1b: noun detection after "det + adj" pattern (e.g. "a large [table]")
- [x] `tests/test_anchor.py` — 4 new tests for Rule 1b; total 253 passed, 5 skipped
- [x] `experiments/pilot/run_pilot.py` — Fixed GQA filter (`"rel"` not `"relation"`), added `--smoke_test_image`, `--device_map`, `--max_gpu_memory` args
- [x] `sgod/utils/model_loader.py` — Added `max_memory: dict | None` param to `load_llava()`; max_memory keys must be **integers** (e.g. `{0: "40GiB"}`) not strings
- [x] `pyproject.toml` — Added `"accelerate"` to core dependencies
- [x] `scripts/download_data.py` — Fixed misleading comment about image requirements
- [x] `scripts/download_gqa_images.py` — New script: downloads only the specific pilot images from VG CDN

### Uncommitted

- [x] `README.md` — Added Quick Start section (5-step guide + validated pilot result). **Needs commit.**

### Key Decisions

| Decision | Rationale | Alternatives Considered |
|---|---|---|
| Use `spin` conda env instead of uv venv | PyTorch in uv venv (py3.13) has NCCL symbol mismatch with server driver | Fixing LD_LIBRARY_PATH (failed), reinstalling cu121 (same error) |
| `max_memory = {0: "40GiB", "cpu": "32GiB"}` | Integer keys required by accelerate; avoids `cudaMemGetInfo` crash | `device_map="cuda:0"` (RuntimeError), `device_map="auto"` without max_memory (CPU offload) |
| `--no_reltr` for pilot | No RelTR checkpoint available on server; GT SG oracle (variant B) is sufficient for gate check | Waiting for checkpoint |
| GQA images in `data/gqa/images/`, copied to `/dev/shm/gqa/images/` | Script hardcodes `gqa_path/images`; symlink approach had a depth bug | Modifying script to accept `--image_dir` |

---

## Files Affected

### Created
- `scripts/download_gqa_images.py` — Downloads ~200 specific pilot images by ID from Visual Genome CDN (tries VG_100K then VG_100K_2 URLs)

### Modified
- `sgod/anchor/detector.py` — Rule 1b at line ~110; also added `_ATTR_WORDS = COLOR_ATTRS | SIZE_ATTRS` module-level constant
- `tests/test_anchor.py` — 4 tests appended at end of file
- `experiments/pilot/run_pilot.py` — `run_pilot()` function: GQA filter line ~75, image loading ~257, model loading ~225–233
- `sgod/utils/model_loader.py` — `load_llava()` signature and kwargs block ~44–74
- `pyproject.toml` — dependencies list
- `scripts/download_data.py` — comment in `gqa` entry's `manual` field
- `README.md` — Added Quick Start section after line 6 (before Setup)

---

## Technical Context

### Server Environment

- **Server:** shared Ubuntu, 2× RTX 4090 (GPU 0 often occupied by others)
- **GPU to use:** `CUDA_VISIBLE_DEVICES=1` (GPU 1, ~42GB free)
- **Working Python:** `spin` conda env — `torch==2.10.0+cu128`, CUDA works
- **uv venv broken for CUDA:** `libtorch_cuda.so` has undefined symbol `ncclCommWindowDeregister` — do NOT use `uv run` for GPU workloads
- **Disk:** ~915GB total, often near-full. `/dev/shm` = 32GB RAM tmpfs (safe for temp data)
- **GQA JSONs:** at `/dev/shm/gqa/` (loaded into RAM for speed)
- **GQA images:** at `/home/uet/truong_vlm/Scene-Graph-Oracle-Decoding/data/gqa/images/` (200 images)

### Run Command Template

```bash
SPIN=/home/uet/anaconda3/envs/spin/bin/python

PYTHONPATH=/home/uet/truong_vlm/Scene-Graph-Oracle-Decoding \
CUDA_VISIBLE_DEVICES=1 $SPIN experiments/pilot/run_pilot.py \
    --gqa_path /dev/shm/gqa/ \
    --num_images 200 --no_reltr \
    --max_gpu_memory 40GiB
```

### Pilot Results (Validated)

```
N questions:    200
Accuracy A (baseline):   15.5%
Accuracy B (GT SG):      47.0%
Delta:                   +31.5 pp
Gate threshold:          +3 pp
Gate PASSED:             YES ✓
```

Output saved to: `outputs/pilot/2026-04-27_00-01-08/` (on the server, not in this repo)

---

## Things to Know

### Gotchas & Pitfalls

- `ln -sf src dst` where `dst` is an existing directory puts the symlink **inside** dst, not replacing it. Use `rm -rf dst && ln -sf src dst` to replace.
- `accelerate` `max_memory` dict requires **integer keys** for GPU devices: `{0: "40GiB"}` not `{"cuda:0": "40GiB"}`.
- GQA semantic type is `"rel"` not `"relation"` — this was a bug that caused 0 relation questions to be selected.
- `uv run` may reset environment variables including `LD_LIBRARY_PATH`.
- `/dev/shm` data is lost on reboot — copy GQA JSONs back if server restarts: `cp data/gqa/val_balanced_questions.json data/gqa/val_sceneGraphs.json /dev/shm/gqa/`

### Known Issues

- uv venv cannot run GPU workloads on this server (NCCL symbol mismatch). Use `spin` conda env for all GPU scripts.
- `transformers` emits `torch_dtype is deprecated, use dtype instead` — harmless warning, low priority to fix.

---

## Current State

### What's Working

- Full unit test suite: **253 passed, 5 skipped** (no GPU required)
- Pilot experiment: runs end-to-end on GPU with `spin` conda env
- Variant A (baseline LLaVA) and Variant B (GT SG oracle): both functional
- Gate criterion: **PASSED** (Δ=+31.5pp with n=200)
- Data download scripts: functional

### What's Not Working / Not Yet Done

- Variant C (RelTR-predicted SG): skipped — no RelTR checkpoint on server
- Main eval benchmarks: not yet run (POPE, Reefknot, AMBER, MMHal, VQA)
- uv venv CUDA: broken (use spin conda env as workaround)

---

## Next Steps

### Immediate (Start Here)

1. **Commit README update:**
   ```bash
   git add README.md
   git commit -m "Add Quick Start guide and validated pilot results to README"
   ```

2. **Download benchmark data** (if not already present):
   ```bash
   SPIN=/home/uet/anaconda3/envs/spin/bin/python
   PYTHONPATH=/home/uet/truong_vlm/Scene-Graph-Oracle-Decoding
   $SPIN scripts/download_data.py --benchmarks pope,reefknot,amber
   ```

3. **Run POPE eval** (first benchmark, fastest):
   ```bash
   PYTHONPATH=/home/uet/truong_vlm/Scene-Graph-Oracle-Decoding \
   CUDA_VISIBLE_DEVICES=1 $SPIN experiments/main_eval/eval_pope.py
   ```

4. **Run Reefknot eval** (main claim benchmark):
   ```bash
   PYTHONPATH=/home/uet/truong_vlm/Scene-Graph-Oracle-Decoding \
   CUDA_VISIBLE_DEVICES=1 $SPIN experiments/main_eval/eval_reefknot.py
   ```

### Subsequent

- Run remaining evals: `eval_amber.py`, `eval_mmhal.py`, `eval_vqa.py`
- Obtain RelTR checkpoint to enable Variant C and full SGOD pipeline
- Pre-compute CLIP vocab cache: `scripts/precompute_clip_vocab.py`
- Fix `torch_dtype` deprecation warning in `model_loader.py`

### Blocked On

- **RelTR checkpoint** (`data/checkpoints/reltr/checkpoint0149.pth`) — needed for Variant C and full SGOD decoder. Source: train via `scripts/train_reltr.py` or download from original RelTR repo.
- **COCO val2014 images** (~6 GB) — needed for POPE eval
- **Reefknot images** — sourced from Visual Genome, see `scripts/download_data.py --show-manual`

---

## Related Resources

### Commands to Run

```bash
# Set up environment for GPU work (every new session)
SPIN=/home/uet/anaconda3/envs/spin/bin/python
PYTHONPATH=/home/uet/truong_vlm/Scene-Graph-Oracle-Decoding

# Verify GPU works
CUDA_VISIBLE_DEVICES=1 $SPIN -c "import torch; print(torch.cuda.is_available())"

# Re-copy GQA JSONs to RAM if lost after reboot
mkdir -p /dev/shm/gqa
cp data/gqa/val_balanced_questions.json data/gqa/val_sceneGraphs.json /dev/shm/gqa/
cp -r data/gqa/images /dev/shm/gqa/images  # optional, images can stay on disk

# Run unit tests
uv run pytest tests/

# Smoke test (5 questions, no checkpoint needed)
CUDA_VISIBLE_DEVICES=1 $SPIN experiments/pilot/run_pilot.py \
    --gqa_path /dev/shm/gqa/ \
    --smoke_test_image test_imgs/image.png \
    --num_images 5 --no_reltr --max_gpu_memory 40GiB

# Full pilot (200 questions)
CUDA_VISIBLE_DEVICES=1 $SPIN experiments/pilot/run_pilot.py \
    --gqa_path /dev/shm/gqa/ \
    --num_images 200 --no_reltr --max_gpu_memory 40GiB
```

### Search Queries

- `grep -n "caching_allocator_warmup"` — find where CUDA init crash occurs in transformers
- `grep -rn "device_map"` in `experiments/` — find all places that pass device config
- `grep -n "img_dir"` in `experiments/pilot/run_pilot.py` — line 257, where image path is constructed

---

## Open Questions

- [ ] Which main eval benchmarks need COCO images vs which are self-contained?
- [ ] Is there a pre-trained RelTR checkpoint available to download (not train from scratch)?
- [ ] Should Variant C (RelTR SG) be validated before running full main eval, or run A/B only?

---

## Session Notes

The majority of this session was spent debugging CUDA compatibility on the shared server:
- uv venv uses Python 3.13 + latest PyTorch which has NCCL ABI mismatch with server driver
- Solution: use `spin` conda env (torch 2.10.0+cu128) which works correctly
- The pilot itself ran fast (~5 min for 200 questions on RTX 4090)
- Delta of +31.5pp is very strong — well above the +3pp gate, good signal for the paper

---

_This handoff was generated at context window capacity. Start a new session and use this document as your initial context._
