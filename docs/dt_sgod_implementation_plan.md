# DT-SGOD: Complete Implementation Plan

**Status:** Plan — pre-implementation
**Date:** 2026-05-18
**Companion docs:** [dt_sgod_proposal.md](dt_sgod_proposal.md) (architecture), [sgod_proposal.md](sgod_proposal.md) (v1 baseline)
**Decision frame:** Phase 1 paper on LLaVA → Phase 2 paper writing → Phase 3 framework refactor → Phase 4 ship

---

## 0. Executive summary

This plan converts the DT-SGOD architectural proposal into 18 weeks of concrete engineering work, organized into 4 phases with explicit go/no-go gates. Phase 1 (weeks 1-4) is paper-critical: build, train, and validate DT-SGOD on LLaVA-1.5-7B. Phase 2 (weeks 5-8) is paper writing + submission. Phase 3 (weeks 9-14) refactors the code into a 3-layer pluggable framework (Oracle / Policy / Backbone) and adds Qwen 2.5-VL + InternVL 2.5 backbones. Phase 4 (weeks 15-18) packages, hubs adapters, and ships. Every phase has explicit exit criteria; failure at any gate triggers a documented pivot path, not "keep trying."

**Core engineering principle:** code with framework abstractions from day 1 (Oracle / Policy / Backbone as ABCs), even when Phase 1 only uses one concrete instance of each. This costs ~1 day up front and saves ~2 weeks of refactoring in Phase 3.

---

## 1. Target code architecture (final state)

```
sgod/
├── core/                       # NEW — shared abstractions & types
│   ├── interfaces.py           # ABCs: Backbone, Oracle, Policy, GenerationState
│   ├── types.py                # SceneGraph (moved from sgg/), TokenContext, OracleEvidence
│   └── registry.py             # decorator-based plugin registry
│
├── backbones/                  # NEW — VLM wrappers (Phase 3 expands)
│   ├── base.py                 # Backbone ABC
│   ├── llava15.py              # LLaVA-1.5 wrapper (Phase 1)
│   ├── qwen25_vl.py            # Qwen 2.5-VL wrapper (Phase 3)
│   └── internvl25.py           # InternVL 2.5 wrapper (Phase 3)
│
├── oracles/                    # RENAMED from oracle/ — pluggable oracle layer
│   ├── base.py                 # Oracle ABC (image → SceneGraph)
│   ├── reltr.py                # RelTR-based oracle (existing logic)
│   ├── grounding_dino.py       # Grounding DINO (already in repo, refactor)
│   ├── owlv2.py                # OWLv2 (Phase 3, optional)
│   └── dual_vision.py          # multi-oracle ensemble (Phase 3, optional)
│
├── policies/                   # NEW — pluggable decoding policy
│   ├── base.py                 # Policy ABC (Backbone, Oracle, state → logit Δ)
│   ├── sgod_v1.py              # SGOD v1 (refactored from decoder/sgod_decoder.py)
│   └── dt_sgod/                # NEW — DT-SGOD policy
│       ├── policy.py           # DTSGODPolicy (orchestrator)
│       ├── grounding_planner.py # Slow module (GNN + GRU + slots)
│       ├── speaker_adapter.py   # Fast module (cross-attn + LoRA + gate)
│       ├── anchor_gate.py       # ATG (learned MLP)
│       └── config.py            # DT-SGOD hyperparameters
│
├── sgg/                        # KEEP — RelTR vendor code stays here
│   └── reltr/                  # vendored, untouched
│
├── anchor/                     # KEEP — rule-based (used as ATG warmup signal)
├── context/                    # KEEP — tracker (used as ATG input feature)
│
├── training/                   # NEW
│   ├── stage0_distill.py       # SGOD v1 → DT-SGOD distillation
│   ├── stage1_finetune.py      # Hallucination-aware fine-tune
│   ├── losses.py               # L_SFT, L_DPO, L_anchor_rate, L_disentangle
│   ├── data/
│   │   ├── pope_style.py       # POVID-style hallucinated negative generation
│   │   ├── sg_corruption.py    # scene-graph-aware corruption (swap object/attr)
│   │   └── distill_collector.py # collect SGOD v1 traces for Stage 0
│   └── callbacks.py            # checkpointing, eval, logging
│
├── eval/                       # NEW — benchmark harness
│   ├── runner.py               # one-command: `sgod eval --policy dt-sgod --backbone llava15 --bench all`
│   ├── benchmarks/
│   │   ├── pope.py             # refactored from experiments/main_eval/eval_pope.py
│   │   ├── amber.py
│   │   ├── mmhal.py
│   │   ├── reefknot.py
│   │   ├── chair.py            # NEW — for long-form drift
│   │   └── hallusion.py        # NEW
│   └── metrics.py
│
├── decoder/                    # KEEP for back-compat, will deprecate after Phase 3
│   └── sgod_decoder.py         # remains as v1 entrypoint
│
└── utils/                      # KEEP — visualization, model_loader, output

scripts/
├── train_dt_sgod.py            # CLI for Stage 0 + Stage 1
├── eval_dt_sgod.py             # CLI for benchmarks
├── precompute_clip_vocab.py    # KEEP
└── distill_sgod_v1.py          # generate Stage 0 distillation data

configs/
├── default.yaml                # KEEP — SGOD v1
├── dt_sgod_llava15.yaml        # NEW — DT-SGOD on LLaVA
└── dt_sgod_qwen25vl.yaml       # Phase 3

tests/
├── ... (existing v1 tests stay)
├── test_dt_grounding_planner.py
├── test_dt_speaker_adapter.py
├── test_dt_anchor_gate.py
├── test_dt_decoder_smoketest.py
├── test_backbones_interface.py
└── test_policy_interface.py
```

**Backward compatibility:** SGOD v1 entrypoint (`sgod.decoder.SGODDecoder`) remains importable through Phase 3. Old tests keep passing. SGOD v1 is reused as ablation baseline and as Stage 0 teacher.

---

## 2. Core abstractions (Phase 1 Day 1)

These are the ABCs that lock in framework-readiness. Define once, never change signatures during Phase 1.

### 2.1 `sgod/core/interfaces.py`

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import torch
from torch import Tensor

# ---- Types ----
@dataclass
class GenerationState:
    """Runtime state passed through every decode step."""
    image: Tensor                          # [3, H, W] preprocessed
    scene_graph: "SceneGraph"              # frozen evidence
    prompt_ids: Tensor                     # [B, L]
    generated_ids: Tensor                  # [B, t]
    hidden_states: Tensor                  # [B, d] last hidden
    step: int
    # Policy-managed slots:
    policy_state: dict                     # opaque dict for Policy use
    # Aggregated context features (for ATG and similar):
    lm_entropy: Tensor                     # [B] scalar entropy of LM logits
    rule_anchor_type: Optional[str]        # "noun" | "rel" | "attr" | None

# ---- Backbone ----
class Backbone(ABC):
    """Wraps a VLM. Provides hidden states + logits per step."""
    @property
    @abstractmethod
    def hidden_dim(self) -> int: ...

    @property
    @abstractmethod
    def vocab_size(self) -> int: ...

    @property
    @abstractmethod
    def lora_target_modules(self) -> list[str]: ...

    @abstractmethod
    def prepare_inputs(self, image, prompt: str) -> dict: ...

    @abstractmethod
    def forward_step(self, inputs: dict, generated_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Returns (hidden_state [B, d], logits [B, V])."""
        ...

    @abstractmethod
    def tokenizer(self): ...

# ---- Oracle ----
class Oracle(ABC):
    """Image → structured belief (scene graph)."""
    @abstractmethod
    def extract(self, image, image_meta: dict) -> "SceneGraph": ...

    @abstractmethod
    def vocab_for(self, tokenizer) -> dict[int, float]:
        """Default per-token scores (for SGOD v1 fallback)."""
        ...

# ---- Policy ----
class Policy(ABC):
    """Maps (Backbone state, Oracle evidence) → logit adjustment."""
    @abstractmethod
    def init_state(self, scene_graph: "SceneGraph") -> dict: ...

    @abstractmethod
    def adjust_logits(self, state: GenerationState) -> Tensor:
        """Returns Δ [B, V] to be added to lm_logits."""
        ...

    @abstractmethod
    def update_state(self, state: GenerationState, sampled_token: int) -> dict: ...
```

**Why these signatures:** every existing SGOD v1 step maps cleanly (decoder = Backbone.forward_step + Policy.adjust_logits). Every DT-SGOD step also maps (Policy.adjust_logits internally runs GP/ATG/Fast Adapter). Every future policy fits.

### 2.2 `sgod/core/registry.py`

```python
_REGISTRY = {"backbone": {}, "oracle": {}, "policy": {}}

def register(kind: str, name: str):
    def deco(cls):
        _REGISTRY[kind][name] = cls
        return cls
    return deco

def build(kind: str, name: str, **kwargs):
    return _REGISTRY[kind][name](**kwargs)
```

Use as `@register("backbone", "llava-1.5-7b")` and `build("backbone", "llava-1.5-7b", ...)`. Lets configs reference components by string.

---

## 3. Phase 1: Core DT-SGOD on LLaVA (Weeks 1-4)

**Goal:** working DT-SGOD on LLaVA-1.5-7B with at-least-equal-to-SGOD-v1 quality and demonstrable wins on Reefknot/AMBER-rel.

### Week 1 — Foundation + skeleton (no training)

**Tasks:**
1. Set up `sgod/core/` abstractions (Section 2). Add unit tests for ABC compliance.
2. Refactor existing components to implement interfaces:
   - Wrap `RelTRModule` → `RelTROracle(Oracle)`.
   - Wrap `GroundingDinoModule` → `GroundingDinoOracle(Oracle)`.
   - Wrap existing `SGODDecoder` → `SGODv1Policy(Policy)` (paper-baseline preserved).
   - Wrap LLaVA-1.5 loading → `LLaVAv15Backbone(Backbone)`.
3. Implement `sgod/policies/dt_sgod/`:
   - `grounding_planner.py`: GAT graph encoder + slot recurrent core + GRU.
   - `speaker_adapter.py`: cross-attention + zero-init gate γ + LoRA hooks.
   - `anchor_gate.py`: 2-layer MLP, straight-through estimator stub.
   - `policy.py`: orchestrates GP + Fast Adapter + ATG, implements `Policy` interface.
4. Smoke test: with γ=0 (closed gate), `DTSGODPolicy.adjust_logits` returns zeros, generation matches LLaVA-base exactly.

**Deliverable:** `tests/test_dt_decoder_smoketest.py` passes — DT-SGOD@init equals LLaVA-base output token-for-token on 10 held-out prompts.

**Exit criterion:** smoke test green. Forward pass shapes verified. Trainable param count ≈ 50M as planned.

**Risks W1:**
- Hidden-state hooking on LLaVA may be fragile (HF API version-sensitive). Mitigation: pin transformers version in `pyproject.toml`, write integration test against pinned version.
- LoRA target detection across LLaVA versions. Mitigation: hard-code module names for LLaVA-1.5 in Backbone, validate with `print(model)`.

### Week 2 — Stage 0 distillation infrastructure

**Tasks:**
1. `scripts/distill_sgod_v1.py`: run SGOD v1 on LLaVA-Instruct-150K subset (100K samples), record `(image, prompt, generated_token_t, lm_logit_t, sgod_v1_logit_t)` to a parquet dataset.
2. `sgod/training/stage0_distill.py`: load distillation traces, train DT-SGOD to match SGOD v1 logits via KL divergence. Only GP + Fast Adapter + ATG trainable; LoRA frozen at zero.
3. `sgod/training/data/distill_collector.py`: streaming dataset loader (don't materialize 100K full logit tensors — store top-k logits or KL-targets).
4. Implement training loop with WandB logging, gradient checkpointing, fp16.

**Deliverable:** Stage 0 training completes on a 10K-sample subset in <6 hours on 1× 4090. KL(DT-SGOD || SGOD v1) < 0.1 on held-out.

**Exit criterion:** DT-SGOD post-Stage-0 matches SGOD v1 within 1pt on POPE (sanity check that distillation faithful).

**Risks W2:**
- Storage cost of distillation traces. 100K samples × avg 30 tokens × top-100 logits ≈ 24 GB. Mitigation: top-50 logits + bf16, ~12 GB; or stream-on-the-fly (slower, no storage).
- Slow data loading. Mitigation: cache scene graphs to disk (image → SG is deterministic).

### Week 3 — Stage 1 fine-tune

**Tasks:**
1. `sgod/training/data/sg_corruption.py`: given (image, caption, scene graph), generate hallucinated negative by:
   - **Object swap**: replace a noun in caption with a class NOT in SG, weighted by COCO frequency.
   - **Attribute swap**: replace an attribute with opposite (red ↔ blue, big ↔ small).
   - **Relation swap**: replace a verb/preposition (on ↔ next to ↔ under).
   - Filter: only keep negatives where SG verifies the swap is wrong.
2. `sgod/training/losses.py`: implement `L_SFT`, `L_DPO`, `L_anchor_rate`, `L_slot_disentangle`.
3. `sgod/training/stage1_finetune.py`: full training loop with all 4 losses.
4. Train DT-SGOD on 200K samples × 2 epochs (~60-80 hours wall on 1× 4090).

**Deliverable:** trained DT-SGOD checkpoint. Loss curves saved. Per-epoch eval on small POPE + AMBER subset shows monotonic improvement.

**Exit criterion:** DT-SGOD AMBER-attribute ≥ SGOD v1 + 1pt **OR** Reefknot ≥ SGOD v1 + 2pt. If neither, **PAUSE** and debug before Week 4 (don't burn benchmark eval on a broken model).

**Risks W3:**
- DPO instability (large gradient spikes). Mitigation: β=0.1 conservative, gradient clipping 1.0, save checkpoints every 1K steps.
- ATG firing rate collapses to 0 or 1. Mitigation: μ=0.5 Lagrangian, monitor `fire_prob` distribution per-step in WandB.
- Slot collapse. Mitigation: orthogonality loss, slot init from K-means on SG node embeddings (diverse init).

### Week 4 — Full ablation + benchmark sweep + decision

**Tasks:**
1. Train 7 ablation variants (Section 5 of proposal). Stage 1 only (3-5 days each on 1× 4090 → run in parallel on 2 GPUs if available, else sequence over remaining time).
2. Run full benchmark suite on all variants: POPE-adv, AMBER, MMHal, Reefknot, CHAIR-long.
3. Build the 4 results tables (main comparison, ablation, per-type, cross-method).
4. **Go/no-go decision** based on go criterion (Section 7 of proposal): full DT-SGOD must beat SGOD v1 by ≥ 1.5pt on ≥ 2 benchmarks AND each major component (slow, slots, ATG, GRU) must contribute ≥ 1pt in ablation.

**Deliverable:** results tables + go/no-go decision recorded in `docs/handoffs/2026-week4-decision.md`.

**Exit criterion (pass):** go criterion met → proceed to Phase 2.
**Exit criterion (fail):** go criterion not met → execute pivot path (Section 9).

**Risks W4:**
- Ablation runs OOM. Mitigation: train ablations with lower r=8 LoRA, accept small accuracy loss for budget.
- Benchmark eval bugs distort results. Mitigation: smoke-test each benchmark on LLaVA-base first; compare to published baseline numbers.

---

## 4. Phase 2: Paper writing (Weeks 5-8)

**Goal:** Q2 journal submission.

| Week | Task | Output |
|---|---|---|
| 5 | Draft Introduction, Related Work, Method (Sections 1-3). Use proposal as starting text. | Draft sections 1-3 |
| 6 | Run additional analyses: per-question-type breakdown, latency benchmark, qualitative examples, OOD eval. Write Experiments section. | Section 4 + figures |
| 7 | Write Theoretical scaffold (Section 3 from proposal), Limitations, Conclusion. Internal review pass 1. | Full draft v1 |
| 8 | Address internal review. Polish figures. Final proofread. Submit. | arXiv preprint + submission |

**Cross-cutting deliverables:**
- All experiment configs versioned and reproducible.
- All figures generated by `scripts/paper/make_fig_<N>.py` for reproducibility.
- Anonymous code release prepared (anonymized fork).

---

## 5. Phase 3: Framework refactor + multi-backbone (Weeks 9-14)

**Goal:** make code framework-grade and support ≥ 3 VLM backbones.

### Week 9-10 — Refactor pass

**Tasks:**
1. Move `sgod/oracle/` → `sgod/oracles/` (singular instances under plural module).
2. Move `sgod/decoder/sgod_decoder.py` logic → `sgod/policies/sgod_v1.py` (preserve back-compat shim).
3. Generalize `sgod/backbones/llava15.py`: extract LLaVA-specific assumptions (hidden_dim=4096, LoRA targets, tokenizer alignment) into the wrapper.
4. Add config-driven instantiation: `HallucinationDecoder.from_config(yaml_path)`.
5. End-to-end test on LLaVA still works (regression-free).

**Deliverable:** refactored codebase passes all existing tests; new `tests/test_policy_backbone_compose.py` verifies arbitrary (Policy, Backbone, Oracle) triples compose without error.

### Week 11-12 — Qwen 2.5-VL backbone + adapter training

**Tasks:**
1. Implement `sgod/backbones/qwen25_vl.py`. Handle dynamic-resolution visual encoder. hidden_dim=3584.
2. Rebuild oracle vocab cache for Qwen tokenizer.
3. Re-run Stage 0 + Stage 1 training on Qwen 2.5-VL-7B-Instruct. Same data, same hyperparameters.
4. Evaluate trained Qwen adapter on full benchmark suite.

**Deliverable:** `sgod-hub/dt-sgod-qwen25vl-7b` checkpoint + results table row.

**Exit criterion:** Qwen + DT-SGOD beats Qwen baseline by ≥ 2pt on ≥ 1 hard benchmark (Reefknot or AMBER-rel). If not, debug interface — likely a hidden-state extraction bug.

### Week 13-14 — InternVL 2.5 backbone + cross-backbone analysis

**Tasks:**
1. Implement `sgod/backbones/internvl25.py`. hidden_dim varies by variant; start with 8B.
2. Train adapter. Evaluate.
3. Build the cross-backbone table (Section 7 of proposal).
4. Write up Phase 3 results as a paper extension or workshop submission.

**Deliverable:** 3 backbones × full benchmark suite. Cross-backbone results table.

**Exit criterion:** DT-SGOD improves ≥ 2 of 3 backbones by ≥ 2pt on at least one hard benchmark.

---

## 6. Phase 4: Adapter hub + lib + ship (Weeks 15-18)

**Goal:** pip-installable, documented, demoable.

| Week | Tasks |
|---|---|
| 15 | Upload trained adapters to HuggingFace Hub under `sgod-hub/` org. Implement `DTSGODPolicy.from_pretrained(hub_id)`. Test download → use round-trip. |
| 16 | Write CLI: `sgod eval`, `sgod generate`, `sgod train`. Pip-installable via `pyproject.toml`. End-to-end smoke test from `pip install sgod` to first benchmark result. |
| 17 | Docs: Sphinx/MkDocs site with quick-start, API reference, examples (LLaVA + Qwen). Demo notebook. |
| 18 | Announce: blog post, X/Twitter thread, paper revision with framework section, optional Hugging Face Space demo. |

**Deliverable:** `pip install sgod` works for a fresh user; quickstart produces first hallucination-reduced caption in < 5 min on 1× 3090.

---

## 7. Cross-cutting concerns

### 7.1 Testing strategy

| Layer | Test type | When run |
|---|---|---|
| Unit (interfaces, math) | pytest, no GPU | every commit |
| Smoke (forward shape, gate=0 = baseline) | pytest, CPU or 1 GPU | every commit |
| Integration (Backbone + Oracle + Policy) | pytest, 1 GPU, mocked checkpoints | per PR |
| Training regression (1 epoch on toy data) | pytest, 1 GPU, 30 min | nightly |
| Benchmark eval (small subset) | scripts, 1 GPU, 2 hours | weekly |

Existing `tests/` markers (`integration`) extend naturally — add `dt_sgod` and `training` markers.

### 7.2 Reproducibility checklist (per training run)

- Config hash logged
- Seed fixed (and `torch.use_deterministic_algorithms`)
- Data manifest version pinned
- Tokenizer + model commit SHA recorded
- Final metric + checkpoint linked to WandB run
- 3 seeds for paper-table runs

### 7.3 Compute orchestration

- Single 4090 (24 GB): default. fp16, gradient checkpointing, batch 4, accum 8.
- Two 4090s: DDP for Stage 1 — halves wall time for paper-critical runs.
- For ablation runs: small jobs queue via simple `nohup` + log file (no need for SLURM).
- Storage: ~200 GB for distillation traces + checkpoints + benchmark caches. External SSD if local disk is tight.

### 7.4 Configuration discipline

All hyperparameters in YAML configs (`configs/dt_sgod_<backbone>.yaml`). No magic numbers in code. Every config gets a corresponding `tests/test_config_<name>.py` that just instantiates the full pipeline from config (asserts no missing keys).

### 7.5 Logging & monitoring

WandB project per phase. Mandatory logs:
- LM logit entropy distribution per epoch
- ATG firing rate (target 0.2 ± 0.05)
- Slot pairwise cosine similarity (should decrease)
- Loss components individually (don't aggregate)
- Validation: POPE-adv F1, AMBER-attr, Reefknot — every 5K steps

---

## 8. Decision gates (explicit go/no-go)

| Gate | When | Pass criterion | Pivot on fail |
|---|---|---|---|
| **G1** | End W1 | Smoke test passes (DT-SGOD@γ=0 == LLaVA-base) | Block on interface bug; cannot proceed |
| **G2** | End W2 | Stage 0 KL < 0.1 on held-out | Reduce data complexity, check trace pipeline |
| **G3** | End W3 | DT-SGOD ≥ SGOD v1 + 1pt on either AMBER-attr OR Reefknot | Pause training; debug ATG firing + slot collapse |
| **G4** | End W4 | Full DT-SGOD beats SGOD v1 by ≥ 1.5pt on ≥ 2 benchmarks AND each major component ≥ 1pt in ablation | Section 9 pivot path |
| **G5** | End W8 | Paper submitted | n/a |
| **G6** | End W12 | Qwen DT-SGOD beats Qwen baseline by ≥ 2pt on ≥ 1 hard benchmark | Debug backbone wrapper |
| **G7** | End W14 | ≥ 2 of 3 backbones improved | Consider whether method is truly backbone-agnostic; may indicate fundamental LLaVA-specific assumption |
| **G8** | End W18 | `pip install sgod` works, docs complete | Polish round |

---

## 9. Pivot paths (if a gate fails)

### If G3/G4 fails (DT-SGOD doesn't beat SGOD v1)

**Likely causes** (in order of probability):
1. Slot collapse / firing-rate collapse — fixable with regularization tuning.
2. Stage 0 distillation locks model into v1 behavior — try Stage 1 with higher LR (5e-4) and shorter Stage 0.
3. Training data quality — synthetic hallucination too easy; try real DPO data (RLHF-V dataset).
4. **Fundamental architecture issue.**

**Pivot order:**
- 1 week: tuning sweep (3-5 LR × 2 reg strengths × 2 data versions).
- If still failing: switch to **Selective-SG (Candidate 2)** from the proposal — Mamba-shaped input-dependent gating. Simpler, less risky.
- If still failing: take Co-DiSG (Candidate 3) seriously, accept compute risk.
- Worst case (4 weeks burned): salvage as "negative result with thorough ablation" — publishable as workshop paper, learn from it.

### If G6/G7 fails (cross-backbone doesn't generalize)

**Likely causes:**
1. LLaVA-specific assumption baked into GP (e.g., hidden-state semantics differ in Qwen).
2. Tokenizer mismatch: oracle vocab alignment differs across BPE schemes.
3. Visual feature distribution differs; ATG entropy signal calibrated to LLaVA.

**Pivot:** add per-backbone calibration step (small extra LR-warmup on entropy normalization). If method only works on LLaVA-family, downgrade framework ambition to "LLaVA-family framework" (still publishable, smaller scope).

---

## 10. Resource estimate

### 10.1 Compute

| Phase | GPU-hours (1× 4090 equivalent) |
|---|---|
| Phase 1 (W1-4): Stage 0 + Stage 1 + 7 ablations + eval | ~500h (~3 weeks wall on 1 GPU; 1.5 weeks on 2 GPUs) |
| Phase 2 (W5-8): paper analyses + figure gen | ~50h |
| Phase 3 (W9-14): refactor + 2 more backbones × Stage 0+1 | ~400h |
| Phase 4 (W15-18): docs + demos | ~20h |
| **Total** | **~970 GPU-hours** |

With 2× 4090 available, wall-clock is comfortable for the 18-week plan. With 1× 4090, ~25% slip risk.

### 10.2 Storage

| Item | Size |
|---|---|
| Distillation traces (per backbone) | ~15 GB |
| Stage 1 training data (POVID-style) | ~10 GB |
| Checkpoints (8 ablation variants × 3 backbones) | ~60 GB |
| Benchmark eval outputs | ~5 GB |
| **Total** | **~90-100 GB** |

### 10.3 Engineering effort (solo)

| Phase | Engineer-weeks |
|---|---|
| Phase 1 | 4 |
| Phase 2 | 4 (mostly writing) |
| Phase 3 | 6 |
| Phase 4 | 4 |
| **Total** | **18 weeks (~4.5 months)** |

---

## 11. Dependencies on existing repo state

**Reuse as-is:**
- `sgod/sgg/reltr/` — vendored RelTR, do not modify.
- `sgod/sgg/grounding_dino_module.py` — already integrated; wrap as `GroundingDinoOracle`.
- `sgod/anchor/detector.py` — used as ATG warmup signal.
- `sgod/context/tracker.py` — feeds ATG as side feature.
- `experiments/main_eval/*` — refactor into `sgod/eval/benchmarks/`, preserve metric logic.
- `tests/` — existing tests stay green throughout.

**New deps (`pyproject.toml`):**
- `peft >= 0.10` (LoRA)
- `torch_geometric` (GAT — alternative: hand-roll GAT layer if want to avoid the dep)
- `trl` (DPO loss implementation — optional, can hand-roll)
- `wandb` (already optional in repo)

**Decisions to confirm before starting:**
1. **GAT library choice**: torch_geometric (full) vs. hand-rolled (~100 LOC). Recommend hand-rolled to avoid heavy dep.
2. **DPO library**: trl (battle-tested) vs. hand-rolled. Recommend trl for correctness.
3. **Test environment**: pytest + GPU markers (existing) vs. add CI? For solo dev, GPU markers enough; CI defers to Phase 4 release.

---

## 12. First-week checklist (what to do Monday)

Concrete, do-this-first list for Week 1 Day 1:

1. `git checkout -b feat/dt-sgod`
2. Create `sgod/core/{__init__,interfaces,types,registry}.py` with stubs from Section 2.
3. Write `tests/test_interfaces.py` verifying ABC compliance.
4. Move `sgg/scene_graph.py::SceneGraph` import path to `sgod.core.types` (add re-export from old location for back-compat).
5. Write 1-line wrapper: `sgod/backbones/llava15.py::LLaVAv15Backbone` that implements `Backbone` over the existing `model_loader.load_llava()` function. Verify forward_step returns correct shapes.
6. Smoke test: `python -c "from sgod.core import build; b = build('backbone', 'llava-1.5-7b'); print(b.hidden_dim)"`.
7. Stub `sgod/policies/dt_sgod/{policy,grounding_planner,speaker_adapter,anchor_gate}.py` with class skeletons + `NotImplementedError`.
8. Add `tests/test_dt_skeleton.py`: instantiate `DTSGODPolicy` with frozen γ=0, verify `adjust_logits` returns zeros.

**End of Day 1 deliverable:** PR or commit `feat(dt): core interfaces + LLaVA backbone wrapper + DT-SGOD skeleton`.

---

## 13. Open design decisions (resolve before Week 1)

| Question | Default if no resolution |
|---|---|
| Use torch_geometric GAT or hand-roll? | Hand-roll (no heavy dep) |
| LoRA on which layers of LLaVA? | Last 8 self-attention layers, both Q and V |
| Slot init strategy? | Random Gaussian (simplest); K-means on SG nodes if collapse observed |
| ATG fires on token-step granularity, or chunk granularity? | Token-step; chunking is a future optimization |
| Oracle for Phase 1: RelTR or Grounding DINO? | RelTR (matches v1, simpler comparison); switch to Grounding DINO in Phase 3 |
| Stage 0 trace storage: parquet of logits, or generate-on-the-fly? | Generate-on-the-fly (slower training, no storage), unless 1 epoch is too slow |
| WandB project layout? | `sgod-dt-phase1`, `sgod-dt-phase3-multibackbone` etc. |

---

## Appendix A — Glossary

- **GP** — Grounding Planner (slow module)
- **ATG** — Anchor-Trigger Gate (learned anchor detector)
- **Fast Adapter** — Speaker Adapter (per-token cross-attention)
- **Stage 0** — distillation warm-start from SGOD v1
- **Stage 1** — hallucination-aware fine-tune with synthetic negatives
- **Backbone / Oracle / Policy** — the three pluggable framework layers
- **G1...G8** — go/no-go decision gates
