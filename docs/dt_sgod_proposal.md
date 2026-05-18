# DT-SGOD: Dual-Timescale Scene-Graph Decoder
### A Breakthrough Architecture Proposal for Vision-Language Hallucination Reduction

**Status:** Proposal — design phase
**Date:** 2026-05-18
**Predecessor:** [sgod_proposal.md](sgod_proposal.md) (SGOD v1, training-free baseline)
**Target venue:** Q2+ journal in computer vision / NLP

---

## Abstract

Existing decoding-time methods for VLM hallucination reduction (OPERA, VCD, HALC, M3ID, and SGOD itself) operate as *logit arithmetic between two distributions*, varying only the negative distribution. They share three implicit assumptions: (i) the grounding oracle is a static per-token scalar, (ii) the scene graph is extracted once and consumed passively, and (iii) decoding is single-pass, single-timescale. **DT-SGOD breaks assumptions (ii) and (iii)** by introducing a *dual-timescale dual-module recurrent decoder* — a slow "Grounding Planner" with slot-based recurrent state over scene-graph evidence, and a fast "Speaker Adapter" cross-attending to that state at every token. Slow-fast coupling is controlled by a learned Anchor-Trigger Gate, replacing the rule-based anchor detection in SGOD v1. The architecture is backbone-agnostic, fits a single consumer GPU via LoRA + gradient checkpointing, and reduces to SGOD v1 as a corner case (S=1 slot, identity GRU, rule-based gate). Variational interpretation gives a principled objective. Target benchmarks are the *open* axes of hallucination — relational (Reefknot, AMBER-relation), compositional binding (AMBER-attribute), long-form drift (CHAIR-long), comparison/spatial (MMHal-Bench) — rather than the saturated POPE family.

---

## 1. Motivation

### 1.1 The "lego problem" with SGOD v1

SGOD v1 is engineered from off-the-shelf components glued by rule-based logic: RelTR for scene graph, CLIP for fallback embedding similarity, regex-style anchor patterns, hand-tuned λ schedule per question type, linear addition `logit_final = logit_lm + λ · oracle_scores`. Every individual component is borrowed; the contribution is the *integration recipe*. For a Q2+ journal, reviewers typically demand a new *computational primitive* — a novel way of computing, not a novel arrangement of existing pieces.

### 1.2 The breakthrough recipe (from verified comparators)

We surveyed four recent breakthrough architecture papers, fetching abstracts directly from arxiv:

- **Mamba** ([arxiv 2312.00752](https://arxiv.org/abs/2312.00752), Gu & Dao). Broken assumption: "SSM parameters must be input-independent." New primitive: input-dependent SSM parameters → content-based selectivity.
- **Mixture-of-Depths** ([arxiv 2404.02258](https://arxiv.org/abs/2404.02258), Raposo et al.). Broken assumption: "Every token gets equal compute at every layer." New primitive: top-k token routing per layer with predictable budget.
- **LLaDA** ([arxiv 2502.09992](https://arxiv.org/abs/2502.09992), Nie et al.). Broken assumption: "Language modeling must be autoregressive." New primitive: discrete diffusion via mask-then-denoise; solves reversal curse.
- **Hierarchical Reasoning Model** ([arxiv 2506.21734](https://arxiv.org/abs/2506.21734), Wang et al., Sapient Inc.). Broken assumption: "Single-module, single-timescale processing is sufficient for reasoning." New primitive: dual interdependent recurrent modules at different timescales, brain-inspired; 27M params, 1000 training samples beats much larger models on ARC.

The pattern is consistent: **(1)** identify an implicit assumption the field accepts, **(2)** invert it with a novel primitive, **(3)** demonstrate a capability gap (something nothing else can do, not just "+2%"), **(4)** wrap in a principled motivation (biological, mathematical, or information-theoretic).

### 1.3 Implicit assumptions in VLM hallucination decoding

Across the decoding-time family (OPERA, VCD, HALC, M3ID, SID, AGLA, RITUAL, VTI, PAI, AvisC, HIO, and SGOD v1), we identify five shared assumptions:

- **A1** Oracle is a static scalar score, additively combined with LM logits.
- **A2** Scene graph (or visual evidence) is extracted once at step 0 and does not evolve.
- **A3** Anchor / token-class detection is rule-based, stateless.
- **A4** Decoding is single-pass forward, single-timescale.
- **A5** Mixing is linear (`logit + λ · oracle`).

DT-SGOD **explicitly breaks A3 and A4** (and partially A5). Slow module breaks A4 (introduces a second timescale); slot-based recurrent state breaks A3 (replaces rule anchor with learned gate over evolving state); cross-attention adapter breaks A5 (mixing becomes a non-linear state-conditioned function).

---

## 2. Background

### 2.1 SGOD v1 recap

SGOD v1 has five components ([sgod_proposal.md](sgod_proposal.md)):

1. **SGG Module** — RelTR extracts scene graph `G = (V_obj ∪ V_attr, E_rel)` once per image.
2. **Visual Oracle** — combines SGG confidence with CLIP soft similarity into `score_map: token → ℝ ∈ [-1, 1]`.
3. **Anchor Detector** — rule-based grammar patterns classify the current token as `noun_anchor | relation_anchor | attr_anchor | neutral`.
4. **Generation Context** — tracks negation depth and question type, returns mixing weight `λ`.
5. **SGOD Decoder** — at anchor positions: `logit_final = logit_lm + λ · score_map[token]`.

Frozen: LLaVA-1.5-7B + RelTR. Trainable: nothing. Mixing: linear, per-token scalar.

### 2.2 What changes in DT-SGOD

| Aspect | SGOD v1 | DT-SGOD |
|---|---|---|
| Oracle representation | scalar score per token | recurrent state `h_slow ∈ ℝ^{S×d}` (S=8 slots) |
| Scene graph use | static lookup | GNN-encoded, slots soft-attend over nodes, state evolves |
| Anchor detection | rule-based grammar | learned MLP gate (ATG) over `[h_t, h_slow_pooled, entropy_lm]` |
| Mixing | linear add at anchor token | non-linear cross-attention adapter, gated, runs every token |
| Trainable params | 0 | ~50M (LoRA + GP + adapter + ATG) |
| Timescales | one (per-token) | two (slow per-K_t tokens, fast per-token) |
| Theoretical frame | heuristic | variational inference with latent grounding state |

SGOD v1 is recovered as a corner case: set S=1, ATG=rule, GRU=identity, cross-attention=scalar projection, gate γ=λ. This makes SGOD an *ablation* of DT-SGOD, not a baseline competitor.

---

## 3. Architecture

### 3.1 High-level data flow

```
                        ┌─────────────────────────────────┐
   Image ──RelTR──►  G  │   SLOW MODULE — Grounding       │
                        │   Planner (GP)                  │◄──┐
                        │   ┌──────────────────────┐     │   │
                        │   │ Graph encoder (GNN) │     │   │ updates every
                        │   │ Prefix summarizer    │     │   │ K_t tokens
                        │   │ Recurrent core       │──►h│   │ (adaptive)
                        │   └──────────────────────┘  slow│  │
                        └─────────────────────────────────┘  │
                                          │                  │
                                          ▼ broadcast        │
                        ┌─────────────────────────────────┐  │
   LLaVA hidden h_t  ──►│  FAST MODULE — Speaker Adapter  │  │
                        │  ┌────────────────────────┐    │  │
                        │  │ Cross-attn (h_t ↔ h_slow│    │  │
                        │  │ + LoRA on output head   │    │  │
                        │  └────────────────────────┘    │  │
                        └─────────────────────────────────┘  │
                                          │                  │
                                          ▼                  │
                                  logit adjustment Δ         │
                                          │                  │
                                          ▼                  │
                                  logit_final = h_lm + Δ     │
                                          │                  │
                                          ▼                  │
                                       sample y_t ───────────┘
                                          │
                          ┌───────────────┴──────────────┐
                          │ Anchor-Trigger Gate (ATG)    │
                          │ — decides if slow re-fires   │
                          └──────────────────────────────┘
```

### 3.2 Slow Module — Grounding Planner (GP)

**State:** `h_slow ∈ ℝ^{S × d}` where S=8 slots, d=4096 (matches LLaVA hidden dim; rescale for other backbones).

**Why 8 slots, not 1?** Compositional hallucination is a *binding* failure: "red car next to small dog" fails when attribute–object pairs collide in a single vector. Slots let DT-SGOD hold multiple bindings simultaneously, analogous to HRM's hierarchical state.

**Fire schedule:** Every `K_t` decoded tokens, where `K_t` is determined by ATG (Section 3.4). Target mean firing rate ≈ 0.2 (one fire per ~5 tokens). Bounded by Lagrangian regularizer during training.

**Per-fire computation (three sub-modules):**

#### 3.2.1 Graph encoder

```
g_v = GAT_2L(V_obj ∪ V_attr, E_rel)   →   ℝ^d  per node
```

2-layer Graph Attention Network. Node features: pretrained embedding of object class (151 entities for RelTR's VG vocabulary) concatenated with bbox-normalized position. Edge features: relation class embedding (51 relations). Computed **once per image** (cached for the entire generation).

#### 3.2.2 Prefix summarizer

```
p_t = MeanPool(LLaVA_hidden[t-K_t : t])   →   ℝ^d
```

Pools the last `K_t` LLaVA hidden states. No additional transformer — LLaVA has already encoded the prefix; pooling is sufficient context for the planner. Cheap (O(K_t · d)).

#### 3.2.3 Recurrent core (the key novelty)

For each slot `s ∈ {1, ..., S}`:

```
α_{s,v} = softmax_v( q(h_slow_s) · k(g_v) / √d )   # slot soft-attends over SG nodes
c_s     = Σ_v α_{s,v} · v(g_v)                     # context vector for slot s
h_slow_s ← GRU(input=[c_s, p_t], state=h_slow_s)   # GRU update
```

`q, k, v` are linear projections (shared across slots). The GRU update is deliberate (not transformer block): GRU's forget/input gates encourage gradual state evolution across many fire events, which a transformer block would tend to overwrite. This is an empirical choice to be ablated.

**Trainable parameters in GP:** GAT (~5M) + projections (~3M) + GRU (~15M) + node/edge embeddings (~7M) ≈ 30M.

### 3.3 Fast Module — Speaker Adapter

Runs *every* token. Input: LLaVA last hidden `h_t ∈ ℝ^d`, current slow state `h_slow ∈ ℝ^{S × d}`. Output: logit adjustment `Δ ∈ ℝ^|V|` added to LM logits.

```
attn_out = CrossAttn(query=h_t, key=h_slow, value=h_slow)    # ℝ^d
Δ        = W_out · tanh(γ) · attn_out                          # ℝ^|V|
logit_final = logit_lm + Δ
```

**Three design choices:**

- **Zero-initialized scalar gate `γ`** (Flamingo-style, [2204.14198](https://arxiv.org/abs/2204.14198)). At initialization, `tanh(0) = 0`, so `Δ = 0` and DT-SGOD is identical to LLaVA-base. The gate opens during training.
- **LoRA rank 16** on the cross-attention Q/K/V matrices and on the output projection `W_out`. Trainable size in fast adapter ≈ 20M.
- **`W_out` initialized as the LLaVA LM head, multiplied by 0.** This biases `Δ` to live in the LM head's column space (no concept-leakage from random init).

### 3.4 Anchor-Trigger Gate (ATG)

Replaces SGOD v1's rule-based `detect_anchor()`. A 2-layer MLP:

```
ATG_input  = [h_t,  Pool(h_slow),  entropy(logit_lm)]   ∈ ℝ^{d + d + 1}
fire_prob  = σ(MLP_2L(ATG_input))   ∈ [0, 1]
```

**Firing decision:**
- Train-time: `fire = (fire_prob > 0.5)` with straight-through estimator for gradient.
- Inference: same threshold, or top-K_t scheduling.

**Why learned, not rule:** Rule-based anchors capture grammatical position (nouns, attributes, relations) but miss two patterns the LLM itself signals — (a) high-entropy decisions (the LLM is unsure → need grounding), (b) topical transitions (the LLM is switching subject → re-ground from SG). ATG can learn both, while subsuming the grammar rules via the warmup loss below.

**Training regularizers:**
- **Rate Lagrangian:** add `μ · (E[fire_prob] − 0.2)²` to keep firing rate ~5 tokens/fire. Without this, ATG collapses to fire-always (≈ single-timescale) or fire-never (≈ LLaVA-base).
- **Warmup supervision:** for the first 1000 training steps, add `λ_warm · BCE(fire_prob, rule_anchor_label)` so ATG learns the SGOD v1 anchor rule as a starting point, then drifts toward task-optimal firing as warmup decays.

ATG parameter count ≈ 1M.

### 3.5 Total budget

| Component | Trainable params | Frozen |
|---|---|---|
| LLaVA-1.5-7B | 0 (LoRA only on cross-attn) | 7B |
| RelTR | 0 | ~50M |
| Grounding Planner (GP) | ~30M | — |
| Fast Adapter | ~20M | — |
| Anchor-Trigger Gate (ATG) | ~1M | — |
| **Total trainable** | **~51M** | 7.05B |

LoRA rank 16 on LLaVA cross-attn adds another ~8M trainable, depending on layer selection; included in the 51M envelope conservatively.

---

## 4. Training procedure

### 4.1 Stage 0 — Distillation warm-start (1–2 days)

**Goal:** Guarantee DT-SGOD at end of Stage 0 is no worse than SGOD v1.

**Data:** 100K samples from LLaVA-Instruct-150K. For each sample, run SGOD v1 to record `logit_final_v1` at every decode step. This becomes the teacher trace.

**Loss:**
```
L_stage0 = KL( softmax(DT_logit_final), softmax(SGOD_v1_logit_final) )
```

After Stage 0, DT-SGOD should approximately reproduce SGOD v1's behavior with the additional capacity unused. This is the safety net: if Stage 1 fails, we still have SGOD v1 reproducibility plus a learned anchor gate.

### 4.2 Stage 1 — Hallucination-aware fine-tune (3–5 days)

**Data:**
- **Positive captions:** ground-truth LLaVA-Instruct + COCO captions.
- **Negative captions (synthetic hallucinations):** generated via *scene-graph-aware corruption* — swap an object in the caption with a non-present object class, or swap an attribute with a contradictory one (red ↔ blue). Verified non-presence via the scene graph. This is in spirit with POVID ([2402.11411](https://arxiv.org/abs/2402.11411)) but uses SG ground truth for the corruption, not random image perturbation.

**Loss:**
```
L_stage1 = L_SFT(y_good | image, G)
         + β  · L_DPO(y_good ≻ y_hallucinated)
         + μ  · L_anchor_rate
         + δ  · L_slot_disentangle
```

- `L_SFT`: standard cross-entropy on the good caption tokens.
- `L_DPO`: Direct Preference Optimization, β=0.1.
- `L_anchor_rate = (E[fire_prob] − 0.2)²`, μ=0.5.
- `L_slot_disentangle = Σ_{s≠s'} cos(h_slow_s, h_slow_s')²`, δ=0.01. Encourages slots to specialize; without it slots collapse to redundant copies.

**Hyperparameters:** AdamW, lr=1e-4 (adapter) / 5e-5 (LoRA), batch=4, grad accum=8, fp16 + gradient checkpointing. Single 4090 (24 GB).

### 4.3 Estimated compute

| Stage | Wall time on 1 × RTX 4090 | Notes |
|---|---|---|
| Stage 0 distillation | ~30 hours | 100K samples × 1 epoch |
| Stage 1 fine-tune | ~70 hours | 200K samples × 2 epochs |
| Ablation runs (×8 variants) | ~140 hours total | Stage 1 only, shorter |
| **Wall total** | **~10 days** | Sequential on single GPU |

With 2 × 4090 in DDP, halve the wall time.

---

## 5. Ablation matrix

The ablation table is the most defensible part of the contribution. Reviewers will check that each component earns its place.

| Variant | GP | ATG | Slots S | GRU | Disentangle | Expected role |
|---|---|---|---|---|---|---|
| **DT-SGOD (full)** | ✓ | learned | 8 | ✓ | ✓ | Main system |
| **–slow** (no GP) | ✗ (static SG soft-prompt) | learned | — | — | — | "Is slow module needed?" |
| **–ATG** (rule fallback) | ✓ | rule | 8 | ✓ | ✓ | "Is learned anchor better than rule?" |
| **–slots** (S=1) | ✓ | learned | 1 | ✓ | n/a | "Does compositional binding need slots?" |
| **–GRU** (identity) | ✓ | learned | 8 | ✗ | ✓ | "Does recurrence matter, or just state?" |
| **–disentangle** | ✓ | learned | 8 | ✓ | ✗ | "Do slots collapse without regularization?" |
| **SGOD v1** | rule oracle | rule | — | — | — | Predecessor baseline |
| **LLaVA-base** | — | — | — | — | — | No grounding |

**Decision rule:** if a component contributes < 0.5pt on the target benchmarks, *cut it* before submission. Reviewers respect minimal sufficient architectures.

---

## 6. Theoretical scaffold

We frame DT-SGOD as *amortized variational inference with a latent grounding state*:

```
p(y_{1:T} | image, G) = ∫ p(h_slow | image, G, y_<t) · p(y_{1:T} | h_slow, image) · dh_slow
```

- `p(h_slow | image, G, y_<t)`: posterior over grounding state, amortized by GP.
- `p(y_{1:T} | h_slow, image)`: LLaVA + Fast Adapter.
- ATG implements *adaptive computation* on the posterior: refine only when the current state is stale or insufficient (high LM entropy, prefix transition).

**Recovering SGOD v1.** Set S=1, GRU=identity, ATG=rule, cross-attention=scalar projection, γ → λ. Then `Δ_t = λ · oracle(token_t)`, recovering SGOD v1. DT-SGOD generalizes the heuristic into a principled posterior decoder.

**Recovering pure LLaVA.** Set γ=0 (closed gate). Then `Δ = 0`, decoding is unmodified LLaVA.

This nesting is the paper's narrative axis: not "another method that beats POPE", but "the *family* SGOD belongs to, parameterized to allow learned grounding."

---

## 7. Benchmark plan

Per the SOTA survey, we *avoid* saturated benchmarks (POPE-random, CHAIR-short) and target the open axes:

| Benchmark | Hallucination type | Hypothesis why DT-SGOD wins |
|---|---|---|
| **Reefknot** ([2408.09429](https://arxiv.org/abs/2408.09429)) | Relational | Slow state retains relation context across tokens |
| **MMHal-Bench**, comparison/spatial categories | Multi-entity, compositional | Slots support simultaneous multi-binding |
| **AMBER-relation, AMBER-attribute** | Relation + attribute | Binding via slot specialization |
| **CHAIR (long-form, length ≥ 50)** | Drift in long captions | State decay protects against accumulating error |
| **HallusionBench (yes/no qPair)** | Reasoning hallucination | ATG fires on uncertain LM, re-grounds |
| **POPE-adversarial** | Object existence (hardest split) | Adapter can veto strong LLM prior |

**Go/no-go signal at week 4.** DT-SGOD (full) must improve ≥ 1.5pt on *at least 2* of the above benchmarks compared to SGOD v1, *and* the ablation must show that slow module + slots each contribute ≥ 1pt. If not, pivot architecture or re-design.

---

## 8. Backbone-agnosticism: applying DT-SGOD to other VLMs

DT-SGOD makes three minimal assumptions about the backbone:

1. **Exposes per-token hidden states.** Required for both ATG input and Fast Adapter query. Standard for all decoder-only VLMs (LLaVA, Qwen-VL, InternVL, MiniCPM-V, etc.).
2. **Can be wrapped in LoRA.** Required for the small trainable cross-attention. All HuggingFace VLMs with peft support qualify.
3. **Has a tokenizer/vocab.** Required to align oracle scores (legacy SGOD v1 fallback) and to compute `Δ` on the logit space.

**Concrete swap to Qwen 2.5-VL-7B-Instruct ([Qwen2-VL paper](https://arxiv.org/abs/2409.12191)):**

| Item | LLaVA-1.5-7B | Qwen 2.5-VL-7B-Instruct |
|---|---|---|
| Hidden dim `d` | 4096 | 3584 |
| Tokenizer | LLaMA BPE | Qwen2 BPE |
| Visual encoder | CLIP-ViT-L/14 (336²) | Qwen2-ViT (dynamic resolution) |
| LoRA target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` | same names, Qwen2 layer paths |
| Output head | LM head | LM head |

**Code changes needed (estimated, modest):**
- Rescale `d=3584` everywhere in GP and Fast Adapter — single config value.
- Rebuild oracle vocab cache (`clip_vocab_cache.pt`) against Qwen tokenizer — re-run `scripts/precompute_clip_vocab.py` with the new tokenizer.
- Adjust LoRA target module names in the adapter config — Qwen 2.5-VL uses `model.language_model.layers.<i>.self_attn.{q,k,v,o}_proj`.
- RelTR + scene graph are **unchanged** (operate on the image, not on the LM).

**Capabilities that may *improve* with Qwen 2.5-VL:**
- Qwen's dynamic-resolution visual encoder produces richer fine-grained features than LLaVA's fixed 336². DT-SGOD's slot attention can benefit because high-resolution attribute detail flows through to GP indirectly via prefix summarizer.
- Qwen 2.5-VL is more instruction-following; ATG warmup converges faster.
- Qwen's stronger relational reasoning baseline raises the floor — Reefknot/MMHal improvements compose.

**Capabilities that may *degrade* or need re-validation:**
- Visual token count varies with image size in Qwen 2.5-VL; if our SG node features ever align to visual patches, that alignment needs re-derivation. Current GP design doesn't depend on visual tokens (only on SG node embeddings from RelTR), so this is OK.

**Realistic effort:** ~2 engineering days to swap backbones once DT-SGOD on LLaVA is working, plus full retrain.

---

## 9. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Slow module collapse (ATG always or never fires) | Medium | Rate Lagrangian + warmup supervision |
| Slots collapse to redundant copies | Medium | Disentanglement loss + diverse init |
| Train cost overruns single 4090 | Low | LoRA r=16 fits 24GB with gradient checkpointing |
| DT-SGOD underperforms SGOD v1 after Stage 1 | Medium | Stage 0 distillation guarantees ≥ SGOD v1 at init; cut Stage 1 if regression |
| RelTR scene graph quality bottlenecks performance | Medium | Hybrid mode with Grounding DINO already in repo (recent commits) |
| Reviewer rejects "yet another VLM hallucination paper" | Medium | Theoretical scaffold + ablation matrix + capability gap on relational benchmarks distinguishes from logit-arithmetic family |

---

## 10. Roadmap (4 weeks to first decision)

| Week | Tasks | Exit criterion |
|---|---|---|
| 1 | Implement GP, ATG, Fast Adapter as new modules under `sgod/dt/`. Verify forward pass shapes. Run inference smoke-test with γ=0 → output identical to LLaVA-base. | Smoke-test passes |
| 2 | Implement Stage 0 distillation loop. Train DT-SGOD@init to match SGOD v1 on 10K samples. | KL < 0.05 on held-out |
| 3 | Implement Stage 1 with full loss. Train 1 epoch on POVID-style data. Evaluate on POPE + AMBER. | DT-SGOD ≥ SGOD v1 on AMBER-attr |
| 4 | Run full ablation matrix (Section 5). Evaluate on Reefknot + MMHal. Apply go/no-go (Section 7). | Pass go criteria, or pivot |

---

## 11. Future extensions (beyond v1 of DT-SGOD)

These are *not* part of the proposed paper but indicate where the architecture can grow:

- **Co-evolving scene graph** (LLaDA-shaped): re-extract SG conditioned on the partial generated caption every K_2 tokens. Pushes A2 in addition to A3/A4.
- **Hierarchical slow modules** (deeper HRM): multiple slow timescales (sentence-level, paragraph-level). Useful for video VLM extension.
- **Energy-based ATG**: replace MLP gate with an energy head that fires when LM hidden and SG state are incompatible. Information-theoretic framing.
- **Causal SG interventions**: do-calculus on scene graph (remove an edge), measure causal contribution of each fact to the next token. Provides faithfulness / interpretability story.

---

## 12. References

- Mamba: Linear-Time Sequence Modeling with Selective State Spaces. Gu & Dao. [arxiv 2312.00752](https://arxiv.org/abs/2312.00752).
- Mixture-of-Depths: Dynamically allocating compute in transformer-based language models. Raposo et al. [arxiv 2404.02258](https://arxiv.org/abs/2404.02258).
- LLaDA: Large Language Diffusion Models. Nie et al. [arxiv 2502.09992](https://arxiv.org/abs/2502.09992).
- Hierarchical Reasoning Model. Wang et al. (Sapient Inc.). [arxiv 2506.21734](https://arxiv.org/abs/2506.21734).
- Flamingo: a Visual Language Model for Few-Shot Learning. Alayrac et al. [arxiv 2204.14198](https://arxiv.org/abs/2204.14198).
- POVID: Aligning Modalities in Vision Large Language Models via Preference Fine-tuning. [arxiv 2402.11411](https://arxiv.org/abs/2402.11411).
- Reefknot: A Comprehensive Benchmark for Relation Hallucination Evaluation. [arxiv 2408.09429](https://arxiv.org/abs/2408.09429).
- Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution. [arxiv 2409.12191](https://arxiv.org/abs/2409.12191).
- SGOD v1 proposal — `docs/sgod_proposal.md` (this repository).

---

## Appendix A — Mapping from SGOD v1 components to DT-SGOD

| SGOD v1 module | Fate in DT-SGOD | Replacement |
|---|---|---|
| `sgod/sgg/SGGModule` | Kept identical | (used by GP graph encoder) |
| `sgod/oracle/VisualOracle` | Kept as Stage 0 distillation teacher; removed at inference | GP + Fast Adapter |
| `sgod/oracle/CLIPScorer` | Kept as fallback in GP (CLIP for nodes not in RelTR vocab) | (auxiliary) |
| `sgod/anchor/detect_anchor` | Kept as ATG warmup supervision signal; removed at inference | ATG |
| `sgod/context/tracker` | Kept as side-feature input to ATG | ATG input |
| `sgod/decoder/SGODDecoder` | Replaced by DT-SGOD decoder | `sgod/dt/DTSGODDecoder` |

New code lives in `sgod/dt/` to keep v1 intact as the baseline for reproducibility.
