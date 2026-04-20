# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Install dependencies:**
```bash
pip install -e ".[dev]"          # core + dev (pytest, ruff)
pip install -e ".[train]"        # adds peft for LoRA training
pip install -e ".[eval]"         # adds lmms-eval for benchmarks
```

**Run tests:**
```bash
pytest tests/                    # unit tests only (no GPU required)
pytest tests/ -m integration     # GPU-intensive tests (requires RelTR checkpoint)
pytest tests/test_anchor.py      # single test file
```

**Lint:**
```bash
ruff check sgod/ tests/
```

**Scripts:**
```bash
python scripts/eval_sgg.py --checkpoint <path> --image <path>
python scripts/precompute_clip_vocab.py      # builds clip_vocab_cache.pt
python scripts/train_reltr.py
python scripts/train_lora.py
```

**Benchmarks** (in `experiments/main_eval/`):
```bash
python experiments/main_eval/eval_pope.py
python experiments/main_eval/eval_amber.py
python experiments/main_eval/eval_mmhal.py
python experiments/main_eval/eval_reefknot.py
python experiments/main_eval/eval_vqa.py
```

## Architecture

SGOD (Scene-Graph Oracle Decoding) is a training-free method to reduce hallucinations in Vision-Language Models. During decoding, it augments LLM logits with scores from an external structured scene graph oracle. One forward pass per token; no self-referential verification.

**VLM backbone:** LLaVA-1.5-7B (frozen). **Scene graph model:** RelTR (Visual Genome vocabulary: 151 entities, 51 relations).

### Five Core Components

**1. SGG Module** (`sgod/sgg/`) — Scene graph generation
- `SGGModule.extract(image)` runs RelTR once before generation and returns a `SceneGraph`
- `SceneGraph` contains `ObjectNode[]`, `RelationEdge[]`, `AttributeNode[]`
- RelTR model code is vendored under `sgod/sgg/reltr/`

**2. Visual Oracle** (`sgod/oracle/`) — Token scoring
- `VisualOracle` combines SGG confidence scores with CLIP soft similarity as a fallback
- Scores range `[-1, 1]`; built once and cached as a score map `{token → score}`
- CLIP vocabulary embeddings are pre-computed to `clip_vocab_cache.pt` via `scripts/precompute_clip_vocab.py`
- `should_activate_oracle()` disables oracle for low-quality scene graphs (average confidence below threshold)

**3. Anchor Detector** (`sgod/anchor/`) — Token classification
- `detect_anchor(prev_tokens, current_token)` returns `noun_anchor | relation_anchor | attr_anchor | neutral`
- Purely rule-based grammar patterns; zero compute overhead; stateless

**4. Generation Context** (`sgod/context/tracker.py`) — Adaptive weighting
- Tracks negation depth (decay-based) and question type (existential/descriptive/comparative/hypothetical)
- `get_lambda()` returns the oracle mixing weight; base values vary by question type (e.g., existential=0.50, descriptive=0.35)

**5. SGOD Decoder** (`sgod/decoder/sgod_decoder.py`) — Orchestrator
- Wires all components together: `logit_final = logit_lm + λ × oracle_scores`
- Applied only at anchor token positions; neutral tokens use unmodified LLM logits

### Data Flow

```
Image
 └→ SGGModule.extract()  [once]  → SceneGraph
      └→ VisualOracle.build()    [once]  → score_map {token → float}

Decode loop [per token]:
  LLaVA forward → logit_lm
  detect_anchor(prev_tokens) → anchor_type
  if anchor_type != neutral:
      λ = context.get_lambda()
      logit_final = logit_lm + λ × oracle_scores
  sample(logit_final)
  context.update(token)
```

### Configuration

YAML configs in `configs/`:
- `default.yaml` — base hyperparameters
- `sgod_plus.yaml` — fine-tuned RelTR variant
- `sgod_plus_plus.yaml` — LoRA-trained LLaVA variant

### Testing Strategy

- Unit tests have no GPU or checkpoint dependency; run with plain `pytest tests/`
- Integration tests are marked `@pytest.mark.integration` and skipped by default; they require a GPU and RelTR checkpoint
- Test markers are declared in `pytest.ini`
