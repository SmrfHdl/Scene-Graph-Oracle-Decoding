"""
Targeted diagnostic: trace oracle behavior step-by-step on disagreement case [1]
(dining table, image 000000283412) to find exactly HOW oracle flips Yes→No.

Instruments: detect_anchor calls, oracle scores, apply_oracle_scores, argmax.
"""
from __future__ import annotations
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Monkey-patch detect_anchor to trace calls ─────────────────────────────────
import sgod.anchor.detector as _det_module
_orig_detect_anchor = _det_module.detect_anchor

_anchor_calls = []  # collects (prev_tokens, current_token, result)

def _traced_detect_anchor(prev_tokens, current_token):
    result = _orig_detect_anchor(prev_tokens, current_token)
    _anchor_calls.append((list(prev_tokens), current_token, result))
    return result

_det_module.detect_anchor = _traced_detect_anchor

# Also patch it in sgod.anchor namespace (the one imported by sgod_decoder)
import sgod.anchor as _anchor_pkg
_anchor_pkg.detect_anchor = _traced_detect_anchor

# And re-patch in sgod_decoder module if already imported
import importlib


# ── Monkey-patch apply_oracle_scores ─────────────────────────────────────────
import sgod.decoder.token_utils as _tok_module
_orig_apply = _tok_module.apply_oracle_scores

_apply_calls = []  # collects (step, top_k_words, oracle_scores, lam)

_current_step = [0]
_current_top_k_words = [[]]

def _traced_apply(logits, top_k_ids, oracle_scores, lam):
    nonzero_mask = oracle_scores.abs() > 1e-6
    print(f"  [apply_oracle_scores called] step={_current_step[0]}  lam={lam}")
    for i, (idx, score) in enumerate(zip(top_k_ids.tolist(), oracle_scores.tolist())):
        if abs(score) > 1e-6:
            word = _current_top_k_words[0][i] if i < len(_current_top_k_words[0]) else "?"
            print(f"    token[{i}] '{word}' (id={idx}) oracle_score={score:.4f}")
    return _orig_apply(logits, top_k_ids, oracle_scores, lam)

_tok_module.apply_oracle_scores = _traced_apply


def main():
    from PIL import Image
    from sgod.utils.model_loader import load_config, load_sgod_decoder

    # Reload sgod_decoder with patched functions
    import sgod.decoder.sgod_decoder as _dec_mod
    importlib.reload(_dec_mod)
    # Patch detect_anchor in the reloaded module
    _dec_mod.detect_anchor = _traced_detect_anchor
    # Patch apply_oracle_scores in the reloaded token_utils (already done above)

    IMG  = Path("data/coco/val2014/COCO_val2014_000000283412.jpg")
    Q    = "Is there a dining table in the image?"
    PROMPT = f"USER: <image>\n{Q}\nAnswer with yes or no.\nASSISTANT:"

    print("Loading SGOD decoder...")
    cfg = load_config("configs/default.yaml")
    decoder = load_sgod_decoder(cfg)

    # Patch detect_anchor in the LOADED decoder module
    import sgod.decoder.sgod_decoder as actual_dec
    actual_dec.detect_anchor = _traced_detect_anchor

    print(f"\nRunning SGOD on: {Q}")
    print(f"Image: {IMG}")
    image = Image.open(IMG).convert("RGB")

    step_counter = [0]

    from sgod.decoder import SGODDecoder

    @staticmethod
    def traced_score_candidates(words, prev_tokens, oracle):
        step = step_counter[0]
        print(f"\n--- Step {step}: _score_candidates ---")
        print(f"  prev_tokens: {prev_tokens}")
        print(f"  top_k words (first 10): {words[:10]}")
        _current_step[0] = step
        _current_top_k_words[0] = words
        scores = torch.zeros(len(words))
        for i, word in enumerate(words):
            anchor = _traced_detect_anchor(prev_tokens, word)
            if anchor != "neutral":
                s = oracle.score(word, anchor)
                scores[i] = s
                print(f"  NON-NEUTRAL: word='{word}' anchor={anchor} score={s:.4f}")
        step_counter[0] += 1
        nonzero = (scores.abs() > 1e-6).sum().item()
        print(f"  Non-zero scores: {nonzero} / {len(words)}")
        print(f"  oracle_scores.abs().sum() = {scores.abs().sum().item():.6f}")
        return scores

    SGODDecoder._score_candidates = traced_score_candidates

    _anchor_calls.clear()
    out = decoder.generate(image, PROMPT, max_new_tokens=4)
    print(f"\n=== SGOD output: {out!r} ===")

    print(f"\nTotal detect_anchor calls: {len(_anchor_calls)}")
    non_neutral = [(p, c, r) for p, c, r in _anchor_calls if r != "neutral"]
    print(f"Non-neutral anchor calls: {len(non_neutral)}")
    for p, c, r in non_neutral:
        print(f"  prev={p} cur='{c}' → {r}")


if __name__ == "__main__":
    main()
