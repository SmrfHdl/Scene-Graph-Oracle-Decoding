"""
Diagnostic: compare baseline `model.generate()` vs SGOD manual loop on the
EXACT same prompts to isolate where they diverge.

For each test image/question:
  1. Run baseline (HF generate, do_sample=False)
  2. Run a "no-oracle" manual loop that mimics decoder.generate but skips
     all oracle work
  3. Run real SGOD decoder.generate
  4. Print first-token logits, argmax, top-5 candidates side-by-side

If (1) == (2) → manual loop matches generate; oracle is the culprit
If (1) != (2) → manual loop itself diverges from generate; root cause is decoding mechanics
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sgod.utils.model_loader import load_config, load_llava, load_sgod_decoder

POPE_FILE    = Path("data/pope/coco_pope_adversarial.json")
RECORDS_FILE = Path("outputs/eval/2026-04-28_12-24-44_pope/records_adversarial.json")
COCO_DIR     = Path("data/coco/val2014/")
N_SAMPLES    = 8  # number of disagreement cases to test


def manual_loop_no_oracle(model, processor, image, prompt, max_new_tokens=4):
    """Mimics SGODDecoder.generate but with oracle entirely disabled.
    Returns: list of (step, top1_id, top1_word, top5_ids, top5_words, top5_logits)
    """
    device = next(model.parameters()).device
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    input_ids    = inputs["input_ids"]
    pixel_values = inputs.get("pixel_values")
    trace = []
    for step in range(max_new_tokens):
        with torch.no_grad():
            out = model(input_ids=input_ids, pixel_values=pixel_values)
            logits = out.logits[0, -1, :].float()
        top5 = torch.topk(logits, k=5)
        top5_ids   = top5.indices.tolist()
        top5_words = processor.tokenizer.batch_decode(top5_ids)
        top5_lg    = [round(x, 3) for x in top5.values.tolist()]
        next_id = int(torch.argmax(logits).item())
        trace.append({
            "step": step,
            "top1_id": next_id,
            "top1_word": processor.tokenizer.decode([next_id]),
            "top5_ids": top5_ids,
            "top5_words": top5_words,
            "top5_logits": top5_lg,
        })
        input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=device)], dim=-1)
        if next_id == processor.tokenizer.eos_token_id:
            break
    return trace


def baseline_generate(model, processor, image, prompt, max_new_tokens=4):
    """HF generate with output_scores so we can compare logits."""
    device = next(model.parameters()).device
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )
    trace = []
    gen_ids = out.sequences[0][inputs["input_ids"].shape[1]:]
    for step, scores in enumerate(out.scores):
        logits = scores[0].float()  # [vocab]
        top5 = torch.topk(logits, k=5)
        top5_ids   = top5.indices.tolist()
        top5_words = processor.tokenizer.batch_decode(top5_ids)
        top5_lg    = [round(x, 3) for x in top5.values.tolist()]
        next_id = int(gen_ids[step].item())
        trace.append({
            "step": step,
            "top1_id": next_id,
            "top1_word": processor.tokenizer.decode([next_id]),
            "top5_ids": top5_ids,
            "top5_words": top5_words,
            "top5_logits": top5_lg,
        })
    return trace


def main():
    from PIL import Image

    print("Loading model...")
    model, processor = load_llava()
    cfg = load_config("configs/default.yaml")
    decoder = load_sgod_decoder(cfg)

    # Load disagreement cases: where baseline != sgod AND sgod got it wrong
    with open(RECORDS_FILE) as f:
        all_records = json.load(f)

    disagree = [
        r for r in all_records
        if r["pred_base"] != r["pred_sgod"]
    ]
    print(f"Total disagreements: {len(disagree)}")
    items = disagree[:N_SAMPLES]

    for idx, item in enumerate(items):
        img_path = COCO_DIR / item["image"]
        if not img_path.exists():
            print(f"Missing: {img_path}")
            continue
        image = Image.open(img_path).convert("RGB")
        question = item["question"]
        prompt = f"USER: <image>\n{question}\nAnswer with yes or no.\nASSISTANT:"

        print("\n" + "=" * 80)
        print(f"[{idx}] {item['image']} | gt={item['label']}")
        print(f"  base={item['pred_base']} sgod={item['pred_sgod']} raw_sgod={item.get('raw_sgod','?')!r}")
        print(f"Question: {question}")
        print(f"Prompt: {prompt!r}")

        print("\n--- BASELINE model.generate ---")
        try:
            tr_base = baseline_generate(model, processor, image, prompt)
            for t in tr_base:
                print(f"  step {t['step']}: top1={t['top1_word']!r} (id={t['top1_id']})  "
                      f"top5_words={t['top5_words']}  logits={t['top5_logits']}")
        except Exception as e:
            print(f"  ERROR: {e}")

        print("\n--- MANUAL loop (no oracle) ---")
        try:
            tr_man = manual_loop_no_oracle(model, processor, image, prompt)
            for t in tr_man:
                print(f"  step {t['step']}: top1={t['top1_word']!r} (id={t['top1_id']})  "
                      f"top5_words={t['top5_words']}  logits={t['top5_logits']}")
        except Exception as e:
            print(f"  ERROR: {e}")

        print("\n--- SGOD decoder.generate (with oracle) ---")
        try:
            sgod_out = decoder.generate(image, prompt, max_new_tokens=4)
            print(f"  raw output: {sgod_out!r}")
        except Exception as e:
            print(f"  ERROR: {e}")

        print("\n--- DIVERGENCE CHECK ---")
        if 'tr_base' in dir() and 'tr_man' in dir():
            for i in range(min(len(tr_base), len(tr_man))):
                if tr_base[i]['top1_id'] != tr_man[i]['top1_id']:
                    print(f"  DIVERGES at step {i}: base={tr_base[i]['top1_word']!r} "
                          f"vs manual={tr_man[i]['top1_word']!r}")
                    break
            else:
                print("  Manual loop matches baseline through compared steps")


if __name__ == "__main__":
    main()
