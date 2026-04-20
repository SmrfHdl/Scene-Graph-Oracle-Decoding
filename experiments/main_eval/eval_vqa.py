"""
General VQA regression check — VQAv2 & GQA (Section 6.2, Benchmark 4).

Ensures SGOD does not degrade general VQA performance:
    - VQAv2 val: 214K questions, VQA Accuracy
    - GQA test-dev: 12.6K balanced questions, Exact match

Usage:
    python experiments/main_eval/eval_vqa.py \
        --config configs/default.yaml \
        --benchmark vqav2 \
        --data_path data/vqav2/
"""
