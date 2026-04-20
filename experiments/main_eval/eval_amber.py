"""
Main Evaluation — AMBER benchmark (Section 6.2, Benchmark 3).

Multi-dimensional hallucination evaluation:
    - Object existence + attribute + relation
    - ~1,000 images from A-OKVQA
    - Metric: F1, CHAIR, Coverage

Usage:
    python experiments/main_eval/eval_amber.py \
        --config configs/default.yaml \
        --amber_path data/amber/
"""
