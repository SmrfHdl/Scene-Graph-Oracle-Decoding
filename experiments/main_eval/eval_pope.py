"""
Main Evaluation — POPE benchmark (Section 6.2, Benchmark 1).

Object hallucination evaluation:
    - adversarial / popular / random splits
    - 9,000 questions, 500 COCO images
    - Metric: F1 Score, Accuracy, Precision, Recall

Usage:
    python experiments/main_eval/eval_pope.py \
        --config configs/default.yaml \
        --pope_path data/pope/
"""
