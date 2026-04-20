"""
Main Evaluation — Reefknot benchmark (Section 6.2, Benchmark 2).

Relation hallucination evaluation (MAIN CLAIM):
    - Perceptive relations (on, under, holding...)
    - Cognitive relations (helping, threatening...)
    - >20,000 samples from Visual Genome
    - Metric: Accuracy

Usage:
    python experiments/main_eval/eval_reefknot.py \
        --config configs/default.yaml \
        --reefknot_path data/reefknot/
"""
