"""
Efficiency benchmarks (Section 10.3).

Measures:
    - Tokens per second
    - Time-to-first-token
    - Peak VRAM usage
    - SGG extraction time (isolated)

Compares: SGOD vs LLaVA baseline vs VCD vs ICD

Usage:
    python experiments/main_eval/eval_efficiency.py \
        --config configs/default.yaml \
        --num_samples 100
"""
