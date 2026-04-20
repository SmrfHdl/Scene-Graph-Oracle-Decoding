"""
Ablation Study (Section 6.3).

Runs ablation experiments:
    1. Remove CLIP fallback (SGG only)
    2. Remove adaptive λ (fixed λ=0.35)
    3. Remove anchor detection (apply to all tokens)
    4. Only noun oracle (no relation/attr)
    5. Replace RelTR with GT scene graph (upper bound)
    6. Apply on InternVL2-8B (generalization)

Usage:
    python experiments/ablation/run_ablation.py \
        --ablation all \
        --config configs/default.yaml
"""
