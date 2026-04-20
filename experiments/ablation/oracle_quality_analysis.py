"""
Oracle Quality Analysis (Section 10.4).

Analyzes SGG quality impact on SGOD performance:
    - SGOD with GT scene graph vs predicted SG vs CLIP-only
    - SGG recall vs Reefknot gain (Pearson correlation)
    - Generates the "most important figure in the paper"

Usage:
    python experiments/ablation/oracle_quality_analysis.py \
        --config configs/default.yaml \
        --gqa_path data/gqa/
"""
