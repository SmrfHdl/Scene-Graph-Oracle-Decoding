"""
LoRA fine-tuning script for LLaVA (Section 9.4 — SGOD++ variant).

Trains LLaVA-1.5-7B with LoRA (rank=16) using hallucination-aware objective:
    L = L_ce + λ₁ * L_orth + λ₂ * L_contrastive

Usage:
    python scripts/train_lora.py \
        --config configs/sgod_plus_plus.yaml \
        --data_mix vqav2,gqa,rlhfv,lrv
"""
