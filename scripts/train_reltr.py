"""
Fine-tune RelTR on GQA scene graph annotations (Section 9.3 — SGOD+ variant).

Usage:
    python scripts/train_reltr.py \
        --backbone resnet50 \
        --pretrained_weights data/checkpoints/reltr/reltr_visual_genome.pth \
        --dataset gqa \
        --data_path data/gqa/ \
        --epochs 10 \
        --lr 1e-5 \
        --batch_size 8
"""
