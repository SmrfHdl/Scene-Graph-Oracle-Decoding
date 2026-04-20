"""
Pilot Experiment (Section 6.1).

Validates that GT scene graph in prompt improves relation hallucination
before committing to full implementation (2-3 days).

3 variants on 200 GQA images with relation-type questions:
    a. LLaVA baseline (no SG)
    b. LLaVA + GT scene graph as text prefix
    c. LLaVA + predicted scene graph (RelTR) as text prefix

Success criterion: variant (b) > baseline by ≥3-5%

Usage:
    python experiments/pilot/run_pilot.py \
        --num_images 200 \
        --gqa_path data/gqa/
"""
