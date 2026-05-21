"""Evaluation utilities for ALVC.

Provides:
    - K-distribution analysis across an eval dataset
    - G5 critical-gate protocol: query-conditional K test

G5 protocol (week 1, critical gate):
    For each image, evaluate K under multiple different queries (simple vs
    complex, factual vs descriptive). Compute |Delta K| per image-pair.

    Pass: median |Delta K| > 0.15 * K_max with bootstrap CI95 excluding 0.
    Fail: paper claim collapses into Matryoshka M3 territory (image-only K).

Other gates:
    G1: training stability (loss converges, no NaN)
    G2: K non-degenerate (std > 0.1 * K_max)
    G3: K correlates with image complexity (rho(K, num_objects) > 0.2)
    G4: task accuracy holds (Delta acc >= -0.02 with 0.7x avg K vs fixed-K baseline)
"""
from __future__ import annotations

# Skeleton — implementation pending.
