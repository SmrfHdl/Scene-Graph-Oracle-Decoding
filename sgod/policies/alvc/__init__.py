"""sgod.policies.alvc — Adaptive-Length Vision Connector.

A VLM connector that outputs a variable number K of vision tokens, where K
depends on both image content and language query. Halting via PonderNet-style
mechanism (Banino 2021), trained with a rate-distortion objective.

Foundational claim: existing VLM connectors (linear projector, Q-Former,
Perceiver IO) output a fixed-length token sequence, which is
information-theoretically wasteful. Different images and different queries
should require different bandwidth.

Theorem candidate (corrected 2026-05-21 after agent review):
    K(x, q) >= ceil( I(X; Y | Q) / C )
where C is per-token continuous channel capacity (nats), estimated empirically.
This is a Shannon-style rate-distortion converse; ALVC is the achievability.

Phase 0 scope: sequence-level halting on TinyLLaVA-2B; G5 (query-conditional K)
is the critical week-1 gate.

Modules:
    connector  — ALVCConnector module (replaces linear projector / Q-Former)
    halting    — HaltingHead + PonderNet halting probability logic
    training   — rate-distortion training loop helpers
    eval       — K-distribution analysis + G5 query-conditional protocol
"""
