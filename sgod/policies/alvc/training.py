"""Training loop helpers for ALVC connector.

Rate-distortion training objective:

    L_total = L_task + beta * L_ponder

where:
    L_task   = sum_n p_n * L_LM(K=n)   (expected LM loss under halting dist)
    L_ponder = KL( p || p_geometric(lambda_p) )

Implementation deferred until halting head is finalized.
"""
from __future__ import annotations

# Skeleton — implementation pending.
