"""PonderNet-style halting head for sequence-level ALVC.

Sequence-level halting: a single halting probability per (image, query) decides
how many vision tokens K to forward to the LM. Per-token (per-patch) halting is
a future generalization.

References:
    Banino et al. 2021, "PonderNet: Learning to Ponder" (arxiv:2107.05407)
    Graves 2016, "Adaptive Computation Time for RNNs" (arxiv:1603.08983)

Phase 0 design (committed 2026-05-21):
    Input: pooled vision features [B, D_v] + pooled query embedding [B, D_q]
    Output: distribution over K in [K_min, K_max], parameterized as a sequence
            of Bernoulli halting probabilities lambda_n = P(halt | n).

The expected loss objective is:
    L = sum_n p_n * L_task(K=n) + beta * KL(p || p_geometric(lambda_p))
where p_n = lambda_n * prod_{i<n} (1 - lambda_i) is the halting distribution.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HaltingConfig:
    """Hyperparameters for HaltingHead.

    Attributes:
        k_min: Minimum K (typically 1).
        k_max: Maximum K (typically 256 = number of patches for SigLIP-SO400M).
        lambda_p: Geometric prior parameter (mean K = 1/lambda_p).
        beta: KL regularization weight.
        d_v: Vision feature dimension.
        d_q: Query embedding dimension.
        d_hidden: Halting head hidden dimension.
        query_conditioned: If False, halting ignores query (image-only baseline
            for ablation — this is the Matryoshka M3 territory).
    """

    k_min: int = 1
    k_max: int = 144
    lambda_p: float = 0.1
    beta: float = 0.01
    d_v: int = 4096
    d_q: int = 3072
    d_hidden: int = 512
    query_conditioned: bool = True


class HaltingHead:
    """Sequence-level halting head.

    NOTE: skeleton only — implementation deferred to next commit.
    The actual nn.Module + forward will be added once the math derivation
    doc (theorem) is signed off and the rate-distortion training objective
    is finalized.
    """

    def __init__(self, config: HaltingConfig) -> None:
        self.config = config
        raise NotImplementedError("HaltingHead skeleton — implementation pending")
