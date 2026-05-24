"""PonderNet-style halting head for sequence-level ALVC.

Sequence-level halting: a single halting distribution per (image, query) decides
how many vision tokens K to forward to the LM. Per-token (per-patch) halting is
a future generalization.

References:
    Banino et al. 2021, "PonderNet: Learning to Ponder" (arxiv:2107.05407)
    Graves 2016, "Adaptive Computation Time for RNNs" (arxiv:1603.08983)

Phase 0 design (committed 2026-05-21):
    Input: pooled vision features [B, D_v] + pooled query embedding [B, D_q]
    Output: distribution over K in [k_min, k_max], parameterized as a sequence
            of Bernoulli halting probabilities lambda_n = P(halt | n).

The expected training loss is:
    L = sum_n p_n * L_task(K=n) + beta * KL(p || p_geometric(lambda_p))
where p_n = lambda_n * prod_{i<n} (1 - lambda_i) is the halting distribution
and the final step is forced to halt (lambda_{k_max} = 1) so p_n sums to 1.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class HaltingConfig:
    """Hyperparameters for HaltingHead.

    Attributes:
        k_min: Minimum K (typically 1).
        k_max: Maximum K (typically = n_visual_patches, e.g. 144 for Phi-3.5-V).
        lambda_p: Geometric prior parameter (mean K under prior ~ 1/lambda_p).
        beta: KL regularization weight (used by training loss, not by the head).
        d_v: Pooled vision feature dimension.
        d_q: Pooled query embedding dimension.
        d_hidden: Halting head hidden dimension.
        query_conditioned: If False, halting ignores query (image-only baseline
            for ablation — this is the Matryoshka M3 territory; G5 ablation
            relies on flipping this flag).
    """

    k_min: int = 1
    k_max: int = 144
    lambda_p: float = 0.1
    beta: float = 0.01
    d_v: int = 4096
    d_q: int = 3072
    d_hidden: int = 512
    query_conditioned: bool = True

    def __post_init__(self) -> None:
        if self.k_min < 1:
            raise ValueError(f"k_min must be >= 1, got {self.k_min}")
        if self.k_max <= self.k_min:
            raise ValueError(f"k_max ({self.k_max}) must be > k_min ({self.k_min})")
        if not (0.0 < self.lambda_p < 1.0):
            raise ValueError(f"lambda_p must be in (0,1), got {self.lambda_p}")
        if self.beta < 0.0:
            raise ValueError(f"beta must be >= 0, got {self.beta}")


class HaltingHead(nn.Module):
    """Sequence-level PonderNet halting head.

    Predicts a distribution over K in [k_min, k_max] from pooled vision and
    (optionally) query features. The output halting distribution p_n is built
    from per-step Bernoulli halting probabilities lambda_n, with the final step
    forced to halt so the distribution is proper.

    Forward returns a dict with:
        lambda_n:   [B, N]  Bernoulli halting probability at each step.
        p_n:        [B, N]  Halting distribution over K, sums to 1 along dim=-1.
        expected_K: [B]     Expected K under p_n (for monitoring).
        k_values:   [N]     The K value each step corresponds to (k_min..k_max).

    Use `kl_to_prior(p_n)` to compute KL(p_n || geometric_prior) per sample for
    the regularization term in the rate-distortion training objective.
    """

    def __init__(self, config: HaltingConfig) -> None:
        super().__init__()
        self.config = config
        self.n_steps = config.k_max - config.k_min + 1

        in_dim = config.d_v + (config.d_q if config.query_conditioned else 0)
        # We predict n_steps - 1 Bernoulli logits; step n_steps is forced halt.
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, config.d_hidden),
            nn.GELU(),
            nn.Linear(config.d_hidden, config.d_hidden),
            nn.GELU(),
            nn.Linear(config.d_hidden, self.n_steps - 1),
        )

        self.register_buffer("k_values", torch.arange(config.k_min, config.k_max + 1))
        self.register_buffer("geom_prior", self._build_geometric_prior())

    def _build_geometric_prior(self) -> torch.Tensor:
        """Geometric(lambda_p) truncated to n_steps, with tail mass on the last step."""
        lam = self.config.lambda_p
        n = self.n_steps
        probs = torch.zeros(n, dtype=torch.float32)
        for i in range(n - 1):
            probs[i] = ((1.0 - lam) ** i) * lam
        probs[n - 1] = max(0.0, 1.0 - probs[: n - 1].sum().item())
        probs = probs.clamp_min(1e-8)
        probs = probs / probs.sum()
        return probs

    def forward(
        self,
        vision_pooled: torch.Tensor,
        query_pooled: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run the halting head.

        Args:
            vision_pooled: [B, d_v] pooled vision features.
            query_pooled:  [B, d_q] pooled query embedding. Required iff
                config.query_conditioned is True.
        """
        cfg = self.config
        if cfg.query_conditioned:
            if query_pooled is None:
                raise ValueError("query_pooled is required when query_conditioned=True")
            if query_pooled.shape[0] != vision_pooled.shape[0]:
                raise ValueError(
                    f"batch mismatch: vision {vision_pooled.shape[0]} vs query {query_pooled.shape[0]}"
                )
            x = torch.cat([vision_pooled, query_pooled], dim=-1)
        else:
            x = vision_pooled

        logits = self.mlp(x)  # [B, N-1]
        lambdas_partial = torch.sigmoid(logits)  # [B, N-1] in (0,1)

        # Force the last step to halt with probability 1.
        ones = torch.ones(
            lambdas_partial.shape[0], 1,
            device=lambdas_partial.device, dtype=lambdas_partial.dtype,
        )
        lambdas = torch.cat([lambdas_partial, ones], dim=-1)  # [B, N]

        # Survival probability entering step n: prod_{i<n} (1 - lambda_i).
        # survival[:, 0] = 1; survival[:, n] = prod over first n entries of (1-lambda).
        not_lambdas = 1.0 - lambdas_partial  # [B, N-1]
        cum_survival = torch.cumprod(not_lambdas, dim=-1)  # [B, N-1]
        survival = torch.cat([torch.ones_like(cum_survival[:, :1]), cum_survival], dim=-1)  # [B, N]

        p_n = lambdas * survival  # [B, N]
        # Numerical safety: renormalize. Analytically sums to 1 because last
        # lambda is forced to 1, but float drift can break this at high N.
        p_n = p_n / p_n.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        expected_K = (p_n * self.k_values.to(p_n.dtype)).sum(dim=-1)

        return {
            "lambda_n": lambdas,
            "p_n": p_n,
            "expected_K": expected_K,
            "k_values": self.k_values,
        }

    def kl_to_prior(self, p_n: torch.Tensor) -> torch.Tensor:
        """KL(p_n || geometric_prior), per-sample. Returns [B]."""
        prior = self.geom_prior.to(p_n.dtype).clamp_min(1e-8)
        p = p_n.clamp_min(1e-8)
        return (p * (p.log() - prior.log())).sum(dim=-1)
