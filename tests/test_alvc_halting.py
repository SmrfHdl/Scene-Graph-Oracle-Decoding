"""Unit tests for sgod.policies.alvc.halting.HaltingHead.

Covers:
    - output shapes & dtypes
    - p_n is a valid probability distribution (sums to 1, non-negative)
    - lambda_n in (0, 1) with the final step forced to 1
    - expected_K is in [k_min, k_max]
    - gradient flows from p_n / expected_K back to MLP parameters
    - query_conditioned True vs False both work, and the True branch actually
      uses the query (different queries -> different distributions)
    - KL to geometric prior is non-negative and zero when p_n == prior
"""
import pytest
import torch

from sgod.policies.alvc.halting import HaltingConfig, HaltingHead


def _small_config(**overrides) -> HaltingConfig:
    """Smaller config so tests run fast and don't hide bugs in big tensors."""
    base = dict(
        k_min=1, k_max=8, lambda_p=0.3, beta=0.01,
        d_v=16, d_q=8, d_hidden=32, query_conditioned=True,
    )
    base.update(overrides)
    return HaltingConfig(**base)


def test_halting_config_validates():
    with pytest.raises(ValueError):
        HaltingConfig(k_min=0)
    with pytest.raises(ValueError):
        HaltingConfig(k_min=4, k_max=4)
    with pytest.raises(ValueError):
        HaltingConfig(lambda_p=0.0)
    with pytest.raises(ValueError):
        HaltingConfig(lambda_p=1.0)
    with pytest.raises(ValueError):
        HaltingConfig(beta=-0.1)


def test_forward_shapes_and_dtypes():
    cfg = _small_config()
    head = HaltingHead(cfg)
    B = 4
    v = torch.randn(B, cfg.d_v)
    q = torch.randn(B, cfg.d_q)

    out = head(v, q)
    N = cfg.k_max - cfg.k_min + 1
    assert out["lambda_n"].shape == (B, N)
    assert out["p_n"].shape == (B, N)
    assert out["expected_K"].shape == (B,)
    assert out["k_values"].shape == (N,)
    assert out["k_values"].tolist() == list(range(cfg.k_min, cfg.k_max + 1))


def test_p_n_is_valid_distribution():
    cfg = _small_config()
    head = HaltingHead(cfg)
    B = 8
    v = torch.randn(B, cfg.d_v)
    q = torch.randn(B, cfg.d_q)

    out = head(v, q)
    p = out["p_n"]
    assert torch.all(p >= 0)
    assert torch.allclose(p.sum(dim=-1), torch.ones(B), atol=1e-5)


def test_lambda_final_step_forced_to_one():
    cfg = _small_config()
    head = HaltingHead(cfg)
    v = torch.randn(2, cfg.d_v)
    q = torch.randn(2, cfg.d_q)
    out = head(v, q)
    lam = out["lambda_n"]
    # All intermediate lambdas strictly in (0,1); final forced to 1.
    assert torch.all(lam[:, -1] == 1.0)
    assert torch.all((lam[:, :-1] > 0) & (lam[:, :-1] < 1))


def test_expected_K_in_range():
    cfg = _small_config()
    head = HaltingHead(cfg)
    v = torch.randn(16, cfg.d_v)
    q = torch.randn(16, cfg.d_q)
    out = head(v, q)
    ek = out["expected_K"]
    assert torch.all(ek >= cfg.k_min - 1e-5)
    assert torch.all(ek <= cfg.k_max + 1e-5)


def test_gradient_flows_through_p_n_and_expected_K():
    cfg = _small_config()
    head = HaltingHead(cfg)
    v = torch.randn(3, cfg.d_v)
    q = torch.randn(3, cfg.d_q)
    out = head(v, q)

    # Loss that depends on both the full distribution and the expected K.
    loss = out["p_n"].sum() + out["expected_K"].mean()
    loss.backward()

    grads_found = [p.grad is not None and p.grad.abs().sum().item() > 0
                   for p in head.parameters()]
    assert all(grads_found), "expected non-zero grads on all halting head params"


def test_query_conditioned_actually_uses_query():
    cfg = _small_config(query_conditioned=True)
    head = HaltingHead(cfg).eval()
    torch.manual_seed(0)
    v = torch.randn(1, cfg.d_v)
    q1 = torch.randn(1, cfg.d_q)
    q2 = torch.randn(1, cfg.d_q) * 5  # markedly different query

    with torch.no_grad():
        p1 = head(v, q1)["p_n"]
        p2 = head(v, q2)["p_n"]

    # Same image, different queries -> non-trivially different distribution.
    assert not torch.allclose(p1, p2, atol=1e-4)


def test_image_only_mode_ignores_query():
    cfg = _small_config(query_conditioned=False)
    head = HaltingHead(cfg).eval()
    v = torch.randn(2, cfg.d_v)
    with torch.no_grad():
        out_no_q = head(v)
        out_with_q = head(v, torch.randn(2, cfg.d_q))
    # query argument is silently ignored when query_conditioned=False
    assert torch.allclose(out_no_q["p_n"], out_with_q["p_n"])


def test_image_only_mode_does_not_need_query():
    cfg = _small_config(query_conditioned=False)
    head = HaltingHead(cfg)
    v = torch.randn(2, cfg.d_v)
    out = head(v)  # should not raise
    assert out["p_n"].shape == (2, cfg.k_max - cfg.k_min + 1)


def test_query_conditioned_requires_query():
    cfg = _small_config(query_conditioned=True)
    head = HaltingHead(cfg)
    v = torch.randn(2, cfg.d_v)
    with pytest.raises(ValueError):
        head(v)  # missing query


def test_query_conditioned_batch_mismatch_raises():
    cfg = _small_config(query_conditioned=True)
    head = HaltingHead(cfg)
    v = torch.randn(2, cfg.d_v)
    q = torch.randn(3, cfg.d_q)
    with pytest.raises(ValueError):
        head(v, q)


def test_geometric_prior_is_valid_distribution():
    cfg = _small_config()
    head = HaltingHead(cfg)
    prior = head.geom_prior
    assert prior.shape == (cfg.k_max - cfg.k_min + 1,)
    assert torch.all(prior > 0)
    assert torch.allclose(prior.sum(), torch.tensor(1.0), atol=1e-6)


def test_kl_to_prior_non_negative_and_zero_at_prior():
    cfg = _small_config()
    head = HaltingHead(cfg)
    B = 5
    v = torch.randn(B, cfg.d_v)
    q = torch.randn(B, cfg.d_q)
    out = head(v, q)
    kl = head.kl_to_prior(out["p_n"])
    assert kl.shape == (B,)
    # KL >= 0 always (allow tiny numerical slack)
    assert torch.all(kl >= -1e-5)

    # If we feed the prior itself, KL should be ~0.
    prior_batch = head.geom_prior.unsqueeze(0).expand(B, -1).contiguous()
    kl_at_prior = head.kl_to_prior(prior_batch)
    assert torch.allclose(kl_at_prior, torch.zeros(B), atol=1e-5)
