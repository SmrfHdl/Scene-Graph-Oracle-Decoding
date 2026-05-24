"""Unit tests for sgod.policies.alvc.connector.ALVCConnector.

Covers:
    - Config validation + halting/connector k_min/k_max sync.
    - Output dict shapes (projected, importance, halting, keep_mask_*).
    - keep_mask_hard is 0/1 and exactly K* tokens are kept per sample,
      where K* = argmax(p_n) + k_min, AND those tokens are the top-K* by
      importance.
    - keep_mask_soft is in [0, 1], monotone non-increasing in rank, and the
      expected number of kept tokens (sum) equals E[K] (which is what we
      want: soft mask = expected survival).
    - Gradient flows from keep_mask_soft back to HaltingHead params (and
      NOT through keep_mask_hard).
    - Input-shape validation (wrong N or D_v -> raises).
    - query_conditioned=False bypass works (no query needed).
"""
import pytest
import torch

from sgod.policies.alvc.connector import ALVCConfig, ALVCConnector
from sgod.policies.alvc.halting import HaltingConfig


def _small_config(**overrides) -> ALVCConfig:
    base = dict(
        n_visual_patches=8, d_v=16, d_lm=12, d_q=10,
        k_min=1, k_max=8,
    )
    base.update(overrides)
    return ALVCConfig(**base)


def test_config_derives_halting_when_none():
    cfg = _small_config()
    assert cfg.halting is not None
    assert cfg.halting.k_min == cfg.k_min
    assert cfg.halting.k_max == cfg.k_max
    assert cfg.halting.d_v == cfg.d_v
    assert cfg.halting.d_q == cfg.d_q


def test_config_rejects_mismatched_halting():
    bad_halting = HaltingConfig(k_min=1, k_max=4, d_v=16, d_q=10)
    with pytest.raises(ValueError):
        _small_config(k_max=8, halting=bad_halting)


def test_config_validates_k_range():
    with pytest.raises(ValueError):
        _small_config(k_min=0)
    with pytest.raises(ValueError):
        _small_config(k_min=4, k_max=4)
    with pytest.raises(ValueError):
        _small_config(n_visual_patches=4, k_max=8)


def test_forward_output_shapes():
    cfg = _small_config()
    conn = ALVCConnector(cfg)
    B, N = 3, cfg.n_visual_patches
    v = torch.randn(B, N, cfg.d_v)
    q = torch.randn(B, cfg.d_q)

    out = conn(v, q)

    assert out["projected"].shape == (B, N, cfg.d_lm)
    assert out["importance"].shape == (B, N)
    assert out["keep_mask_hard"].shape == (B, N)
    assert out["keep_mask_soft"].shape == (B, N)
    halting = out["halting"]
    n_steps = cfg.k_max - cfg.k_min + 1
    assert halting["p_n"].shape == (B, n_steps)


def test_forward_rejects_wrong_shape():
    cfg = _small_config()
    conn = ALVCConnector(cfg)
    q = torch.randn(2, cfg.d_q)
    with pytest.raises(ValueError):
        conn(torch.randn(2, cfg.d_v), q)  # missing N dim
    with pytest.raises(ValueError):
        conn(torch.randn(2, cfg.n_visual_patches + 1, cfg.d_v), q)  # wrong N
    with pytest.raises(ValueError):
        conn(torch.randn(2, cfg.n_visual_patches, cfg.d_v + 1), q)  # wrong D_v


def test_keep_mask_hard_keeps_exactly_K_star_top_importance():
    torch.manual_seed(42)
    cfg = _small_config()
    conn = ALVCConnector(cfg).eval()
    B, N = 4, cfg.n_visual_patches
    v = torch.randn(B, N, cfg.d_v)
    q = torch.randn(B, cfg.d_q)

    with torch.no_grad():
        out = conn(v, q)
    mask = out["keep_mask_hard"]
    importance = out["importance"]
    p_n = out["halting"]["p_n"]
    k_star = p_n.argmax(dim=-1) + cfg.k_min  # [B]

    # Mask is exactly 0/1.
    assert torch.all((mask == 0) | (mask == 1))

    # Number kept matches k_star per sample.
    assert torch.equal(mask.sum(dim=-1).long(), k_star.long())

    # The kept tokens are precisely the top-k_star by importance.
    for b in range(B):
        k = int(k_star[b].item())
        topk_idx = torch.topk(importance[b], k=k).indices
        expected = torch.zeros(N)
        expected[topk_idx] = 1
        assert torch.equal(mask[b], expected)


def test_keep_mask_soft_bounded_and_monotone_in_rank():
    torch.manual_seed(0)
    cfg = _small_config()
    conn = ALVCConnector(cfg).eval()
    B, N = 4, cfg.n_visual_patches
    v = torch.randn(B, N, cfg.d_v)
    q = torch.randn(B, cfg.d_q)
    with torch.no_grad():
        out = conn(v, q)
    soft = out["keep_mask_soft"]
    importance = out["importance"]

    # Bounded in [0, 1].
    assert torch.all(soft >= 0.0 - 1e-6)
    assert torch.all(soft <= 1.0 + 1e-6)

    # When sorted by importance descending, soft mask must be non-increasing
    # (higher-importance patches are kept with at least as much probability
    # as lower-importance ones).
    sorted_idx = torch.argsort(importance, dim=-1, descending=True)
    sorted_soft = torch.gather(soft, dim=-1, index=sorted_idx)
    diffs = sorted_soft[:, 1:] - sorted_soft[:, :-1]
    assert torch.all(diffs <= 1e-6), "keep_mask_soft must be non-increasing in rank"


def test_keep_mask_soft_sum_equals_expected_K():
    """E[# kept tokens] under soft mask = sum_r P(K >= r) = E[K]. This is the
    whole point of the soft mask being the right differentiable surrogate."""
    torch.manual_seed(7)
    cfg = _small_config()
    conn = ALVCConnector(cfg).eval()
    B, N = 6, cfg.n_visual_patches
    v = torch.randn(B, N, cfg.d_v)
    q = torch.randn(B, cfg.d_q)
    with torch.no_grad():
        out = conn(v, q)
    soft_sum = out["keep_mask_soft"].sum(dim=-1)
    expected_K = out["halting"]["expected_K"]
    assert torch.allclose(soft_sum, expected_K, atol=1e-5)


def test_gradient_flows_through_soft_mask_to_halting_head():
    cfg = _small_config()
    conn = ALVCConnector(cfg)
    v = torch.randn(2, cfg.n_visual_patches, cfg.d_v)
    q = torch.randn(2, cfg.d_q)
    out = conn(v, q)
    loss = out["keep_mask_soft"].sum()  # E[K] essentially
    loss.backward()

    # Halting head params must receive gradient.
    halting_params = list(conn.halting_head.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in halting_params)


def test_no_gradient_through_hard_mask():
    cfg = _small_config()
    conn = ALVCConnector(cfg)
    v = torch.randn(2, cfg.n_visual_patches, cfg.d_v)
    q = torch.randn(2, cfg.d_q)
    out = conn(v, q)
    assert not out["keep_mask_hard"].requires_grad


def test_query_conditioned_false_no_query_needed():
    halting = HaltingConfig(k_min=1, k_max=8, d_v=16, d_q=10, query_conditioned=False)
    cfg = _small_config(halting=halting)
    conn = ALVCConnector(cfg)
    v = torch.randn(2, cfg.n_visual_patches, cfg.d_v)
    out = conn(v)  # no query
    assert out["projected"].shape == (2, cfg.n_visual_patches, cfg.d_lm)


def test_phi35v_default_shapes_smoke():
    """Smoke test at Phi-3.5-V defaults; shape sanity only, no GPU."""
    cfg = ALVCConfig()  # defaults: 144 / 4096 / 3072
    # Use cfg.n_visual_patches=4 patches via override to keep memory tiny? No —
    # the point is to verify defaults parse and a tiny batch can forward.
    conn = ALVCConnector(cfg)
    B = 1
    v = torch.randn(B, cfg.n_visual_patches, cfg.d_v)
    q = torch.randn(B, cfg.d_q)
    out = conn(v, q)
    assert out["projected"].shape == (B, cfg.n_visual_patches, cfg.d_lm)
    assert out["keep_mask_soft"].shape == (B, cfg.n_visual_patches)
    # E[K] should be inside [k_min, k_max] at init.
    ek = out["halting"]["expected_K"].item()
    assert cfg.k_min <= ek <= cfg.k_max
