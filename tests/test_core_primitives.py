"""Core primitives: discretize, segsum, conv, norms, RoPE."""

import torch

from torchmamba.core.conv import causal_depthwise_conv1d
from torchmamba.core.discretize import exp_euler, softplus_dt, trap_coeffs, zoh
from torchmamba.core.norms import group_rms_norm, rms_norm
from torchmamba.core.rope import (
    accumulate_angles,
    apply_pairwise_rotation,
    pairwise_angles_to_cos_sin,
)
from torchmamba.core.segsum_util import segsum


def test_segsum_matches_naive_loop():
    torch.manual_seed(0)
    a = torch.randn(2, 3, 7)
    S = segsum(a)
    assert S.shape == (2, 3, 7, 7)
    # Upper triangle is -inf; diagonal is 0.
    assert torch.isinf(S[..., 0, 1]).all() and (S[..., 0, 1] < 0).all()
    assert torch.allclose(
        S.diagonal(dim1=-2, dim2=-1), torch.zeros(2, 3, 7), atol=0.0
    )
    # Lower triangle equals cumsum[i] - cumsum[j].
    c = a.cumsum(-1)
    for i in range(7):
        for j in range(i + 1):
            assert torch.allclose(S[..., i, j], c[..., i] - c[..., j])
    # T=0 and T=1 edges.
    assert segsum(torch.empty(2, 0)).shape == (2, 0, 0)
    assert torch.equal(segsum(torch.zeros(1, 1)), torch.zeros(1, 1, 1))


def test_trap_lam1_equals_exp_euler():
    torch.manual_seed(0)
    dt = torch.rand(2, 5, 4) + 0.1
    A = -torch.exp(torch.randn(4, 3))
    alpha_e, dt_out = exp_euler(dt.unsqueeze(-1), A.unsqueeze(0))
    alpha_t, beta_t, gamma_t = trap_coeffs(
        dt.unsqueeze(-1), A.unsqueeze(0), torch.ones(2, 5, 4, 1)
    )
    assert torch.equal(alpha_t, alpha_e)
    assert torch.equal(beta_t, torch.zeros_like(beta_t))
    assert torch.equal(gamma_t, dt.unsqueeze(-1))
    # lam=None path is the EE 2-term path.
    alpha_n, beta_n, gamma_n = trap_coeffs(dt.unsqueeze(-1), A.unsqueeze(0), None)
    assert torch.equal(alpha_n, alpha_e)
    assert torch.equal(beta_n, torch.zeros_like(beta_n))
    assert torch.equal(gamma_n, dt.unsqueeze(-1))
    assert torch.equal(dt_out, dt.unsqueeze(-1))


def test_zoh_small_branch_matches_series():
    torch.manual_seed(0)
    dt = torch.full((3, 4), 1e-6)
    A = torch.full((4,), -2.0)
    B = torch.ones(3, 4)
    alpha, Bd = zoh(dt, A, B)
    # Small |dt*A|: Bd ~= dt (series) to 1e-6 relative.
    assert torch.allclose(Bd, dt, rtol=1e-6, atol=1e-12)
    assert torch.allclose(alpha, torch.exp(dt * A))
    # A == 0 is finite (no NaN).
    _, Bd0 = zoh(
        torch.ones(2, 2) * 0.5, torch.zeros(2, 2), torch.ones(2, 2)
    )
    assert torch.isfinite(Bd0).all()
    assert torch.allclose(Bd0, torch.full((2, 2), 0.5))


def test_softplus_dt():
    p = torch.tensor([0.0, 1.0])
    b = torch.tensor([0.5])
    assert torch.allclose(softplus_dt(p, b), torch.nn.functional.softplus(p + b))


def test_causal_conv_matches_sliding_window():
    torch.manual_seed(0)
    B, D, L, K = 2, 3, 7, 4
    x = torch.randn(B, D, L)
    w = torch.randn(D, 1, K)
    b = torch.randn(D)
    y = causal_depthwise_conv1d(x, w, b)
    assert y.shape == (B, D, L)
    for t in range(L):
        lo = max(0, t - K + 1)
        seg = x[:, :, lo : t + 1]
        kval = w[:, 0, -(t - lo + 1) :]
        expect = (seg * kval.unsqueeze(0)).sum(-1) + b
        assert torch.allclose(y[:, :, t], expect, atol=1e-6)
    # K=1 reduces to per-step scale.
    w1 = torch.randn(D, 1, 1)
    assert torch.allclose(
        causal_depthwise_conv1d(x, w1, b),
        x * w1.squeeze(-1).squeeze(-1).view(1, D, 1) + b.view(1, D, 1),
    )


def test_norms():
    torch.manual_seed(0)
    x = torch.randn(2, 8)
    w = torch.rand(8) + 0.5
    y = rms_norm(x, w)
    expect = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w
    assert torch.allclose(y, expect)
    yg = group_rms_norm(x, 2, w)
    # GroupRMS with 1 group equals RMS.
    assert torch.allclose(group_rms_norm(x, 1, w), y)
    assert yg.shape == x.shape
    try:
        group_rms_norm(x, 3)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for 8 % 3")


def test_rope_composition_and_accumulation():
    torch.manual_seed(0)
    a = torch.rand(2, 5, 3, 4) * 0.7
    b = torch.rand(2, 5, 3, 4) * 0.7
    x = torch.randn(2, 5, 3, 8)
    acc = accumulate_angles(a) + accumulate_angles(b)
    both = accumulate_angles(a + b)
    assert torch.allclose(acc, both)
    # apply(R(a+b)) == apply(R(b)) o apply(R(a)).
    cos_a, sin_a = pairwise_angles_to_cos_sin(accumulate_angles(a))
    cos_b, sin_b = pairwise_angles_to_cos_sin(accumulate_angles(b))
    cos_ab, sin_ab = pairwise_angles_to_cos_sin(both)
    seq = apply_pairwise_rotation(apply_pairwise_rotation(x, cos_a, sin_a), cos_b, sin_b)
    direct = apply_pairwise_rotation(x, cos_ab, sin_ab)
    assert torch.allclose(seq, direct, atol=1e-6)
    # Rotation preserves norm.
    assert torch.allclose(direct.norm(dim=-1), x.norm(dim=-1), atol=1e-5)
    # Odd N raises.
    try:
        apply_pairwise_rotation(torch.randn(2, 3), torch.ones(2, 1), torch.ones(2, 1))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for odd N")
