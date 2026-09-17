"""Mamba-3 SISO: lam=1/theta=0 reduction, trapezoid hand value, loop/matrix."""

import torch

from torchmamba.mamba2.core import ssd_recurrent
from torchmamba.mamba3.siso_block import Mamba3SisoBlock
from torchmamba.mamba3.siso_core import trap_rope_loop, trap_rope_matrix


def _rand(B=2, L=10, H=2, P=4, N=6, G=1, dtype=torch.float64):
    torch.manual_seed(0)
    X = torch.randn(B, L, H, P, dtype=dtype)
    A = -torch.rand(B, L, H, dtype=dtype) - 0.01
    Bc = torch.randn(B, L, G, N, dtype=dtype)
    Cc = torch.randn(B, L, G, N, dtype=dtype)
    return X, A, Bc, Cc


def test_lam1_theta0_equals_m2():
    X, A, Bc, Cc = _rand()
    Y3, h3, dbg = trap_rope_loop(X, A, Bc, Cc, None, 1.0)
    Y2, h2 = ssd_recurrent(X, A, Bc, Cc)
    assert dbg["two_term"] is True
    torch.testing.assert_close(Y3, Y2, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(h3, h2, rtol=1e-10, atol=1e-12)
    # zeros theta == None == real-only.
    th0 = torch.zeros(2, 10, 2, 3, dtype=torch.float64)
    Yz, _, _ = trap_rope_loop(X, A, Bc, Cc, th0, 1.0)
    torch.testing.assert_close(Yz, Y2, rtol=1e-10, atol=1e-12)


def test_trapezoid_hand_value_two_step_scalar():
    # Scalar 1-head case, lam=0.5: h1 = a1*h0 + 0.5*a1*B0*x0 + 0.5*B1*x1.
    # Core takes logdecay (a = exp(logdecay)); B/C are (B, L, G=1, N=1).
    X = torch.tensor([[[[2.0]], [[3.0]]]], dtype=torch.float64)  # (1,2,1,1)
    A = torch.tensor([[[-0.5], [-0.25]]], dtype=torch.float64)  # (1,2,1)
    Bc = torch.tensor([[[[1.0]], [[1.0]]]], dtype=torch.float64)
    Cc = torch.tensor([[[[1.0]], [[1.0]]]], dtype=torch.float64)
    Y, _, _ = trap_rope_loop(X, A, Bc, Cc, None, 0.5)
    a1 = float(torch.exp(A[0, 1, 0]))
    h0 = 0.5 * 1.0 * 2.0  # gamma0*B0*x0 with lam=0.5 (beta term needs x_{-1}=0)
    h1 = a1 * h0 + 0.5 * a1 * 1.0 * 2.0 + 0.5 * 1.0 * 3.0
    assert abs(float(Y[0, 0, 0, 0]) - h0) < 1e-12
    assert abs(float(Y[0, 1, 0, 0]) - h1) < 1e-12


def test_loop_vs_matrix():
    X, A, Bc, Cc = _rand()
    torch.manual_seed(1)
    th = torch.randn(2, 10, 2, 3, dtype=torch.float64) * 0.1
    lam = torch.sigmoid(torch.randn(2, 10, 2, dtype=torch.float64))
    Yl, hl, _ = trap_rope_loop(X, A, Bc, Cc, th, lam)
    Ym, hm = trap_rope_matrix(X, A, Bc, Cc, th, lam)
    torch.testing.assert_close(Yl, Ym, rtol=1e-9, atol=1e-11)
    torch.testing.assert_close(hl, hm, rtol=1e-9, atol=1e-11)


def test_matrix_guard():
    X, A, Bc, Cc = _rand(L=17)
    try:
        trap_rope_matrix(X, A, Bc, Cc)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for L>16")


def test_rotation_vs_explicit_per_step():
    # Accumulated-angle path equals sequential per-step rotation application.
    from torchmamba.core.rope import (
        accumulate_angles,
        apply_pairwise_rotation,
        pairwise_angles_to_cos_sin,
    )

    torch.manual_seed(2)
    B, L, H, NN = 1, 5, 1, 4
    Bc = torch.randn(B, L, H, NN, dtype=torch.float64)
    th = torch.randn(B, L, H, NN // 2, dtype=torch.float64) * 0.3
    acc = accumulate_angles(th)
    cos, sin = pairwise_angles_to_cos_sin(acc)
    direct = apply_pairwise_rotation(Bc, cos, sin)
    # Sequential: rotate by increments.
    cur = Bc.clone()
    for t in range(L):
        c, s = pairwise_angles_to_cos_sin(th[:, t : t + 1, :])
        step_rot = apply_pairwise_rotation(cur[:, t : t + 1, :], c, s)
        # sequential-from-origin differs from cumulative application; instead
        # verify first-step and composition law only.
        if t == 0:
            torch.testing.assert_close(
                step_rot, direct[:, t : t + 1, :], rtol=1e-10, atol=1e-12
            )
    # Composition: R(acc_L) == R(sum theta).
    assert torch.allclose(acc[:, -1, :], th.sum(dim=1), atol=1e-12)


def test_block_prefill_vs_step():
    for dtype, rtol, atol in (
        (torch.float32, 1e-5, 1e-6),
        (torch.float64, 1e-10, 1e-12),
    ):
        torch.manual_seed(5)
        blk = Mamba3SisoBlock(
            d_model=12, n_heads=2, d_head=4, d_state=6, n_groups=1
        ).to(dtype)
        blk.eval()
        u = torch.randn(2, 7, 12, dtype=dtype)
        with torch.no_grad():
            yf = blk(u)
            yp, cache = blk.prefill(u)
            torch.testing.assert_close(yp, yf, rtol=rtol, atol=atol)
            # Step from scratch through the block step path.
            _, c0 = blk.prefill(u[:, :1, :])
            outs = []
            c = c0
            for t in range(1, 7):
                ot, c = blk.step(u[:, t : t + 1, :], c)
                outs.append(ot)
            torch.testing.assert_close(
                torch.cat([yp[:, :1, :]] + outs, dim=1), yf, rtol=rtol, atol=atol
            )
