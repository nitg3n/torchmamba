"""Mamba-3 MIMO: R=1 reduction, R^2 SISOs, chunkwise, prefill/step, gradcheck."""

import torch

from torchmamba.mamba3.mimo_block import Mamba3MimoBlock
from torchmamba.mamba3.mimo_core import mimo_as_r2_sisos, mimo_chunkwise, mimo_loop
from torchmamba.mamba3.siso_core import trap_rope_loop


def _rand(B=2, L=24, H=2, P=4, N=6, G=1, R=2, dtype=torch.float64):
    torch.manual_seed(0)
    X = torch.randn(B, L, H, P, R, dtype=dtype)
    A = -torch.rand(B, L, H, dtype=dtype) - 0.01
    Bc = torch.randn(B, L, G, N, R, dtype=dtype)
    Cc = torch.randn(B, L, G, N, R, dtype=dtype)
    return X, A, Bc, Cc


def test_r1_squeezed_equals_siso():
    X, A, Bc, Cc = _rand(R=1)
    Ym, hm = mimo_loop(X, A, Bc, Cc, None, 1.0)
    Ys, hs, _ = trap_rope_loop(
        X.squeeze(-1), A, Bc.squeeze(-1), Cc.squeeze(-1), None, 1.0
    )
    torch.testing.assert_close(Ym.squeeze(-1), Ys, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(hm, hs, rtol=1e-10, atol=1e-12)


def test_fused_loop_equals_r2_sisos():
    X, A, Bc, Cc = _rand()
    Yf, _ = mimo_loop(X, A, Bc, Cc, None, 1.0)
    Yr2 = mimo_as_r2_sisos(X, A, Bc, Cc)
    torch.testing.assert_close(Yf, Yr2, rtol=1e-10, atol=1e-12)


def test_chunkwise_vs_loop_with_tail():
    X, A, Bc, Cc = _rand(L=70, R=4)
    Yl, hl = mimo_loop(X, A, Bc, Cc, None, 1.0)
    Yc, hc = mimo_chunkwise(X, A, Bc, Cc, None, 1.0, chunk_len=16)
    torch.testing.assert_close(Yc, Yl, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(hc, hl, rtol=1e-10, atol=1e-12)
    # Default CR rule chunk (64//4=16) matches.
    Yd, _ = mimo_chunkwise(X, A, Bc, Cc, None, 1.0)
    torch.testing.assert_close(Yd, Yl, rtol=1e-10, atol=1e-12)


def test_chunkwise_with_trap():
    X, A, Bc, Cc = _rand(L=20, R=2)
    lam = torch.sigmoid(torch.randn(2, 20, 2, dtype=torch.float64))
    Yl, _ = mimo_loop(X, A, Bc, Cc, None, lam)
    Yc, _ = mimo_chunkwise(X, A, Bc, Cc, None, lam, chunk_len=7)
    torch.testing.assert_close(Yc, Yl, rtol=1e-10, atol=1e-12)


def test_block_prefill_vs_step():
    for dtype, rtol, atol in (
        (torch.float32, 1e-5, 1e-6),
        (torch.float64, 1e-10, 1e-12),
    ):
        torch.manual_seed(11)
        blk = Mamba3MimoBlock(
            d_model=12, n_heads=2, d_head=4, d_state=6, n_groups=1, mimo_rank=2
        ).to(dtype)
        blk.eval()
        u = torch.randn(2, 7, 12, dtype=dtype)
        with torch.no_grad():
            yf = blk(u)
            yfc = blk.forward_chunkwise(u)
            torch.testing.assert_close(yfc, yf, rtol=rtol, atol=atol)
            yp, state = blk.prefill(u)
            torch.testing.assert_close(yp, yf, rtol=rtol, atol=atol)
            _, s0 = blk.prefill(u[:, :1, :])
            outs, st = [], s0
            for t in range(1, 7):
                ot, st = blk.step(u[:, t : t + 1, :], st)
                outs.append(ot)
            torch.testing.assert_close(
                torch.cat([yp[:, :1, :]] + outs, dim=1), yf, rtol=rtol, atol=atol
            )


def test_mimo_gradcheck():
    torch.manual_seed(0)
    B, L, H, P, N, R = 1, 3, 1, 2, 2, 2
    X = torch.randn(B, L, H, P, R, dtype=torch.float64, requires_grad=True)
    A = (-torch.rand(B, L, H, dtype=torch.float64) - 0.01).requires_grad_()
    Bc = torch.randn(B, L, 1, N, R, dtype=torch.float64, requires_grad=True)
    Cc = torch.randn(B, L, 1, N, R, dtype=torch.float64, requires_grad=True)

    def fn(X_, A_, B_, C_):
        Y, _ = mimo_loop(X_, A_, B_, C_, None, 1.0)
        return Y

    assert torch.autograd.gradcheck(fn, (X, A, Bc, Cc), eps=1e-6, atol=1e-4)
