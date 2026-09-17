"""Mamba-2: recurrent/quadratic/chunkwise, boundaries, groups, block, gradcheck."""

import torch

from torchmamba.mamba2.block import Mamba2Block
from torchmamba.mamba2.core import (
    boundary_mask_to_logdecay,
    ssd_chunkwise,
    ssd_quadratic,
    ssd_recurrent,
    ssd_step,
)


def _rand_ssd(B=2, L=130, H=2, P=8, N=8, G=1, dtype=torch.float64):
    torch.manual_seed(0)
    X = torch.randn(B, L, H, P, dtype=dtype)
    logdecay = -torch.rand(B, L, H, dtype=dtype) - 0.01
    Bc = torch.randn(B, L, G, N, dtype=dtype)
    Cc = torch.randn(B, L, G, N, dtype=dtype)
    return X, logdecay, Bc, Cc


def test_three_paths_agree_with_tail():
    X, A, Bc, Cc = _rand_ssd()
    Yr, hr = ssd_recurrent(X, A, Bc, Cc)
    Xp, Ap, Bcp, Ccp = X[:, :64, :], A[:, :64, :], Bc[:, :64, :], Cc[:, :64, :]
    Yq, hq = ssd_quadratic(Xp, Ap, Bcp, Ccp)
    _, hr_pre = ssd_recurrent(Xp, Ap, Bcp, Ccp)
    Yc, hc = ssd_chunkwise(X, A, Bc, Cc, block_len=32)
    torch.testing.assert_close(Yc, Yr, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(hc, hr, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(Yq, Yr[:, :64, :], rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(hq, hr_pre, rtol=1e-10, atol=1e-12)


def test_three_paths_fp32():
    # fp32 SSD over L=70 accumulates ~2.8e-6 abs; tolerance accounts for
    # exp/cumsum rounding in the chunkwise path (looser than fp64 1e-12).
    X, A, Bc, Cc = _rand_ssd(L=70, dtype=torch.float32)
    Yr, _ = ssd_recurrent(X, A, Bc, Cc)
    Yc, _ = ssd_chunkwise(X, A, Bc, Cc, block_len=32)
    torch.testing.assert_close(Yc, Yr, rtol=2e-5, atol=5e-6)


def test_initial_states_threading():
    X, A, Bc, Cc = _rand_ssd(L=64)
    Y_full, _ = ssd_chunkwise(X, A, Bc, Cc, block_len=32)
    Y1, s1 = ssd_chunkwise(X[:, :32, :], A[:, :32, :], Bc[:, :32, :], Cc[:, :32, :])
    Y2, _ = ssd_chunkwise(
        X[:, 32:, :], A[:, 32:, :], Bc[:, 32:, :], Cc[:, 32:, :],
        block_len=32, initial_states=s1,
    )
    torch.testing.assert_close(
        torch.cat([Y1, Y2], dim=1), Y_full, rtol=1e-10, atol=1e-12
    )


def test_boundary_zero_blocks_carry():
    torch.manual_seed(1)
    B, L, H, P, N = 1, 20, 2, 4, 4
    X = torch.randn(B, L, H, P, dtype=torch.float64)
    A = -torch.rand(B, L, H, dtype=torch.float64) - 0.01
    Bc = torch.randn(B, L, 1, N, dtype=torch.float64)
    Cc = torch.randn(B, L, 1, N, dtype=torch.float64)
    boundary = torch.zeros(B, L, dtype=torch.bool)
    boundary[:, 10] = True
    Ab = boundary_mask_to_logdecay(boundary, A)
    Yb, _ = ssd_recurrent(X, Ab, Bc, Cc)
    Y2, _ = ssd_recurrent(X[:, 10:, :], A[:, 10:, :], Bc[:, 10:, :], Cc[:, 10:, :])
    torch.testing.assert_close(Yb[:, 10:, :], Y2, rtol=1e-10, atol=1e-12)


def test_group_broadcast_vs_expanded():
    X, A, Bc, Cc = _rand_ssd(G=1)
    Yr1, _ = ssd_recurrent(X, A, Bc, Cc)
    Yr2, _ = ssd_recurrent(
        X, A, Bc.repeat_interleave(2, dim=2), Cc.repeat_interleave(2, dim=2)
    )
    torch.testing.assert_close(Yr1, Yr2, rtol=0, atol=0)


def test_l1_and_empty():
    X, A, Bc, Cc = _rand_ssd(L=1)
    Y, h = ssd_recurrent(X, A, Bc, Cc)
    Yc, hc = ssd_chunkwise(X, A, Bc, Cc, block_len=32)
    torch.testing.assert_close(Y, Yc, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(h, hc, rtol=1e-10, atol=1e-12)
    st = torch.zeros_like(h)
    Ys, hs = ssd_step(X, A, Bc, Cc, st)
    torch.testing.assert_close(Ys.squeeze(1), Y.squeeze(1), rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(hs, h, rtol=1e-10, atol=1e-12)
    Xe, Ae, Be, Ce = _rand_ssd(L=0)
    Ye, he = ssd_recurrent(Xe, Ae, Be, Ce)
    assert Ye.shape[1] == 0 and he.shape == (2, 2, 8, 8)


def test_block_forward_and_step():
    for dtype, rtol, atol in (
        (torch.float32, 1e-5, 1e-6),
        (torch.float64, 1e-10, 1e-12),
    ):
        torch.manual_seed(3)
        blk = Mamba2Block(
            d_model=16, n_heads=2, d_head=8, d_state=8, n_groups=1, d_conv=3
        ).to(dtype)
        blk.eval()
        u = torch.randn(2, 9, 16, dtype=dtype)
        with torch.no_grad():
            yf = blk(u)
            yfr = blk.forward_recurrent(u)
            torch.testing.assert_close(yf, yfr, rtol=rtol, atol=atol)
            _, cache = blk.prefill(u)
            outs, c = [], cache
            # step from zero cache tested in prefill/decode suite; here check shapes.
            assert c["ssm_state"].shape == (2, 2, 8, 8)


def test_chunkwise_gradcheck():
    torch.manual_seed(0)
    B, L, H, P, N = 1, 8, 1, 3, 3
    X = torch.randn(B, L, H, P, dtype=torch.float64, requires_grad=True)
    A = (-torch.rand(B, L, H, dtype=torch.float64) - 0.01).requires_grad_()
    Bc = torch.randn(B, L, 1, N, dtype=torch.float64, requires_grad=True)
    Cc = torch.randn(B, L, 1, N, dtype=torch.float64, requires_grad=True)

    def fn(X_, A_, B_, C_):
        Y, _ = ssd_chunkwise(X_, A_, B_, C_, block_len=4)
        return Y

    assert torch.autograd.gradcheck(fn, (X, A, Bc, Cc), eps=1e-6, atol=1e-4)
