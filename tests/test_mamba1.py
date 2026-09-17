"""Mamba-1: loop-vs-matrix, D_skip, prefill/step, gating theorem, gradcheck."""

import torch

from torchmamba.core.policy import compute_dtype_of
from torchmamba.mamba1.block import Mamba1Block, Mamba1Cache
from torchmamba.mamba1.core import selective_scan_loop, selective_scan_matrix

def _rand_scan(B=2, L=8, D=4, N=3, dtype=torch.float64):
    torch.manual_seed(0)
    x = torch.randn(B, L, D, dtype=dtype)
    dt = torch.rand(B, L, D, dtype=dtype) + 0.1
    A = -torch.exp(torch.randn(D, N, dtype=dtype))
    Bc = torch.randn(B, L, N, dtype=dtype)
    Cc = torch.randn(B, L, N, dtype=dtype)
    return x, dt, A, Bc, Cc


def test_loop_vs_matrix_fp64():
    x, dt, A, Bc, Cc = _rand_scan()
    D_skip = torch.randn(x.size(-1), dtype=torch.float64)
    y_loop, h_loop = selective_scan_loop(x, dt, A, Bc, Cc, D_skip)
    y_mat, h_mat = selective_scan_matrix(x, dt, A, Bc, Cc, D_skip)
    torch.testing.assert_close(y_loop, y_mat, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(h_loop, h_mat, rtol=1e-10, atol=1e-12)


def test_matrix_guard_and_empty():
    x, dt, A, Bc, Cc = _rand_scan(L=33)
    try:
        selective_scan_matrix(x, dt, A, Bc, Cc)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for L>32")
    xe, dte, Ae, Be, Ce = _rand_scan(L=0)
    y, h = selective_scan_loop(xe, dte, Ae, Be, Ce)
    assert y.shape == (xe.size(0), 0, xe.size(2))
    assert h.shape == (xe.size(0), xe.size(2), Ae.size(1))


def test_dskip_none_vs_zeros():
    x, dt, A, Bc, Cc = _rand_scan()
    y_none, _ = selective_scan_loop(x, dt, A, Bc, Cc, None)
    y_zero, _ = selective_scan_loop(
        x, dt, A, Bc, Cc, torch.zeros(x.size(-1), dtype=torch.float64)
    )
    torch.testing.assert_close(y_none, y_zero, rtol=0, atol=0)


def test_block_prefill_vs_step_fp32_and_fp64():
    for dtype, rtol, atol in (
        (torch.float32, 1e-5, 1e-6),
        (torch.float64, 1e-10, 1e-12),
    ):
        torch.manual_seed(7)
        blk = Mamba1Block(d_model=6, expand=2, d_state=4, d_conv=3).to(dtype)
        blk.eval()
        u = torch.randn(2, 9, 6, dtype=dtype)
        comp = compute_dtype_of(dtype)
        with torch.no_grad():
            y_full = blk(u)
            y_pre, _ = blk.prefill(u)
            torch.testing.assert_close(y_pre, y_full, rtol=rtol, atol=atol)
            zero = Mamba1Cache(
                conv_state=torch.zeros(2, blk.d_inner, blk.d_conv - 1, dtype=comp),
                ssm_state=torch.zeros(2, blk.d_inner, blk.d_state, dtype=comp),
            )
            outs, cache_t = [], zero
            for t in range(9):
                ot, cache_t = blk.step(u[:, t : t + 1, :], cache_t)
                outs.append(ot)
            torch.testing.assert_close(
                torch.cat(outs, dim=1), y_full, rtol=rtol, atol=atol
            )


def test_gating_theorem_spot_check():
    # N=1, A=-1, B=1. Decay part: exp(-softplus(pre)) == sigmoid(-pre) == 1-g.
    # EE input weight is dt=softplus(pre) (not g); g-weighting holds under ZOH
    # where (alpha-1)/A = 1-alpha = g. Both facts checked here.
    from torchmamba.core.discretize import zoh

    torch.manual_seed(0)
    B_, L, D = 1, 5, 3
    x = torch.randn(B_, L, D, dtype=torch.float64)
    lin_w = torch.randn(D, dtype=torch.float64)
    lin_b = torch.randn((), dtype=torch.float64)
    pre = x * lin_w + lin_b
    dt = torch.nn.functional.softplus(pre)
    A = -torch.ones(D, 1, dtype=torch.float64)
    Bone = torch.ones(B_, L, 1, dtype=torch.float64)
    C = torch.ones(B_, L, 1, dtype=torch.float64)
    y, _ = selective_scan_loop(x, dt, A, Bone, C, None)
    g = torch.sigmoid(pre)
    h = torch.zeros(B_, D, dtype=torch.float64)
    hs = []
    for t in range(L):
        h = (1 - g[:, t, :]) * h + dt[:, t, :] * x[:, t, :]
        hs.append(h.clone())
    torch.testing.assert_close(y, torch.stack(hs, dim=1), rtol=1e-10, atol=1e-12)
    # ZOH cross-check: B_disc coefficient equals g exactly.
    _, Bd = zoh(dt.unsqueeze(-1), A.unsqueeze(0), Bone.unsqueeze(-1))
    torch.testing.assert_close(
        Bd.squeeze(-1), g, rtol=1e-10, atol=1e-12
    )


def test_loop_gradcheck():
    torch.manual_seed(0)
    B_, L, D, N = 1, 3, 2, 2
    x = torch.randn(B_, L, D, dtype=torch.float64, requires_grad=True)
    dt = (torch.rand(B_, L, D, dtype=torch.float64) + 0.1).requires_grad_()
    A = (-torch.exp(torch.randn(D, N, dtype=torch.float64))).requires_grad_()
    Bc = torch.randn(B_, L, N, dtype=torch.float64, requires_grad=True)
    Cc = torch.randn(B_, L, N, dtype=torch.float64, requires_grad=True)

    def fn(x_, dt_, A_, B_, C_):
        out, _ = selective_scan_loop(x_, dt_, A_, B_, C_, None)
        return out

    assert torch.autograd.gradcheck(fn, (x, dt, A, Bc, Cc), eps=1e-6, atol=1e-4)
