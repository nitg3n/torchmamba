"""Mamba-3 MIMO core: rank-R matmul update on shared (N, P) state.

Per-head update (M3 Sec. 3.3, Table 2): H += B @ X with
B: (N, R), X: (P, R) -> einsum over R; Y = C^T H per r.
Trap/RoPE apply per-rank exactly as SISO. R=1 squeezes to SISO.
"""

import torch
from torch import Tensor

from torchmamba.core.policy import compute_dtype_of, from_compute, to_compute
from torchmamba.mamba3.siso_core import _expand_groups, _is_one, _rotated_BC


def _check_mimo(
    X: Tensor, logdecay: Tensor, B: Tensor, C: Tensor, theta_dt: Tensor | None
):
    Bx, Lx, Hx, Px, Rx = X.shape
    assert logdecay.shape == (Bx, Lx, Hx)
    assert B.shape[:2] == (Bx, Lx) and C.shape[:2] == (Bx, Lx)
    assert B.shape == C.shape, "B/C shape mismatch"
    assert B.size(-1) == Rx, "B/C rank must match X rank"
    assert Hx % B.size(2) == 0
    if theta_dt is not None:
        assert theta_dt.shape[:3] == (Bx, Lx, Hx)
        assert theta_dt.size(-1) * 2 == B.size(-2)
    return Bx, Lx, Hx, Px, Rx


def _as_lam(lam, like: Tensor) -> Tensor:
    from torch import Tensor as T

    import torch

    if isinstance(lam, T):
        return to_compute(lam).to(like.dtype)
    return torch.as_tensor(float(lam), dtype=like.dtype, device=like.device)


def _rotated_mimo(
    Bc: Tensor, Cc: Tensor, th, Bx: int, L: int, H: int, R: int, N: int
):
    """Rotate per (rank, n-pair): fold R into the head axis, rotate, unfold."""
    Bc_pairs = Bc.permute(0, 1, 2, 4, 3).reshape(Bx, L, H * R, N)
    Cc_pairs = Cc.permute(0, 1, 2, 4, 3).reshape(Bx, L, H * R, N)
    th_pairs = th.repeat_interleave(R, dim=2) if th is not None else None
    Bc_r, Cc_r = _rotated_BC(Bc_pairs, Cc_pairs, th_pairs)
    Bc_r = Bc_r.view(Bx, L, H, R, N).permute(0, 1, 2, 4, 3)
    Cc_r = Cc_r.view(Bx, L, H, R, N).permute(0, 1, 2, 4, 3)
    return Bc_r, Cc_r


def _mimo_scan(
    Xc: Tensor,
    a: Tensor,
    Bc_r: Tensor,
    Cc_r: Tensor,
    lamc,
    two_term: bool,
    h_init: Tensor,
    xb_init: tuple[Tensor, Tensor] | None = None,
) -> tuple[Tensor, Tensor]:
    """Shared seeded scan used by loop/chunk/seeded paths (compute dtype).

    xb_init carries (x_prev, b_prev) for the beta term across chunk
    boundaries; None (loop path) means sequence-start zeros.
    """
    import torch

    Bx, L, H, P, R = Xc.shape
    N = Bc_r.size(-2)
    h = h_init
    ys: list[Tensor] = []
    if xb_init is None:
        x_prev = torch.zeros(Bx, H, P, R, dtype=Xc.dtype, device=Xc.device)
        b_prev = torch.zeros(Bx, H, N, R, dtype=Xc.dtype, device=Xc.device)
    else:
        x_prev, b_prev = xb_init
    for t in range(L):
        at = a[:, t, :].view(Bx, H, 1, 1)
        add_now = torch.einsum("bhnr,bhpr->bhnp", Bc_r[:, t, :], Xc[:, t, :])
        if two_term:
            h = at * h + add_now
        else:
            lt = lamc if lamc.dim() == 0 else lamc[:, t, :]
            lv = lt.view(Bx, H, 1, 1) if lt.dim() > 0 else lt
            h = (
                at * h
                + (1 - lv) * at * torch.einsum("bhnr,bhpr->bhnp", b_prev, x_prev)
                + lv * add_now
            )
        ys.append(torch.einsum("bhnr,bhnp->bhpr", Cc_r[:, t, :], h))
        x_prev = Xc[:, t, :]
        b_prev = Bc_r[:, t, :]
    Y = torch.stack(ys, dim=1) if L > 0 else Xc.new_empty((Bx, 0, H, P, R))
    return Y, h


def _prepare(X, logdecay, B, C, theta_dt):
    Bx, L, H, P, R = _check_mimo(X, logdecay, B, C, theta_dt)
    Xc, Ac, Bc, Cc = (to_compute(t) for t in (X, logdecay, B, C))
    Bc, Cc = _expand_groups_mimo(Bc, Cc, H)
    comp = Xc.dtype
    N = Bc.size(-2)
    th = to_compute(theta_dt).to(comp) if theta_dt is not None else None
    Bc_r, Cc_r = _rotated_mimo(Bc, Cc, th, Bx, L, H, R, N)
    return (Bx, L, H, P, R, N), Xc, Ac, Bc_r, Cc_r


def _expand_groups_mimo(B: Tensor, C: Tensor, H: int):
    if B.size(2) == H:
        return B, C
    rep = H // B.size(2)
    return B.repeat_interleave(rep, dim=2), C.repeat_interleave(rep, dim=2)


def mimo_loop(
    X: Tensor,
    logdecay: Tensor,
    B: Tensor,
    C: Tensor,
    theta_dt: Tensor | None = None,
    lam=1.0,
) -> tuple[Tensor, Tensor]:
    """Fused MIMO loop; shared state (B, H, N, P), outputs (B, L, H, P, R)."""
    import torch

    if X.size(-1) == 1:
        # Structural R=1 reduction: squeeze, run SISO, unsqueeze.
        from torchmamba.mamba3.siso_core import trap_rope_loop

        Y, h, _ = trap_rope_loop(
            X.squeeze(-1), logdecay, B.squeeze(-1), C.squeeze(-1), theta_dt, lam
        )
        return Y.unsqueeze(-1), h
    io_dtype = X.dtype
    (Bx, L, H, P, R, N), Xc, Ac, Bc_r, Cc_r = _prepare(X, logdecay, B, C, theta_dt)
    a = torch.exp(Ac)
    two_term = _is_one(lam)
    lamc = None if two_term else _as_lam(lam, Xc)
    h0 = torch.zeros(Bx, H, N, P, dtype=Xc.dtype, device=Xc.device)
    Y, h = _mimo_scan(Xc, a, Bc_r, Cc_r, lamc, two_term, h0)
    return from_compute(Y, io_dtype), h


def mimo_chunkwise(
    X: Tensor,
    logdecay: Tensor,
    B: Tensor,
    C: Tensor,
    theta_dt: Tensor | None = None,
    lam=1.0,
    chunk_len: int | None = None,
    siso_block_len: int = 64,
) -> tuple[Tensor, Tensor]:
    """Chunked MIMO: chunk_len defaults to max(1, siso_block_len // R)."""
    import torch

    R = X.size(-1)
    if chunk_len is None:
        chunk_len = max(1, siso_block_len // R)
    io_dtype = X.dtype
    (Bx, L, H, P, _R, N), Xc, Ac, Bc_r, Cc_r = _prepare(X, logdecay, B, C, theta_dt)
    comp = Xc.dtype
    a = torch.exp(Ac)
    two_term = _is_one(lam)
    lamc_full = None if two_term else _as_lam(lam, Xc)
    h = torch.zeros(Bx, H, N, P, dtype=comp, device=Xc.device)
    Rr = Xc.size(-1)
    Nr = Bc_r.size(-2)
    x_prev = torch.zeros(Bx, H, P, Rr, dtype=comp, device=Xc.device)
    b_prev = torch.zeros(Bx, H, Nr, Rr, dtype=comp, device=Xc.device)
    outs: list[Tensor] = []
    for s in range(0, L, chunk_len):
        e = min(s + chunk_len, L)
        lamc = None if two_term else _slice_lam(lamc_full, s, e)
        Yc, h = _mimo_scan(
            Xc[:, s:e, :], a[:, s:e, :], Bc_r[:, s:e, :], Cc_r[:, s:e, :],
            lamc, two_term, h, (x_prev, b_prev),
        )
        # Carry last rotated (x, B) of this chunk into the next chunk's beta.
        x_prev = Xc[:, e - 1, :]
        b_prev = Bc_r[:, e - 1, :]
        outs.append(Yc)
    Y = torch.cat(outs, dim=1) if outs else Xc.new_empty((Bx, 0, H, P, R))
    return from_compute(Y, io_dtype), h


def _slice_lam(lamc, s: int, e: int):
    if lamc.dim() == 0:
        return lamc
    return lamc[s:e] if lamc.dim() == 1 else lamc[:, s:e, :]


def mimo_as_r2_sisos(
    X: Tensor,
    logdecay: Tensor,
    B: Tensor,
    C: Tensor,
) -> Tensor:
    """Test-only: R^2 SISO runs summed per (M3-12)-(M3-14); Euler, no RoPE."""
    from torchmamba.mamba3.siso_core import trap_rope_loop as siso_loop

    Bx, L, H, P, R = _check_mimo(X, logdecay, B, C, None)
    Y = torch.zeros(Bx, L, H, P, R, dtype=X.dtype)
    for i in range(R):
        acc = None
        for j in range(R):
            # Shared alpha => sum_j SSM(x_j) == SSM with summed (B_j x_j);
            # run SISO per j with (X_j, B_j) read by C_i, sum over j.
            Yij, _ = siso_loop(X[..., j], logdecay, B[..., j], C[..., i], None, 1.0)[:2]
            acc = Yij if acc is None else acc + Yij
        Y[..., i] = acc
    return Y
