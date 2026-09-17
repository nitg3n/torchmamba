"""Mamba-1 selective scan S6 (M1 Secs. 2-3; exponential-Euler per M3 Sec. 3.1).

Ground truth discretization is exponential-Euler (EE):

    h = exp(dt * A) * h_prev + (dt * B) * x,   y = sum_N(h * C) + D_skip * x

Reference clarity over speed: Python loop over L, fully differentiable,
no in-place writes to saved tensors. fp16/bf16/fp32 inputs compute in fp32,
fp64 stays fp64; data outputs return to input dtype, states stay >= fp32.
"""

import torch
from torch import Tensor

from torchmamba.core.discretize import exp_euler
from torchmamba.core.policy import compute_dtype_of, from_compute, to_compute


def selective_scan_loop(
    x: Tensor,
    dt: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D_skip: Tensor | None = None,
    h0: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Selective scan loop (EE), the M1 ground-truth oracle.

    Args:
        x: (B, L, D) inputs.
        dt: (B, L, D) step sizes (post-softplus).
        A: (D, N) continuous decays (data-independent).
        B: (B, L, N) input projections.
        C: (B, L, N) output projections.
        D_skip: (D,) or None (None == no skip).
        h0: (B, D, N) or None (None == zeros).

    Returns:
        (y: (B, L, D) in input dtype, h_last: (B, D, N) in compute dtype).
    """
    io_dtype = x.dtype
    comp = compute_dtype_of(io_dtype)
    xc, dtc, Ac, Bc, Cc = (to_compute(t) for t in (x, dt, A, B, C))
    Dc = to_compute(D_skip) if D_skip is not None else None
    B_, L, Dd = xc.shape
    N = Ac.shape[-1]
    h = (
        to_compute(h0).to(comp)
        if h0 is not None
        else torch.zeros(B_, Dd, N, dtype=comp, device=xc.device)
    )
    ys: list[Tensor] = []
    A4 = Ac.unsqueeze(0)  # (1, D, N)
    for t in range(L):
        xt = xc[:, t, :]  # (B, D)
        dtt = dtc[:, t, :]  # (B, D)
        Bt = Bc[:, t, :]  # (B, N)
        Ct = Cc[:, t, :]  # (B, N)
        alpha, _ = exp_euler(dtt.unsqueeze(-1), A4)  # (B, D, N)
        h = alpha * h + dtt.unsqueeze(-1) * Bt.unsqueeze(1) * xt.unsqueeze(-1)
        yt = (h * Ct.unsqueeze(1)).sum(-1)
        if Dc is not None:
            yt = yt + Dc.unsqueeze(0) * xt
        ys.append(yt)
    y = torch.stack(ys, dim=1) if L > 0 else xc.new_empty((B_, 0, Dd))
    return from_compute(y, io_dtype), h


def selective_scan_matrix(
    x: Tensor,
    dt: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D_skip: Tensor | None = None,
    h0: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Explicit (L, L, D) decay-times-CB mixer; quadratic oracle, L <= 32.

    Shares exp_euler with selective_scan_loop. h0 supported:
    h_t += (prod_{k<=t} alpha_k) * h0.
    """
    io_dtype = x.dtype
    comp = compute_dtype_of(io_dtype)
    xc, dtc, Ac, Bc, Cc = (to_compute(t) for t in (x, dt, A, B, C))
    Dc = to_compute(D_skip) if D_skip is not None else None
    B_, L, Dd = xc.shape
    if L > 32:
        raise ValueError(f"selective_scan_matrix only supports L<=32, got {L}")
    if L == 0:
        h = (
            to_compute(h0).to(comp)
            if h0 is not None
            else torch.zeros(B_, Dd, Ac.shape[-1], dtype=comp, device=xc.device)
        )
        return from_compute(xc.new_empty((B_, 0, Dd)), io_dtype), h
    A4 = Ac.view(1, 1, Dd, -1)  # (1, 1, D, N)
    alpha, _ = exp_euler(dtc.unsqueeze(-1), A4)  # (B, L, D, N)
    inp = dtc.unsqueeze(-1) * Bc.unsqueeze(2) * xc.unsqueeze(-1)  # (B, L, D, N)
    cum = alpha.cumprod(dim=1)  # prod_{k<=t} alpha_k
    ratio = cum.unsqueeze(2) / cum.unsqueeze(1)  # M[t,s] = cum[t]/cum[s]
    tril = torch.tril(
        torch.ones(L, L, device=xc.device, dtype=torch.bool)
    ).view(1, L, L, 1, 1)
    mix = ratio.masked_fill(~tril, 0.0)
    h = (mix * inp.unsqueeze(1)).sum(dim=2)  # (B, L, D, N)
    if h0 is not None:
        h = h + cum * to_compute(h0).to(comp).unsqueeze(1)
    y = (h * Cc.unsqueeze(2)).sum(-1)
    if Dc is not None:
        y = y + Dc.unsqueeze(0).unsqueeze(0) * xc
    return from_compute(y, io_dtype), h[:, -1, :]
