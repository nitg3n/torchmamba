"""Mamba-3 SISO core: exponential-trapezoidal + RoPE trick.

Shipped recurrence (M3 Prop. 1 Eqs. 5-6 + Prop. 4 Eq. 11), in logdecay units where the
M2 core already folded dt into B (i.e. gamma = lam, beta = (1-lam)*alpha):

    h = alpha h + beta B_prev x_prev + gamma B x,   y = C . h

(alpha, beta, gamma) share trap_coeffs semantics; (B, C) are rotated by
accumulated angles. lam=1 dispatches to the exact M2 2-term path
(no B_prev contribution). theta=None/zeros equals the real-only path.
"""

import torch
from torch import Tensor

from torchmamba.core.policy import compute_dtype_of, from_compute, to_compute
from torchmamba.core.rope import (
    accumulate_angles,
    apply_pairwise_rotation,
    pairwise_angles_to_cos_sin,
)
from torchmamba.core.segsum_util import segsum


def _check(
    X: Tensor, logdecay: Tensor, B: Tensor, C: Tensor, theta_dt: Tensor | None
):
    Bx, Lx, Hx, Px = X.shape
    assert logdecay.shape == (Bx, Lx, Hx), "logdecay must be (B, L, H)"
    assert B.shape[:2] == (Bx, Lx) and C.shape[:2] == (Bx, Lx), "length mismatch"
    assert B.shape == C.shape, "B/C shape mismatch"
    assert Hx % B.size(2) == 0, "groups must divide heads"
    if theta_dt is not None:
        assert theta_dt.shape[:3] == (Bx, Lx, Hx), "theta_dt must be (B, L, H, N/2)"
        assert theta_dt.size(-1) * 2 == B.size(-1), "theta pairs must match N"
        if B.size(-1) % 2 != 0:
            raise ValueError("N must be even when theta_dt is given")
    return Bx, Lx, Hx, Px


def _expand_groups(B: Tensor, C: Tensor, H: int) -> tuple[Tensor, Tensor]:
    if B.size(2) == H:
        return B, C
    rep = H // B.size(2)
    return B.repeat_interleave(rep, dim=2), C.repeat_interleave(rep, dim=2)


def _is_one(lam) -> bool:
    if isinstance(lam, Tensor):
        return bool(torch.equal(lam, torch.ones_like(lam)))
    return float(lam) == 1.0


def _as_lam(lam, like: Tensor) -> Tensor:
    if isinstance(lam, Tensor):
        return to_compute(lam).to(like.dtype)
    return torch.as_tensor(float(lam), dtype=like.dtype, device=like.device)


def _rotated_BC(
    Bc: Tensor, Cc: Tensor, theta_dt: Tensor | None
) -> tuple[Tensor, Tensor]:
    """Rotate (B, C) pairs by accumulated angles; identity when None/all-zero."""
    if theta_dt is None or not bool((theta_dt != 0).any()):
        return Bc, Cc
    angles = accumulate_angles(theta_dt)  # (B, L, H, N/2)
    cos, sin = pairwise_angles_to_cos_sin(angles)
    return (
        apply_pairwise_rotation(Bc, cos, sin),
        apply_pairwise_rotation(Cc, cos, sin),
    )


def trap_rope_loop(
    X: Tensor,
    logdecay: Tensor,
    B: Tensor,
    C: Tensor,
    theta_dt: Tensor | None = None,
    lam=1.0,
) -> tuple[Tensor, Tensor, dict]:
    """3-term trap+RoPE loop; lam=1 == exact M2 2-term path."""
    io_dtype = X.dtype
    Bx, L, H, P = _check(X, logdecay, B, C, theta_dt)
    Xc, Ac, Bc, Cc = (to_compute(t) for t in (X, logdecay, B, C))
    Bc, Cc = _expand_groups(Bc, Cc, H)
    comp = Xc.dtype
    N = Bc.size(-1)
    th = to_compute(theta_dt).to(comp) if theta_dt is not None else None
    Bcr, Ccr = _rotated_BC(Bc, Cc, th)
    a = torch.exp(Ac)  # (B, L, H)
    two_term = _is_one(lam)
    lamc = None if two_term else _as_lam(lam, Xc)
    h = torch.zeros(Bx, H, N, P, dtype=comp, device=Xc.device)
    ys: list[Tensor] = []
    x_prev = torch.zeros(Bx, H, P, dtype=comp, device=Xc.device)
    b_prev = torch.zeros(Bx, H, N, dtype=comp, device=Xc.device)
    for t in range(L):
        at = a[:, t, :].view(Bx, H, 1, 1)
        if two_term:
            h = at * h + torch.einsum("bhn,bhp->bhnp", Bcr[:, t, :], Xc[:, t, :])
        else:
            lt = lamc if lamc.dim() == 0 else lamc[:, t, :]
            lv = lt.view(Bx, H, 1, 1) if lt.dim() > 0 else lt
            h = (
                at * h
                + (1 - lv) * at * torch.einsum("bhn,bhp->bhnp", b_prev, x_prev)
                + lv * torch.einsum("bhn,bhp->bhnp", Bcr[:, t, :], Xc[:, t, :])
            )
        ys.append(torch.einsum("bhn,bhnp->bhp", Ccr[:, t, :], h))
        x_prev = Xc[:, t, :]
        b_prev = Bcr[:, t, :]
    Y = torch.stack(ys, dim=1) if L > 0 else Xc.new_empty((Bx, 0, H, P))
    return from_compute(Y, io_dtype), h, {"two_term": two_term}


def trap_rope_matrix(
    X: Tensor,
    logdecay: Tensor,
    B: Tensor,
    C: Tensor,
    theta_dt: Tensor | None = None,
    lam=1.0,
) -> tuple[Tensor, Tensor]:
    """Explicit (M3-7) mask: (1-SS o 2-band) times CB^T. L <= 16 guard."""
    io_dtype = X.dtype
    Bx, L, H, P = _check(X, logdecay, B, C, theta_dt)
    if L > 16:
        raise ValueError(f"trap_rope_matrix only supports L<=16, got {L}")
    Xc, Ac, Bc, Cc = (to_compute(t) for t in (X, logdecay, B, C))
    Bc, Cc = _expand_groups(Bc, Cc, H)
    comp = Xc.dtype
    th = to_compute(theta_dt).to(comp) if theta_dt is not None else None
    Bcr, Ccr = _rotated_BC(Bc, Cc, th)
    if L == 0:
        return (
            from_compute(Xc.new_empty((Bx, 0, H, P)), io_dtype),
            torch.zeros(Bx, H, Bc.size(-1), P, dtype=comp, device=Xc.device),
        )
    a = torch.exp(Ac)  # (B, L, H)
    one_ss = torch.exp(segsum(Ac.permute(0, 2, 1)))  # (B, H, L, L)
    if _is_one(lam):
        band = (
            torch.eye(L, dtype=comp, device=Xc.device)
            .view(1, 1, L, L)
            .expand(Bx, H, L, L)
        )
    else:
        lamc = _as_lam(lam, Xc)
        lam_full = (
            lamc
            if lamc.dim() == 3
            else torch.full((Bx, L, H), float(lamc), dtype=comp, device=Xc.device)
        )
        band = torch.zeros(Bx, H, L, L, dtype=comp, device=Xc.device)
        # lam_full/a are (B, L, H); band is (B, H, L, L): index as
        # band[b, h, t, t] = lam[b, t, h], NO transpose (both (B, H) slices).
        lt = lam_full.permute(0, 2, 1)  # (B, H, L)
        at = a.permute(0, 2, 1)  # (B, H, L)
        for t in range(L):
            band[:, :, t, t] = lt[:, :, t]
            if t > 0:
                band[:, :, t, t - 1] = (1 - lt[:, :, t]) * at[:, :, t]
    # (M3-7): CIRC is matrix multiplication (NOT Hadamard): L = 1SS @ band.
    # Check: (1SS @ band)[t,s] = 1SS[t,s] g_s + 1SS[t,s+1] b_{s+1}
    #   = alpha_{t..s} g_s + alpha_{t..s+1} b_{s+1}. Band alone is diag(gamma)
    #   + subdiag(beta); lam=1 (beta=0, gamma=1) reduces L to 1SS exactly.
    Lm = one_ss @ band
    G = torch.einsum("blhn,bshn->bhls", Ccr, Bcr)
    M = G * Lm
    Y = torch.einsum("bhls,bshp->blhp", M, Xc)
    _, h, _ = trap_rope_loop(X, logdecay, B, C, theta_dt, lam)
    return from_compute(Y, io_dtype), h
