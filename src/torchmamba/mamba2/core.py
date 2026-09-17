"""Mamba-2 SSD core (M2 Sec. 6, Listing 1).

Entry points: ssd_recurrent (ground truth loop), ssd_quadratic (explicit mask),
ssd_chunkwise (Listing-1 port, tail pad-trim), ssd_step (single-step wrapper).
G broadcasts over H by repeat; H % G == 0 required.
"""

import torch
import torch.nn.functional as Fpad
from torch import Tensor

from torchmamba.core.policy import compute_dtype_of, from_compute, to_compute
from torchmamba.core.segsum_util import segsum


def _check_shapes(
    X: Tensor, logdecay: Tensor, B: Tensor, C: Tensor
) -> tuple[int, int, int, int, int]:
    Bx, Lx, Hx, Px = X.shape
    Bl, Ll, Hl = logdecay.shape
    Bb, Lb, Gb, Nb = B.shape
    Bc, Lc, Gc, Nc = C.shape
    assert Bx == Bl == Bb == Bc, "batch mismatch"
    assert Lx == Ll == Lb == Lc, "length mismatch"
    assert Hx == Hl, "heads mismatch"
    assert Gb == Gc and Nb == Nc, "B/C group/state mismatch"
    if Hx % Gb != 0:
        raise ValueError(f"n_heads ({Hx}) must be divisible by n_groups ({Gb})")
    return Bx, Lx, Hx, Px, Nb


def _expand_groups(B: Tensor, C: Tensor, H: int) -> tuple[Tensor, Tensor]:
    """Repeat (B, L, G, N) -> (B, L, H, N) when G < H."""
    if B.size(2) == H:
        return B, C
    rep = H // B.size(2)
    return (
        B.repeat_interleave(rep, dim=2),
        C.repeat_interleave(rep, dim=2),
    )


def boundary_mask_to_logdecay(
    boundary: Tensor, base: Tensor, blocked: float = -1e30
) -> Tensor:
    """Set logdecay to blocked (-inf equivalent) where boundary is True.

    Args:
        boundary: (B, L) bool, True at sequence starts (no carry from past).
        base: (B, L, H) log-decays to mask.
    """
    out = base.clone()
    out[boundary.unsqueeze(-1).expand_as(out)] = blocked
    return out


def ssd_recurrent(
    X: Tensor, logdecay: Tensor, B: Tensor, C: Tensor
) -> tuple[Tensor, Tensor]:
    """Per-step scalar-decay loop; SSD ground truth.

    h_t = a_t h_{t-1} + B_t x_t (outer over (N, P)), y_t = C_t^T h_t.
    a_t = exp(logdecay_t). logdecay -> -inf blocks carry exactly (a=0).
    """
    io_dtype = X.dtype
    Bx, L, H, P, N = _check_shapes(X, logdecay, B, C)
    Xc, Ac, Bc, Cc = (to_compute(t) for t in (X, logdecay, B, C))
    Bc, Cc = _expand_groups(Bc, Cc, H)
    a = torch.exp(Ac)  # (B, L, H)
    h = torch.zeros(Bx, H, N, P, dtype=Xc.dtype, device=Xc.device)
    ys: list[Tensor] = []
    for t in range(L):
        h = a[:, t, :].view(Bx, H, 1, 1) * h + torch.einsum(
            "bhn,bhp->bhnp", Bc[:, t, :], Xc[:, t, :]
        )
        ys.append(torch.einsum("bhn,bhnp->bhp", Cc[:, t, :], h))
    Y = torch.stack(ys, dim=1) if L > 0 else Xc.new_empty((Bx, 0, H, P))
    return from_compute(Y, io_dtype), h


def ssd_quadratic(
    X: Tensor, logdecay: Tensor, B: Tensor, C: Tensor
) -> tuple[Tensor, Tensor]:
    """Explicit (L, L) mask: G = C B^T, M = G o L, Y = M V. L <= 512 guard."""
    io_dtype = X.dtype
    Bx, L, H, P, N = _check_shapes(X, logdecay, B, C)
    if L > 512:
        raise ValueError(f"ssd_quadratic only supports L<=512, got {L}")
    Xc, Ac, Bc, Cc = (to_compute(t) for t in (X, logdecay, B, C))
    Bc, Cc = _expand_groups(Bc, Cc, H)
    if L == 0:
        return (
            from_compute(Xc.new_empty((Bx, 0, H, P)), io_dtype),
            torch.zeros(Bx, H, N, P, dtype=Xc.dtype, device=Xc.device),
        )
    Lm = torch.exp(segsum(Ac.permute(0, 2, 1)))  # (B, H, L, L)
    G = torch.einsum("blhn,bshn->bhls", Cc, Bc)
    M = G * Lm
    Y = torch.einsum("bhls,bshp->blhp", M, Xc)
    a = torch.exp(Ac)
    h = torch.zeros(Bx, H, N, P, dtype=Xc.dtype, device=Xc.device)
    for t in range(L):
        h = a[:, t, :].view(Bx, H, 1, 1) * h + torch.einsum(
            "bhn,bhp->bhnp", Bc[:, t, :], Xc[:, t, :]
        )
    return from_compute(Y, io_dtype), h


def ssd_chunkwise(
    X: Tensor,
    logdecay: Tensor,
    B: Tensor,
    C: Tensor,
    block_len: int = 64,
    initial_states: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Chunkwise SSD, M2 Sec. 6 Listing 1 port (no einops).

    Tail chunks (block_len not dividing L) are zero-padded: X/B/C with 0
    (no contribution) and logdecay with 0.0 (a=1 carry-through), then outputs
    trimmed to L. NOTE: -inf-equivalent logdecay padding would zero the final
    state through the center scan; 0.0 preserves carry exactly.
    """
    io_dtype = X.dtype
    Bx, L, H, P, N = _check_shapes(X, logdecay, B, C)
    Xc, Ac, Bc, Cc = (to_compute(t) for t in (X, logdecay, B, C))
    Bc, Cc = _expand_groups(Bc, Cc, H)
    comp = Xc.dtype
    if L == 0:
        return (
            from_compute(Xc.new_empty((Bx, 0, H, P)), io_dtype),
            torch.zeros(Bx, H, N, P, dtype=comp, device=Xc.device),
        )
    n_chunks = (L + block_len - 1) // block_len
    Lp = n_chunks * block_len
    pad = Lp - L
    if pad > 0:
        Xc = Fpad.pad(Xc.permute(0, 2, 3, 1), (0, pad)).permute(0, 3, 1, 2)
        Bc = Fpad.pad(Bc.permute(0, 2, 3, 1), (0, pad)).permute(0, 3, 1, 2)
        Cc = Fpad.pad(Cc.permute(0, 2, 3, 1), (0, pad)).permute(0, 3, 1, 2)
        Ac = Fpad.pad(Ac.permute(0, 2, 1), (0, pad), value=0.0).permute(0, 2, 1)
    # Chunk views: (B, n_chunks, l, ...) with heads-last where Listing uses h.
    Xc_ = Xc.view(Bx, n_chunks, block_len, H, P)
    Bc_ = Bc.view(Bx, n_chunks, block_len, H, N)
    Cc_ = Cc.view(Bx, n_chunks, block_len, H, N)
    Ac_ = Ac.view(Bx, n_chunks, block_len, H).permute(0, 3, 1, 2)  # (B,H,C,l)
    # 1. intra-chunk (diagonal): Y_diag = einsum(C, B, L, X)
    Lm = torch.exp(segsum(Ac_))  # (B, H, C, l, l)
    Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", Cc_, Bc_, Lm, Xc_)
    # 2. right factor (B terms): per-chunk terminal states from zero init.
    # Listing 1 einsum "bclhn,bhcl,bclhp->bchpn" yields (B, C, H, P, N).
    A_cumsum = torch.cumsum(Ac_, dim=-1)  # (B, H, C, l)
    decay_states = torch.exp(A_cumsum[..., -1:] - A_cumsum)  # (B,H,C,l)
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", Bc_, decay_states, Xc_)
    # Convention (matches M2 Listing 1 verbatim): chunk states are (B, C, H,
    # P, N) everywhere inside; only the RETURNED final_state is (B, H, N, P)
    # to match ssd_recurrent. Incoming initial_states is (B, H, N, P).
    if initial_states is not None:
        init = to_compute(initial_states).to(comp)
        # (B, H, N, P) -> (B, 1, H, P, N).
        init = init.permute(0, 1, 3, 2).unsqueeze(1).contiguous()
    else:
        init = torch.zeros_like(states[:, :1])
    states_cat = torch.cat([init, states], dim=1)  # (B, C+1, H, P, N)
    # 3. center factor (A terms): inter-chunk scalar scan over chunk means
    decay_chunk = torch.exp(
        segsum(Fpad.pad(A_cumsum[..., -1], (1, 0)))
    )  # (B, H, C+1, C+1)
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states_cat)
    states_out, final_pn = new_states[:, :-1], new_states[:, -1]
    # new_states is (B, C, H, P, N): transpose last two -> (B, H, N, P).
    final_state = final_pn.permute(0, 1, 3, 2).contiguous()  # (B, H, N, P)
    # states_out: (B, C, H, P, N) -> (B, C, l, H, N) broadcast over l below.
    # 4. left factor (C terms): boundary states -> outputs
    state_decay_out = torch.exp(A_cumsum)  # (B, H, C, l)
    Y_off = torch.einsum("bclhn,bchpn,bhcl->bclhp", Cc_, states_out, state_decay_out)
    Y = Y_diag + Y_off
    Y = Y.reshape(Bx, Lp, H, P)[:, :L, :]
    return from_compute(Y, io_dtype), final_state


def ssd_step(
    X: Tensor,
    logdecay: Tensor,
    B: Tensor,
    C: Tensor,
    state: Tensor,
) -> tuple[Tensor, Tensor]:
    """Single-step (L=1): Y_1, h_1 from incoming state h_0."""
    io_dtype = X.dtype
    Bx, L, H, P, N = _check_shapes(X, logdecay, B, C)
    if L != 1:
        raise ValueError(f"ssd_step requires L==1, got {L}")
    Xc, Ac, Bc, Cc = (to_compute(t) for t in (X, logdecay, B, C))
    Bc, Cc = _expand_groups(Bc, Cc, H)
    sc = to_compute(state).to(Xc.dtype)
    a = torch.exp(Ac)  # (B, 1, H)
    h1 = a[:, 0, :].view(Bx, H, 1, 1) * sc + torch.einsum(
        "bhn,bhp->bhnp", Bc[:, 0, :], Xc[:, 0, :]
    )
    Y1 = torch.einsum("bhn,bhnp->bhp", Cc[:, 0, :], h1).unsqueeze(1)
    return from_compute(Y1, io_dtype), h1
