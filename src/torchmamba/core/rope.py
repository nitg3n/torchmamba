"""Pairwise (2x2-block) rotations for the Mamba-3 RoPE trick.

Pairs are (2i, 2i+1) along the last dim; N must be even.
All ops are out-of-place.
"""

import torch
from torch import Tensor


def _check_even(n: int) -> None:
    if n % 2 != 0:
        raise ValueError(f"last dim must be even for pairwise rotation, got {n}")


def pairwise_angles_to_cos_sin(angles: Tensor) -> tuple[Tensor, Tensor]:
    """Split (..., N/2) angles into (cos, sin) pair."""
    return torch.cos(angles), torch.sin(angles)


def apply_pairwise_rotation(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply 2x2 rotations pairing (2i, 2i+1); out-of-place.

    Args:
        x: (..., N), N even.
        cos, sin: broadcastable to (..., N/2).
    """
    _check_even(x.size(-1))
    xe = x[..., 0::2]
    xo = x[..., 1::2]
    out_e = xe * cos - xo * sin
    out_o = xe * sin + xo * cos
    out = torch.stack((out_e, out_o), dim=-1)
    return out.reshape(*x.shape[:-1], x.size(-1))


def accumulate_angles(theta_dt: Tensor) -> Tensor:
    """Cumulative rotation angles: cumsum over the L axis of (B, L, H, N/2)."""
    return torch.cumsum(theta_dt, dim=1)
