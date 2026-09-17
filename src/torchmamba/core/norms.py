"""RMS normalizations in pure torch (no fused kernels)."""

import torch
from torch import Tensor


def rms_norm(
    x: Tensor, weight: Tensor | None = None, eps: float = 1e-6
) -> Tensor:
    """RMSNorm over the last dim: x / sqrt(mean(x^2) + eps) [* weight]."""
    var = x.pow(2).mean(dim=-1, keepdim=True)
    out = x * torch.rsqrt(var + eps)
    return out * weight if weight is not None else out


def group_rms_norm(
    x: Tensor,
    num_groups: int,
    weight: Tensor | None = None,
    eps: float = 1e-6,
) -> Tensor:
    """GroupRMSNorm: independent RMS per group of last-dim channels.

    Args:
        x: (..., D) with D % num_groups == 0.
        num_groups: number of groups; else ValueError.
        weight: optional (D,) affine weight applied after norm.
    """
    D = x.size(-1)
    if D % num_groups != 0:
        raise ValueError(
            f"num_groups ({num_groups}) must divide last dim ({D})"
        )
    W = D // num_groups
    g = x.reshape(*x.shape[:-1], num_groups, W)
    var = g.pow(2).mean(dim=-1, keepdim=True)
    g = g * torch.rsqrt(var + eps)
    out = g.reshape(*x.shape[:-1], D)
    return out * weight if weight is not None else out
