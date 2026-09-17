"""Segment-sum for 1-SS masks (M2 Sec. 6, Listing 1).

exp(segsum(log_decay)) is the 1-SS matrix, i.e. the scalar SSM scan.
-inf above the diagonal BEFORE any exp is causality itself.
"""

import torch
from torch import Tensor


def segsum(log_decay: Tensor) -> Tensor:
    """Segment sums: out[..., i, j] = cumsum[i] - cumsum[j], lower-triangular.

    Args:
        log_decay: Tensor[..., T] of log-decays (<= 0 normally, -inf blocks carry).

    Returns:
        Tensor[..., T, T] with -inf strictly above the diagonal.
        T=0 -> (..., 0, 0); T=1 -> zeros (1, 1).
    """
    T = log_decay.size(-1)
    if T == 0:
        return log_decay.new_empty((*log_decay.shape[:-1], 0, 0))
    cumsum = torch.cumsum(log_decay, dim=-1)
    out = cumsum[..., :, None] - cumsum[..., None, :]
    mask = torch.tril(
        torch.ones(T, T, device=log_decay.device, dtype=torch.bool), diagonal=0
    )
    return out.masked_fill(~mask, float("-inf"))
