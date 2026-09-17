"""Causal depthwise 1D convolution (M1 Sec. 3.4, Fig. 3, x branch).

Left-padded, groups=D: output[t] depends only on inputs[<=t].
"""

import torch.nn.functional as F
from torch import Tensor


def causal_depthwise_conv1d(
    x: Tensor, weight: Tensor, bias: Tensor | None = None
) -> Tensor:
    """Causal depthwise conv1d.

    Args:
        x: (B, D, L) block I/O (channel-first for conv).
        weight: (D, 1, K) depthwise kernel.
        bias: (D,) or None.

    Returns:
        (B, D, L). K=1 reduces to a per-step affine scale.
    """
    K = weight.size(-1)
    if K > 1:
        x = F.pad(x, (K - 1, 0))
    return F.conv1d(x, weight, bias, groups=x.size(1))
