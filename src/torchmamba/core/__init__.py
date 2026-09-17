"""Shared primitives: dtype policy, discretization, segsum, conv, norms, RoPE."""

from torchmamba.core.conv import causal_depthwise_conv1d
from torchmamba.core.discretize import exp_euler, softplus_dt, trap_coeffs, zoh
from torchmamba.core.norms import group_rms_norm, rms_norm
from torchmamba.core.policy import (
    COMPUTE_DTYPE,
    ORACLE_DTYPE,
    compute_dtype_of,
    from_compute,
    to_compute,
)
from torchmamba.core.rope import (
    accumulate_angles,
    apply_pairwise_rotation,
    pairwise_angles_to_cos_sin,
)
from torchmamba.core.segsum_util import segsum

__all__ = [
    "COMPUTE_DTYPE",
    "ORACLE_DTYPE",
    "compute_dtype_of",
    "to_compute",
    "from_compute",
    "softplus_dt",
    "exp_euler",
    "zoh",
    "trap_coeffs",
    "segsum",
    "causal_depthwise_conv1d",
    "rms_norm",
    "group_rms_norm",
    "pairwise_angles_to_cos_sin",
    "apply_pairwise_rotation",
    "accumulate_angles",
]
