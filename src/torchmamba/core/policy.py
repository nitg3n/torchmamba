"""Global dtype/compute policy for TorchMamba reference kernels.

Rule (plan Step 0): fp16/bf16/fp32 inputs are upcast to fp32 for all
exp/cumsum/segsum/scan/state math, outputs downcast to the input dtype;
fp64 inputs stay fp64 end-to-end. States are never kept below fp32.
"""

import torch
from torch import Tensor

COMPUTE_DTYPE = torch.float32
ORACLE_DTYPE = torch.float64

_LOW_PRECISION = (torch.float16, torch.bfloat16, torch.float32)


def compute_dtype_of(dtype: torch.dtype) -> torch.dtype:
    """Compute dtype for a given input dtype: fp64 in -> fp64, else fp32."""
    return torch.float64 if dtype == torch.float64 else torch.float32


def to_compute(t: Tensor) -> Tensor:
    """Upcast a tensor to its compute dtype (no-op if already there)."""
    target = compute_dtype_of(t.dtype)
    return t.to(target) if t.dtype != target else t


def from_compute(t: Tensor, dtype: torch.dtype) -> Tensor:
    """Downcast a compute-dtype tensor back to an I/O dtype."""
    return t.to(dtype) if t.dtype != dtype else t
