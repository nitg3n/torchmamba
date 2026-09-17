"""Dtype policy: fp16/bf16/fp32 compute in fp32, fp64 stays fp64."""

import torch

from torchmamba.mamba1.core import selective_scan_loop


def test_policy_upcast_downcast():
    torch.manual_seed(0)
    x = torch.randn(2, 5, 4, dtype=torch.float16)
    dt = torch.rand(2, 5, 4, dtype=torch.float16) + 0.1
    A = -torch.exp(torch.randn(4, 3, dtype=torch.float16))
    B = torch.randn(2, 5, 3, dtype=torch.float16)
    C = torch.randn(2, 5, 3, dtype=torch.float16)
    y, h = selective_scan_loop(x, dt, A, B, C)
    # Data output returns to input dtype; states never below fp32.
    assert y.dtype == torch.float16
    assert h.dtype in (torch.float32, torch.float64)
    assert h.dtype == torch.float32


def test_policy_fp64_stays():
    torch.manual_seed(0)
    x = torch.randn(1, 4, 2, dtype=torch.float64)
    dt = torch.rand(1, 4, 2, dtype=torch.float64) + 0.1
    A = -torch.exp(torch.randn(2, 2, dtype=torch.float64))
    B = torch.randn(1, 4, 2, dtype=torch.float64)
    C = torch.randn(1, 4, 2, dtype=torch.float64)
    y, h = selective_scan_loop(x, dt, A, B, C)
    assert y.dtype == torch.float64
    assert h.dtype == torch.float64


def test_policy_bf16_compute_fp32():
    torch.manual_seed(1)
    x = torch.randn(1, 3, 2, dtype=torch.bfloat16)
    dt = torch.rand(1, 3, 2, dtype=torch.bfloat16) + 0.1
    A = -torch.exp(torch.randn(2, 2, dtype=torch.bfloat16))
    B = torch.randn(1, 3, 2, dtype=torch.bfloat16)
    C = torch.randn(1, 3, 2, dtype=torch.bfloat16)
    y, h = selective_scan_loop(x, dt, A, B, C)
    assert y.dtype == torch.bfloat16
    assert h.dtype == torch.float32
