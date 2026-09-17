"""Mamba-1 block (M1 Sec. 3.4, Fig. 3).

forward = in_proj -> split(x, z) -> causal conv -> SiLU -> S6 scan
          -> SiLU(z) gate -> out_proj.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torchmamba.core.conv import causal_depthwise_conv1d
from torchmamba.core.policy import compute_dtype_of
from torchmamba.mamba1.core import selective_scan_loop


@dataclass
class Mamba1Cache:
    conv_state: Tensor  # (B, d_inner, d_conv - 1); empty when d_conv == 1
    ssm_state: Tensor  # (B, d_inner, d_state), compute dtype


class Mamba1Block(nn.Module):
    """Mamba-1 selective SSM block with prefill/step caches."""

    def __init__(
        self,
        d_model: int,
        expand: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        dt_rank: int | str = "auto",
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.expand = expand
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.dt_rank: int = (
            max(1, d_model // 16) if dt_rank == "auto" else int(dt_rank)
        )
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=bias)
        self.conv_weight = nn.Parameter(torch.empty(self.d_inner, 1, d_conv))
        self.conv_bias = nn.Parameter(torch.empty(self.d_inner))
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + 2 * d_state, bias=bias
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.A_log = nn.Parameter(torch.empty(self.d_inner, d_state))
        self.D_skip = nn.Parameter(torch.empty(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.in_proj.weight, a=5**0.5)
        nn.init.kaiming_uniform_(self.x_proj.weight, a=5**0.5)
        nn.init.kaiming_uniform_(self.dt_proj.weight, a=5**0.5)
        nn.init.kaiming_uniform_(self.out_proj.weight, a=5**0.5)
        for lin in (self.in_proj, self.x_proj, self.dt_proj, self.out_proj):
            if lin.bias is not None:
                nn.init.zeros_(lin.bias)
        nn.init.kaiming_uniform_(self.conv_weight, a=5**0.5)
        nn.init.zeros_(self.conv_bias)
        # S4D-Real: A = -(1..N) broadcast over channels (M1 Sec. 3.6).
        A = -torch.arange(1, self.d_state + 1, dtype=torch.float32)
        with torch.no_grad():
            self.A_log.copy_(torch.log(-A).expand_as(self.A_log).contiguous())
        nn.init.ones_(self.D_skip)
        # dt bias = inv_softplus(U(0.001, 0.1)) via log(exp(u) - 1).
        with torch.no_grad():
            u = torch.empty_like(self.dt_proj.bias).uniform_(0.001, 0.1)
            self.dt_proj.bias.copy_(torch.log(torch.exp(u) - 1))

    def _A(self) -> Tensor:
        return -torch.exp(self.A_log)

    def forward_core(
        self, x_c: Tensor, dt: Tensor, B: Tensor, C: Tensor
    ) -> tuple[Tensor, Tensor]:
        return selective_scan_loop(x_c, dt, self._A(), B, C, self.D_skip)

    def _project_ssm(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        proj = self.x_proj(x)
        r = self.dt_rank
        dt_in, B, C = proj.split([r, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt_in))
        return dt, B, C

    def _scan_one_step(
        self,
        xc: Tensor,
        dt: Tensor,
        B: Tensor,
        C: Tensor,
        h_prev: Tensor,
        comp: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        """One (EE) step in compute dtype; mirrors mamba1/core selective_scan_loop."""
        A = self._A().to(comp)
        alpha = torch.exp(dt.to(comp).unsqueeze(-1) * A.unsqueeze(0))
        h = (
            alpha * h_prev.to(comp)
            + dt.to(comp).unsqueeze(-1)
            * B.to(comp).unsqueeze(1)
            * xc.to(comp).unsqueeze(-1)
        )
        y = (h * C.to(comp).unsqueeze(1)).sum(-1)
        y = y + self.D_skip.to(comp).unsqueeze(0) * xc.to(comp)
        return y, h

    def forward(self, u: Tensor) -> Tensor:
        xz = self.in_proj(u)
        x, z = xz.chunk(2, dim=-1)
        x_c = causal_depthwise_conv1d(
            x.transpose(1, 2), self.conv_weight, self.conv_bias
        ).transpose(1, 2)
        x_c = F.silu(x_c)
        dt, B, C = self._project_ssm(x_c)
        y, _ = self.forward_core(x_c, dt, B, C)
        return self.out_proj(y * F.silu(z))

    def prefill(self, u: Tensor) -> tuple[Tensor, Mamba1Cache]:
        comp = compute_dtype_of(u.dtype)
        xz = self.in_proj(u)
        x, z = xz.chunk(2, dim=-1)
        xc_t = x.transpose(1, 2)
        x_c = causal_depthwise_conv1d(
            xc_t, self.conv_weight, self.conv_bias
        ).transpose(1, 2)
        x_c = F.silu(x_c)
        dt, B, C = self._project_ssm(x_c)
        y, h_last = self.forward_core(x_c, dt, B, C)
        out = self.out_proj(y * F.silu(z))
        k = self.d_conv - 1
        conv_state = (
            xc_t[:, :, -k:].detach().to(comp)
            if k > 0
            else xc_t.new_empty((xc_t.size(0), self.d_inner, 0)).to(comp)
        )
        return out, Mamba1Cache(
            conv_state=conv_state, ssm_state=h_last.detach().to(comp)
        )

    def step(self, u_t: Tensor, cache: Mamba1Cache) -> tuple[Tensor, Mamba1Cache]:
        """Single-step decode; no future leak (uses conv shift register)."""
        comp = compute_dtype_of(u_t.dtype)
        xz = self.in_proj(u_t)
        x, z = xz.chunk(2, dim=-1)  # (B, 1, d_inner)
        xt = x[:, 0, :]  # (B, d_inner)
        if self.d_conv > 1:
            window = torch.cat([cache.conv_state, xt.unsqueeze(-1)], dim=-1)
            # (B, d, K) * (d, K) + (d,) == causal_depthwise_conv1d last step.
            xc = (window * self.conv_weight.squeeze(1).unsqueeze(0)).sum(
                -1
            ) + self.conv_bias
            new_conv = window[:, :, 1:].detach().to(comp)
        else:
            # K=1 causal conv == per-step scale by weight[..., 0] + bias.
            xc = xt * self.conv_weight.squeeze(-1).squeeze(-1) + self.conv_bias
            new_conv = cache.conv_state
        xc = F.silu(xc)
        dt, B, C = self._project_ssm(xc.unsqueeze(1))
        y, h = self._scan_one_step(
            xc, dt.squeeze(1), B.squeeze(1), C.squeeze(1), cache.ssm_state, comp
        )
        out = self.out_proj(y.to(u_t.dtype).unsqueeze(1) * F.silu(z))
        return out, Mamba1Cache(conv_state=new_conv, ssm_state=h.detach())
