"""Mamba-2 block (M2 Sec. 7.1, Fig. 6; Sec. 8.1).

(X, B, C, dt) projected in parallel from block input u (NOT from x_c).
Depthwise conv applies to X only; psi on B, C; SiLU gate with z;
terminal norm; out_proj.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torchmamba.core.conv import causal_depthwise_conv1d
from torchmamba.core.norms import group_rms_norm, rms_norm
from torchmamba.core.policy import compute_dtype_of
from torchmamba.mamba2.core import ssd_chunkwise, ssd_recurrent, ssd_step


class Mamba2Block(nn.Module):
    """Mamba-2 SSD block with prefill/step caches."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_head: int = 64,
        d_state: int = 128,
        n_groups: int = 1,
        d_conv: int = 4,
        bias: bool = True,
        norm: str = "group",
        use_normalizer: bool = False,
        psi=None,
    ) -> None:
        super().__init__()
        if n_heads % n_groups != 0:
            raise ValueError("n_heads must be divisible by n_groups")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_state = d_state
        self.n_groups = n_groups
        self.d_conv = d_conv
        self.norm_kind = norm
        self.use_normalizer = use_normalizer
        self.psi = (lambda t: F.silu(t)) if psi is None else psi
        d_inner = n_heads * d_head
        self.d_inner = d_inner
        # Parallel projections from u: X, z, B, C, dt.
        self.in_proj_xz = nn.Linear(d_model, 2 * d_inner, bias=bias)
        self.in_proj_bcdt = nn.Linear(
            d_model, 2 * n_groups * d_state + n_heads, bias=bias
        )
        self.conv_weight = nn.Parameter(torch.empty(d_inner, 1, d_conv))
        self.conv_bias = nn.Parameter(torch.empty(d_inner))
        # Log-decay base per head; dt scales it: logdecay = dt * A_base (<0).
        self.A_log = nn.Parameter(torch.empty(n_heads))
        self.norm_weight = nn.Parameter(torch.empty(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model, bias=bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.in_proj_xz.weight, a=5**0.5)
        nn.init.kaiming_uniform_(self.in_proj_bcdt.weight, a=5**0.5)
        if self.in_proj_xz.bias is not None:
            nn.init.zeros_(self.in_proj_xz.bias)
        if self.in_proj_bcdt.bias is not None:
            nn.init.zeros_(self.in_proj_bcdt.bias)
        nn.init.kaiming_uniform_(self.conv_weight, a=5**0.5)
        nn.init.zeros_(self.conv_bias)
        # A_base in (0, 1] via exp(-U); logdecay = -dt * exp(A_log) <= 0.
        with torch.no_grad():
            self.A_log.copy_(
                torch.empty_like(self.A_log).uniform_(0.0, 3.0).neg()
            )
        nn.init.ones_(self.norm_weight)
        nn.init.kaiming_uniform_(self.out_proj.weight, a=5**0.5)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def _logdecay(self, dt: Tensor) -> Tensor:
        # dt: (B, L, H) positive; returns (B, L, H) <= 0.
        return -dt * torch.exp(self.A_log)

    def _norm(self, t: Tensor) -> Tensor:
        if self.norm_kind == "group":
            return group_rms_norm(t, self.n_heads, self.norm_weight)
        if self.norm_kind == "rms":
            return rms_norm(t, self.norm_weight)
        raise ValueError(f"unknown norm {self.norm_kind!r}")

    def _core_inputs(self, u: Tensor):
        xz = self.in_proj_xz(u)
        x, z = xz.chunk(2, dim=-1)
        B_, L, _ = x.shape
        bcdt = self.in_proj_bcdt(u)
        g, n = self.n_groups, self.d_state
        Bc, Cc, dt = bcdt.split([g * n, g * n, self.n_heads], dim=-1)
        Bc = self.psi(Bc.view(B_, L, g, n))
        Cc = self.psi(Cc.view(B_, L, g, n))
        dt = F.softplus(dt)
        X = causal_depthwise_conv1d(
            x.transpose(1, 2), self.conv_weight, self.conv_bias
        ).transpose(1, 2)
        X = X.view(B_, L, self.n_heads, self.d_head)
        if self.use_normalizer:
            X = torch.cat(
                [X, X.new_ones((B_, L, self.n_heads, 1))], dim=-1
            )
        return X, self._logdecay(dt), Bc, Cc, z

    def forward(self, u: Tensor) -> Tensor:
        X, logdecay, Bc, Cc, z = self._core_inputs(u)
        Y, _ = ssd_chunkwise(X, logdecay, Bc, Cc)
        Y = Y.reshape(Y.size(0), Y.size(1), -1)
        if self.use_normalizer:
            Y, Ynorm = Y[..., :-1], Y[..., -1:]
            Y = Y / (Ynorm + 1e-6)
        return self.out_proj(self._norm(Y * F.silu(z)))

    def prefill(self, u: Tensor):
        comp = compute_dtype_of(u.dtype)
        xz = self.in_proj_xz(u)
        x, z = xz.chunk(2, dim=-1)
        B_, L, _ = x.shape
        xc_t = x.transpose(1, 2)
        X = causal_depthwise_conv1d(
            xc_t, self.conv_weight, self.conv_bias
        ).transpose(1, 2)
        Xh = X.view(B_, L, self.n_heads, self.d_head)
        bcdt = self.in_proj_bcdt(u)
        g, n = self.n_groups, self.d_state
        Bc, Cc, dt = bcdt.split([g * n, g * n, self.n_heads], dim=-1)
        Bc = self.psi(Bc.view(B_, L, g, n))
        Cc = self.psi(Cc.view(B_, L, g, n))
        dt = F.softplus(dt)
        logdecay = self._logdecay(dt)
        Xn = Xh
        if self.use_normalizer:
            Xn = torch.cat(
                [Xh, Xh.new_ones((B_, L, self.n_heads, 1))], dim=-1
            )
        Y, final = ssd_chunkwise(Xn, logdecay, Bc, Cc)
        Y = Y.reshape(B_, L, -1)
        if self.use_normalizer:
            Y, Ynorm = Y[..., :-1], Y[..., -1:]
            Y = Y / (Ynorm + 1e-6)
        out = self.out_proj(self._norm(Y * F.silu(z)))
        k = self.d_conv - 1
        conv_state = (
            xc_t[:, :, -k:].detach().to(comp)
            if k > 0
            else xc_t.new_empty((B_, self.d_inner, 0)).to(comp)
        )
        cache = {
            "conv_state": conv_state,
            "ssm_state": final.detach().to(comp),
        }
        return out, cache

    def step(self, u_t: Tensor, cache: dict):
        comp = compute_dtype_of(u_t.dtype)
        xz = self.in_proj_xz(u_t)
        x, z = xz.chunk(2, dim=-1)  # (B, 1, d_inner)
        B_ = x.size(0)
        xt = x[:, 0, :]
        if self.d_conv > 1:
            window = torch.cat(
                [cache["conv_state"], xt.unsqueeze(-1)], dim=-1
            )
            xc = (window * self.conv_weight.squeeze(1).unsqueeze(0)).sum(
                -1
            ) + self.conv_bias
            new_conv = window[:, :, 1:].detach().to(comp)
        else:
            xc = xt * self.conv_weight.squeeze(-1).squeeze(-1) + self.conv_bias
            new_conv = cache["conv_state"]
        Xh = xc.view(B_, 1, self.n_heads, self.d_head)
        bcdt = self.in_proj_bcdt(u_t)
        g, n = self.n_groups, self.d_state
        Bc, Cc, dt = bcdt.split([g * n, g * n, self.n_heads], dim=-1)
        Bc = self.psi(Bc.view(B_, 1, g, n))
        Cc = self.psi(Cc.view(B_, 1, g, n))
        dt = F.softplus(dt)
        logdecay = self._logdecay(dt)
        Xn = Xh
        if self.use_normalizer:
            Xn = torch.cat(
                [Xh, Xh.new_ones((B_, 1, self.n_heads, 1))], dim=-1
            )
        Y1, h1 = ssd_step(Xn, logdecay, Bc, Cc, cache["ssm_state"])
        Y = Y1.reshape(B_, 1, -1)
        if self.use_normalizer:
            Y, Ynorm = Y[..., :-1], Y[..., -1:]
            Y = Y / (Ynorm + 1e-6)
        out = self.out_proj(self._norm(Y * F.silu(z)))
        return out, {"conv_state": new_conv, "ssm_state": h1.detach()}

    # Test escape hatch: exact recurrent path (same math, no chunking).
    def forward_recurrent(self, u: Tensor) -> Tensor:
        X, logdecay, Bc, Cc, z = self._core_inputs(u)
        Y, _ = ssd_recurrent(X, logdecay, Bc, Cc)
        Y = Y.reshape(Y.size(0), Y.size(1), -1)
        return self.out_proj(self._norm(Y * F.silu(z)))
