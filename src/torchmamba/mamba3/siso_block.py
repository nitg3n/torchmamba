"""Mamba-3 SISO block: in_proj -> BC RMSNorm + bias -> trap-RoPE core.

NO external short convolution (M3 Sec. 3.4).
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torchmamba.core.norms import rms_norm
from torchmamba.core.policy import compute_dtype_of
from torchmamba.mamba3.siso_core import trap_rope_loop


@dataclass
class Mamba3SisoCache:
    angle: Tensor  # (B, H, N/2) accumulated rotation, compute dtype
    ssm: Tensor  # (B, H, N, P) fp32 minimum
    x_prev: Tensor  # (B, H, P)
    b_prev: Tensor  # (B, G, N)


class Mamba3SisoBlock(nn.Module):
    """Mamba-3 SISO block with prefill/step caches."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_head: int = 64,
        d_state: int = 128,
        n_groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if n_heads % n_groups != 0:
            raise ValueError("n_heads must divide n_groups")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_state = d_state
        self.n_groups = n_groups
        d_inner = n_heads * d_head
        self.d_inner = d_inner
        self.in_proj_xz = nn.Linear(d_model, 2 * d_inner, bias=bias)
        self.in_proj_bcdt_theta = nn.Linear(
            d_model, 2 * n_groups * d_state + n_heads + n_heads * (d_state // 2),
            bias=bias,
        )
        self.bc_norm_weight = nn.Parameter(torch.empty(2 * n_groups * d_state))
        self.bc_bias = nn.Parameter(torch.empty(2 * n_groups * d_state))
        self.A_log = nn.Parameter(torch.empty(n_heads))
        self.out_proj = nn.Linear(d_inner, d_model, bias=bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.in_proj_xz.weight, a=5**0.5)
        nn.init.kaiming_uniform_(self.in_proj_bcdt_theta.weight, a=5**0.5)
        if self.in_proj_xz.bias is not None:
            nn.init.zeros_(self.in_proj_xz.bias)
        if self.in_proj_bcdt_theta.bias is not None:
            nn.init.zeros_(self.in_proj_bcdt_theta.bias)
        nn.init.ones_(self.bc_norm_weight)
        nn.init.zeros_(self.bc_bias)
        with torch.no_grad():
            self.A_log.copy_(
                torch.empty_like(self.A_log).uniform_(0.0, 3.0).neg()
            )
        nn.init.kaiming_uniform_(self.out_proj.weight, a=5**0.5)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def _core_inputs(self, u: Tensor):
        B_, L, _ = u.shape
        xz = self.in_proj_xz(u)
        x, z = xz.chunk(2, dim=-1)
        X = x.view(B_, L, self.n_heads, self.d_head)
        proj = self.in_proj_bcdt_theta(u)
        g, n, h = self.n_groups, self.d_state, self.n_heads
        Bc, Cc, dt, theta = proj.split([g * n, g * n, h, h * (n // 2)], dim=-1)
        bc = torch.cat([Bc, Cc], dim=-1)
        bc = rms_norm(bc, self.bc_norm_weight) + self.bc_bias
        Bc, Cc = bc.chunk(2, dim=-1)
        Bc = Bc.view(B_, L, g, n)
        Cc = Cc.view(B_, L, g, n)
        theta_dt = F.softplus(theta.view(B_, L, h, n // 2))
        dt = F.softplus(dt)
        logdecay = -dt * torch.exp(self.A_log)
        lam = torch.sigmoid(dt)  # data-dependent convex weight in [0, 1]
        return X, logdecay, Bc, Cc, theta_dt, lam, z

    def forward(self, u: Tensor) -> Tensor:
        X, logdecay, Bc, Cc, theta_dt, lam, z = self._core_inputs(u)
        Y, _, _ = trap_rope_loop(X, logdecay, Bc, Cc, theta_dt, lam)
        Y = Y.reshape(Y.size(0), Y.size(1), -1)
        return self.out_proj(Y * F.silu(z))

    def prefill(self, u: Tensor):
        comp = compute_dtype_of(u.dtype)
        X, logdecay, Bc, Cc, theta_dt, lam, z = self._core_inputs(u)
        Y, final, _ = trap_rope_loop(X, logdecay, Bc, Cc, theta_dt, lam)
        B_ = u.size(0)
        out = self.out_proj(
            Y.reshape(B_, Y.size(1), -1) * F.silu(z)
        )
        g, n, h = self.n_groups, self.d_state, self.n_heads
        Xc = X.to(comp)
        Bc_ = Bc.to(comp)
        th = theta_dt.to(comp)
        cache = Mamba3SisoCache(
            angle=th.sum(dim=1).detach(),
            ssm=final.detach().to(comp),
            x_prev=Xc[:, -1, :].detach(),
            b_prev=Bc_[:, -1, :].detach(),
        )
        _ = g, n, h
        return out, cache

    def step(self, u_t: Tensor, cache: Mamba3SisoCache):
        comp = compute_dtype_of(u_t.dtype)
        B_ = u_t.size(0)
        X, logdecay, Bc, Cc, theta_dt, lam, z = self._core_inputs(u_t)
        # Single step with 1-step (x, B) carry for the beta term.
        Xc = X.to(comp)
        Bc_ = Bc.to(comp)
        Cc_ = Cc.to(comp)
        th = theta_dt.to(comp)
        lamc = lam.to(comp)
        # Expand groups for per-head math.
        rep = self.n_heads // self.n_groups
        Bcr_full = Bc_.repeat_interleave(rep, dim=2)
        Ccr_full = Cc_.repeat_interleave(rep, dim=2)
        b_prev_full = cache.b_prev.to(comp).repeat_interleave(rep, dim=1)
        from torchmamba.core.rope import (
            apply_pairwise_rotation,
            pairwise_angles_to_cos_sin,
        )

        new_angle = cache.angle.to(comp) + th[:, 0, :]
        cos, sin = pairwise_angles_to_cos_sin(new_angle)
        Bt = apply_pairwise_rotation(Bcr_full[:, 0, :], cos, sin)
        Ct = apply_pairwise_rotation(Ccr_full[:, 0, :], cos, sin)
        cos_p, sin_p = pairwise_angles_to_cos_sin(cache.angle.to(comp))
        Bp = apply_pairwise_rotation(b_prev_full, cos_p, sin_p)
        a = torch.exp(logdecay.to(comp))[:, 0, :].view(B_, self.n_heads, 1, 1)
        lv = lamc[:, 0, :].view(B_, self.n_heads, 1, 1)
        h = (
            a * cache.ssm.to(comp)
            + (1 - lv) * a * torch.einsum("bhn,bhp->bhnp", Bp, cache.x_prev.to(comp))
            + lv * torch.einsum("bhn,bhp->bhnp", Bt, Xc[:, 0, :])
        )
        Y = torch.einsum("bhn,bhnp->bhp", Ct, h).unsqueeze(1)
        out = self.out_proj(Y.reshape(B_, 1, -1).to(u_t.dtype) * F.silu(z))
        new_cache = Mamba3SisoCache(
            angle=new_angle.detach(),
            ssm=h.detach(),
            x_prev=Xc[:, 0, :].detach(),
            b_prev=Bc_[:, 0, :].detach(),
        )
        return out, new_cache
