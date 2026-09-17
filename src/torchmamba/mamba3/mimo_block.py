"""Mamba-3 MIMO block: SISO projections + data-independent R expansion.

B, C shared across heads (MVA): full DNR projection.
x, y, z per-head: SISO projection then elementwise R-expansion vectors
(DP + PR params, M3 Sec. 3.3, App. C). Same BCNorm+bias/gate/out as SISO.
State has no R axis: same cache shapes as SISO.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torchmamba.core.norms import rms_norm
from torchmamba.core.policy import compute_dtype_of
from torchmamba.mamba3.mimo_core import mimo_chunkwise, mimo_loop
from torchmamba.mamba3.siso_block import Mamba3SisoCache


class Mamba3MimoBlock(nn.Module):
    """Mamba-3 MIMO block with prefill/step caches."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_head: int = 64,
        d_state: int = 128,
        n_groups: int = 1,
        mimo_rank: int = 4,
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
        self.mimo_rank = mimo_rank
        d_inner = n_heads * d_head
        self.d_inner = d_inner
        self.in_proj_xz = nn.Linear(d_model, 2 * d_inner, bias=bias)
        self.in_proj_bcdt_theta = nn.Linear(
            d_model, 2 * n_groups * d_state + n_heads + n_heads * (d_state // 2),
            bias=bias,
        )
        # SISO B/C projections produce (..., N); rank expansion to R:
        self.bc_rank_proj = nn.Linear(
            n_groups * d_state, n_groups * d_state * mimo_rank, bias=False
        )
        self.bc_norm_weight = nn.Parameter(torch.empty(2 * n_groups * d_state))
        self.bc_bias = nn.Parameter(torch.empty(2 * n_groups * d_state))
        self.A_log = nn.Parameter(torch.empty(n_heads))
        # Data-independent R-expansion vectors for per-head x, z, out.
        self.x_expand = nn.Parameter(torch.empty(d_inner, mimo_rank))
        self.z_expand = nn.Parameter(torch.empty(d_inner, mimo_rank))
        self.out_contract = nn.Parameter(torch.empty(d_inner, mimo_rank))
        self.out_proj = nn.Linear(d_inner, d_model, bias=bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.in_proj_xz.weight, a=5**0.5)
        nn.init.kaiming_uniform_(self.in_proj_bcdt_theta.weight, a=5**0.5)
        if self.in_proj_xz.bias is not None:
            nn.init.zeros_(self.in_proj_xz.bias)
        if self.in_proj_bcdt_theta.bias is not None:
            nn.init.zeros_(self.in_proj_bcdt_theta.bias)
        nn.init.kaiming_uniform_(self.bc_rank_proj.weight, a=5**0.5)
        nn.init.ones_(self.bc_norm_weight)
        nn.init.zeros_(self.bc_bias)
        with torch.no_grad():
            self.A_log.copy_(
                torch.empty_like(self.A_log).uniform_(0.0, 3.0).neg()
            )
        nn.init.ones_(self.x_expand)
        nn.init.ones_(self.z_expand)
        nn.init.ones_(self.out_contract)
        nn.init.kaiming_uniform_(self.out_proj.weight, a=5**0.5)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def _core_inputs(self, u: Tensor):
        B_, L, _ = u.shape
        g, n, h, dh, r = (
            self.n_groups, self.d_state, self.n_heads, self.d_head, self.mimo_rank,
        )
        xz = self.in_proj_xz(u)
        x, z = xz.chunk(2, dim=-1)
        # (B, L, H, P, R): SISO projection then elementwise R expansion.
        X = (
            x.view(B_, L, h, dh).unsqueeze(-1)
            * self.x_expand.view(1, 1, h, dh, r)
        )
        Z = z.view(B_, L, h, dh).unsqueeze(-1) * self.z_expand.view(1, 1, h, dh, r)
        proj = self.in_proj_bcdt_theta(u)
        Bc, Cc, dt, theta = proj.split([g * n, g * n, h, h * (n // 2)], dim=-1)
        bc = rms_norm(torch.cat([Bc, Cc], dim=-1), self.bc_norm_weight) + self.bc_bias
        Bc, Cc = bc.chunk(2, dim=-1)
        # Full DNR-style rank expansion for shared B, C.
        Bc = self.bc_rank_proj(Bc).view(B_, L, g, n, r)
        Cc = self.bc_rank_proj(Cc).view(B_, L, g, n, r)
        theta_dt = F.softplus(theta.view(B_, L, h, n // 2))
        dt = F.softplus(dt)
        logdecay = -dt * torch.exp(self.A_log)
        lam = torch.sigmoid(dt)
        return X, logdecay, Bc, Cc, theta_dt, lam, Z

    def _contract(self, Y: Tensor) -> Tensor:
        # (B, L, H, P, R) -> (B, L, H*P) via learned R contraction.
        w = self.out_contract.view(1, 1, self.n_heads, self.d_head, self.mimo_rank)
        return (Y * w).sum(-1).reshape(Y.size(0), Y.size(1), -1)

    def _gate(self, Y: Tensor, Z: Tensor) -> Tensor:
        g = F.silu(Z)
        w = self.out_contract.view(1, 1, self.n_heads, self.d_head, self.mimo_rank)
        return ((Y * g * w).sum(-1)).reshape(Y.size(0), Y.size(1), -1)

    def forward(self, u: Tensor) -> Tensor:
        X, logdecay, Bc, Cc, theta_dt, lam, Z = self._core_inputs(u)
        Y, _ = mimo_loop(X, logdecay, Bc, Cc, theta_dt, lam)
        return self.out_proj(self._gate(Y, Z))

    def forward_chunkwise(self, u: Tensor) -> Tensor:
        X, logdecay, Bc, Cc, theta_dt, lam, Z = self._core_inputs(u)
        Y, _ = mimo_chunkwise(X, logdecay, Bc, Cc, theta_dt, lam)
        return self.out_proj(self._gate(Y, Z))

    def prefill(self, u: Tensor):
        comp = compute_dtype_of(u.dtype)
        X, logdecay, Bc, Cc, theta_dt, lam, Z = self._core_inputs(u)
        Y, final = mimo_loop(X, logdecay, Bc, Cc, theta_dt, lam)
        out = self.out_proj(self._gate(Y, Z))
        th = theta_dt.to(comp)
        # Cache matches SISO shapes (state has no R); keep rank-summed x/b carry.
        Xc = X.to(comp)
        Bc_ = Bc.to(comp)
        cache = Mamba3SisoCache(
            angle=th.sum(dim=1).detach(),
            ssm=final.detach().to(comp),
            x_prev=Xc[:, -1, :].mean(dim=-1).detach(),
            b_prev=Bc_[:, -1, :].mean(dim=-1).detach(),
        )
        aux = {
            "x_prev_full": Xc[:, -1, :].detach(),
            "b_prev_full": Bc_[:, -1, :].detach(),
        }
        return out, (cache, aux)

    def step(self, u_t: Tensor, state) -> Tensor:
        comp = compute_dtype_of(u_t.dtype)
        cache, aux = state
        X, logdecay, Bc, Cc, theta_dt, lam, Z = self._core_inputs(u_t)
        Xc = X.to(comp)
        Bc_ = Bc.to(comp)
        Cc_ = Cc.to(comp)
        th = theta_dt.to(comp)
        rep = self.n_heads // self.n_groups
        from torchmamba.core.rope import (
            apply_pairwise_rotation,
            pairwise_angles_to_cos_sin,
        )

        B_, _, h, dh, r = Xc.shape
        n = self.d_state
        Bc_g = Bc_.repeat_interleave(rep, dim=2)  # (B,1,H,N,R)
        Cc_g = Cc_.repeat_interleave(rep, dim=2)
        new_angle = cache.angle.to(comp) + th[:, 0, :]
        cos, sin = pairwise_angles_to_cos_sin(new_angle)
        # Rotate per (rank, pair).
        Bc_p = Bc_g.permute(0, 1, 2, 4, 3).reshape(B_, 1, h * r, n)
        Cc_p = Cc_g.permute(0, 1, 2, 4, 3).reshape(B_, 1, h * r, n)
        # (B, H*R, N/2) -> (B, 1, H*R, N/2) to align with (B, 1, H*R, N).
        cos_r = cos.repeat_interleave(r, dim=1).unsqueeze(1)
        sin_r = sin.repeat_interleave(r, dim=1).unsqueeze(1)
        Bt = apply_pairwise_rotation(Bc_p, cos_r, sin_r).view(B_, 1, h, r, n).permute(
            0, 1, 2, 4, 3
        )
        Ct = apply_pairwise_rotation(Cc_p, cos_r, sin_r).view(B_, 1, h, r, n).permute(
            0, 1, 2, 4, 3
        )
        cos_p, sin_p = pairwise_angles_to_cos_sin(cache.angle.to(comp))
        # aux b_prev_full: (B, H, N, R) -> pairs
        bpp = (
            aux["b_prev_full"]
            .to(comp)
            .repeat_interleave(rep, dim=1)
            .permute(0, 1, 3, 2)
            .reshape(B_, h * r, n)
        )
        cos_pr = cos_p.repeat_interleave(r, dim=1)
        sin_pr = sin_p.repeat_interleave(r, dim=1)
        Bp = apply_pairwise_rotation(bpp, cos_pr, sin_pr).view(B_, h, r, n).permute(
            0, 1, 3, 2
        )
        x_prev = aux["x_prev_full"].to(comp)  # (B,H,P,R), no L dim
        a = torch.exp(logdecay.to(comp))[:, 0, :].view(B_, h, 1, 1)
        lv = lam.to(comp)[:, 0, :].view(B_, h, 1, 1)
        h = (
            a * cache.ssm.to(comp)
            + (1 - lv) * a * torch.einsum("bhnr,bhpr->bhnp", Bp, x_prev)
            + lv * torch.einsum("bhnr,bhpr->bhnp", Bt[:, 0, :], Xc[:, 0, :])
        )
        Y = torch.einsum("bhnr,bhnp->bhpr", Ct[:, 0, :], h).unsqueeze(1)
        out = self.out_proj(self._gate(Y.to(u_t.dtype), Z))
        new_cache = Mamba3SisoCache(
            angle=new_angle.detach(),
            ssm=h.detach(),
            x_prev=Xc[:, 0, :].mean(dim=-1).detach(),
            b_prev=Bc_[:, 0, :].mean(dim=-1).detach(),
        )
        new_aux = {"x_prev_full": Xc[:, 0, :].detach(), "b_prev_full": Bc_[:, 0, :].detach()}
        return out, (new_cache, new_aux)
