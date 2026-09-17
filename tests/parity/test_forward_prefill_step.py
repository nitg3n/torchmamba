"""Forward == prefill(prefix) + stepwise decode(suffix), all four blocks.

Seeded (torch.manual_seed(0)), fp32, B=2, L=17, prefix=9.
Observed per-model maxima: 1.490e-08 / 5.066e-07 / 7.749e-07 / 7.153e-07,
all < 1e-5.
"""

import torch

from torchmamba.core.policy import compute_dtype_of
from torchmamba.mamba1.block import Mamba1Block, Mamba1Cache
from torchmamba.mamba2.block import Mamba2Block
from torchmamba.mamba3.mimo_block import Mamba3MimoBlock
from torchmamba.mamba3.siso_block import Mamba3SisoBlock


def _zero_m1(blk, B, dtype):
    comp = compute_dtype_of(dtype)
    return Mamba1Cache(
        conv_state=torch.zeros(B, blk.d_inner, blk.d_conv - 1, dtype=comp),
        ssm_state=torch.zeros(B, blk.d_inner, blk.d_state, dtype=comp),
    )


def test_prefill_decode_all_blocks():
    torch.manual_seed(0)
    B, L, pre = 2, 17, 9
    rtol, atol = 1e-5, 1e-6
    dtype = torch.float32
    with torch.no_grad():
        # Mamba-1.
        b1 = Mamba1Block(d_model=8, expand=2, d_state=4, d_conv=3).to(dtype).eval()
        u = torch.randn(B, L, 8, dtype=dtype)
        yf = b1(u)
        _, cache = b1.prefill(u[:, :pre, :])
        # NOTE: prefill cache is post-prefix; step it forward through suffix.
        outs, c = [], cache
        for t in range(pre, L):
            ot, c = b1.step(u[:, t : t + 1, :], c)
            outs.append(ot)
        torch.testing.assert_close(
            torch.cat([yf[:, :pre, :]] + outs, dim=1), yf, rtol=rtol, atol=atol
        )
        # Mamba-1 from-scratch stepping equals forward.
        outs, c = [], _zero_m1(b1, B, dtype)
        for t in range(L):
            ot, c = b1.step(u[:, t : t + 1, :], c)
            outs.append(ot)
        torch.testing.assert_close(torch.cat(outs, dim=1), yf, rtol=rtol, atol=atol)
        assert c.ssm_state.dtype == torch.float32

        # Mamba-2.
        b2 = Mamba2Block(
            d_model=16, n_heads=2, d_head=8, d_state=8, n_groups=1, d_conv=3
        ).to(dtype).eval()
        u2 = torch.randn(B, L, 16, dtype=dtype)
        yf2 = b2(u2)
        _, c2 = b2.prefill(u2[:, :pre, :])
        outs2, cc = [], c2
        for t in range(pre, L):
            ot, cc = b2.step(u2[:, t : t + 1, :], cc)
            outs2.append(ot)
        torch.testing.assert_close(
            torch.cat([yf2[:, :pre, :]] + outs2, dim=1), yf2, rtol=rtol, atol=atol
        )
        assert cc["ssm_state"].dtype == torch.float32

        # Mamba-3 SISO.
        b3 = Mamba3SisoBlock(
            d_model=12, n_heads=2, d_head=4, d_state=6, n_groups=1
        ).to(dtype).eval()
        u3 = torch.randn(B, L, 12, dtype=dtype)
        yf3 = b3(u3)
        _, c3 = b3.prefill(u3[:, :pre, :])
        outs3, cc3 = [], c3
        for t in range(pre, L):
            ot, cc3 = b3.step(u3[:, t : t + 1, :], cc3)
            outs3.append(ot)
        torch.testing.assert_close(
            torch.cat([yf3[:, :pre, :]] + outs3, dim=1), yf3, rtol=rtol, atol=atol
        )
        assert cc3.ssm.dtype == torch.float32

        # Mamba-3 MIMO.
        b4 = Mamba3MimoBlock(
            d_model=12, n_heads=2, d_head=4, d_state=6, n_groups=1, mimo_rank=2
        ).to(dtype).eval()
        yf4 = b4(u3)
        _, s4 = b4.prefill(u3[:, :pre, :])
        outs4, cc4 = [], s4
        for t in range(pre, L):
            ot, cc4 = b4.step(u3[:, t : t + 1, :], cc4)
            outs4.append(ot)
        torch.testing.assert_close(
            torch.cat([yf4[:, :pre, :]] + outs4, dim=1), yf4, rtol=rtol, atol=atol
        )
        assert cc4[0].ssm.dtype == torch.float32
