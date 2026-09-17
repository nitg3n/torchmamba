"""Cross-model reductions: M1->M2 (P=1), M2->SISO, SISO->MIMO, ZOH-vs-EE."""

import torch

from torchmamba.core.discretize import exp_euler, zoh
from torchmamba.mamba1.core import selective_scan_loop
from torchmamba.mamba2.core import ssd_recurrent
from torchmamba.mamba3.mimo_core import mimo_loop
from torchmamba.mamba3.siso_core import trap_rope_loop


def test_m1_vs_m2_mis_reduction():
    # M1 S6 with P=1 per-channel dynamics == M2 scalar head with tied decay.
    # Setup: B=1, D=H=2 channels/heads, N=3, L=6. M1 A:(D,N) rows tied per
    # channel to a scalar a_d (same decay across N, matching M2 scalar head).
    torch.manual_seed(0)
    B, L, D, N = 1, 6, 2, 3
    x = torch.randn(B, L, D, dtype=torch.float64)
    # Per-channel scalar decays -> dt*A with A_d constant over N.
    a_ch = -torch.rand(B, L, D, dtype=torch.float64) - 0.05  # log-decays
    dt = -a_ch  # with A=-1..., alpha = exp(-dt) = exp(a_ch)
    A = -torch.ones(D, N, dtype=torch.float64)
    Bc = torch.randn(B, L, N, dtype=torch.float64)
    Cc = torch.randn(B, L, N, dtype=torch.float64)
    y1, _ = selective_scan_loop(x, dt, A, Bc, Cc, None)
    # B-folding (M3 Sec. 2.3 fn. 2): M2 B means discretized gamma*B, so
    # dt folded into B, so per-head B2[b,l,h,n] = dt[b,l,h]*Bc[b,l,n] (G=H),
    # C shared by expansion. X:(B,L,H=D,P=1), logdecay=a_ch.
    B2 = dt.unsqueeze(-1) * Bc.unsqueeze(2).expand(B, L, D, N)
    C2 = Cc.unsqueeze(2).expand(B, L, D, N)
    Y2, _ = ssd_recurrent(x.unsqueeze(-1), a_ch, B2, C2)
    torch.testing.assert_close(Y2.squeeze(-1), y1, rtol=1e-10, atol=1e-12)


def test_m2_vs_siso_both_paths():
    torch.manual_seed(0)
    B, L, H, P, N = 2, 12, 2, 4, 6
    X = torch.randn(B, L, H, P, dtype=torch.float64)
    A = -torch.rand(B, L, H, dtype=torch.float64) - 0.01
    Bc = torch.randn(B, L, 1, N, dtype=torch.float64)
    Cc = torch.randn(B, L, 1, N, dtype=torch.float64)
    Y2, _ = ssd_recurrent(X, A, Bc, Cc)
    Y3, _, _ = trap_rope_loop(X, A, Bc, Cc, None, 1.0)
    torch.testing.assert_close(Y3, Y2, rtol=1e-10, atol=1e-12)
    # Chunk paths agree too.
    from torchmamba.mamba2.core import ssd_chunkwise

    Y2c, _ = ssd_chunkwise(X, A, Bc, Cc, block_len=5)
    torch.testing.assert_close(Y2c, Y2, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(Y2c, Y3, rtol=1e-10, atol=1e-12)


def test_siso_vs_mimo_r1_block():
    torch.manual_seed(0)
    B, L, H, P, N = 1, 8, 2, 3, 4
    X = torch.randn(B, L, H, P, dtype=torch.float64)
    A = -torch.rand(B, L, H, dtype=torch.float64) - 0.01
    Bc = torch.randn(B, L, 1, N, dtype=torch.float64)
    Cc = torch.randn(B, L, 1, N, dtype=torch.float64)
    th = torch.randn(B, L, H, N // 2, dtype=torch.float64) * 0.2
    lam = torch.sigmoid(torch.randn(B, L, H, dtype=torch.float64))
    Ys, _, _ = trap_rope_loop(X, A, Bc, Cc, th, lam)
    Ym, _ = mimo_loop(
        X.unsqueeze(-1), A, Bc.unsqueeze(-1), Cc.unsqueeze(-1), th, lam
    )
    torch.testing.assert_close(Ym.squeeze(-1), Ys, rtol=1e-10, atol=1e-12)


def test_zoh_vs_ee_documented_difference():
    # Guards the default: EE and ZOH MUST differ (>1e-6) on a fixed case,
    # so the ground-truth choice can never silently flip.
    torch.manual_seed(0)
    dt = torch.rand(2, 5, 4, 1) + 0.5
    A = -torch.exp(torch.randn(1, 1, 4, 3))
    Bc = torch.randn(2, 5, 4, 3)
    # Clean direct comparison on shared shapes:
    d = torch.full((4, 3), 0.7)
    a = torch.full((4, 3), -1.5)
    b = torch.ones(4, 3)
    _, be = exp_euler(d, a, b)
    _, bz = zoh(d, a, b)
    assert (be - bz).abs().max().item() > 1e-6
