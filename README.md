# TorchMamba

Clean, complete, numerically self-consistent **PyTorch reference kernels** for
Mamba-1 (S6), Mamba-2 (SSD), Mamba-3 SISO, and Mamba-3 MIMO — derived from the
papers, not from optimized kernels.

> **Scope: parity with paper math, not with fast kernels.**
> Every recurrence, mask, and block equation follows the papers (section/equation
> cited in each module docstring). No Triton/CUDA/TileLang, no `causal-conv1d`
> package, no `mamba-ssm` import. CPU-runnable, autograd-safe.
> Internal equivalence (loop vs quadratic vs chunkwise vs stepwise, plus
> cross-model reductions) is proven in fp32/fp64 across **41 tests**.
> Bit-parity with `mamba-ssm` kernels and paper benchmark numbers
> (perplexity/throughput) are explicitly out of scope.

## Models

| Model | Paper | Core recurrence | Entry points |
|---|---|---|---|
| Mamba-1 (S6) | Gu & Dao 2023 | `h = exp(dt·A)·h + (dt·B)·x` (exponential-Euler) | `selective_scan_loop`, `selective_scan_matrix`, `Mamba1Block` |
| Mamba-2 (SSD) | Dao & Gu 2024 | scalar-decay `h = a·h + B·x`, chunkwise Listing-1 | `ssd_recurrent`, `ssd_quadratic`, `ssd_chunkwise`, `ssd_step`, `Mamba2Block` |
| Mamba-3 SISO | Lahoti et al. 2026 | 3-term trap `h = αh + βB₋x₋ + γBx` + data-dependent RoPE, no external conv | `trap_rope_loop`, `trap_rope_matrix`, `Mamba3SisoBlock` |
| Mamba-3 MIMO | Lahoti et al. 2026 | rank-R matmul update on shared `(N,P)` state | `mimo_loop`, `mimo_chunkwise`, `mimo_as_r2_sisos`, `Mamba3MimoBlock` |

Reductions (proven in `tests/test_cross_model.py`):
M1→M2 at `P=1` with dt folded into B; M2→SISO at `λ=1, θ=0`;
SISO→MIMO at `R=1` (structural, squeezed internally).

## Requirements

- Python ≥ 3.12
- `torch>=2.14.0` (CPU is enough; CUDA also works)
- `uv` (or any PEP 517 installer + `pytest` for tests)

## Install

```bash
git clone https://github.com/nitg3n/torchmamba.git
cd torchmamba
uv sync              # creates .venv, installs torch + package
```

From PyPI (once published):

```bash
pip install torchmamba
```

## Quickstart

Block-level — `forward` equals `prefill` + stepwise `step` decode for every model:

```python
import torch
from torchmamba import Mamba2Block

torch.manual_seed(0)
blk = Mamba2Block(d_model=16, n_heads=2, d_head=8, d_state=8, n_groups=1, d_conv=3).eval()
u = torch.randn(2, 17, 16)
with torch.no_grad():
    y = blk(u)                          # (2, 17, 16)
    y_pre, cache = blk.prefill(u[:, :9])
    outs = []
    for t in range(9, 17):
        o, cache = blk.step(u[:, t:t+1], cache)
        outs.append(o)
    assert torch.allclose(torch.cat([y[:, :9], *outs], 1), y, rtol=1e-5, atol=1e-6)
```

Functional cores — shapes in, shapes out:

```python
from torchmamba import (
    selective_scan_loop,   # M1: x (B,L,D)   -> y (B,L,D),   h (B,D,N)
    ssd_recurrent,          # M2: X (B,L,H,P) -> Y (B,L,H,P), h (B,H,N,P)
    trap_rope_loop,         # M3 SISO: same as M2 + theta (B,L,H,N/2), lam
    mimo_loop,              # M3 MIMO: X (B,L,H,P,R) -> Y (B,L,H,P,R), h (B,H,N,P)
)

y, h = selective_scan_loop(x, dt, A, B, C)
y, h = ssd_recurrent(X, logdecay, B, C)
y, h, dbg = trap_rope_loop(X, logdecay, B, C, theta_dt, lam)
y, h = mimo_loop(Xr, logdecay, Br, Cr, theta_dt, lam)
```

Subpackage imports work too
(`torchmamba.mamba1`, `torchmamba.mamba2`, `torchmamba.mamba3`,
`torchmamba.core` for `segsum`, discretization, norms, RoPE).

## API reference

Blocks — all expose `forward(u)`, `prefill(u)`, `step(u_t, cache)`:

| Class | Key params | Cache |
|---|---|---|
| `Mamba1Block` | `d_model, expand=2, d_state=16, d_conv=4, dt_rank="auto", bias=True` | `Mamba1Cache(conv_state, ssm_state)` |
| `Mamba2Block` | `d_model, n_heads, d_head=64, d_state=128, n_groups=1, d_conv=4, norm="group", use_normalizer=False, psi=SiLU` | dict `{conv_state, ssm_state}` |
| `Mamba3SisoBlock` | `d_model, n_heads, d_head, d_state, n_groups` (no conv, by design) | `Mamba3SisoCache(angle, ssm, x_prev, b_prev)` |
| `Mamba3MimoBlock` | SISO params + `mimo_rank=4` | `(Mamba3SisoCache, aux)`; state has no `R` axis |

Core functions take/return `(B,L,…)` block I/O (M1 conv helper takes `(B,D,L)`).
Oracle size guards raise `ValueError` above their limits:
`selective_scan_matrix` `L≤32`, `trap_rope_matrix` `L≤16`, `ssd_quadratic` `L≤512`.
Semantics: `lam=1` takes the exact 2-term code path; `theta=None`/zeros is the
real-only path; MIMO `R=1` dispatches to the SISO loop structurally.

Numerics policy (`torchmamba.core.policy`): fp16/bf16/fp32 compute in fp32 with
outputs downcast to input dtype; fp64 stays fp64 end-to-end; states never below
fp32. `H % G == 0` required wherever groups appear; RoPE needs even `N`.

## Math notes (one paragraph per model)

- **M1.** Continuous SSM → exponential-Euler discretization `α=exp(dt·A)`,
  `B_disc=dt·B`. (The paper text claims ZOH; the shipped implementation is EE —
  EE is the ground truth here, ZOH a documented alternative guarded by a test so
  the default can never silently flip.) Input-dependent `(Δ,B,C)` make it LTV:
  scan-only, no convolution view.
- **M2.** Scalar-identity `A=aI` splits decay from content: `M = L∘(CBᵀ)`,
  quadratic dual `Y=(L∘QKᵀ)V`, chunkwise intra-quadratic plus `B→A→C` factored
  inter-chunk carry (`O(TN²)` FLOPs, matmul-bound). Heads follow MVA/GVA.
- **M3 SISO.** Exponential-trapezoidal 3-term recurrence with data-dependent `λ`;
  the parallel mask is `L = 1SS @ band` — a matrix product, not Hadamard.
  Complex state enters via the RoPE trick (accumulated pairwise rotations on
  `B,C`); BCNorm+bias replace the external short convolution (removed).
- **MIMO.** Rank-`R` expansion turns the `B·xᵀ` outer product into a matmul on a
  shared `(N,P)` state (`Θ(R)` intensity at fixed state size); training is `R²`
  SISO calls, chunked `C_MIMO=C_SISO/R` cuts it to `R×`; `R=1` is SISO.

## Repository layout

```text
torchmamba/
├── README.md
├── LICENSE
├── pyproject.toml
├── uv.lock
├── src/torchmamba/
│   ├── __init__.py           # top-level re-exports (19 symbols)
│   ├── core/                 # policy, discretize, segsum, conv, norms, rope
│   ├── mamba1/               # scan loop/matrix + block
│   ├── mamba2/               # recurrent/quadratic/chunkwise/step + block
│   └── mamba3/               # siso_core/block, mimo_core/block
└── tests/                    # 41 tests: primitives, 4 models, reductions,
    ├── parity/               # forward == prefill + stepwise decode
    └── test_*.py
```

## Verification

```bash
uv run pytest -q
# 41 passed
```

Per-model forward-vs-decode maxima live in
`tests/parity/test_forward_prefill_step.py` (seeded, B=2, L=17, prefix=9):
1.490e-08 / 5.066e-07 / 7.749e-07 / 7.153e-07, all < 1e-5.

Tolerances: fp64 `rtol=1e-10, atol=1e-12`; fp32 `rtol=1e-5, atol=1e-6`
(the M2 long-sequence fp32 path documents `2e-5/5e-6` for exp/cumsum rounding).
The suite includes fp64 gradchecks per core, oracle-vs-loop-vs-chunkwise-vs-step
agreement, cross-model reductions, and the ZOH-vs-EE differ-guard.

## Non-goals

- No fused/fast kernels, no varlen `cu_seqlens` API, no distributed sharding.
- No `mamba-ssm` weight compatibility or kernel bit-parity harness.
- No training runs or paper Table 3–7 reproduction.

## References

- Mamba-1: Gu & Dao, *Mamba: Linear-Time Sequence Modeling with Selective State
  Spaces*, arXiv:2312.00752.
- Mamba-2: Dao & Gu, *Transformers are SSMs*, arXiv:2405.21060.
- Mamba-3: Lahoti et al., *Mamba-3: Improved Sequence Modeling using State Space
  Principles*, arXiv:2603.15569.
- Implementation cross-check target (not a dependency):
  [`state-spaces/mamba`](https://github.com/state-spaces/mamba)
  (+ [`causal-conv1d`](https://github.com/Dao-AILab/causal-conv1d)).

## License

MIT — see [LICENSE](LICENSE).
