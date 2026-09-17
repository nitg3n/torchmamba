"""Discretization rules (M1 Sec. 2; M3 Sec. 3.1, Table 1).

Ground truth for M1/M2 is exponential-Euler (EE):

    alpha = exp(dt * A),  B_disc = dt * B_cont

ZOH is a documented alternative only, never the default path.
Trap (M3) generalizes EE with a data-dependent convex weight lam.
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def softplus_dt(proj: Tensor, bias: Tensor) -> Tensor:
    """dt = softplus(proj + bias), broadcastable."""
    return F.softplus(proj + bias)


def exp_euler(dt: Tensor, A: Tensor, B_cont: Tensor | None = None):
    """Exponential-Euler discretization.

    Args:
        dt: step sizes, broadcastable with A.
        A: continuous decay, broadcastable with dt.
        B_cont: continuous B, broadcastable with dt (or None).

    Returns:
        (alpha, B_disc) if B_cont given else (alpha, dt),
        with alpha = exp(dt * A), B_disc = dt * B_cont.
    """
    alpha = torch.exp(dt * A)
    if B_cont is None:
        return alpha, dt
    return alpha, dt * B_cont


def zoh(dt: Tensor, A: Tensor, B_cont: Tensor):
    """Zero-order-hold discretization (documented alternative).

    B_disc = where(|dt*A| < 1e-4, dt, (alpha - 1) / A) * B_cont.
    Safe against A == 0 via guarded division.
    """
    alpha = torch.exp(dt * A)
    prod = dt * A
    small = prod.abs() < 1e-4
    A_safe = torch.where(A == 0, torch.ones_like(A), A)
    # (alpha - 1) / A_safe is NaN-free; selected only where not small and A != 0.
    coeff = (alpha - 1) / A_safe
    coeff = torch.where(small, dt, coeff)
    return alpha, coeff * B_cont


def trap_coeffs(dt: Tensor, A: Tensor, lam: Tensor | float | None = None):
    """Exponential-trapezoidal coefficients (M3 Prop. 1).

    alpha = exp(dt*A), beta = (1-lam)*dt*alpha, gamma = lam*dt.
    lam=None is bitwise-identical to exp_euler (2-term path).
    """
    if lam is None:
        alpha, dt_out = exp_euler(dt, A, None)
        # beta = 0, gamma = dt exactly matching EE output object.
        beta = torch.zeros_like(alpha)
        return alpha, beta, dt_out
    alpha = torch.exp(dt * A)
    # lam may be scalar float or tensor broadcastable to dt.
    if not isinstance(lam, Tensor):
        lam_t = torch.as_tensor(lam, dtype=alpha.dtype, device=alpha.device)
    else:
        lam_t = lam.to(dtype=alpha.dtype)
    beta = (1 - lam_t) * dt * alpha
    gamma = lam_t * dt
    return alpha, beta, gamma
