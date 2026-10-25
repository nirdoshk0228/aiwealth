"""Equal-risk-contribution (risk parity) portfolio, long-only.

Pure functions: no I/O, no network, no imports from app.data or app.ai.

Target: every asset contributes the same share of portfolio volatility,

    RC_i = w_i (Sigma w)_i / sigma_p = sigma_p / n       (Euler components, see
                                                          app.engine.risk.attribution)

Method (Spinu 2013, "An algorithm for computing risk parity weights"): the ERC portfolio is
y / sum(y) where y solves the strictly convex problem

    min_{y > 0}  1/2 y'Sigma y - b' ln y,      b_i = 1/n (the risk budget)

whose first-order condition y_i (Sigma y)_i = b_i is exactly "equal risk contribution". It is
solved by cyclical coordinate descent (Griveau-Billion, Richard and Roncalli 2013): fixing the
other coordinates, the condition for y_i is the quadratic
    Sigma_ii y_i^2 + (sum_{j != i} Sigma_ij y_j) y_i - b_i = 0
with one positive root, taken in closed form. Each sweep lowers the convex objective; it stops
when the relative spread of risk contributions is below `tol`.

Properties: needs no expected returns (unlike mean-variance), so it is far less sensitive to
estimation error; with a diagonal Sigma it gives w_i ~ 1/sigma_i (inverse volatility, tested).
A weight cap is not applied (risk parity weights are what they are; the result reports them).
"""

import math
from dataclasses import dataclass

import numpy as np

from app.engine.risk.attribution import euler_contributions

DEFAULT_TOL = 1e-10
MAX_SWEEPS = 10_000


@dataclass(frozen=True)
class RiskParityResult:
    weights: np.ndarray  # sums to 1, all > 0
    risk_shares: np.ndarray  # component / sigma_p, all ~ 1/n
    volatility: float  # in the units of sqrt(cov)
    sweeps: int
    max_share_error: float  # max_i |share_i - 1/n|


def risk_parity_weights(
    cov: np.ndarray, *, tol: float = DEFAULT_TOL, max_sweeps: int = MAX_SWEEPS
) -> RiskParityResult:
    """Equal-risk-contribution weights for covariance `cov` (any time unit)."""
    cov = np.asarray(cov, dtype=float)
    n = cov.shape[0]
    if cov.ndim != 2 or cov.shape != (n, n) or n < 1:
        raise ValueError(f"cov must be a square matrix, got {cov.shape}")
    if not np.isfinite(cov).all():
        raise ValueError("cov must be finite")
    diag = np.diag(cov)
    if (diag <= 0).any():
        raise ValueError("every asset needs a positive variance for risk parity")
    cov = 0.5 * (cov + cov.T)
    b = np.full(n, 1.0 / n)
    y = 1.0 / np.sqrt(diag)  # inverse-volatility start
    y *= math.sqrt(1.0 / float(y @ cov @ y))  # scale so that y'Sigma y = 1 (= sum b)

    sweeps = 0
    while sweeps < max_sweeps:
        sweeps += 1
        for i in range(n):
            s = float(cov[i] @ y) - cov[i, i] * y[i]  # sum_{j != i} Sigma_ij y_j
            y[i] = (-s + math.sqrt(s * s + 4.0 * cov[i, i] * b[i])) / (2.0 * cov[i, i])
        rc = y * (cov @ y)
        spread = float((rc.max() - rc.min()) / rc.mean())
        if spread < tol:
            break
    w = y / y.sum()
    sigma, _, comp = euler_contributions(cov, w)
    shares = comp / sigma
    return RiskParityResult(
        weights=w,
        risk_shares=shares,
        volatility=sigma,
        sweeps=sweeps,
        max_share_error=float(np.max(np.abs(shares - 1.0 / n))),
    )
