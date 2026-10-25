"""Value-at-Risk and Conditional VaR (expected shortfall) for a portfolio.

Pure functions: no I/O, no network, no imports from app.data or app.ai (enforced by
tests/test_architecture.py). Inputs come from app.engine.returns.simple_returns.

Conventions
-----------
- Inputs: `returns` is a DataFrame of **simple** daily returns (rows = days, columns =
  asset ids, decimals, no NaN); `weights` maps asset id -> portfolio weight.
- Portfolio return: R_p,t = sum_i w_i * r_i,t. This assumes **fixed weights**, i.e. the
  portfolio is rebalanced back to `weights` every day. A buy-and-hold portfolio drifts
  away from its starting weights; over a 1-day horizon the difference is negligible.
- Weights: any finite numbers summing to 1 (negative = short). Long-only is a product rule
  enforced at the API layer, not here.
- Output: VaR and CVaR are **positive loss fractions** of portfolio value (0.0234 = a
  2.34% loss), always reported with a confidence level and a horizon in trading days.
  They are not clipped at zero: a negative VaR means the portfolio gains even at the
  chosen quantile (large mean relative to volatility) and is reported as such.
- With alpha = 1 - confidence and loss L = -R_p:
      VaR_c  = the loss exceeded with probability (at most) alpha
      CVaR_c = E[L | L >= VaR_c]      (>= VaR_c by construction)
- Horizon: 1 day is the base; longer horizons use the square-root-of-time rule, see
  `scale_to_horizon` for its assumptions and limits.

Methods
-------
- historical:  empirical quantile of the realised portfolio returns.
- parametric:  normal distribution with the sample mean and covariance.
- monte_carlo: simulate asset returns from a multivariate normal with the sample mean and
  covariance, then take the empirical quantile of the simulated portfolio returns.
  NOTE: for a fixed-weight (linear) portfolio, w'X with X ~ N(mu, Sigma) is exactly
  N(w'mu, w'Sigma w), so this is the *same model* as `parametric` plus sampling noise. It
  exists as the scaffolding for non-normal (e.g. Student-t) or non-linear simulation.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from statistics import NormalDist
from typing import Literal

import numpy as np
import pandas as pd

# MAX_HORIZON_DAYS and WEIGHT_SUM_TOL are re-exported: tests and app.engine.stress import them here.
from app.engine.risk._validation import MAX_HORIZON_DAYS as MAX_HORIZON_DAYS
from app.engine.risk._validation import WEIGHT_SUM_TOL as WEIGHT_SUM_TOL
from app.engine.risk._validation import check_returns_frame, check_var_params, weights_vector

DEFAULT_CONFIDENCE = 0.95
DEFAULT_MIN_OBS = 252  # one year of trading days
DEFAULT_N_SIMS = 100_000
DEFAULT_SEED = 0
MIN_N_SIMS = 1_000

Method = Literal["historical", "parametric", "monte_carlo"]
_STD_NORMAL = NormalDist()


@dataclass(frozen=True)
class VaRResult:
    """One VaR/CVaR estimate. `var` and `cvar` are positive loss fractions."""

    method: Method
    confidence: float  # e.g. 0.99
    horizon_days: int  # trading days; 1 = base
    var: float  # 0.0234 = 2.34% of portfolio value
    cvar: float  # expected loss given loss >= var, same units
    n_obs: int  # return observations used for estimation
    n_sims: int | None = None  # monte_carlo only
    seed: int | None = None  # monte_carlo only


def _prepare(
    returns: pd.DataFrame,
    weights: Mapping[str, float] | pd.Series,
    confidence: float,
    horizon_days: int,
    min_obs: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    check_var_params(confidence, horizon_days, min_obs)
    check_returns_frame(returns, min_obs)
    return weights_vector(returns, weights)


# --- shared building blocks ---


def tail_count(n: int, confidence: float) -> int:
    """Number of observations in the loss tail: k = ceil(n * alpha), alpha = 1 - confidence.

    n * alpha is rounded to 9 decimals before the ceiling because binary floating point
    makes e.g. 100 * (1 - 0.95) = 5.000000000000004, which would otherwise give k = 6.
    """
    return math.ceil(round(n * (1 - confidence), 9))


def empirical_var_cvar(portfolio_returns: np.ndarray, confidence: float) -> tuple[float, float]:
    """Order-statistic VaR and CVaR of a sample of portfolio returns.

        alpha = 1 - confidence,  k = ceil(n * alpha)   (see `tail_count`)
        L_(1) >= L_(2) >= ... >= L_(n)                  losses L = -R sorted, worst first
        VaR   = L_(k)                                   k-th worst loss
        CVaR  = (1/k) * sum_{j=1..k} L_(j)              mean of the k worst losses

    VaR is -F_n^{-1}(alpha) with the generalised inverse F_n^{-1}(a) = inf{x : F_n(x) >= a}
    of the empirical CDF, i.e. np.quantile(R, alpha, method="inverted_cdf") up to the
    rounding of n * alpha. Other conventions differ slightly: np.quantile's default
    ("linear") and Excel's PERCENTILE.INC interpolate between neighbouring order statistics,
    so they give a value between L_(k) and L_(k+1). CVaR uses whole observations (no
    fractional weight on L_(k) when n * alpha is not an integer). CVaR >= VaR always.

    Requires n * alpha >= 1 (at least one full observation in the tail); raises otherwise.
    """
    n = len(portfolio_returns)
    if n * (1 - confidence) < 1 - 1e-9:
        raise ValueError(
            f"{n} observations leave no full observation in the {1 - confidence:.2%} tail; "
            f"need at least {math.ceil(round(1 / (1 - confidence), 9))}"
        )
    k = tail_count(n, confidence)
    worst_first = np.sort(-np.asarray(portfolio_returns, dtype=float))[::-1]
    return float(worst_first[k - 1]), float(worst_first[:k].mean())


def scale_to_horizon(one_day: float, horizon_days: int) -> float:
    """Square-root-of-time rule: X_h = sqrt(h) * X_1 for X in {VaR, CVaR}.

    Exact only if daily portfolio returns are i.i.d. normal with **zero mean**: then the
    h-day sum is N(0, h * sigma^2) and every quantile scales with sqrt(h). Limits:
    - A non-zero mean scales with h, not sqrt(h); for 1-day VaR it is small, but the rule
      misstates the drift over longer horizons.
    - Volatility clustering (calm periods followed by turbulent ones) and autocorrelation
      break the i.i.d. assumption; after a shock, sqrt(h) understates multi-day risk.
    - Fat tails do not scale like a normal's (sums of fat-tailed returns get *less*
      fat-tailed), so scaling a historical 1-day quantile is an approximation.
    - Compounding is ignored (h-day simple return != sum of daily returns).
    Hence horizon_days is capped at MAX_HORIZON_DAYS; use 1 day unless you need e.g. the
    regulatory-style 10-day figure.
    """
    return math.sqrt(horizon_days) * one_day


def portfolio_returns(returns: pd.DataFrame, weights: Mapping[str, float] | pd.Series) -> pd.Series:
    """Daily portfolio simple returns with fixed weights (rebalanced daily).

        R_p,t = sum_i w_i * r_i,t

    Inputs: `returns` simple daily returns (decimals, no NaN), `weights` asset id -> weight
    (finite, summing to 1; ids matched by label, not position). Columns of `returns`
    without a weight are ignored. Returns a Series on the same index, named "portfolio".
    """
    check_returns_frame(returns, min_obs=1)
    sub, w = weights_vector(returns, weights)
    return pd.Series(sub.to_numpy(dtype=float) @ w, index=returns.index, name="portfolio")


# --- the three methods ---


def historical_var(
    returns: pd.DataFrame,
    weights: Mapping[str, float] | pd.Series,
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    horizon_days: int = 1,
    min_obs: int = DEFAULT_MIN_OBS,
) -> VaRResult:
    """Historical-simulation VaR and CVaR.

    Formula: R_p,t = w . r_t for every observed day, then the order statistics of
    `empirical_var_cvar` (VaR = k-th worst loss, CVaR = mean of the k worst losses,
    k = ceil(n * (1 - confidence))), scaled by sqrt(horizon_days).

    Inputs/units: simple daily returns (decimals), weights summing to 1, confidence in
    (0.5, 1), horizon in trading days (1..MAX_HORIZON_DAYS). Output: positive loss fractions.

    Assumptions: the last n days are a representative sample of tomorrow's return
    distribution (no distributional assumption, but no weighting of recent days either);
    fixed weights. Caveat on sample size: with n = 252 (the default minimum) a 99% VaR rests
    on k = ceil(2.52) = 3 tail observations, so it is noisy and jumps when one extreme day
    enters or leaves the window. Prefer >= 3 years of data for 99%.
    """
    sub, w = _prepare(returns, weights, confidence, horizon_days, min_obs)
    var, cvar = empirical_var_cvar(sub.to_numpy(dtype=float) @ w, confidence)
    return VaRResult(
        method="historical",
        confidence=confidence,
        horizon_days=horizon_days,
        var=scale_to_horizon(var, horizon_days),
        cvar=scale_to_horizon(cvar, horizon_days),
        n_obs=len(sub),
    )


def parametric_var(
    returns: pd.DataFrame,
    weights: Mapping[str, float] | pd.Series,
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    horizon_days: int = 1,
    min_obs: int = DEFAULT_MIN_OBS,
) -> VaRResult:
    """Parametric (variance-covariance, normal) VaR and CVaR.

        mu_p    = w' mu_hat                 mu_hat = sample mean of daily returns
        sigma_p = sqrt(w' Sigma_hat w)      Sigma_hat = sample covariance (ddof = 1)
        z       = Phi^{-1}(confidence)      e.g. 1.6449 (95%), 2.3263 (99%)
        VaR     = z * sigma_p - mu_p
        CVaR    = sigma_p * phi(z) / alpha - mu_p,   alpha = 1 - confidence

    then both scaled by sqrt(horizon_days). Phi^{-1} and phi come from the standard
    library's statistics.NormalDist (no scipy dependency).

    Inputs/units: as `historical_var`. Output: positive loss fractions.

    Assumptions: portfolio returns are normal (so fat tails and skew are ignored and VaR at
    high confidence is typically *understated* for equities); mean and covariance are
    constant over the sample and estimated without error; fixed weights.
    """
    sub, w = _prepare(returns, weights, confidence, horizon_days, min_obs)
    mu_p = float(sub.mean().to_numpy() @ w)
    variance = float(w @ sub.cov(ddof=1).to_numpy() @ w)
    sigma_p = math.sqrt(max(variance, 0.0))  # guard tiny negative rounding
    alpha = 1 - confidence
    z = _STD_NORMAL.inv_cdf(confidence)
    var = z * sigma_p - mu_p
    cvar = sigma_p * _STD_NORMAL.pdf(z) / alpha - mu_p
    return VaRResult(
        method="parametric",
        confidence=confidence,
        horizon_days=horizon_days,
        var=scale_to_horizon(var, horizon_days),
        cvar=scale_to_horizon(cvar, horizon_days),
        n_obs=len(sub),
    )


def monte_carlo_var(
    returns: pd.DataFrame,
    weights: Mapping[str, float] | pd.Series,
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    horizon_days: int = 1,
    min_obs: int = DEFAULT_MIN_OBS,
    n_sims: int = DEFAULT_N_SIMS,
    seed: int = DEFAULT_SEED,
) -> VaRResult:
    """Monte Carlo VaR and CVaR from a multivariate normal fitted to the sample.

        X_s ~ N(mu_hat, Sigma_hat),  s = 1..n_sims      (simulated daily asset returns)
        R_s = w . X_s                                    (simulated portfolio returns)
        VaR, CVaR = empirical_var_cvar(R, confidence)    (order statistics, as historical)

    then both scaled by sqrt(horizon_days). Draws use np.random.default_rng(seed), so the
    same inputs and seed give bit-identical results. Sampling uses an SVD factorisation of
    Sigma_hat, which works for singular covariances (e.g. two perfectly correlated assets)
    where a Cholesky factorisation would fail.

    Inputs/units: as `historical_var`, plus n_sims >= MIN_N_SIMS and an integer seed.
    Output: positive loss fractions. Sampling error of the 99% VaR is roughly
    sqrt(alpha(1-alpha)/n_sims) / phi(z) ~ 0.3% of sigma_p at n_sims = 100,000.

    Assumptions: same as `parametric_var`. For this fixed-weight linear portfolio the model
    is *identical* to the parametric one (w'X is exactly normal), so the result equals the
    parametric VaR up to sampling noise; it adds information only once the simulated
    distribution or the portfolio valuation is not normal/linear.
    """
    if n_sims < MIN_N_SIMS:
        raise ValueError(f"n_sims must be >= {MIN_N_SIMS}, got {n_sims}")
    sub, w = _prepare(returns, weights, confidence, horizon_days, min_obs)
    rng = np.random.default_rng(seed)
    sims = rng.multivariate_normal(
        sub.mean().to_numpy(), sub.cov(ddof=1).to_numpy(), size=n_sims, method="svd"
    )
    var, cvar = empirical_var_cvar(sims @ w, confidence)
    return VaRResult(
        method="monte_carlo",
        confidence=confidence,
        horizon_days=horizon_days,
        var=scale_to_horizon(var, horizon_days),
        cvar=scale_to_horizon(cvar, horizon_days),
        n_obs=len(sub),
        n_sims=n_sims,
        seed=seed,
    )
