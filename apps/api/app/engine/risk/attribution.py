"""Risk attribution: how much of a portfolio's risk each position is responsible for.

Pure functions: no I/O, no network, no imports from app.data or app.ai (enforced by
tests/test_architecture.py). Inputs come from app.engine.returns.simple_returns.

Conventions
-----------
- Inputs: as app.engine.risk.var. `returns` is a DataFrame of **simple** daily returns
  (rows = days, columns = asset ids, decimals, no NaN); `weights` maps asset id -> weight
  (finite, summing to 1, negative = short). Columns without a weight are ignored. Fixed
  weights (rebalanced daily), so R_p = w'r and every figure below is a function of w.
- Covariance: the sample covariance Sigma_hat of daily returns (ddof = 1), the same
  estimate `var.parametric_var` uses, so the VaR components here add up to its output.
- Volatility figures are annualised (x sqrt(252)), as in app.engine.risk.metrics. Scaling
  every term by the same constant leaves the percent shares unchanged.
- Output order: per-asset entries follow the order of `weights`; the numbers for an asset
  do not depend on that order or on the column order of `returns`.

The Euler decomposition
-----------------------
A risk measure rho(w) that is homogeneous of degree 1 (rho(k w) = k rho(w), true for
volatility and for parametric VaR/CVaR) satisfies Euler's theorem:

    rho(w) = sum_i w_i * d rho / d w_i

so component_i = w_i * marginal_i splits the total *exactly*. For volatility,
sigma_p = sqrt(w' Sigma w), marginal_i = (Sigma w)_i / sigma_p = cov(r_i, R_p) / sigma_p and
component_i = w_i (Sigma w)_i / sigma_p = w_i * beta_i * sigma_p, with beta_i the beta of
asset i to the portfolio.

What the decomposition means, and what it does not
--------------------------------------------------
- It is conditional on the covariance estimate. A different window, frequency or a
  shrunk/EWMA covariance gives different contributions; they are estimates, not facts.
- The marginal is a local derivative: the change in portfolio risk per unit of extra
  weight in asset i, all other weights held fixed, valid for a *small* change. It does not
  say how the extra weight is financed (a real trade sells something else, and that
  trade's effect is the difference of two marginals). Large reallocations change Sigma w
  and therefore the marginals themselves.
- A component is an allocation of the total, not a stand-alone risk. An asset with
  negative covariance to the portfolio (a hedge) has a **negative** component, so percent
  shares can be below 0 and above 100%. sum_i component_i still equals the total.
- Components are not "the risk removed if the asset is sold". Selling changes the weights
  of everything else (they must be renormalised) and the result is the incremental risk,
  which differs from the component except for infinitesimal changes.
- Only parametric (normal) VaR/CVaR decompose in closed form. Historical and Monte Carlo
  VaR are quantiles of a sample: their Euler components are E[-w_i r_i | L_p = VaR], which
  must be estimated from the handful of observations at (or near) the VaR quantile. That
  estimate is noisy and needs a kernel or window choice, so it is not provided here.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd

from app.engine.risk._validation import check_var_params, prepare_weighted
from app.engine.risk.metrics import TRADING_DAYS, ZERO_TOL
from app.engine.risk.var import DEFAULT_CONFIDENCE, DEFAULT_MIN_OBS, scale_to_horizon

_SQRT_T = math.sqrt(TRADING_DAYS)
_STD_NORMAL = NormalDist()


# --- result types ---


@dataclass(frozen=True)
class AssetVolContribution:
    """One asset's share of portfolio volatility. Volatility units: annualised decimals."""

    asset_id: str
    weight: float
    marginal: float  # d sigma_p / d w_i, annualised (vol per unit of weight)
    component: float  # w_i * marginal; sums to portfolio_volatility over assets
    percent: float  # component / portfolio_volatility; sums to 1, may be < 0 or > 1


@dataclass(frozen=True)
class VolatilityAttribution:
    portfolio_volatility: float  # annualised
    contributions: tuple[AssetVolContribution, ...]  # in the order of `weights`
    n_obs: int

    def to_frame(self) -> pd.DataFrame:
        """One row per asset (index = asset id): weight, marginal, component, percent."""
        return _frame(self.contributions)


@dataclass(frozen=True)
class AssetVaRContribution:
    """One asset's share of parametric VaR/CVaR. Positive loss fractions, like VaRResult."""

    asset_id: str
    weight: float
    component_var: float  # sums to portfolio_var over assets
    component_cvar: float  # sums to portfolio_cvar over assets
    percent_var: float | None  # None if |portfolio_var| <= ZERO_TOL
    percent_cvar: float | None


@dataclass(frozen=True)
class VaRAttribution:
    confidence: float
    horizon_days: int
    portfolio_var: float  # equals var.parametric_var(...).var
    portfolio_cvar: float  # equals var.parametric_var(...).cvar
    contributions: tuple[AssetVaRContribution, ...]
    n_obs: int

    def to_frame(self) -> pd.DataFrame:
        return _frame(self.contributions)


@dataclass(frozen=True)
class RiskBudgetRow:
    asset_id: str
    weight: float  # share of capital (weights sum to 1)
    risk_share: float  # share of volatility (percent contribution; sums to 1)
    ratio: float | None  # risk_share / weight; 1 = risk in line with capital. None if w = 0


@dataclass(frozen=True)
class RiskBudgetReport:
    rows: tuple[RiskBudgetRow, ...]
    max_abs_deviation: float  # max_i |risk_share_i - weight_i|; 0 for equal risk budget
    portfolio_volatility: float  # annualised
    n_obs: int

    def to_frame(self) -> pd.DataFrame:
        return _frame(self.rows)


def _frame(rows: tuple[object, ...]) -> pd.DataFrame:
    df = pd.DataFrame([vars(r) for r in rows])
    return df.set_index("asset_id")


# --- helpers ---


def _ids(weights: Mapping[str, float] | pd.Series) -> list[str]:
    return [str(k) for k in pd.Series(weights, dtype=float).index]


# --- core (array level; also the building block for risk parity) ---


def euler_contributions(cov: np.ndarray, w: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Euler decomposition of portfolio volatility for a covariance matrix and weights.

        sigma_p     = sqrt(w' Sigma w)
        marginal_i  = (Sigma w)_i / sigma_p            = d sigma_p / d w_i
        component_i = w_i * marginal_i,   sum_i component_i = w' Sigma w / sigma_p = sigma_p

    Inputs/units: `cov` an n x n symmetric covariance matrix (any return unit, e.g. daily),
    `w` a length-n weight vector (no sum constraint here). Outputs are in the units of
    sqrt(cov). Returns (sigma_p, marginal, component).

    Assumptions: none beyond `cov` being positive semi-definite. Sigma is never inverted, so
    a singular covariance (perfectly correlated or perfectly hedged assets) is fine.
    Raises ValueError when sigma_p <= ZERO_TOL (1e-12, sized for daily-scale or larger
    covariances): the marginals divide by sigma_p, so a fully hedged, zero-volatility
    portfolio has no defined risk split (it is undefined, not zero).
    """
    cov = np.asarray(cov, dtype=float)
    w = np.asarray(w, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or cov.shape[0] != w.shape[0] or w.ndim != 1:
        raise ValueError(f"cov must be n x n and w length n, got {cov.shape} and {w.shape}")
    if not (np.isfinite(cov).all() and np.isfinite(w).all()):
        raise ValueError("cov and w must be finite")
    sigma_w = cov @ w
    sigma_p = math.sqrt(max(float(w @ sigma_w), 0.0))  # guard tiny negative rounding
    if sigma_p <= ZERO_TOL:
        raise ValueError(
            "portfolio volatility is zero (fully hedged); risk contributions are undefined"
        )
    marginal = sigma_w / sigma_p
    return sigma_p, marginal, w * marginal


# --- public functions ---


def volatility_attribution(
    returns: pd.DataFrame,
    weights: Mapping[str, float] | pd.Series,
    *,
    min_obs: int = DEFAULT_MIN_OBS,
) -> VolatilityAttribution:
    """Marginal, component and percent contribution of each asset to portfolio volatility.

        Sigma_hat     = sample covariance of daily returns (ddof = 1)
        sigma_p       = sqrt(w' Sigma_hat w) * sqrt(252)
        marginal_i    = (Sigma_hat w)_i / sqrt(w' Sigma_hat w) * sqrt(252)
        component_i   = w_i * marginal_i            sum_i component_i = sigma_p (exact)
        percent_i     = component_i / sigma_p       sum_i percent_i   = 1

    Inputs/units: simple daily returns (decimals), weights summing to 1. Output annualised.

    Assumptions: Sigma_hat is the true covariance (no estimation error), constant over the
    sample; fixed weights. See the module docstring for how to read the numbers (negative
    components for hedges, local meaning of the marginal). Raises ValueError if the
    portfolio's volatility is zero.
    """
    sub, w = prepare_weighted(returns, weights, min_obs)
    sigma_d, marginal_d, component_d = euler_contributions(sub.cov(ddof=1).to_numpy(), w)
    rows = tuple(
        AssetVolContribution(
            asset_id=a,
            weight=float(w[i]),
            marginal=float(marginal_d[i]) * _SQRT_T,
            component=float(component_d[i]) * _SQRT_T,
            percent=float(component_d[i]) / sigma_d,
        )
        for i, a in enumerate(_ids(weights))
    )
    return VolatilityAttribution(
        portfolio_volatility=sigma_d * _SQRT_T, contributions=rows, n_obs=len(sub)
    )


def parametric_var_attribution(
    returns: pd.DataFrame,
    weights: Mapping[str, float] | pd.Series,
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    horizon_days: int = 1,
    min_obs: int = DEFAULT_MIN_OBS,
) -> VaRAttribution:
    """Component parametric (normal) VaR and CVaR; components add up to the portfolio figures.

    With mu_hat, Sigma_hat the sample mean and covariance of daily returns,
    sigma_p = sqrt(w' Sigma_hat w), z = Phi^{-1}(confidence), alpha = 1 - confidence:

        VaR     = z sigma_p - w' mu_hat                       (as var.parametric_var)
        CVaR    = (phi(z) / alpha) sigma_p - w' mu_hat
        cVaR_i  = w_i * ( z            (Sigma_hat w)_i / sigma_p - mu_i )
        cCVaR_i = w_i * ( phi(z)/alpha (Sigma_hat w)_i / sigma_p - mu_i )

    then everything is multiplied by sqrt(horizon_days), the same scaling (mean term
    included) var.parametric_var applies, so sum_i cVaR_i = parametric_var(...).var and
    sum_i cCVaR_i = parametric_var(...).cvar to floating-point precision. Both VaR and CVaR
    are degree-1 homogeneous in w (the mean term is linear), so Euler applies exactly.
    percent_* = component / total; None when the total is within ZERO_TOL of zero.

    Inputs/units: as var.parametric_var. Output: positive loss fractions of portfolio value.

    Assumptions: those of var.parametric_var (normal returns, constant and known mean and
    covariance, fixed weights). The mean term makes the VaR shares differ slightly from the
    volatility shares; at a 1-day horizon the difference is usually small. Historical and
    Monte Carlo VaR do not decompose in closed form (see the module docstring).
    """
    check_var_params(confidence, horizon_days, min_obs)
    sub, w = prepare_weighted(returns, weights, min_obs)
    mu = sub.mean().to_numpy()
    _, marginal, _ = euler_contributions(sub.cov(ddof=1).to_numpy(), w)
    z = _STD_NORMAL.inv_cdf(confidence)
    k_cvar = _STD_NORMAL.pdf(z) / (1 - confidence)
    comp_var = scale_to_horizon(1.0, horizon_days) * w * (z * marginal - mu)
    comp_cvar = scale_to_horizon(1.0, horizon_days) * w * (k_cvar * marginal - mu)
    total_var, total_cvar = float(comp_var.sum()), float(comp_cvar.sum())

    def pct(c: float, total: float) -> float | None:
        return None if abs(total) <= ZERO_TOL else c / total

    rows = tuple(
        AssetVaRContribution(
            asset_id=a,
            weight=float(w[i]),
            component_var=float(comp_var[i]),
            component_cvar=float(comp_cvar[i]),
            percent_var=pct(float(comp_var[i]), total_var),
            percent_cvar=pct(float(comp_cvar[i]), total_cvar),
        )
        for i, a in enumerate(_ids(weights))
    )
    return VaRAttribution(
        confidence=confidence,
        horizon_days=horizon_days,
        portfolio_var=total_var,
        portfolio_cvar=total_cvar,
        contributions=rows,
        n_obs=len(sub),
    )


def diversification_ratio(
    returns: pd.DataFrame,
    weights: Mapping[str, float] | pd.Series,
    *,
    min_obs: int = DEFAULT_MIN_OBS,
) -> float | None:
    """Diversification ratio: gross weighted-average asset volatility / portfolio volatility.

        DR = sum_i |w_i| sigma_i / sigma_p
        sigma_i = sqrt(Sigma_hat_ii),   sigma_p = sqrt(w' Sigma_hat w)

    For a long-only portfolio |w_i| = w_i and this is exactly Choueifaty & Coignard's (2008)
    definition. With shorts, |w| makes the numerator a *gross* average (a short position
    adds its volatility, it does not cancel a long one). DR >= 1 always, by the triangle
    inequality sd(sum_i w_i r_i) <= sum_i |w_i| sd(r_i). A signed average could be zero or
    negative and would not be interpretable.

    Reading: DR = 1 when all assets are perfectly correlated (no diversification);
    n equal-volatility uncorrelated assets with equal weights give DR = sqrt(n). Scale-free,
    so daily and annualised volatilities give the same value.

    Inputs/units: as `volatility_attribution`. Returns None if sigma_p <= ZERO_TOL (fully
    hedged; the ratio is undefined), following metrics.py's rule for undefined ratios.
    Assumptions: Sigma_hat is the true covariance, as elsewhere in this module.
    """
    sub, w = prepare_weighted(returns, weights, min_obs)
    cov = sub.cov(ddof=1).to_numpy()
    sigma_p = math.sqrt(max(float(w @ cov @ w), 0.0))
    if sigma_p <= ZERO_TOL:
        return None
    return float(np.abs(w) @ np.sqrt(np.diag(cov))) / sigma_p


def risk_budget_report(
    returns: pd.DataFrame,
    weights: Mapping[str, float] | pd.Series,
    *,
    min_obs: int = DEFAULT_MIN_OBS,
) -> RiskBudgetReport:
    """Capital share vs volatility-risk share per asset.

        weight_i     = w_i                                (sum to 1)
        risk_share_i = w_i (Sigma_hat w)_i / (w' Sigma_hat w)   (= percent contribution, sum to 1)
        ratio_i      = risk_share_i / weight_i            (None if w_i = 0)
                     = (Sigma_hat w)_i / (w' Sigma_hat w) = beta of asset i to the portfolio

    ratio > 1: the asset carries more risk than capital (high volatility and/or high
    correlation with the rest); ratio < 1: less; ratio < 0: it hedges the portfolio. An
    equal-risk-contribution (risk parity) portfolio has risk_share_i = 1/n for all i.
    `max_abs_deviation` = max_i |risk_share_i - weight_i| summarises how far risk is from
    capital. Shares are volatility shares; parametric-VaR shares differ only by the mean term.

    Inputs/units and assumptions: as `volatility_attribution`.
    """
    attr = volatility_attribution(returns, weights, min_obs=min_obs)
    rows = tuple(
        RiskBudgetRow(
            asset_id=c.asset_id,
            weight=c.weight,
            risk_share=c.percent,
            ratio=None if c.weight == 0 else c.percent / c.weight,
        )
        for c in attr.contributions
    )
    return RiskBudgetReport(
        rows=rows,
        max_abs_deviation=max(abs(r.risk_share - r.weight) for r in rows),
        portfolio_volatility=attr.portfolio_volatility,
        n_obs=attr.n_obs,
    )
