"""Factor regressions: OLS of excess returns on factor returns with Newey-West HAC errors.

Pure numpy/pandas: no I/O, no network, no imports from app.data or app.ai (enforced by
tests/test_architecture.py). Factors come from app.services.factors.load_aligned_factors.

Model
-----
    r_t - RF_t = alpha + sum_k beta_k f_{k,t} + e_t,        t = 1..n

with f = (MKT_RF, SMB, HML, WML) by default: the Carhart (1997) four-factor model on the IIMA
Indian factors (docs/data.md). r is a **simple** daily return in decimals; RF is the daily
91-day T-bill return from the factor file, row by row. (Sharpe and Sortino in
app/engine/risk/metrics.py use a constant annual rate instead; the two are not the same RF.)

Estimation (X = [1, f_1, ..., f_K], p = K + 1 columns)
----------
    b            = argmin ||y - X b||^2               (np.linalg.lstsq)
    e            = y - X b
    R^2          = 1 - e'e / sum (y_t - mean(y))^2
    adj R^2      = 1 - (1 - R^2) (n - 1) / (n - p)
    residual vol = sqrt(e'e / (n - p)) * sqrt(252)    (annualised std of e)
    alpha_annual = 252 * alpha                         (arithmetic, like Jensen's alpha in
                                                        metrics.py; not compounded)

Newey-West (1987) HAC covariance
--------------------------------
    u_t = e_t x_t
    S   = sum_t u_t u_t'
          + sum_{l=1..L} w_l sum_{t=l+1..n} (u_t u_{t-l}' + u_{t-l} u_t'),   w_l = 1 - l/(L + 1)
    V   = n / (n - p) * (X'X)^-1 S (X'X)^-1
    se  = sqrt(diag V),  t = coefficient / se,  p-value two-sided from the standard normal
          (HAC inference is asymptotic)

- Why HAC: daily returns are heteroskedastic (volatility clusters) and mildly autocorrelated
  (stale prices, NAVs struck from closes), so classical OLS standard errors are too small.
- Lag L: Newey and West (1994) plug-in rule L = floor(4 (n/100)^(2/9)), i.e. 8 for the 2,713
  daily rows of release 2025-12. Override with `hac_lags`.
- Small-sample convention: the n / (n - p) factor (HC1-style). It is what statsmodels applies
  with cov_type="HAC", cov_kwds={"maxlags": L, "use_correction": True}; see
  scripts/crosscheck_factor_regression.py. Without it, V is smaller by (n - p) / n.
- With L = 0, V is White's HC1, **not** the classical s^2 (X'X)^-1. The two coincide when every
  residual has the same absolute value (tests/engine/factors/test_regression.py).

Inputs and data window
----------------------
`returns` and `factors` must have **identical** indexes (same dates, same order) and contain no
NaN; nothing is re-aligned or dropped silently. With release 2025-12 that means the 2,713
rows on which align_factors produced data (2015-01-05 to 2025-12-31): the caller selects
`aligned.factors.dropna()` and the matching return rows. The rows from 2026-01-01 on have no
factor data yet and are excluded.

Validation
----------
- n >= min_obs (default 504, two years). Five parameters plus a long-run (HAC) variance with
  about 8 lags need far more rows than parameters; one year gives unstable WML and HML betas.
- Collinearity: condition number of X with every column scaled to unit length (Belsley, Kuh
  and Welsch 1980). Above 1e6 the regression raises; above 30 it adds a warning to the result.
  A constant factor column is collinear with the intercept and raises with its own message.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252
DEFAULT_FACTORS = ("MKT_RF", "SMB", "HML", "WML")
RF_COLUMN = "RF"
DEFAULT_MIN_OBS = 504
COND_WARN = 30.0
COND_RAISE = 1e6


# --- result types ---


@dataclass(frozen=True)
class OlsHacFit:
    """Raw numpy fit. params[0] is the intercept (daily alpha)."""

    params: np.ndarray
    cov: np.ndarray  # HAC covariance of params, with the n / (n - p) correction
    resid: np.ndarray
    lags: int


@dataclass(frozen=True)
class FactorRegressionResult:
    """One regression. Coefficient dicts are keyed "alpha" (daily) and the factor names."""

    factors: tuple[str, ...]
    alpha_daily: float
    alpha_annual: float  # 252 * alpha_daily
    betas: Mapping[str, float]
    std_errors: Mapping[str, float]  # HAC; "alpha" in daily units
    t_stats: Mapping[str, float]  # NaN where the standard error is exactly 0
    p_values: Mapping[str, float]  # two-sided, normal approximation
    r_squared: float
    adj_r_squared: float
    residual_vol_annual: float
    n_obs: int
    hac_lags: int
    condition_number: float  # of the column-scaled design matrix
    start: pd.Timestamp
    end: pd.Timestamp
    warnings: tuple[str, ...]

    def to_row(self) -> dict[str, float]:
        """Flat dict for a table: alpha, t-stats and betas, then fit statistics."""
        row = {"alpha_annual": self.alpha_annual, "t_alpha": self.t_stats["alpha"]}
        for f in self.factors:
            row[f"beta_{f}"] = self.betas[f]
            row[f"t_{f}"] = self.t_stats[f]
        row |= {
            "r_squared": self.r_squared,
            "adj_r_squared": self.adj_r_squared,
            "residual_vol_annual": self.residual_vol_annual,
            "n_obs": float(self.n_obs),
            "hac_lags": float(self.hac_lags),
            "condition_number": self.condition_number,
        }
        return row


# --- numerical core (numpy only) ---


def newey_west_lags(n: int) -> int:
    """Newey-West (1994) plug-in lag: floor(4 (n/100)^(2/9)). 4 at n=100, 8 at n=2,713."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    return math.floor(4 * (n / 100) ** (2 / 9))


def scaled_condition_number(X: np.ndarray) -> float:
    """2-norm condition number of X after scaling each column to unit length; inf if a
    column is all zeros or X is rank-deficient."""
    norms = np.linalg.norm(X, axis=0)
    if (norms == 0).any():
        return math.inf
    return float(np.linalg.cond(X / norms))


def hac_covariance(X: np.ndarray, resid: np.ndarray, lags: int) -> np.ndarray:
    """Newey-West covariance of OLS coefficients, Bartlett kernel, n/(n-p) correction.

    See the module docstring for the formula. lags = 0 gives White's HC1.
    """
    n, p = X.shape
    if not 0 <= lags < n:
        raise ValueError(f"lags must be in [0, n), got {lags} with n = {n}")
    if n <= p:
        raise ValueError(f"need more observations than parameters (n = {n}, p = {p})")
    u = X * resid[:, None]
    S = u.T @ u
    for lag in range(1, lags + 1):
        gamma = u[lag:].T @ u[:-lag]
        S += (1 - lag / (lags + 1)) * (gamma + gamma.T)
    bread = np.linalg.inv(X.T @ X)
    return n / (n - p) * (bread @ S @ bread)


def ols_hac(y: np.ndarray, X: np.ndarray, lags: int) -> OlsHacFit:
    """OLS coefficients, residuals and HAC covariance. No input validation (see
    `factor_regression`)."""
    params, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ params
    return OlsHacFit(params=params, cov=hac_covariance(X, resid, lags), resid=resid, lags=lags)


# --- validation ---


def _check_inputs(
    returns: pd.Series,
    factors: pd.DataFrame,
    factor_names: Sequence[str],
    rf_column: str,
    min_obs: int,
) -> None:
    if not isinstance(returns, pd.Series):
        raise TypeError(
            "returns must be a pandas Series; use factor_regression_by_asset for a frame"
        )
    if not isinstance(factors, pd.DataFrame):
        raise TypeError("factors must be a pandas DataFrame")
    if not factor_names:
        raise ValueError("factor_names must not be empty")
    if len(set(factor_names)) != len(factor_names):
        raise ValueError(f"duplicate factor names: {list(factor_names)}")
    if rf_column in factor_names:
        raise ValueError(f"{rf_column!r} is the risk-free column, not a factor")
    missing = [c for c in (*factor_names, rf_column) if c not in factors.columns]
    if missing:
        raise ValueError(f"factors: missing columns {missing}")
    if min_obs < len(factor_names) + 2:
        raise ValueError(f"min_obs must be >= number of parameters + 1, got {min_obs}")

    if not isinstance(returns.index, pd.DatetimeIndex):
        raise ValueError("returns: index must be a DatetimeIndex")
    if not returns.index.is_monotonic_increasing or not returns.index.is_unique:
        raise ValueError("returns: dates must be strictly increasing")
    if not returns.index.equals(factors.index):
        only_r = len(returns.index.difference(factors.index))
        only_f = len(factors.index.difference(returns.index))
        raise ValueError(
            "returns and factors must have identical indexes (same dates, same order): "
            f"{only_r} return dates not in factors, {only_f} factor dates not in returns; "
            "select the rows with factor data first (aligned.factors.dropna())"
        )


def _check_values(values: np.ndarray, label: str, hint: str) -> None:
    if np.isnan(values).any():
        raise ValueError(f"{label} contain NaN; {hint}")
    if not np.isfinite(values).all():
        raise ValueError(f"{label} contain inf")


# --- public API ---


def factor_regression(
    returns: pd.Series,
    factors: pd.DataFrame,
    *,
    factor_names: Sequence[str] = DEFAULT_FACTORS,
    rf_column: str = RF_COLUMN,
    hac_lags: int | None = None,
    min_obs: int = DEFAULT_MIN_OBS,
) -> FactorRegressionResult:
    """Regress one return series (a portfolio or an asset) in excess of RF on the factors.

    Args:
        returns: simple daily returns (decimals), DatetimeIndex, no NaN.
        factors: decimal factor returns with `factor_names` and `rf_column`, index identical
            to `returns` (rows with factor data only; see the module docstring).
        factor_names: regressors, in output order. Default MKT_RF, SMB, HML, WML.
        rf_column: daily risk-free return column, subtracted from `returns`.
        hac_lags: Newey-West lag L; None uses floor(4 (n/100)^(2/9)).
        min_obs: minimum number of rows (default 504).

    Returns:
        FactorRegressionResult (alpha daily and annual = 252 x daily, betas, HAC standard
        errors, t-statistics, p-values, R^2, adjusted R^2, annualised residual volatility).

    Raises:
        TypeError / ValueError: wrong types, misaligned indexes, NaN or inf, too few rows,
            invalid lag, a constant factor, or a scaled condition number above 1e6.
    """
    names = tuple(factor_names)
    _check_inputs(returns, factors, names, rf_column, min_obs)
    r = returns.to_numpy(dtype=float)
    F = factors[list(names)].to_numpy(dtype=float)
    rf = factors[rf_column].to_numpy(dtype=float)
    _check_values(r, "returns", "build them with align_prices + simple_returns")
    hint = "select the rows with factor data first (aligned.factors.dropna())"
    _check_values(F, "factors", hint)
    _check_values(rf, rf_column, hint)

    n, p = len(r), len(names) + 1
    if n < min_obs:
        raise ValueError(f"need at least {min_obs} observations, got {n}")
    lags = newey_west_lags(n) if hac_lags is None else hac_lags
    if isinstance(lags, bool) or not isinstance(lags, int) or not 0 <= lags < n:
        raise ValueError(f"hac_lags must be an int in [0, n), got {hac_lags!r} with n = {n}")

    for j, name in enumerate(names):
        if np.ptp(F[:, j]) == 0:
            raise ValueError(f"factor {name!r} is constant: collinear with the intercept")
    X = np.column_stack([np.ones(n), F])
    cond = scaled_condition_number(X)
    if not math.isfinite(cond) or cond > COND_RAISE:
        raise ValueError(
            f"collinear factors: scaled condition number {cond:.3g} > {COND_RAISE:.0e} "
            f"(regressors: intercept, {', '.join(names)})"
        )
    warnings = []
    if cond > COND_WARN:
        warnings.append(
            f"scaled condition number {cond:.1f} > {COND_WARN:.0f}: factors are strongly "
            "correlated, so individual betas are imprecise"
        )

    y = r - rf
    fit = ols_hac(y, X, lags)
    se = np.sqrt(np.maximum(np.diag(fit.cov), 0.0))
    t = np.divide(fit.params, se, out=np.full(p, np.nan), where=se > 0)
    pv = np.array([math.erfc(abs(x) / math.sqrt(2)) if math.isfinite(x) else math.nan for x in t])

    rss = float(fit.resid @ fit.resid)
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - rss / tss if tss > 0 else math.nan
    keys = ("alpha", *names)
    return FactorRegressionResult(
        factors=names,
        alpha_daily=float(fit.params[0]),
        alpha_annual=TRADING_DAYS * float(fit.params[0]),
        betas={k: float(v) for k, v in zip(names, fit.params[1:], strict=True)},
        std_errors=dict(zip(keys, map(float, se), strict=True)),
        t_stats=dict(zip(keys, map(float, t), strict=True)),
        p_values=dict(zip(keys, map(float, pv), strict=True)),
        r_squared=r2,
        adj_r_squared=1 - (1 - r2) * (n - 1) / (n - p),
        residual_vol_annual=math.sqrt(rss / (n - p)) * math.sqrt(TRADING_DAYS),
        n_obs=n,
        hac_lags=lags,
        condition_number=cond,
        start=returns.index[0],
        end=returns.index[-1],
        warnings=tuple(warnings),
    )


def factor_regression_by_asset(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    *,
    factor_names: Sequence[str] = DEFAULT_FACTORS,
    rf_column: str = RF_COLUMN,
    hac_lags: int | None = None,
    min_obs: int = DEFAULT_MIN_OBS,
) -> pd.DataFrame:
    """`factor_regression` for each column of `returns`.

    One row per asset with the columns of `FactorRegressionResult.to_row`. Same arguments
    and validation as `factor_regression`.
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame (columns = asset ids)")
    if returns.shape[1] == 0:
        raise ValueError("returns has no columns")
    if not returns.columns.is_unique:
        raise ValueError("returns: duplicate column names")
    rows = {
        col: factor_regression(
            returns[col],
            factors,
            factor_names=factor_names,
            rf_column=rf_column,
            hac_lags=hac_lags,
            min_obs=min_obs,
        ).to_row()
        for col in returns.columns
    }
    return pd.DataFrame.from_dict(rows, orient="index")
