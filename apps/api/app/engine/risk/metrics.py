"""Risk and performance metrics for a portfolio or a single asset.

Pure functions: no I/O, no network, no imports from app.data or app.ai (enforced by
tests/test_architecture.py). Inputs come from app.engine.returns.simple_returns; a
portfolio's return series comes from app.engine.risk.var.portfolio_returns(returns, weights).

Conventions
-----------
- Inputs: one Series of **simple** daily returns (decimals, no NaN, no inf), i.e. one
  portfolio or one asset. `asset_metrics` and `correlation_matrix` take the whole
  DataFrame (rows = days, columns = asset ids). A benchmark must have exactly the same
  index as the series it is compared with; nothing is re-aligned silently.
- Annualisation uses TRADING_DAYS = 252 and n = the number of return rows:
      volatility-type figures  x sqrt(252)
      mean-type figures        x 252        (arithmetic)
      compounded growth        ^(252 / n)   (geometric, `annualised_return`)
- Rates: the risk-free rate and the minimum acceptable return (MAR) are **annual decimals
  with no default** (0.065 = 6.5% a year), passed by the caller. They are converted to a
  daily rate geometrically, r_d = (1 + r_annual)^(1/252) - 1, consistent with CAGR.
  Sourcing (not done here): for India, the 91-day T-bill yield (RBI auctions / FBIL
  T-bill benchmark) or FBIL Overnight MIBOR, *averaged over the measurement window*.
- Losses are positive: max drawdown 0.35 = a 35% fall from the peak, as for VaR.
  The drawdown *series* is <= 0 (the usual charting convention).
- Undefined ratios: when a denominator (a daily standard deviation, downside deviation or
  benchmark variance) is <= ZERO_TOL, the ratio is None (NaN in `asset_metrics`). It never
  raises and never returns inf. A tolerance is needed because the sample std of a constant
  series is ~1e-18 in floating point, not exactly 0. The dispersion itself is reported 0.0.

Known data limitations (see docs/data.md and docs/methodology.md)
-----------------------------------------------------------------
- ^NSEI is a *price* index; stocks (adjusted closes) and Growth NAVs are total-return
  style. Alpha, active return and information ratio against ^NSEI are therefore biased
  **upward** by roughly the index dividend yield (about 1-1.5% a year). Beta, correlation
  and tracking error are barely affected (the dividend is a near-constant drift).
- ^NSEI is missing 7 real sessions since 2015, which carry a filled 0% benchmark return
  (and a two-day return on the next row). This slightly distorts beta, correlation and
  tracking error, typically pulling beta and correlation a little toward 0.
"""

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.engine.risk._validation import (
    check_annual_rate,
    check_metrics_frame,
    check_same_index,
    check_series,
    check_wealth_domain,
)

TRADING_DAYS = 252
DEFAULT_MIN_OBS = 252  # one year of trading days, as in var.py
ZERO_TOL = 1e-12  # daily dispersion at or below this counts as zero
DD_TOL = 1e-12  # relative tolerance for "back at the peak" in drawdowns

_SQRT_T = math.sqrt(TRADING_DAYS)


# --- result types ---


@dataclass(frozen=True)
class DrawdownEpisode:
    """One underwater period: from a peak, down to a trough, back to the peak (or not yet).

    Lengths are in trading days (return rows). `peak_date` is None when the peak is the
    starting value, i.e. the series fell on its first day. If not recovered,
    `duration_days` runs from the peak to the last observation (underwater so far).
    """

    depth: float  # positive: 0.20 = 20% below the peak at the trough
    peak_date: pd.Timestamp | None
    trough_date: pd.Timestamp
    recovery_date: pd.Timestamp | None  # first date back at (or above) the peak value
    decline_days: int  # peak -> trough
    recovery_days: int | None  # trough -> recovery; None if not recovered
    duration_days: int  # peak -> recovery, or peak -> last observation

    @property
    def recovered(self) -> bool:
        return self.recovery_date is not None


@dataclass(frozen=True)
class DrawdownStats:
    max_drawdown: float  # positive; 0.0 if the series never fell below a previous peak
    max_drawdown_episode: DrawdownEpisode | None  # the deepest episode
    longest_drawdown: DrawdownEpisode | None  # the longest underwater episode
    n_episodes: int
    n_obs: int


@dataclass(frozen=True)
class BenchmarkStats:
    """Single-index regression r_p = a + b * r_b + e and active-return statistics.

    All annualised; see `benchmark_stats` for the formulas.
    """

    beta: float | None  # None if the benchmark has zero variance
    alpha: float | None  # Jensen's alpha, annual, arithmetic
    correlation: float | None  # None if either series has zero variance
    r_squared: float | None
    residual_volatility: float | None  # annualised std of r_p - beta * r_b
    tracking_error: float  # annualised std of r_p - r_b
    active_return: float  # annual arithmetic mean of r_p - r_b
    information_ratio: float | None  # None if tracking error is zero
    risk_free_annual: float
    n_obs: int


@dataclass(frozen=True)
class PerformanceSummary:
    start: pd.Timestamp  # first return date
    end: pd.Timestamp  # last return date
    n_obs: int
    annualised_return: float  # CAGR
    arithmetic_annual_return: float  # mean daily return x 252
    annualised_volatility: float
    downside_deviation: float  # annualised, below mar_annual
    sharpe: float | None
    sortino: float | None
    risk_free_annual: float
    mar_annual: float  # = risk_free_annual in this summary
    drawdown: DrawdownStats
    benchmark: BenchmarkStats | None


# --- helpers ---


def _daily_rate(annual: float) -> float:
    """Geometric de-annualisation: (1 + annual)^(1/252) - 1."""
    return (1 + annual) ** (1 / TRADING_DAYS) - 1


def _std(values: np.ndarray) -> float:
    """Sample std (ddof=1), snapped to 0.0 at or below ZERO_TOL."""
    s = float(np.std(values, ddof=1))
    return 0.0 if s <= ZERO_TOL else s


# --- volatility and return ---


def annualised_volatility(returns: pd.Series, *, min_obs: int = DEFAULT_MIN_OBS) -> float:
    """Annualised volatility.

        sigma = std(r_t, ddof=1) * sqrt(252)

    Inputs/units: simple daily returns (decimals). Output: annual decimal (0.18 = 18%).
    Assumptions: i.i.d. daily returns, so variance scales linearly with time. Volatility
    clustering and autocorrelation (e.g. stale fund NAVs) break this; positive
    autocorrelation makes sqrt(252) scaling *understate* annual risk. Returns exactly 0.0
    if the daily std is <= ZERO_TOL.
    """
    return _std(check_series(returns, min_obs)) * _SQRT_T


def annualised_return(returns: pd.Series, *, min_obs: int = DEFAULT_MIN_OBS) -> float:
    """Geometric annualised return (CAGR) from compounded daily returns.

        G    = prod_t (1 + r_t)              total growth factor over n days
        CAGR = G^(252 / n) - 1

    This is the constant annual rate that turns 1 into G over n/252 years, i.e. what a
    buy-and-hold investor actually earned (for a portfolio: with daily rebalancing).

    It is NOT mean(r) * 252 (`arithmetic_annual_return`). Compared on the same
    (continuously compounded) scale, the geometric growth rate is below the arithmetic
    mean by about half the variance ("volatility drag"):
        ln(1 + CAGR) = 252 * mean(ln(1 + r_t))  ~=  252 * mean(r_t) - sigma_annual^2 / 2
    e.g. +10% then -10% has an arithmetic mean of 0 but loses 1%. CAGR itself compounds
    while mean * 252 does not, so for low-volatility, high-return series CAGR can be the
    *larger* number (constant 0.1% a day: CAGR 28.6% vs mean * 252 = 25.2%); do not read
    "CAGR < arithmetic" as a rule. The arithmetic mean is the input for Sharpe/alpha
    (expected one-period return); CAGR answers "what did it return per year".

    Inputs/units: simple daily returns, all >= -1. Output: annual decimal. n is the number
    of return rows (trading days), not calendar time; with exactly 252 rows a year the two
    agree. Short histories are extrapolated, hence min_obs defaults to 252.
    """
    values = check_series(returns, min_obs)
    check_wealth_domain(values)
    if (values == -1).any():
        return -1.0
    log_growth = float(np.log1p(values).sum())  # sum of logs: no overflow on long series
    return math.expm1(log_growth * TRADING_DAYS / len(values))


def arithmetic_annual_return(returns: pd.Series, *, min_obs: int = DEFAULT_MIN_OBS) -> float:
    """Arithmetic annualised mean return: mean(r_t) * 252.

    Overstates what was actually earned when returns are volatile (see
    `annualised_return`). Used as the numerator of Sharpe and in alpha.
    """
    return float(np.mean(check_series(returns, min_obs))) * TRADING_DAYS


# --- risk-adjusted ratios ---


def downside_deviation(
    returns: pd.Series, *, mar_annual: float, min_obs: int = DEFAULT_MIN_OBS
) -> float:
    """Annualised downside deviation below a minimum acceptable return (MAR).

        m    = (1 + mar_annual)^(1/252) - 1           daily MAR
        DD_d = sqrt( (1/n) * sum_t min(r_t - m, 0)^2 )
        DD   = DD_d * sqrt(252)

    The sum runs over **all n days** (days above the MAR contribute 0), as in Sortino and
    Price (1994); it is not the std of the losing days only, and there is no ddof
    correction. `mar_annual` is required: 0.0 means "losing money", the risk-free rate
    means "underperforming cash". Output: annual decimal; 0.0 if no day is below the MAR.
    """
    check_annual_rate(mar_annual, "mar_annual")
    values = check_series(returns, min_obs)
    shortfall = np.minimum(values - _daily_rate(mar_annual), 0.0)
    dd = math.sqrt(float(np.mean(shortfall**2)))
    return (0.0 if dd <= ZERO_TOL else dd) * _SQRT_T


def sharpe_ratio(
    returns: pd.Series, *, risk_free_annual: float, min_obs: int = DEFAULT_MIN_OBS
) -> float | None:
    """Annualised ex-post Sharpe ratio.

        rf_d   = (1 + risk_free_annual)^(1/252) - 1
        Sharpe = mean(r_t - rf_d) / std(r_t, ddof=1) * sqrt(252)

    rf is constant, so std(r - rf_d) = std(r). Arithmetic mean excess return over its
    volatility, annualised by sqrt(252) (= 252 / sqrt(252)). Some fund factsheets instead
    use (CAGR - rf) / sigma, which differs by (CAGR - 252 * mean(r)) / sigma and can be
    higher or lower (see `annualised_return`); compare like with like. The constant rf
    enters as 252 * rf_d (~ ln(1 + rf)), i.e. exactly as the arithmetic mean of an asset
    that earns rf every day, so a cash-like series has zero excess return.

    Inputs: simple daily returns; risk_free_annual is an annual decimal with no default
    (it must be sourced for the window, see module docstring). Output: dimensionless, or
    None if volatility is zero. Assumes i.i.d. returns; it treats upside and downside
    volatility alike and says nothing about tails (see `sortino_ratio`, drawdown, VaR).
    """
    check_annual_rate(risk_free_annual, "risk_free_annual")
    values = check_series(returns, min_obs)
    sd = _std(values)
    if sd == 0.0:
        return None
    return float(np.mean(values - _daily_rate(risk_free_annual))) / sd * _SQRT_T


def sortino_ratio(
    returns: pd.Series, *, mar_annual: float, min_obs: int = DEFAULT_MIN_OBS
) -> float | None:
    """Annualised Sortino ratio.

        m       = (1 + mar_annual)^(1/252) - 1
        Sortino = mean(r_t - m) * 252 / DD,        DD = `downside_deviation` (annualised)

    Like Sharpe, but only shortfalls below the MAR count as risk. `mar_annual` is
    required (`performance_summary` uses MAR = risk-free rate). Output: dimensionless, or
    None if no day is below the MAR (downside deviation 0).
    """
    dd = downside_deviation(returns, mar_annual=mar_annual, min_obs=min_obs)
    if dd == 0.0:
        return None
    excess = float(np.mean(returns.to_numpy(dtype=float) - _daily_rate(mar_annual)))
    return excess * TRADING_DAYS / dd


# --- drawdowns ---


def _wealth(values: np.ndarray) -> np.ndarray:
    """Wealth index W_t = prod_{s<=t} (1 + r_s), starting from W_0 = 1 before day 1."""
    return np.cumprod(1 + values)


def _drawdown_values(values: np.ndarray) -> np.ndarray:
    wealth = _wealth(values)
    running_peak = np.maximum.accumulate(np.concatenate([[1.0], wealth]))[1:]
    dd = wealth / running_peak - 1
    dd[dd > -DD_TOL] = 0.0  # back at the peak up to floating-point noise
    return dd


def drawdown_series(returns: pd.Series, *, min_obs: int = DEFAULT_MIN_OBS) -> pd.Series:
    """Drawdown from the running peak, on the returns' index.

        W_0 = 1,  W_t = prod_{s<=t} (1 + r_s)
        D_t = W_t / max_{0<=s<=t} W_s - 1          (<= 0; 0 = at a new high)

    The starting value W_0 counts as a peak, so a loss on the first day is a drawdown.
    Values within DD_TOL of the peak are set to exactly 0 (a return to the old peak can
    otherwise be 1e-16 short in floating point). Output: Series named "drawdown".
    """
    values = check_series(returns, min_obs)
    check_wealth_domain(values)
    return pd.Series(_drawdown_values(values), index=returns.index, name="drawdown")


def drawdown_episodes(
    returns: pd.Series, *, min_obs: int = DEFAULT_MIN_OBS
) -> tuple[DrawdownEpisode, ...]:
    """All underwater episodes, in date order.

    An episode is a maximal run of days with D_t < 0. Its peak is the day before the run
    (or the starting value), its trough the first day of minimum D_t in the run, and its
    recovery the day after the run (if the series gets back to the peak). An episode still
    underwater at the last observation has recovery_date = None.
    """
    values = check_series(returns, min_obs)
    check_wealth_domain(values)
    dd = _drawdown_values(values)
    index = returns.index
    n = len(dd)
    underwater = np.concatenate([[False], dd < 0, [False]])
    starts = np.flatnonzero(~underwater[:-1] & underwater[1:])  # first underwater position
    ends = np.flatnonzero(underwater[:-1] & ~underwater[1:])  # one past the last one
    episodes = []
    for s, e in zip(starts, ends, strict=True):
        peak = s - 1  # -1 = the starting value W_0
        trough = s + int(np.argmin(dd[s:e]))
        recovery = e if e < n else None
        episodes.append(
            DrawdownEpisode(
                depth=float(-dd[trough]),
                peak_date=index[peak] if peak >= 0 else None,
                trough_date=index[trough],
                recovery_date=index[recovery] if recovery is not None else None,
                decline_days=int(trough - peak),
                recovery_days=int(recovery - trough) if recovery is not None else None,
                duration_days=int((recovery if recovery is not None else n - 1) - peak),
            )
        )
    return tuple(episodes)


def drawdown_stats(returns: pd.Series, *, min_obs: int = DEFAULT_MIN_OBS) -> DrawdownStats:
    """Maximum drawdown, its episode, and the longest underwater period.

        MDD = -min_t D_t = max_t (1 - W_t / max_{s<=t} W_s)        (positive)

    - `max_drawdown_episode`: the deepest episode (earliest one on ties), with its peak,
      trough and recovery dates and lengths.
    - `longest_drawdown`: the episode with the most trading days underwater (earliest on
      ties); it can be a shallower episode than the deepest one. An unrecovered episode
      counts its length up to the last observation, so it is a lower bound.

    Inputs: simple daily returns (>= -1) with a DatetimeIndex, so dates can be reported.
    Output units: fractions of the peak value and trading days. Path-dependent and
    sample-dependent: a longer history can only show an equal or deeper MDD.
    """
    episodes = drawdown_episodes(returns, min_obs=min_obs)
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise ValueError("returns must have a DatetimeIndex to report drawdown dates")
    if not episodes:
        return DrawdownStats(0.0, None, None, 0, len(returns))
    deepest = max(episodes, key=lambda ep: ep.depth)  # max() keeps the first on ties
    longest = max(episodes, key=lambda ep: ep.duration_days)
    return DrawdownStats(
        max_drawdown=deepest.depth,
        max_drawdown_episode=deepest,
        longest_drawdown=longest,
        n_episodes=len(episodes),
        n_obs=len(returns),
    )


def max_drawdown(returns: pd.Series, *, min_obs: int = DEFAULT_MIN_OBS) -> float:
    """Maximum drawdown as a positive fraction (0.0 if never below a previous peak)."""
    values = check_series(returns, min_obs)
    check_wealth_domain(values)
    return float(-_drawdown_values(values).min())


# --- benchmark-relative ---


def benchmark_stats(
    returns: pd.Series,
    benchmark: pd.Series,
    *,
    risk_free_annual: float,
    min_obs: int = DEFAULT_MIN_OBS,
) -> BenchmarkStats:
    """Beta, alpha, correlation, tracking error and information ratio vs a benchmark.

    With p = portfolio and b = benchmark daily simple returns, rf_d the daily risk-free
    rate, and all (co)variances sample estimates with ddof = 1:

        beta      = cov(p, b) / var(b)                          OLS slope of p on b
        rho       = cov(p, b) / (sd(p) sd(b)),     R^2 = rho^2
        alpha     = [mean(p - rf_d) - beta * mean(b - rf_d)] * 252     (Jensen, arithmetic)
        resid vol = sd(p - beta * b) * sqrt(252)  = sd(p) sqrt(1 - rho^2) sqrt(252)
        TE        = sd(p - b) * sqrt(252)                      tracking error
        active    = mean(p - b) * 252
        IR        = active / TE                                 information ratio

    Residual volatility is the risk the benchmark does not explain (idiosyncratic risk);
    tracking error is the risk relative to holding the benchmark *one-for-one*. A
    portfolio of 2x the benchmark has beta 2, residual vol 0, but TE = sd(b) sqrt(252).
    Residual vol uses ddof = 1 for consistency (OLS would use n - 2; negligible here).

    Note: with free leverage, p = 2b gives alpha = rf exactly (it earns 2 x (b - rf) + rf
    against beta x (b - rf)), because no borrowing cost is charged. Real alpha needs
    funding costs and fees in p.

    Data limitations: vs ^NSEI (a price index) alpha, active return and IR are biased
    upward by about the index dividend yield (~1-1.5% a year, multiplied by beta for
    alpha); the 7 filled 0% benchmark days slightly distort beta, rho and TE.

    Inputs: two simple daily return Series with identical indexes; risk_free_annual an
    annual decimal (required). Output: annualised decimals; None where undefined.
    """
    check_annual_rate(risk_free_annual, "risk_free_annual")
    p = check_series(returns, min_obs)
    b = check_series(benchmark, min_obs, label="benchmark")
    check_same_index(returns, benchmark)
    rf_d = _daily_rate(risk_free_annual)

    sd_p, sd_b = _std(p), _std(b)
    cov = float(np.cov(p, b, ddof=1)[0, 1])
    beta = cov / sd_b**2 if sd_b > 0 else None
    correlation = None
    if sd_p > 0 and sd_b > 0:
        correlation = max(-1.0, min(1.0, cov / (sd_p * sd_b)))
    alpha = residual_vol = None
    if beta is not None:
        alpha = (float(np.mean(p - rf_d)) - beta * float(np.mean(b - rf_d))) * TRADING_DAYS
        residual_vol = _std(p - beta * b) * _SQRT_T
    te = _std(p - b) * _SQRT_T
    active = float(np.mean(p - b)) * TRADING_DAYS
    return BenchmarkStats(
        beta=beta,
        alpha=alpha,
        correlation=correlation,
        r_squared=correlation**2 if correlation is not None else None,
        residual_volatility=residual_vol,
        tracking_error=te,
        active_return=active,
        information_ratio=active / te if te > 0 else None,
        risk_free_annual=risk_free_annual,
        n_obs=len(p),
    )


def correlation_matrix(returns: pd.DataFrame, *, min_obs: int = DEFAULT_MIN_OBS) -> pd.DataFrame:
    """Pearson correlation matrix of daily simple returns.

        rho_ij = cov(r_i, r_j) / (sd(r_i) sd(r_j))         (ddof cancels)

    A column with zero variance (<= ZERO_TOL) has NaN correlations with everything,
    including itself; other diagonals are exactly 1. Inputs: DataFrame, rows = days,
    columns = asset ids, no NaN. Output: DataFrame indexed and labelled by asset id.

    Daily correlations understate co-movement between assets priced at different times
    (e.g. the gold FoFs' NAV timing, docs/data.md), and they are averages over the window:
    in sell-offs correlations tend to rise exactly when diversification is needed.
    """
    values = check_metrics_frame(returns, min_obs)
    centred = values - values.mean(axis=0)
    sd = np.sqrt((centred**2).sum(axis=0) / (len(values) - 1))
    ok = sd > ZERO_TOL
    corr = np.full((values.shape[1], values.shape[1]), np.nan)
    if ok.any():
        z = centred[:, ok] / sd[ok]
        sub = np.clip(z.T @ z / (len(values) - 1), -1.0, 1.0)
        np.fill_diagonal(sub, 1.0)
        corr[np.ix_(ok, ok)] = sub
    return pd.DataFrame(corr, index=returns.columns, columns=returns.columns)


# --- summaries ---


def performance_summary(
    returns: pd.Series,
    *,
    risk_free_annual: float,
    benchmark: pd.Series | None = None,
    min_obs: int = DEFAULT_MIN_OBS,
) -> PerformanceSummary:
    """All metrics for one return series. Sortino uses MAR = risk_free_annual, so Sharpe
    and Sortino share a numerator. `benchmark` (same index) adds BenchmarkStats."""
    check_series(returns, min_obs)
    rf = risk_free_annual
    return PerformanceSummary(
        start=returns.index[0],
        end=returns.index[-1],
        n_obs=len(returns),
        annualised_return=annualised_return(returns, min_obs=min_obs),
        arithmetic_annual_return=arithmetic_annual_return(returns, min_obs=min_obs),
        annualised_volatility=annualised_volatility(returns, min_obs=min_obs),
        downside_deviation=downside_deviation(returns, mar_annual=rf, min_obs=min_obs),
        sharpe=sharpe_ratio(returns, risk_free_annual=rf, min_obs=min_obs),
        sortino=sortino_ratio(returns, mar_annual=rf, min_obs=min_obs),
        risk_free_annual=rf,
        mar_annual=rf,
        drawdown=drawdown_stats(returns, min_obs=min_obs),
        benchmark=(
            benchmark_stats(returns, benchmark, risk_free_annual=rf, min_obs=min_obs)
            if benchmark is not None
            else None
        ),
    )


def asset_metrics(
    returns: pd.DataFrame,
    *,
    risk_free_annual: float,
    benchmark: pd.Series | None = None,
    min_obs: int = DEFAULT_MIN_OBS,
) -> pd.DataFrame:
    """One row per asset (column of `returns`) with the headline metrics.

    Columns: cagr, volatility, sharpe, sortino, max_drawdown, and with a benchmark also
    beta, alpha, correlation, tracking_error, information_ratio. Undefined ratios are NaN.
    Same conventions as `performance_summary` (Sortino MAR = risk_free_annual).
    """
    check_metrics_frame(returns, min_obs)
    rows = {}
    for col in returns.columns:
        s = performance_summary(
            returns[col], risk_free_annual=risk_free_annual, benchmark=benchmark, min_obs=min_obs
        )
        row: dict[str, float | None] = {
            "cagr": s.annualised_return,
            "volatility": s.annualised_volatility,
            "sharpe": s.sharpe,
            "sortino": s.sortino,
            "max_drawdown": s.drawdown.max_drawdown,
        }
        if s.benchmark is not None:
            row |= {
                "beta": s.benchmark.beta,
                "alpha": s.benchmark.alpha,
                "correlation": s.benchmark.correlation,
                "tracking_error": s.benchmark.tracking_error,
                "information_ratio": s.benchmark.information_ratio,
            }
        rows[col] = row
    return pd.DataFrame.from_dict(rows, orient="index").astype(float)
