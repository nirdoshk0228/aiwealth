"""Out-of-sample VaR backtesting: rolling VaR forecasts vs realised portfolio returns.

Pure functions: no I/O, no network, no imports from app.data or app.ai (enforced by
tests/test_architecture.py). Inputs come from app.engine.returns.simple_returns; the VaR
models are the ones in app.engine.risk.var.

Conventions
-----------
- Inputs: as app.engine.risk.var. `returns` is a DataFrame of **simple** daily returns
  (rows = days, columns = asset ids, decimals, no NaN); `weights` maps asset id -> weight
  (finite, summing to 1). Fixed weights (rebalanced daily), so R_p,t = w'r_t every day.
- Rolling window of fixed length N (`window`, default 504 = about two years). For test day
  t (row t, t = N..T-1) the VaR is estimated from rows t-N..t-1 only, i.e. strictly
  **before** t, and then compared with the realised R_p,t. Row t never enters its own
  forecast (no look-ahead); the first N rows only serve as the first estimation window.
- Horizon: **1 trading day only**. h-day VaR checked daily would use overlapping h-day
  returns, whose breaches are autocorrelated by construction; that breaks the independence
  assumption behind both tests below.
- Breach (exception, violation): I_t = 1 if L_t = -R_p,t > VaR_t, strictly greater; a loss
  exactly equal to the VaR is not a breach. Expected rate p = alpha = 1 - confidence.
- Monte Carlo: n_sims defaults to 10,000 per day (not var.py's 100,000) because the model
  is re-estimated on every test day. The same seed is used every day (common random
  numbers), so day-to-day VaR changes come from the data, not from new simulation noise.
  Results still depend on (n_sims, seed): a different seed shifts every VaR by sampling
  error of roughly sqrt(alpha(1-alpha)/n_sims) / phi(z) * sigma_p (~2.7% of sigma_p at 99%
  with 10,000 sims), which can move a borderline day across the breach line.

Tests (chi-square p-values without scipy)
------------------------------------------
- Kupiec (1995) proportion of failures, LR_uc ~ chi2(1): is the breach *rate* right?
- Christoffersen (1998) independence, LR_ind ~ chi2(1): do breaches cluster (is a breach
  today more likely after a breach yesterday)? Conditional coverage LR_cc = LR_uc + LR_ind
  ~ chi2(2) tests both at once.
- Survival functions: chi2(1): P(X > x) = erfc(sqrt(x / 2)); chi2(2): P(X > x) = exp(-x / 2)
  (chi2(2) is the exponential distribution with mean 2), both exact.
- All three are *asymptotic* tests. At 99% with ~250-2,400 test days only a handful to a
  few dozen breaches are expected, so their power is low: "not rejected" means the data
  cannot tell the model is wrong, not that the model is right.

Basel traffic light
-------------------
The Basel Committee's zones (green 0-4, yellow 5-9, red 10+ exceptions) are defined for
**250 trading days of 1-day 99% VaR**, the regulatory convention for that setup only.
They are reported for the last 250 test days when confidence == 0.99 and there are at
least 250 test days, and are None otherwise; they do not translate to other confidence
levels or sample sizes.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Literal

import numpy as np
import pandas as pd

from app.engine.risk.var import (
    DEFAULT_CONFIDENCE,
    DEFAULT_SEED,
    MIN_N_SIMS,
    Method,
    VaRResult,
    _check_params,
    _check_returns,
    historical_var,
    monte_carlo_var,
    parametric_var,
    portfolio_returns,
)

DEFAULT_WINDOW = 504  # about two years of trading days
DEFAULT_BACKTEST_N_SIMS = 10_000  # per test day; see the module docstring
BASEL_DAYS = 250
BASEL_CONFIDENCE = 0.99
BASEL_GREEN_MAX = 4  # 0-4 exceptions in 250 days
BASEL_YELLOW_MAX = 9  # 5-9; 10+ is red

BaselZone = Literal["green", "yellow", "red"]
_METHODS: tuple[Method, ...] = ("historical", "parametric", "monte_carlo")


# --- result types ---


@dataclass(frozen=True)
class KupiecTest:
    """Kupiec proportion-of-failures (unconditional coverage) test."""

    n: int  # observations (test days)
    x: int  # breaches
    expected_rate: float  # p = 1 - confidence
    lr_uc: float  # likelihood-ratio statistic, ~ chi2(1) under H0: rate = p
    p_value: float


@dataclass(frozen=True)
class ChristoffersenTest:
    """Christoffersen independence and conditional coverage tests.

    Transition counts n_ij = number of days t with I_{t-1} = i and I_t = j (n - 1 pairs).
    lr_ind / lr_cc and their p-values are None when a transition probability is undefined
    (no day in state 0, or none in state 1, before the last day); see `christoffersen_test`.
    """

    n00: int
    n01: int
    n10: int
    n11: int
    pi01: float | None  # P(breach | no breach yesterday)
    pi11: float | None  # P(breach | breach yesterday)
    lr_ind: float | None  # ~ chi2(1) under H0: pi01 = pi11
    p_ind: float | None
    lr_cc: float | None  # lr_uc + lr_ind, ~ chi2(2) under H0: rate = p and independent
    p_cc: float | None


@dataclass(frozen=True)
class BacktestResult:
    """Rolling 1-day VaR backtest of a fixed-weight portfolio.

    `dates`, `var` and `realised` are aligned, one entry per test day (as tuples so the
    frozen result stays immutable and comparable); `to_frame()` gives the plotting view.
    """

    method: Method
    confidence: float
    window: int  # estimation window length N (rows before each test day)
    horizon_days: int  # always 1
    n_test_days: int
    n_breaches: int
    breach_rate: float  # n_breaches / n_test_days
    expected_rate: float  # 1 - confidence
    kupiec: KupiecTest
    christoffersen: ChristoffersenTest
    # Basel regulatory convention for 250 days of 1-day 99% VaR only; None otherwise.
    basel_breaches_last_250: int | None
    basel_zone_last_250: BaselZone | None
    breach_dates: tuple[date, ...]
    dates: tuple[date, ...]  # test days
    var: tuple[float, ...]  # VaR_t forecast for each test day (positive loss fraction)
    realised: tuple[float, ...]  # realised portfolio simple return R_p,t
    n_sims: int | None = None  # monte_carlo only
    seed: int | None = None  # monte_carlo only

    @property
    def expected_breaches(self) -> float:
        """n_test_days * (1 - confidence)."""
        return self.n_test_days * self.expected_rate

    def to_frame(self) -> pd.DataFrame:
        """One row per test day (DatetimeIndex): var, realised, loss, breach."""
        df = pd.DataFrame(
            {"var": self.var, "realised": self.realised},
            index=pd.DatetimeIndex(self.dates, name="date"),
        )
        df["loss"] = -df["realised"]
        df["breach"] = df["loss"] > df["var"]
        return df


# --- helpers ---


def _xlogy(x: float, y: float) -> float:
    """x * ln(y) with the convention 0 * ln(0) = 0 (the limit of x ln x as x -> 0+)."""
    return 0.0 if x == 0 else x * math.log(y)


def chi2_sf_1dof(x: float) -> float:
    """P(X > x) for X ~ chi2(1): erfc(sqrt(x / 2)), since X = Z^2 with Z standard normal."""
    return math.erfc(math.sqrt(max(x, 0.0) / 2))


def chi2_sf_2dof(x: float) -> float:
    """P(X > x) for X ~ chi2(2): exp(-x / 2) (exponential distribution with mean 2)."""
    return math.exp(-max(x, 0.0) / 2)


def _check_rate(p: float) -> None:
    if not 0 < p < 1:
        raise ValueError(f"expected_rate must be in (0, 1), got {p}")


# --- statistical tests ---


def kupiec_test(n: int, x: int, expected_rate: float) -> KupiecTest:
    """Kupiec (1995) proportion-of-failures likelihood-ratio test.

    With n test days, x breaches, p = expected_rate and pi_hat = x / n:

        LR_uc = -2 [ (n - x) ln(1 - p) + x ln p
                     - (n - x) ln(1 - pi_hat) - x ln(pi_hat) ]

    i.e. -2 ln of the binomial likelihood at p over the likelihood at the MLE pi_hat.
    Under H0 (breaches are Bernoulli(p)) LR_uc ~ chi2(1) asymptotically; p-value =
    erfc(sqrt(LR_uc / 2)).

    Edge cases use 0 * ln(0) = 0 explicitly:
        x = 0:  LR_uc = -2 n ln(1 - p)     (e.g. n = 250, p = 1%: 5.025, p-value 0.025)
        x = n:  LR_uc = -2 n ln p
    so zero breaches is itself evidence against the model when n p is large enough: a VaR
    that is never breached is too conservative.

    Inputs: n >= 1, 0 <= x <= n, p in (0, 1). Assumptions: i.i.d. Bernoulli breaches (the
    test ignores their timing; see `christoffersen_test`); asymptotic chi2 approximation,
    poor when n p is small. Power is low at 99% (exact binomial, 5% test level): with
    n = 2,400 (24 expected breaches) a model whose true breach rate is 1.5% is rejected
    only 59% of the time; with n = 250 a true rate of 2% is rejected 24% of the time.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if not 0 <= x <= n:
        raise ValueError(f"x must be in [0, n] = [0, {n}], got {x}")
    _check_rate(expected_rate)
    p = expected_rate
    pi_hat = x / n
    ll_h0 = _xlogy(n - x, 1 - p) + _xlogy(x, p)
    ll_mle = _xlogy(n - x, 1 - pi_hat) + _xlogy(x, pi_hat)
    lr = max(-2 * (ll_h0 - ll_mle), 0.0)  # >= 0 in exact arithmetic; guard rounding
    return KupiecTest(n=n, x=x, expected_rate=p, lr_uc=lr, p_value=chi2_sf_1dof(lr))


def christoffersen_test(breaches: Sequence[bool], expected_rate: float) -> ChristoffersenTest:
    """Christoffersen (1998) Markov independence and conditional coverage tests.

    Count transitions over the n - 1 consecutive pairs (I_{t-1}, I_t):

        pi01 = n01 / (n00 + n01)          P(breach today | no breach yesterday)
        pi11 = n11 / (n10 + n11)          P(breach today | breach yesterday)
        pi   = (n01 + n11) / (n - 1)      unconditional breach rate over the pairs
        ln L0 = (n00 + n10) ln(1 - pi) + (n01 + n11) ln pi
        ln L1 = n00 ln(1 - pi01) + n01 ln pi01 + n10 ln(1 - pi11) + n11 ln pi11
        LR_ind = -2 (ln L0 - ln L1)                       ~ chi2(1), H0: pi01 = pi11
        LR_cc  = LR_uc + LR_ind                           ~ chi2(2)

    with 0 * ln(0) = 0 and LR_uc the Kupiec statistic on all n days (the usual practice:
    LR_uc uses n observations, LR_ind the n - 1 transitions, so the sum is an
    approximation of the joint LR, exact only up to the first observation).

    Undefined cases return None for lr_ind, p_ind, lr_cc and p_cc (counts are still given):
    - n00 + n01 = 0: no non-breach day before the last day, so pi01 is undefined;
    - n10 + n11 = 0: no breach before the last day (zero breaches, or a single breach on
      the last day), so pi11 is undefined. Some references set pi11 = 0 here and get
      LR_ind = 0; that hides the fact that there is no information about clustering.

    Assumptions: first-order Markov alternative only (it detects breaches on *consecutive*
    days, not e.g. two breaches a week apart in a volatile month); asymptotic chi2; low
    power with few breaches.
    """
    seq = np.asarray(breaches, dtype=bool)
    if seq.ndim != 1 or len(seq) < 2:
        raise ValueError(f"need a 1-d breach sequence of length >= 2, got shape {seq.shape}")
    _check_rate(expected_rate)
    prev, curr = seq[:-1], seq[1:]
    n00 = int(np.sum(~prev & ~curr))
    n01 = int(np.sum(~prev & curr))
    n10 = int(np.sum(prev & ~curr))
    n11 = int(np.sum(prev & curr))
    pi01 = n01 / (n00 + n01) if n00 + n01 else None
    pi11 = n11 / (n10 + n11) if n10 + n11 else None
    if pi01 is None or pi11 is None:
        return ChristoffersenTest(n00, n01, n10, n11, pi01, pi11, None, None, None, None)
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    ll0 = _xlogy(n00 + n10, 1 - pi) + _xlogy(n01 + n11, pi)
    ll1 = _xlogy(n00, 1 - pi01) + _xlogy(n01, pi01) + _xlogy(n10, 1 - pi11) + _xlogy(n11, pi11)
    lr_ind = max(-2 * (ll0 - ll1), 0.0)
    lr_cc = kupiec_test(len(seq), int(seq.sum()), expected_rate).lr_uc + lr_ind
    return ChristoffersenTest(
        n00=n00,
        n01=n01,
        n10=n10,
        n11=n11,
        pi01=pi01,
        pi11=pi11,
        lr_ind=lr_ind,
        p_ind=chi2_sf_1dof(lr_ind),
        lr_cc=lr_cc,
        p_cc=chi2_sf_2dof(lr_cc),
    )


def basel_zone(n_breaches: int) -> BaselZone:
    """Basel traffic-light zone for the number of exceptions of 1-day 99% VaR in 250 days.

        green: 0-4    yellow: 5-9    red: 10+

    Regulatory convention for exactly that setup (99%, 250 trading days); it is not a
    statistical test and does not apply to other confidence levels or sample sizes.
    """
    if n_breaches < 0:
        raise ValueError(f"n_breaches must be >= 0, got {n_breaches}")
    if n_breaches <= BASEL_GREEN_MAX:
        return "green"
    if n_breaches <= BASEL_YELLOW_MAX:
        return "yellow"
    return "red"


# --- the backtest ---


def var_backtest(
    returns: pd.DataFrame,
    weights: Mapping[str, float] | pd.Series,
    *,
    method: Method = "historical",
    confidence: float = DEFAULT_CONFIDENCE,
    window: int = DEFAULT_WINDOW,
    n_sims: int = DEFAULT_BACKTEST_N_SIMS,
    seed: int = DEFAULT_SEED,
) -> BacktestResult:
    """Rolling out-of-sample backtest of 1-day VaR for a fixed-weight portfolio.

    For each test day t = N..T-1 (N = window, T = len(returns)):

        VaR_t  = <method>_var(returns[t-N : t], weights, confidence).var   (rows before t)
        R_p,t  = w' r_t                                                     (realised)
        I_t    = 1 if -R_p,t > VaR_t else 0                                 (breach)

    then breach_rate = sum I_t / (T - N), compared with p = 1 - confidence by
    `kupiec_test`, `christoffersen_test` and (99% only, last 250 days) `basel_zone`.

    Inputs/units: simple daily returns (decimals, no NaN), weights summing to 1 (columns
    without a weight are ignored), method in {"historical", "parametric", "monte_carlo"},
    confidence in (0.5, 1), window >= 2 with len(returns) > window. `n_sims` and `seed`
    apply to monte_carlo only (same seed every day; results depend on both).

    Assumptions and caveats:
    - Rolling (not expanding) window of equal weight on all N days, re-estimated daily.
    - Historical VaR at window N takes the k-th worst loss, k = ceil(N alpha). For i.i.d.
      continuous returns the chance that the next loss exceeds the k-th largest of N is
      exactly k / (N + 1), so its expected breach rate is k / (N + 1), not alpha
      (N = 504: 26/505 = 5.15% at 95%, 6/505 = 1.19% at 99%). Normal-based methods also
      breach slightly more than alpha from estimation error in mu and Sigma.
    - Weights are the *current* weights applied to the whole history (a backtest of
      today's portfolio, not of the portfolio actually held in the past).
    """
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    _check_params(confidence, 1, window)
    _check_returns(returns, window + 1)  # >= 1 test day
    if method == "monte_carlo" and n_sims < MIN_N_SIMS:
        raise ValueError(f"n_sims must be >= {MIN_N_SIMS}, got {n_sims}")
    realised = portfolio_returns(returns, weights)  # validates weights

    def estimate(sample: pd.DataFrame) -> VaRResult:
        if method == "historical":
            return historical_var(sample, weights, confidence=confidence, min_obs=window)
        if method == "parametric":
            return parametric_var(sample, weights, confidence=confidence, min_obs=window)
        return monte_carlo_var(
            sample, weights, confidence=confidence, min_obs=window, n_sims=n_sims, seed=seed
        )

    n = len(returns)
    var = np.array([estimate(returns.iloc[t - window : t]).var for t in range(window, n)])
    r_test = realised.to_numpy()[window:]
    breaches = -r_test > var
    test_dates = tuple(ts.date() for ts in pd.DatetimeIndex(returns.index[window:]))
    n_test = n - window
    x = int(breaches.sum())
    p = 1 - confidence

    basel_count: int | None = None
    zone: BaselZone | None = None
    if math.isclose(confidence, BASEL_CONFIDENCE) and n_test >= BASEL_DAYS:
        basel_count = int(breaches[-BASEL_DAYS:].sum())
        zone = basel_zone(basel_count)

    return BacktestResult(
        method=method,
        confidence=confidence,
        window=window,
        horizon_days=1,
        n_test_days=n_test,
        n_breaches=x,
        breach_rate=x / n_test,
        expected_rate=p,
        kupiec=kupiec_test(n_test, x, p),
        christoffersen=(
            christoffersen_test(breaches, p)
            if n_test >= 2
            else ChristoffersenTest(0, 0, 0, 0, None, None, None, None, None, None)
        ),
        basel_breaches_last_250=basel_count,
        basel_zone_last_250=zone,
        breach_dates=tuple(d for d, b in zip(test_dates, breaches, strict=True) if b),
        dates=test_dates,
        var=tuple(float(v) for v in var),
        realised=tuple(float(r) for r in r_test),
        n_sims=n_sims if method == "monte_carlo" else None,
        seed=seed if method == "monte_carlo" else None,
    )
