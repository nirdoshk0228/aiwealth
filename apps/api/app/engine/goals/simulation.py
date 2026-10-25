"""Monte Carlo SIP (systematic investment plan) simulation under geometric Brownian motion.

Pure functions: no I/O, no network, no imports from app.data or app.ai.

Model
-----
Portfolio value follows a GBM. Over a step of length dt (years) the log growth is normal:

    ln(V_{t+dt} / V_t) ~ N(m dt, s^2 dt)

Calibration (`calibrate_gbm`) uses **log** returns of the portfolio's daily simple returns,
because log returns add up over time (docs/data.md, "Return conventions"):

    x_t = ln(1 + R_p,t),   m = 252 * mean(x),   s = sqrt(252) * sd(x, ddof = 1)

m is the continuously compounded growth rate; the arithmetic mean is m + s^2/2 and the median
outcome of a lump sum grows at m (volatility drag), which is why using simple returns here
would overstate growth.

SIP (`simulate_sip`), monthly steps dt = 1/12 over M = 12 * years months:

    V_0 = initial
    c_k = monthly * (1 + step_up)^floor(k / 12)                 contribution in month k
    V_{k+1} = (V_k + c_k) * exp(m dt + s sqrt(dt) Z_k),  Z_k ~ N(0, 1) i.i.d.

i.e. contributions are invested at the start of each month. The goal is stated in today's
rupees and inflated at a constant rate to the horizon:

    goal_nominal = goal_today * (1 + inflation)^years
    P(reach)     = share of paths with V_M >= goal_nominal

with Monte Carlo standard error sqrt(p (1 - p) / n_paths).

Assumptions and limits (stated with every result)
-------------------------------------------------
- The future looks like the calibration window: constant m and s (no regime changes), normal
  log returns (real returns have fatter tails, so extreme outcomes are understated), monthly
  steps independent of each other.
- Fixed portfolio weights (rebalanced), no taxes, exit loads or extra fees (fund NAVs are
  already net of expense ratios), constant inflation, contributions never missed.
- It describes a range of outcomes under these assumptions, not a forecast.
"""

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252
MONTHS = 12
DEFAULT_PATHS = 10_000
DEFAULT_SEED = 0
MAX_YEARS = 40
MIN_CALIBRATION_OBS = 252
PERCENTILES = (5, 10, 25, 50, 75, 90, 95)


@dataclass(frozen=True)
class GbmParams:
    m: float  # annual log drift (continuously compounded growth rate of the median)
    s: float  # annual volatility of log returns
    n_obs: int

    @property
    def arithmetic_mean(self) -> float:
        """Annual expected simple growth rate e^(m + s^2/2) - 1."""
        return math.expm1(self.m + 0.5 * self.s**2)

    @property
    def median_growth(self) -> float:
        """Annual growth of the median outcome e^m - 1."""
        return math.expm1(self.m)


@dataclass(frozen=True)
class SipPlan:
    initial: float  # rupees invested today
    monthly: float  # rupees per month in the first year
    years: int
    goal_today: float  # target in today's rupees
    inflation_annual: float = 0.0
    step_up_annual: float = 0.0  # yearly increase of the monthly contribution


@dataclass(frozen=True)
class SipSimulation:
    plan: SipPlan
    params: GbmParams
    n_paths: int
    seed: int
    goal_nominal: float
    total_invested: float  # sum of all contributions plus the initial amount (nominal)
    probability: float  # P(final value >= goal_nominal)
    probability_se: float  # Monte Carlo standard error
    final_percentiles: dict[int, float]  # nominal rupees
    final_percentiles_real: dict[int, float]  # deflated to today's rupees
    years: tuple[int, ...]  # 0..years
    fan: dict[int, tuple[float, ...]]  # percentile -> value at each year end (nominal)
    invested_by_year: tuple[float, ...]  # cumulative contributions at each year end


def calibrate_gbm(returns: pd.Series, *, min_obs: int = MIN_CALIBRATION_OBS) -> GbmParams:
    """GBM drift and volatility from daily **simple** returns via their logs (see module)."""
    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series of daily simple returns")
    r = returns.to_numpy(dtype=float)
    if len(r) < min_obs:
        raise ValueError(f"need at least {min_obs} return observations, got {len(r)}")
    if not np.isfinite(r).all():
        raise ValueError("returns contain NaN or inf")
    if (r <= -1).any():
        raise ValueError("a return of -100% or less has no logarithm")
    x = np.log1p(r)
    return GbmParams(
        m=float(x.mean()) * TRADING_DAYS,
        s=float(x.std(ddof=1)) * math.sqrt(TRADING_DAYS),
        n_obs=len(r),
    )


def _check_plan(plan: SipPlan) -> None:
    for name in ("initial", "monthly", "goal_today"):
        v = getattr(plan, name)
        if not math.isfinite(v) or v < 0:
            raise ValueError(f"{name} must be a finite amount >= 0, got {v}")
    if plan.initial == 0 and plan.monthly == 0:
        raise ValueError("invest something: initial amount or monthly SIP must be > 0")
    if not plan.goal_today > 0:
        raise ValueError("goal must be > 0")
    if not (isinstance(plan.years, int) and 1 <= plan.years <= MAX_YEARS):
        raise ValueError(f"years must be an integer in [1, {MAX_YEARS}], got {plan.years}")
    if not -0.05 < plan.inflation_annual < 0.5:
        raise ValueError(
            f"inflation must be an annual decimal in (-0.05, 0.5), got {plan.inflation_annual}"
        )
    if not 0 <= plan.step_up_annual <= 0.5:
        raise ValueError(
            f"step-up must be an annual decimal in [0, 0.5], got {plan.step_up_annual}"
        )


def contributions(plan: SipPlan) -> np.ndarray:
    """Monthly contributions c_k, k = 0..12*years-1 (step-up applied each 12 months)."""
    k = np.arange(MONTHS * plan.years)
    return plan.monthly * (1 + plan.step_up_annual) ** (k // MONTHS)


def deterministic_value(plan: SipPlan, m: float) -> float:
    """Final value with zero volatility: V_M = initial g^M + sum_k c_k g^(M-k), g = e^(m/12).
    The s = 0 limit of `simulate_sip`, used as an exact check."""
    c = contributions(plan)
    months = len(c)
    g = math.exp(m / MONTHS)
    return plan.initial * g**months + float(np.sum(c * g ** (months - np.arange(months))))


def simulate_sip(
    params: GbmParams,
    plan: SipPlan,
    *,
    n_paths: int = DEFAULT_PATHS,
    seed: int = DEFAULT_SEED,
) -> SipSimulation:
    """Simulate `n_paths` monthly GBM paths of the SIP (see the module docstring).

    Deterministic for a given seed (np.random.default_rng). Values are nominal rupees unless
    stated; `final_percentiles_real` divides by (1 + inflation)^years.
    """
    _check_plan(plan)
    if not (math.isfinite(params.m) and math.isfinite(params.s) and params.s >= 0):
        raise ValueError("GBM parameters must be finite with s >= 0")
    if not 1_000 <= n_paths <= 200_000:
        raise ValueError(f"n_paths must be in [1000, 200000], got {n_paths}")

    c = contributions(plan)
    months = len(c)
    dt = 1.0 / MONTHS
    rng = np.random.default_rng(seed)
    drift = params.m * dt
    vol = params.s * math.sqrt(dt)

    v = np.full(n_paths, float(plan.initial))
    yearly = np.empty((plan.years + 1, n_paths))
    yearly[0] = v
    for k in range(months):
        v = (v + c[k]) * np.exp(drift + vol * rng.standard_normal(n_paths))
        if (k + 1) % MONTHS == 0:
            yearly[(k + 1) // MONTHS] = v

    deflator = (1 + plan.inflation_annual) ** plan.years
    goal_nominal = plan.goal_today * deflator
    p = float(np.mean(v >= goal_nominal))
    final = {q: float(np.percentile(v, q)) for q in PERCENTILES}
    invested_cum = plan.initial + np.concatenate([[0.0], np.cumsum(c)])
    return SipSimulation(
        plan=plan,
        params=params,
        n_paths=n_paths,
        seed=seed,
        goal_nominal=goal_nominal,
        total_invested=float(invested_cum[-1]),
        probability=p,
        probability_se=math.sqrt(p * (1 - p) / n_paths),
        final_percentiles=final,
        final_percentiles_real={q: x / deflator for q, x in final.items()},
        years=tuple(range(plan.years + 1)),
        fan={
            q: tuple(float(x) for x in np.percentile(yearly, q, axis=1))
            for q in (5, 25, 50, 75, 95)
        },
        invested_by_year=tuple(float(invested_cum[MONTHS * y]) for y in range(plan.years + 1)),
    )
