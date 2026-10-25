"""Long-only mean-variance optimisation: minimum variance, efficient frontier, maximum Sharpe.

Pure functions: no I/O, no network, no imports from app.data or app.ai.

Conventions
-----------
- Inputs: `returns` is a DataFrame of **simple** daily returns (rows = days, columns = asset
  ids, no NaN), as everywhere in the engine; `asset_ids` selects the investable set.
- Estimates (annual, arithmetic, from the same window):
      mu    = 252 * mean(r)                 expected return vector
      Sigma = 252 * cov(r, ddof = 1)        covariance matrix
  Portfolio:  E[R_p] = w'mu,  sigma_p = sqrt(w'Sigma w),  Sharpe = (w'mu - rf) / sigma_p.
- Constraints: sum w = 1, 0 <= w_i <= max_weight (long-only; the cap is optional, default 1).
- Risk-free rate: annual decimal supplied by the caller (no default in the engine).

Problems solved (all with app.engine.optimize.qp, an exact active-set QP)
--------------------------------------------------------------------------
- minimum variance:  min w'Sigma w
- mean-variance:     max w'mu - (gamma/2) w'Sigma w   for a risk aversion gamma > 0.
  Every solution is on the efficient frontier, and as gamma falls from large to small the
  solution moves from the minimum-variance portfolio towards the maximum-return portfolio,
  so a sweep over gamma traces the whole long-only frontier.
- maximum return (gamma -> 0): put max_weight in the highest-mu asset, then the next, ...
- maximum Sharpe: maximise (w'mu - rf) / sigma_p over the frontier. On a concave frontier the
  Sharpe ratio is unimodal along the curve, so a golden-section search over log gamma
  (bracketed by the best point of the sweep) finds it. Without a cap and with an interior
  solution it equals the closed-form tangency portfolio w ~ Sigma^-1 (mu - rf 1) (tested).

What the results mean, and what they do not
--------------------------------------------
- **Expected returns are the weakest input.** mu is the historical average over the window; a
  few good years for one fund make it dominate the "optimal" portfolio. Mean-variance weights
  are notoriously sensitive to mu (Michaud 1989, "error maximisation"). Results describe the
  trade-off *in the sample*, not a forecast, and are shown next to risk parity, which needs no
  return estimate.
- The same caveats as the rest of the engine apply: survivorship bias in the universe, a
  price-index benchmark, fixed weights rebalanced daily.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.engine.optimize.qp import check_feasible, solve_simplex_qp

TRADING_DAYS = 252
DEFAULT_MIN_OBS = 252
FRONTIER_POINTS = 120  # gamma values in the sweep (before removing duplicates)
GOLDEN_ITER = 80
WEIGHT_TOL = 1e-10  # weights below this are reported as 0


@dataclass(frozen=True)
class Estimates:
    asset_ids: tuple[str, ...]
    mu: np.ndarray  # annual expected returns (arithmetic)
    cov: np.ndarray  # annual covariance
    n_obs: int


@dataclass(frozen=True)
class PortfolioPoint:
    weights: dict[str, float]  # asset id -> weight, sums to 1, all >= 0
    expected_return: float  # annual, w'mu
    volatility: float  # annual, sqrt(w'Sigma w)
    sharpe: float | None  # (w'mu - rf) / volatility; None if volatility is 0
    kkt_residual: float  # optimality certificate of the QP (0 = exact); 0 for closed forms


@dataclass(frozen=True)
class FrontierResult:
    min_variance: PortfolioPoint
    max_sharpe: PortfolioPoint
    max_return: PortfolioPoint
    frontier: tuple[PortfolioPoint, ...]  # by increasing volatility, min-variance first
    risk_free_annual: float
    max_weight: float
    n_obs: int
    warnings: tuple[str, ...]


def estimate(
    returns: pd.DataFrame, asset_ids: Sequence[str], *, min_obs: int = DEFAULT_MIN_OBS
) -> Estimates:
    """Annualised mean vector and covariance of the selected assets' simple daily returns.

    mu = 252 * mean(r),   Sigma = 252 * cov(r, ddof = 1)
    """
    ids = tuple(str(a) for a in asset_ids)
    if len(ids) < 1:
        raise ValueError("select at least one asset")
    if len(set(ids)) != len(ids):
        raise ValueError("asset ids must be unique")
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame (columns = asset ids)")
    missing = [a for a in ids if a not in returns.columns]
    if missing:
        raise ValueError(f"assets not in returns: {missing}")
    sub = returns.loc[:, list(ids)].to_numpy(dtype=float)
    if np.isnan(sub).any() or not np.isfinite(sub).all():
        raise ValueError(
            "returns contain NaN or inf; build them with align_prices + simple_returns"
        )
    if len(sub) < min_obs:
        raise ValueError(f"need at least {min_obs} return observations, got {len(sub)}")
    mu = sub.mean(axis=0) * TRADING_DAYS
    cov = np.atleast_2d(np.cov(sub, rowvar=False, ddof=1)) * TRADING_DAYS
    return Estimates(asset_ids=ids, mu=mu, cov=cov, n_obs=len(sub))


def _point(est: Estimates, w: np.ndarray, rf: float, kkt: float = 0.0) -> PortfolioPoint:
    w = np.where(w < WEIGHT_TOL, 0.0, w)
    w = w / w.sum()
    ret = float(w @ est.mu)
    vol = math.sqrt(max(float(w @ est.cov @ w), 0.0))
    return PortfolioPoint(
        weights={a: float(x) for a, x in zip(est.asset_ids, w, strict=True)},
        expected_return=ret,
        volatility=vol,
        sharpe=(ret - rf) / vol if vol > 1e-12 else None,
        kkt_residual=kkt,
    )


def max_return_weights(mu: np.ndarray, cap: float) -> np.ndarray:
    """Linear programme max mu'w on the capped simplex: fill the cap greedily by mu."""
    w = np.zeros_like(mu)
    left = 1.0
    for i in np.argsort(-mu, kind="stable"):
        take = min(cap, left)
        w[i] = take
        left -= take
        if left <= 1e-15:
            break
    return w


def min_variance(est: Estimates, *, max_weight: float = 1.0, rf: float = 0.0) -> PortfolioPoint:
    """Long-only minimum-variance portfolio: min w'Sigma w on the capped simplex."""
    res = solve_simplex_qp(est.cov, np.zeros(len(est.mu)), cap=max_weight)
    return _point(est, res.w, rf, res.kkt_residual)


def mean_variance(
    est: Estimates,
    gamma: float,
    *,
    max_weight: float = 1.0,
    rf: float = 0.0,
    w0: np.ndarray | None = None,
) -> PortfolioPoint:
    """max w'mu - (gamma/2) w'Sigma w on the capped simplex (gamma > 0 = risk aversion)."""
    if not gamma > 0:
        raise ValueError(f"gamma must be > 0, got {gamma}")
    res = solve_simplex_qp(gamma * est.cov, est.mu, cap=max_weight, w0=w0)
    return _point(est, res.w, rf, res.kkt_residual)


def _gamma_scale(est: Estimates) -> float:
    """gamma at which the return and risk terms are of similar size (problem-invariant unit)."""
    mu_scale = float(np.max(np.abs(est.mu - est.mu.mean()))) or float(np.max(np.abs(est.mu)))
    var_scale = float(np.trace(est.cov)) / len(est.mu)
    if var_scale <= 0:
        raise ValueError("all selected assets have zero variance")
    return max(mu_scale, 1e-12) / var_scale


def efficient_frontier(
    est: Estimates,
    *,
    risk_free_annual: float,
    max_weight: float = 1.0,
    n_points: int = FRONTIER_POINTS,
) -> FrontierResult:
    """Minimum variance, maximum Sharpe, maximum return and the frontier between them.

    Sweep: gamma = kappa * scale for kappa log-spaced over [1e-3, 1e5], where scale makes the
    two terms comparable (`_gamma_scale`); solutions are warm-started from the previous one.
    Duplicate points (same weights to 1e-9) are removed and the minimum-variance and maximum-
    return portfolios are added as the two ends. Max Sharpe: golden-section search on
    log kappa around the best swept point, compared with both ends.
    """
    if not math.isfinite(risk_free_annual):
        raise ValueError("risk_free_annual must be finite")
    check_feasible(len(est.mu), max_weight)
    rf = risk_free_annual
    warnings: list[str] = []

    mv = min_variance(est, max_weight=max_weight, rf=rf)
    mr = _point(est, max_return_weights(est.mu, max_weight), rf)

    scale = _gamma_scale(est)
    kappas = np.logspace(5, -3, n_points)  # high aversion (near min var) to low
    points: list[PortfolioPoint] = [mv]
    sweep_sharpes: list[float] = []
    w_prev = np.array(list(mv.weights.values()))
    for k in kappas:
        p = mean_variance(est, k * scale, max_weight=max_weight, rf=rf, w0=w_prev)
        sweep_sharpes.append(p.sharpe if p.sharpe is not None else -math.inf)
        w_new = np.array(list(p.weights.values()))
        if np.max(np.abs(w_new - np.array(list(points[-1].weights.values())))) > 1e-9:
            points.append(p)
        w_prev = w_new
    if np.max(np.abs(np.array(list(mr.weights.values())) - w_prev)) > 1e-9:
        points.append(mr)
    points.sort(key=lambda q: (q.volatility, q.expected_return))

    # Max Sharpe: golden-section search on log kappa around the best swept point.
    def sharpe_at(log_k: float) -> float:
        p = mean_variance(est, math.exp(log_k) * scale, max_weight=max_weight, rf=rf)
        return p.sharpe if p.sharpe is not None else -math.inf

    logs = np.log(kappas)
    i = int(np.argmax(sweep_sharpes))
    best = mean_variance(est, kappas[i] * scale, max_weight=max_weight, rf=rf)
    lo, hi = logs[min(i + 1, len(logs) - 1)], logs[max(i - 1, 0)]
    phi = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    x1, x2 = b - phi * (b - a), a + phi * (b - a)
    f1, f2 = sharpe_at(x1), sharpe_at(x2)
    for _ in range(GOLDEN_ITER):
        if b - a < 1e-10:
            break
        if f1 < f2:
            a, x1, f1 = x1, x2, f2
            x2 = a + phi * (b - a)
            f2 = sharpe_at(x2)
        else:
            b, x2, f2 = x2, x1, f1
            x1 = b - phi * (b - a)
            f1 = sharpe_at(x1)
    refined = mean_variance(est, math.exp(0.5 * (a + b)) * scale, max_weight=max_weight, rf=rf)
    options = [q for q in (refined, best, mv, mr) if q.sharpe is not None]
    ms = max(options, key=lambda q: q.sharpe or -math.inf) if options else mv

    if ms.sharpe is not None and ms.sharpe <= 0:
        warnings.append(
            "No portfolio of the selected assets beat the risk-free rate over the window; the "
            "'maximum Sharpe' portfolio has a non-positive Sharpe ratio."
        )
    if est.n_obs < 3 * TRADING_DAYS:
        warnings.append(
            f"Expected returns are averages over only {est.n_obs} days; mean-variance weights "
            "are very sensitive to them."
        )
    return FrontierResult(
        min_variance=mv,
        max_sharpe=ms,
        max_return=mr,
        frontier=tuple(points),
        risk_free_annual=rf,
        max_weight=max_weight,
        n_obs=est.n_obs,
        warnings=tuple(warnings),
    )


def evaluate(est: Estimates, weights: dict[str, float], rf: float) -> PortfolioPoint:
    """Expected return, volatility and Sharpe of given weights under the same estimates.
    Assets of `est` missing from `weights` get 0; weights must sum to 1."""
    unknown = [a for a in weights if a not in est.asset_ids]
    if unknown:
        raise ValueError(f"weights reference assets outside the estimates: {unknown}")
    w = np.array([float(weights.get(a, 0.0)) for a in est.asset_ids])
    if abs(w.sum() - 1) > 1e-8:
        raise ValueError(f"weights must sum to 1, got {w.sum():.10f}")
    ret = float(w @ est.mu)
    vol = math.sqrt(max(float(w @ est.cov @ w), 0.0))
    return PortfolioPoint(
        weights={a: float(x) for a, x in zip(est.asset_ids, w, strict=True)},
        expected_return=ret,
        volatility=vol,
        sharpe=(ret - rf) / vol if vol > 1e-12 else None,
        kkt_residual=0.0,
    )
