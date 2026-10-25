"""Input validation shared by the risk modules (var, metrics, attribution, backtest).

Pure functions: no I/O, no network, no imports from app.data or app.ai. Every check raises
ValueError (or TypeError for a wrong container type) with a message that names the problem
and, where useful, how to fix it. The modules import these instead of each keeping a copy,
so a rule (e.g. "weights sum to 1 within 1e-8") is defined once.

Two input shapes exist:
- a returns **DataFrame** (rows = days, columns = asset ids) plus weights: var, attribution,
  backtest, and metrics' `asset_metrics` / `correlation_matrix`;
- one returns **Series** (a portfolio or a single asset): the rest of metrics.
"""

import math
from collections.abc import Mapping

import numpy as np
import pandas as pd

MAX_HORIZON_DAYS = 20  # about one trading month; sqrt-time scaling degrades beyond that
WEIGHT_SUM_TOL = 1e-8


# --- parameters ---


def check_min_obs(min_obs: int) -> None:
    """min_obs >= 2 (a sample standard deviation needs two observations)."""
    if min_obs < 2:
        raise ValueError(f"min_obs must be >= 2, got {min_obs}")


def check_var_params(confidence: float, horizon_days: int, min_obs: int) -> None:
    """confidence in (0.5, 1), horizon_days in [1, MAX_HORIZON_DAYS], min_obs >= 2."""
    if not 0.5 < confidence < 1:
        raise ValueError(f"confidence must be in (0.5, 1), got {confidence}")
    if not 1 <= horizon_days <= MAX_HORIZON_DAYS:
        raise ValueError(
            f"horizon_days must be in [1, {MAX_HORIZON_DAYS}], got {horizon_days} "
            "(square-root-of-time scaling is not reliable for longer horizons)"
        )
    check_min_obs(min_obs)


def check_annual_rate(rate: float, label: str) -> None:
    """An annual decimal rate (risk-free, MAR): finite and > -1."""
    if not math.isfinite(rate) or rate <= -1:
        raise ValueError(f"{label} must be a finite annual decimal > -1, got {rate}")


def check_probability(p: float, label: str = "expected_rate") -> None:
    """A probability strictly inside (0, 1), e.g. a backtest's expected breach rate."""
    if not 0 < p < 1:
        raise ValueError(f"{label} must be in (0, 1), got {p}")


# --- return data ---


def check_values(values: np.ndarray, n: int, min_obs: int, label: str) -> None:
    """Non-empty, no NaN, no inf, at least `min_obs` rows."""
    if n == 0:
        raise ValueError(f"{label} is empty")
    if np.isnan(values).any():
        raise ValueError(f"{label} contain NaN; build them with align_prices + simple_returns")
    if not np.isfinite(values).all():
        raise ValueError(f"{label} contain inf")
    if n < min_obs:
        raise ValueError(f"need at least {min_obs} return observations, got {n}")


def check_returns_frame(returns: pd.DataFrame, min_obs: int) -> np.ndarray:
    """Validate a returns DataFrame and return its values as float64.

    A frame with no rows or no columns is "empty" (pandas' DataFrame.empty).
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame (columns = asset ids)")
    if returns.empty:
        raise ValueError("returns is empty")
    values = returns.to_numpy(dtype=float)
    check_values(values, len(returns), min_obs, "returns")
    return values


def check_metrics_frame(returns: pd.DataFrame, min_obs: int) -> np.ndarray:
    """As `check_returns_frame`, plus min_obs >= 2, with a specific message for 0 columns."""
    check_min_obs(min_obs)
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame (columns = asset ids)")
    if returns.shape[1] == 0:
        raise ValueError("returns has no columns")
    values = returns.to_numpy(dtype=float)
    check_values(values, len(returns), min_obs, "returns")
    return values


def check_series(returns: pd.Series, min_obs: int, label: str = "returns") -> np.ndarray:
    """Validate one return Series (a portfolio or an asset) and return its float64 values."""
    check_min_obs(min_obs)
    if not isinstance(returns, pd.Series):
        raise TypeError(
            f"{label} must be a pandas Series (one portfolio or asset); "
            "use var.portfolio_returns for a weighted portfolio"
        )
    values = returns.to_numpy(dtype=float)
    check_values(values, len(returns), min_obs, label)
    return values


def check_wealth_domain(values: np.ndarray) -> None:
    """Simple returns of a long position cannot be below -100%."""
    if (values < -1).any():
        raise ValueError("returns below -100% are not possible for a long position")


def check_same_index(returns: pd.Series, benchmark: pd.Series) -> None:
    """Benchmark-relative statistics need identical indexes; nothing is re-aligned silently."""
    if not returns.index.equals(benchmark.index):
        raise ValueError(
            "returns and benchmark must have identical indexes (same dates, same order); "
            "align them first"
        )


# --- weights ---


def weights_vector(
    returns: pd.DataFrame, weights: Mapping[str, float] | pd.Series
) -> tuple[pd.DataFrame, np.ndarray]:
    """Select the weighted columns and return (sub-frame, weight array in column order).

    Rules: non-empty, unique ids, every id a column of `returns`, finite, summing to 1
    within WEIGHT_SUM_TOL (negative = short; long-only is enforced at the API layer).
    Columns without a weight are ignored, so the whole universe frame can be passed.
    """
    w = pd.Series(weights, dtype=float)
    if w.empty:
        raise ValueError("weights is empty")
    if not w.index.is_unique:
        raise ValueError(f"duplicate asset ids in weights: {sorted(w.index[w.index.duplicated()])}")
    missing = [k for k in w.index if k not in returns.columns]
    if missing:
        raise ValueError(f"weights reference assets not in returns: {missing}")
    if not np.isfinite(w.to_numpy()).all():
        raise ValueError("weights must be finite")
    total = float(w.sum())
    if abs(total - 1.0) > WEIGHT_SUM_TOL:
        raise ValueError(f"weights must sum to 1, got {total:.10f}")
    sub = returns.loc[:, list(w.index)]
    return sub, w.to_numpy()


def prepare_weighted(
    returns: pd.DataFrame, weights: Mapping[str, float] | pd.Series, min_obs: int
) -> tuple[pd.DataFrame, np.ndarray]:
    """min_obs >= 2, a valid returns frame and valid weights -> (sub-frame, weight array)."""
    check_min_obs(min_obs)
    check_returns_frame(returns, min_obs)
    return weights_vector(returns, weights)
