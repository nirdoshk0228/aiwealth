"""Helpers shared by the analytics services: windows, risk-free rate, number conversion."""

import math
from collections.abc import Callable

import numpy as np
import pandas as pd

from app.schemas.common import AnalysisWindow, RiskFreeRate, TimeSeries
from app.schemas.portfolio import PortfolioIn
from app.services.errors import AnalyticsInputError
from app.services.market import MarketData

TRADING_DAYS = 252
MIN_RF_COVERAGE = 0.5  # share of window rows with factor data needed to average over them
RF_FALLBACK_ROWS = 252


def num(x: float | None) -> float | None:
    """NaN/inf -> None, so responses are valid JSON; other values as float."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def window_of(index: pd.Index) -> AnalysisWindow:
    idx = pd.DatetimeIndex(index)
    return AnalysisWindow(start=idx[0].date(), end=idx[-1].date(), n_obs=len(idx))


def series(s: pd.Series) -> TimeSeries:
    return TimeSeries(
        dates=[d.date() for d in pd.DatetimeIndex(s.index)],
        values=[float(v) for v in s.to_numpy(dtype=float)],
    )


def weights_for(md: MarketData, portfolio: PortfolioIn) -> dict[str, float]:
    """Validated weights whose assets all exist in the aligned data."""
    w = portfolio.weights()
    md.check_assets(list(w))
    return w


def engine_call[T](fn: Callable[[], T]) -> T:
    """Run an engine computation; its ValueError (bad input for this data) becomes a 422."""
    try:
        return fn()
    except ValueError as exc:
        raise AnalyticsInputError(str(exc)) from exc


def risk_free_for(md: MarketData, index: pd.Index, override: float | None) -> RiskFreeRate:
    """The annual risk-free rate for a window: the caller's value, or the data's T-bill rate.

    Default = geometric average of the IIMA daily 91-day T-bill return (RF) over the window's
    rows that have factor data, annualised over 252 rows:
        rf_annual = prod(1 + RF_t)^(252 / n) - 1
    If factor data covers less than half of the window (the factor file ends before the
    window does), an average over a few months would stand in for the whole window, so the
    latest RF_FALLBACK_ROWS rows of factor data are used instead and the source says so.
    """
    if override is not None:
        return RiskFreeRate(annual=override, source="supplied by the caller")
    if md.factors is None:
        raise AnalyticsInputError(
            "risk_free_annual is required: factor data (the default T-bill source) is not loaded"
        )
    rf_all = md.factors.factors["RF"].dropna()
    rf = md.factors.factors["RF"].reindex(pd.DatetimeIndex(index)).dropna()
    fallback = len(rf) < MIN_RF_COVERAGE * len(index)
    if fallback:
        rf = rf_all.iloc[-RF_FALLBACK_ROWS:]
    annual = float(np.prod(1 + rf.to_numpy()) ** (TRADING_DAYS / len(rf)) - 1)
    release = md.factor_meta.release if md.factor_meta else "unknown"
    source = (
        f"91-day T-bill (RF in the IIMA factor file, release {release}), geometric average "
        f"over {rf.index[0].date()} to {rf.index[-1].date()} ({len(rf)} rows)"
    )
    if fallback:
        source += (
            f"; factor data covers less than {MIN_RF_COVERAGE:.0%} of the analysis window, "
            "so the latest year is used"
        )
    return RiskFreeRate(annual=annual, source=source)
