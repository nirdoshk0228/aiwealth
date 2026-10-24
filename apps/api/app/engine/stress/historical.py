"""Historical replay: what a past window would do to today's portfolio.

Pure numpy/pandas: no I/O, no network, no imports from app.data or app.ai (enforced by
tests/test_architecture.py). Prices come from app.services.market_data.load_aligned_prices.

Model (buy-and-hold inside the window)
--------------------------------------
Invest today's weights w_i at the close of `start` and hold the units, without rebalancing,
until the close of `end`:

    R_i      = P_i(end) / P_i(start) - 1                   window return of asset i
    V_t / V0 = sum_i w_i P_i(t) / P_i(start)                portfolio value path, V0 = 1
    P&L      = V_end / V0 - 1 = sum_i w_i R_i               exact for buy-and-hold
    c_i      = w_i R_i                                      sum_i c_i = P&L exactly

    max drawdown in window = max_t (1 - V_t / max_{s<=t} V_s),  t in [start, end]

Why buy-and-hold rather than the fixed weights of var.py: over a multi-week crash an
investor does not rebalance daily, and with buy-and-hold the per-asset split w_i R_i is exact.
With daily rebalancing the P&L would be prod_t (1 + w' r_t) - 1, which differs (by about
+12.5% vs 0% for an asset that doubles and halves; see the tests), and its per-asset split is
not exact. For a one-day window the two are identical.

Assumptions and limits
----------------------
- Today's weights are applied to past returns: the replay says how the *current* assets
  behaved, not what any past portfolio lost. Funds' holdings and managers have changed.
- Each window is one realised path. It is not a probability or a worst case, and it carries
  everything else that happened in it (e.g. the 2016 US election inside the demonetisation
  window). Cause and window choice are documented per scenario in data/scenarios/.
- Prices are aligned closes and NAVs: a fund NAV is struck at the day's close, and calendar
  days filled by align_prices (<= 3 days) carry a 0% move that lands on the next real price.
- Assets must have real data at both ends; nothing is proxied or backfilled.
"""

from collections.abc import Mapping

import numpy as np
import pandas as pd

from app.engine.stress.result import AssetImpact, StressResult, WindowDrawdown
from app.engine.stress.scenarios import HistoricalScenario, check_portfolio_value, check_weights

NOTE_BUY_AND_HOLD = (
    "Historical replay: today's weights are invested at the start close and held without "
    "rebalancing to the end close (buy-and-hold); P&L = sum of weight x window return."
)


def _window(prices: pd.DataFrame, scenario: HistoricalScenario) -> pd.DataFrame:
    """Rows from scenario.start to scenario.end inclusive; both must be rows of `prices`."""
    idx = prices.index
    start, end = pd.Timestamp(scenario.start), pd.Timestamp(scenario.end)
    if start < idx[0] or end > idx[-1]:
        raise ValueError(
            f"{scenario.id}: window {scenario.start} -> {scenario.end} is outside the data "
            f"({idx[0].date()} -> {idx[-1].date()})"
        )
    for label, ts in (("start", start), ("end", end)):
        if ts not in idx:
            pos = idx.searchsorted(ts)
            raise ValueError(
                f"{scenario.id}: {label} {ts.date()} is not a trading day in the data; nearest "
                f"rows are {idx[pos - 1].date()} and {idx[pos].date()}"
            )
    return prices.loc[start:end]


def window_drawdown(values: pd.Series) -> WindowDrawdown:
    """Max drawdown of a positive value path; the first value counts as a peak.

        dd_t = 1 - V_t / max_{s<=t} V_s,   max drawdown = max_t dd_t   (positive)

    Peak date = the date of the running maximum in force at the trough (its first occurrence).
    """
    v = values.to_numpy(dtype=float)
    running_max = np.maximum.accumulate(v)
    dd = 1 - v / running_max
    t = int(np.argmax(dd))
    if dd[t] <= 0:
        return WindowDrawdown(max_drawdown=0.0, peak_date=None, trough_date=None)
    peak = int(np.argmax(v[: t + 1]))
    return WindowDrawdown(
        max_drawdown=float(dd[t]),
        peak_date=values.index[peak].date(),
        trough_date=values.index[t].date(),
    )


def historical_replay(
    prices: pd.DataFrame,
    weights: Mapping[str, float] | pd.Series,
    scenario: HistoricalScenario,
    *,
    portfolio_value: float,
) -> StressResult:
    """Replay `scenario`'s window on `weights` (buy-and-hold; see the module docstring).

    Args:
        prices: aligned prices (rows = trading days, columns = asset ids), as produced by
            align_prices. Columns without a weight are ignored.
        weights: asset id -> weight, finite and summing to 1.
        scenario: the window; start and end must both be rows of `prices`.
        portfolio_value: current portfolio value in rupees (> 0), for rupee P&L.

    Returns:
        StressResult with P&L (fraction and rupees), per-asset window returns and contributions
        w_i R_i, the in-window max drawdown of the portfolio value and the number of sessions.

    Raises:
        TypeError / ValueError: wrong types, invalid weights or value, a window outside the data
            or on a non-trading day, missing/NaN/non-positive prices in the window, or a
            non-positive portfolio value path (possible only with shorts).
    """
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame (columns = asset ids)")
    if not isinstance(prices.index, pd.DatetimeIndex) or prices.empty:
        raise ValueError("prices: index must be a non-empty DatetimeIndex")
    if not prices.index.is_monotonic_increasing or not prices.index.is_unique:
        raise ValueError("prices: dates must be strictly increasing")
    w = check_weights(weights)
    check_portfolio_value(portfolio_value)
    missing = [a for a in w.index if a not in prices.columns]
    if missing:
        raise ValueError(f"weights reference assets not in prices: {missing}")

    window = _window(prices, scenario)[list(w.index)]
    values = window.to_numpy(dtype=float)
    if np.isnan(values).any():
        bad = list(window.columns[np.isnan(values).any(axis=0)])
        raise ValueError(f"{scenario.id}: prices missing inside the window for {bad}")
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError(f"{scenario.id}: prices must be finite and > 0")

    relative = window / window.iloc[0]
    asset_returns = relative.iloc[-1] - 1
    path = relative @ w
    if (path <= 0).any():
        raise ValueError(f"{scenario.id}: portfolio value path is <= 0; drawdown is undefined")
    contrib = w * asset_returns
    pnl = float(contrib.sum())
    impacts = tuple(
        AssetImpact(
            asset_id=a,
            weight=float(w[a]),
            asset_return=float(asset_returns[a]),
            contribution=float(contrib[a]),
            contribution_rupees=float(contrib[a]) * portfolio_value,
            source="historical",
        )
        for a in w.index
    )
    return StressResult(
        scenario_id=scenario.id,
        scenario_name=scenario.name,
        scenario_type="historical",
        portfolio_value=float(portfolio_value),
        pnl=pnl,
        pnl_rupees=pnl * portfolio_value,
        impacts=impacts,
        notes=(NOTE_BUY_AND_HOLD,),
        warnings=(),
        start=scenario.start,
        end=scenario.end,
        sessions=len(window) - 1,
        drawdown=window_drawdown(path),
    )
