"""Price alignment and return calculation.

Pure functions on pandas objects: no I/O, no network, no imports from app.data or app.ai
(enforced by tests/test_architecture.py). Loading from the cache happens in
app/services/market_data.py.

Conventions
-----------
- Prices: one Series per asset, tz-naive DatetimeIndex (calendar dates), strictly
  increasing, finite and > 0. Stocks are dividend/split-adjusted closes, funds are Growth
  NAVs, the benchmark (^NSEI) is a price index.
- Returns: **simple** daily returns, r_t = P_t / P_{t-1} - 1, used throughout the platform.
  `log_returns` exists for time-aggregation/GBM calibration only (see its docstring).
- A "day" is a row of the master calendar. When a calendar day is skipped (e.g. a gap row
  is dropped), the next return spans two sessions; this is the same for every asset.
- Annualisation (elsewhere) uses 252 trading days.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

DEFAULT_MIN_CALENDAR_COVERAGE = 0.5
DEFAULT_MAX_FFILL_DAYS = 3
DEFAULT_MIN_OBS = 756  # about 3 years x 252 trading days
GAP_ROW_WARNING_SHARE = 0.01  # warn if one asset forces >1% of rows to be dropped


# --- result types ---


@dataclass(frozen=True)
class DroppedAsset:
    asset_id: str
    observations: int
    reason: str


@dataclass(frozen=True)
class AlignmentReport:
    """Everything `align_prices` did to the data, so nothing changes silently."""

    start: date
    max_ffill_days: int
    min_obs: int
    min_coverage: float
    calendar_ids: tuple[str, ...]
    benchmark_id: str
    calendar_days: int  # master-calendar days on/after `start`
    # Differences vs the benchmark's own dates, within the final common window:
    calendar_only_days: tuple[date, ...]  # on our calendar, missing from the benchmark
    benchmark_only_days: tuple[date, ...]  # benchmark has a bar, our calendar does not
    weekend_sessions: tuple[date, ...]  # Sat/Sun calendar days kept (e.g. Muhurat, budget day)
    common_start: date
    start_set_by: tuple[str, ...]
    rows_trimmed_start: int
    common_end: date
    end_set_by: tuple[str, ...]
    rows_trimmed_end: int
    dropped_assets: tuple[DroppedAsset, ...]
    filled_values: Mapping[str, int]  # forward-filled cells per asset, in the final window
    off_calendar_obs: Mapping[str, int]  # real prices discarded because not on the calendar
    gap_rows_dropped: tuple[date, ...]  # rows removed because some asset had a gap > max_ffill
    gap_rows_by_asset: Mapping[str, int]
    final_rows: int
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def summary(self) -> str:
        def days(ds: Sequence[date], limit: int = 12) -> str:
            shown = ", ".join(str(d) for d in ds[:limit])
            return f"{len(ds)}" + (f" [{shown}{', ...' if len(ds) > limit else ''}]" if ds else "")

        n_assets = len(self.filled_values)

        def who(ids: Sequence[str]) -> str:
            if len(ids) == n_assets:
                return f"all {n_assets} assets"
            return ", ".join(ids[:5]) + (f" and {len(ids) - 5} more" if len(ids) > 5 else "")

        def nonzero(m: Mapping[str, int]) -> str:
            items = [f"{k}={v}" for k, v in sorted(m.items(), key=lambda kv: -kv[1]) if v]
            return ", ".join(items) if items else "none"

        lines = [
            "Alignment report",
            f"  parameters: start>={self.start}, calendar = dates where >= {self.min_coverage:.0%}"
            f" of {len(self.calendar_ids)} calendar assets have a bar, ffill <= "
            f"{self.max_ffill_days} days, min_obs = {self.min_obs}",
            f"  master calendar days (from start): {self.calendar_days}",
            f"  common window: {self.common_start} -> {self.common_end}, {self.final_rows} rows",
            f"    start set by {who(self.start_set_by)} "
            f"({self.rows_trimmed_start} calendar rows trimmed)",
            f"    end set by {who(self.end_set_by)} "
            f"({self.rows_trimmed_end} calendar rows trimmed)",
            f"  vs {self.benchmark_id} calendar (within window):",
            f"    on our calendar, not in benchmark: {days(self.calendar_only_days)}",
            f"    in benchmark, not on our calendar: {days(self.benchmark_only_days)}",
            f"  weekend sessions kept: {days(self.weekend_sessions)}",
            "  dropped assets: "
            + (
                "; ".join(f"{d.asset_id} ({d.reason})" for d in self.dropped_assets)
                if self.dropped_assets
                else "none"
            ),
            f"  forward-filled values: {nonzero(self.filled_values)}",
            f"  off-calendar prices discarded: {nonzero(self.off_calendar_obs)}",
            f"  rows dropped for gaps > {self.max_ffill_days} days: "
            f"{days(self.gap_rows_dropped)}; by asset: {nonzero(self.gap_rows_by_asset)}",
        ]
        lines += [f"  WARNING: {w}" for w in self.warnings] or ["  warnings: none"]
        return "\n".join(lines)


@dataclass(frozen=True)
class AlignedPrices:
    prices: pd.DataFrame  # rows = cleaned calendar, columns = asset ids, no NaN
    report: AlignmentReport


# --- helpers ---


def _validate_series(s: pd.Series, label: str) -> None:
    if not isinstance(s.index, pd.DatetimeIndex):
        raise ValueError(f"{label}: index must be a DatetimeIndex")
    if s.index.tz is not None:
        raise ValueError(f"{label}: index must be tz-naive")
    if not s.index.is_monotonic_increasing or not s.index.is_unique:
        raise ValueError(f"{label}: dates must be strictly increasing")
    values = s.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError(f"{label}: prices must be finite and > 0")


def _fill_short_gaps(s: pd.Series, max_days: int) -> tuple[pd.Series, pd.Series]:
    """Forward-fill interior gaps of at most `max_days` consecutive NaNs.

    Gaps longer than `max_days` are left entirely NaN (never half-filled). Only the span
    between the first and last real price is touched, so nothing is filled before an
    asset starts or after its last real price. Returns (filled series, mask of filled cells).
    """
    first, last = s.first_valid_index(), s.last_valid_index()
    inside = (s.index >= first) & (s.index <= last)
    isna = s.isna() & inside
    run_id = (~s.isna()).cumsum()  # each gap shares an id with the price before it
    run_len = isna.groupby(run_id).transform("sum")
    fill_mask = isna & (run_len <= max_days)
    return s.where(~fill_mask, s.ffill()), fill_mask


def _dates(idx: pd.Index) -> tuple[date, ...]:
    return tuple(d.date() for d in pd.DatetimeIndex(idx))


# --- public API ---


def build_calendar(
    prices: Mapping[str, pd.Series],
    calendar_ids: Sequence[str],
    *,
    min_coverage: float = DEFAULT_MIN_CALENDAR_COVERAGE,
) -> pd.DatetimeIndex:
    """Master trading calendar: dates on which enough calendar assets have a bar.

    A date d is included iff  #{i in calendar_ids : d in prices[i].index}
                              >= ceil(min_coverage * len(calendar_ids)).

    The denominator is fixed at len(calendar_ids) (not "assets active that day") so that an
    incomplete latest session, where most stocks have no bar yet, cannot pass the rule.
    Using many stocks instead of the index's own dates protects against gaps in a single
    series (Yahoo's ^NSEI misses several real sessions) and against single-stock phantom bars.
    """
    if not calendar_ids:
        raise ValueError("calendar_ids must not be empty")
    if not 0 < min_coverage <= 1:
        raise ValueError("min_coverage must be in (0, 1]")
    missing = [i for i in calendar_ids if i not in prices]
    if missing:
        raise ValueError(f"calendar assets missing from prices: {missing}")
    needed = math.ceil(min_coverage * len(calendar_ids))
    stacked = pd.concat([pd.Series(1, index=prices[i].index) for i in calendar_ids])
    counts = stacked.groupby(level=0).sum()
    return pd.DatetimeIndex(counts.index[counts >= needed], name="date").sort_values()


def align_prices(
    prices: Mapping[str, pd.Series],
    *,
    calendar_ids: Sequence[str],
    benchmark_id: str,
    start: date,
    max_ffill_days: int = DEFAULT_MAX_FFILL_DAYS,
    min_obs: int = DEFAULT_MIN_OBS,
    min_coverage: float = DEFAULT_MIN_CALENDAR_COVERAGE,
) -> AlignedPrices:
    """Align price series of different assets onto one clean trading calendar.

    Steps (each one is recorded in the returned AlignmentReport):
      1. Validate inputs; cut every series at `start`.
      2. Master calendar = `build_calendar(calendar_ids)` (the stocks). The benchmark is an
         ordinary column: it never adds or removes calendar days.
      3. Reindex each asset onto the calendar. Prices on dates that are not calendar days
         (liquid-fund weekend NAVs, year-end NAVs on holidays) are discarded and counted.
         For accruing assets this is correct: the next calendar price includes the accrual.
      4. Drop assets with fewer than `min_obs` real prices on the calendar, before the window
         is computed so a dropped asset cannot move it. The benchmark is mandatory: if it is
         too short, raise.
      5. Common window = [latest first price, earliest last price] over the kept assets.
      6. Forward-fill interior gaps of <= `max_ffill_days` consecutive calendar days with the
         last real price (a filled day is a 0% return; the move lands on the next real day).
         Longer gaps are not filled at all. Filling never goes past an asset's last price.
      7. Rows still containing NaN (a gap > max_ffill_days in some asset) are dropped for
         every asset. If one asset forces more than 1% of the window's rows out, a warning
         is added to the report.

    Args:
        prices: asset id -> price Series (see module conventions).
        calendar_ids: assets that define the calendar (the stocks).
        benchmark_id: benchmark asset id (must be in `prices`).
        start: earliest date used (inclusive).
        max_ffill_days: longest gap (consecutive calendar days) that is forward-filled.
        min_obs: minimum real prices on the calendar for an asset to be kept.
        min_coverage: share of calendar assets that must have a bar for a date to count.

    Returns:
        AlignedPrices with a NaN-free DataFrame (columns in input order, minus dropped
        assets) and the report.
    """
    if benchmark_id not in prices:
        raise ValueError(f"benchmark {benchmark_id!r} missing from prices")
    if benchmark_id in calendar_ids:
        raise ValueError("the benchmark must not be one of the calendar assets")
    if max_ffill_days < 0 or min_obs < 1:
        raise ValueError("max_ffill_days must be >= 0 and min_obs >= 1")
    for asset_id, s in prices.items():
        _validate_series(s, asset_id)

    start_ts = pd.Timestamp(start)
    cut = {k: s[s.index >= start_ts] for k, s in prices.items()}
    calendar = build_calendar(cut, calendar_ids, min_coverage=min_coverage)
    if calendar.empty:
        raise ValueError("master calendar is empty")

    # 3. reindex onto the calendar
    on_cal = {k: s.reindex(calendar) for k, s in cut.items()}
    off_calendar = {k: int(len(cut[k]) - on_cal[k].notna().sum()) for k in prices}

    # 4. drop short histories
    dropped: list[DroppedAsset] = []
    kept: list[str] = []
    for k in prices:
        n = int(on_cal[k].notna().sum())
        if n >= min_obs:
            kept.append(k)
            continue
        reason = f"only {n} observations on the calendar since {start} (< {min_obs})"
        if k == benchmark_id:
            raise ValueError(f"benchmark {k}: {reason}")
        dropped.append(DroppedAsset(asset_id=k, observations=n, reason=reason))

    # 5. common window
    firsts = {k: on_cal[k].first_valid_index() for k in kept}
    lasts = {k: on_cal[k].last_valid_index() for k in kept}
    common_start, common_end = max(firsts.values()), min(lasts.values())
    if common_start > common_end:
        raise ValueError(f"no common window: start {common_start} > end {common_end}")
    window = calendar[(calendar >= common_start) & (calendar <= common_end)]

    # 6. forward-fill short gaps (per asset, inside its own first..last range)
    columns: dict[str, pd.Series] = {}
    filled: dict[str, int] = {}
    for k in kept:
        s, mask = _fill_short_gaps(on_cal[k], max_ffill_days)
        columns[k] = s.loc[window]
        filled[k] = int(mask.loc[window].sum())
    frame = pd.DataFrame(columns, index=window)

    # 7. drop rows with remaining gaps
    nan_rows = frame.isna().any(axis=1)
    gap_by_asset = {k: int(frame[k].isna().sum()) for k in kept}
    warnings = [
        f"{k} caused {n} of {len(window)} rows ({n / len(window):.1%}) to be dropped "
        f"for gaps > {max_ffill_days} days"
        for k, n in gap_by_asset.items()
        if n / len(window) > GAP_ROW_WARNING_SHARE
    ]
    clean = frame.loc[~nan_rows]
    assert not clean.isna().any().any()

    bench_dates = cut[benchmark_id].index
    bench_in_window = bench_dates[(bench_dates >= common_start) & (bench_dates <= common_end)]
    report = AlignmentReport(
        start=start,
        max_ffill_days=max_ffill_days,
        min_obs=min_obs,
        min_coverage=min_coverage,
        calendar_ids=tuple(calendar_ids),
        benchmark_id=benchmark_id,
        calendar_days=len(calendar),
        calendar_only_days=_dates(window.difference(bench_in_window)),
        benchmark_only_days=_dates(bench_in_window.difference(window)),
        weekend_sessions=_dates(window[window.dayofweek >= 5]),
        common_start=common_start.date(),
        start_set_by=tuple(k for k in kept if firsts[k] == common_start),
        rows_trimmed_start=int((calendar < common_start).sum()),
        common_end=common_end.date(),
        end_set_by=tuple(k for k in kept if lasts[k] == common_end),
        rows_trimmed_end=int((calendar > common_end).sum()),
        dropped_assets=tuple(dropped),
        filled_values=filled,
        off_calendar_obs=off_calendar,
        gap_rows_dropped=_dates(window[nan_rows.to_numpy()]),
        gap_rows_by_asset=gap_by_asset,
        final_rows=len(clean),
        warnings=tuple(warnings),
    )
    return AlignedPrices(prices=clean, report=report)


def _check_clean_prices(prices: pd.DataFrame | pd.Series) -> None:
    values = prices.to_numpy(dtype=float)
    if np.isnan(values).any():
        raise ValueError("prices contain NaN; align them first (align_prices)")
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("prices must be finite and > 0")
    if not prices.index.is_monotonic_increasing:
        raise ValueError("prices must be sorted by date")


def simple_returns[T: (pd.DataFrame, pd.Series)](prices: T) -> T:
    """Simple (arithmetic) returns between consecutive rows.

        r_t = P_t / P_{t-1} - 1          (decimal, e.g. 0.01 = +1%)

    The first row has no return and is dropped, so the output has len(prices) - 1 rows.
    A return spans whatever lies between two consecutive calendar rows (normally one
    session; two if a session was removed during alignment).

    Why simple returns are the platform convention: they aggregate **across assets**
    exactly. A portfolio holding weights w_i at t-1 has return
        R_p,t = sum_i w_i * r_i,t,
    so portfolio returns, P&L (value x R_p) and VaR stated as a loss in rupees follow
    directly. Log returns do not have this property: ln(1 + R_p) != sum_i w_i ln(1 + r_i).

    Requires aligned prices: no NaN, finite, > 0, sorted (raises ValueError otherwise).
    """
    _check_clean_prices(prices)
    return (prices / prices.shift(1) - 1).iloc[1:]


def log_returns[T: (pd.DataFrame, pd.Series)](prices: T) -> T:
    """Log (continuously compounded) returns between consecutive rows.

        l_t = ln(P_t / P_{t-1}) = ln(1 + r_t)

    Caveat: NOT for portfolio aggregation (a weighted sum of log returns is not the
    portfolio's log return). Use it where additivity **over time** matters:
    l_(t..t+k) = sum of daily l, e.g. calibrating a GBM for Monte Carlo goal planning.
    Everything else on the platform uses `simple_returns`.

    Same input requirements and output shape as `simple_returns`.
    """
    _check_clean_prices(prices)
    return np.log(prices / prices.shift(1)).iloc[1:]
