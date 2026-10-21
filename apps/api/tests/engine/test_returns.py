"""Hand-checkable tests for app.engine.returns.

Dates: D[i] is the i-th business day from Monday 2025-01-06 (D[0]=Mon 6 Jan, D[4]=Fri 10 Jan,
D[5]=Mon 13 Jan, ...). Prices are 100 + i on D[i] unless stated, so every expected value
can be read off directly.
"""

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from app.engine.returns import (
    _fill_short_gaps,
    align_prices,
    build_calendar,
    log_returns,
    simple_returns,
)

D = pd.bdate_range("2025-01-06", periods=30)
START = date(2025, 1, 1)


def series(idx: list[int], base: float = 100.0) -> pd.Series:
    """Price base + i on each D[i] in idx."""
    return pd.Series([base + i for i in idx], index=D[idx], dtype=float)


def full(n: int = 15, base: float = 100.0) -> pd.Series:
    return series(list(range(n)), base)


def align(prices: dict[str, pd.Series], calendar_ids: list[str], **kw):
    kw.setdefault("min_obs", 3)
    return align_prices(prices, calendar_ids=calendar_ids, benchmark_id="B", start=START, **kw)


# --- calendar ---


def test_calendar_needs_at_least_half_of_calendar_assets() -> None:
    every = list(range(10))
    prices = {
        "A": series(every),
        "B": series([i for i in every if i != 5]),
        "C": series([i for i in every if i not in (2, 5)]),
        "D": series([i for i in every if i not in (2, 5)]),
    }
    cal = build_calendar(prices, ["A", "B", "C", "D"], min_coverage=0.5)
    assert D[2] in cal  # 2 of 4 have a bar -> 50% -> kept
    assert D[5] not in cal  # 1 of 4 -> 25% -> excluded
    assert len(cal) == 9


def test_benchmark_never_defines_the_calendar_and_differences_are_reported() -> None:
    prices = {
        "S1": series([i for i in range(10) if i != 5]),
        "S2": series([i for i in range(10) if i != 5]),
        "B": series([i for i in range(10) if i != 7]),  # has D5 (not on calendar), lacks D7
    }
    r = align(prices, ["S1", "S2"]).report
    assert r.calendar_only_days == (D[7].date(),)  # we have it, benchmark doesn't
    assert r.benchmark_only_days == (D[5].date(),)  # benchmark has it, calendar doesn't
    assert r.filled_values["B"] == 1  # benchmark gap on D7 filled
    assert r.off_calendar_obs["B"] == 1  # its D5 bar is discarded


def test_weekend_session_kept_and_listed() -> None:
    saturday = pd.Timestamp("2025-01-11")  # e.g. a Muhurat / budget-day session
    stock = pd.concat([full(10), pd.Series([104.5], index=[saturday])]).sort_index()
    prices = {"S1": stock, "S2": stock.copy(), "B": full(10)}
    out = align(prices, ["S1", "S2"])
    assert saturday in out.prices.index
    assert out.report.weekend_sessions == (saturday.date(),)
    assert out.prices.loc[saturday, "B"] == 104.0  # benchmark filled from Fri D4 = 104
    assert out.report.calendar_only_days == (saturday.date(),)


# --- forward-fill and gaps ---


def test_gap_of_three_is_filled_and_gap_of_four_is_not() -> None:
    x_idx = [i for i in range(15) if i not in (3, 4, 5, 8, 9, 10, 11)]  # gaps: 3 days, 4 days
    prices = {"S": full(15), "B": full(15), "X": series(x_idx)}
    out = align(prices, ["S"])

    # 3-day gap: D3..D5 carry the D2 price (102) -> 0% returns, then D6 jumps to 106.
    assert out.prices.loc[D[3:6], "X"].tolist() == [102.0, 102.0, 102.0]
    assert out.report.filled_values["X"] == 3
    # 4-day gap: not filled at all; those rows are removed for every asset.
    assert out.report.gap_rows_dropped == tuple(d.date() for d in D[8:12])
    assert out.report.gap_rows_by_asset == {"S": 0, "B": 0, "X": 4}
    assert all(d not in out.prices.index for d in D[8:12])
    assert out.report.final_rows == 15 - 4
    # Across the dropped rows the return spans D7 -> D12: 112/107 - 1, same span for all.
    r = simple_returns(out.prices)
    assert r.loc[D[12], "X"] == pytest.approx(112 / 107 - 1)
    assert r.loc[D[12], "S"] == pytest.approx(112 / 107 - 1)


def test_no_fill_past_last_real_price_and_end_trimmed() -> None:
    prices = {"S": full(15), "B": full(15), "X": series(list(range(12)))}  # X ends at D11
    out = align(prices, ["S"])
    r = out.report
    assert r.common_end == D[11].date()
    assert r.end_set_by == ("X",)
    assert r.rows_trimmed_end == 3  # D12, D13, D14
    assert out.prices.index[-1] == D[11]
    assert r.filled_values["X"] == 0  # nothing invented after the last price


def test_fill_helper_never_fills_before_first_or_after_last_price() -> None:
    # Defence in depth: trimming already hides trailing fills in align_prices, so test the
    # helper directly. NaN, 10, NaN, 12, NaN, NaN -> only the interior gap is filled.
    s = pd.Series([np.nan, 10.0, np.nan, 12.0, np.nan, np.nan], index=D[:6])
    filled, mask = _fill_short_gaps(s, max_days=3)
    assert filled.tolist()[1:4] == [10.0, 10.0, 12.0]
    assert np.isnan(filled.iloc[0]) and filled.iloc[4:].isna().all()
    assert mask.tolist() == [False, False, True, False, False, False]


def test_start_trimmed_to_latest_first_price() -> None:
    prices = {"S": full(15), "B": full(15), "Y": series(list(range(3, 15)))}  # Y starts D3
    r = align(prices, ["S"]).report
    assert r.common_start == D[3].date()
    assert r.start_set_by == ("Y",)
    assert r.rows_trimmed_start == 3


def test_data_before_start_is_ignored() -> None:
    prices = {"S": full(15), "B": full(15)}
    out = align_prices(prices, calendar_ids=["S"], benchmark_id="B", start=D[5].date(), min_obs=3)
    assert out.prices.index[0] == D[5]
    assert out.report.calendar_days == 10


def test_gap_guard_warns_above_one_percent_of_rows() -> None:
    def run(n_days: int):
        idx = pd.bdate_range("2025-01-06", periods=n_days)
        base = pd.Series(np.arange(n_days, dtype=float) + 100, index=idx)
        x = base.drop(idx[10:14])  # one 4-day gap -> 4 rows dropped
        return align_prices(
            {"S": base, "B": base.copy(), "X": x},
            calendar_ids=["S"],
            benchmark_id="B",
            start=START,
            min_obs=3,
        ).report

    assert run(500).warnings == ()  # 4/500 = 0.8%
    (w,) = run(300).warnings  # 4/300 = 1.3%
    assert w.startswith("X caused 4 of 300 rows (1.3%)")
    assert "WARNING: X caused 4 of 300 rows" in run(300).summary()


# --- dropping short histories ---


def test_short_history_asset_dropped_without_moving_the_window() -> None:
    prices = {"S": full(15), "B": full(15), "Z": series([10, 11, 12, 13])}  # 4 obs, late start
    out = align(prices, ["S"], min_obs=5)
    (d,) = out.report.dropped_assets
    assert (d.asset_id, d.observations) == ("Z", 4)
    assert "only 4 observations" in d.reason
    assert "Z" not in out.prices.columns
    assert out.report.common_start == D[0].date()  # Z's late start did not shrink the data


def test_benchmark_with_too_little_history_raises() -> None:
    prices = {"S": full(15), "B": series([0, 1])}
    with pytest.raises(ValueError, match="benchmark B: only 2 observations"):
        align(prices, ["S"], min_obs=5)


# --- output shape and input validation ---


def test_output_shape_no_nan_and_column_order() -> None:
    prices = {"S": full(15), "FUND": full(15, base=10.0), "B": full(15)}
    out = align(prices, ["S"])
    assert list(out.prices.columns) == ["S", "FUND", "B"]
    assert out.prices.shape == (15, 3)
    assert not out.prices.isna().any().any()
    assert out.report.final_rows == 15


def test_off_calendar_prices_are_discarded_and_counted() -> None:
    sunday = pd.Timestamp("2025-01-12")  # e.g. a liquid fund's weekend NAV
    fund = pd.concat([full(10, base=10.0), pd.Series([14.2], index=[sunday])]).sort_index()
    out = align({"S": full(10), "B": full(10), "F": fund}, ["S"])
    assert sunday not in out.prices.index
    assert out.report.off_calendar_obs["F"] == 1


@pytest.mark.parametrize(
    "bad",
    [
        pd.Series([100.0, np.nan], index=D[:2]),
        pd.Series([100.0, 0.0], index=D[:2]),
        pd.Series([100.0, 101.0], index=D[[1, 0]]),
        pd.Series([100.0], index=D[:1].tz_localize("Asia/Kolkata")),
    ],
    ids=["nan", "zero", "unsorted", "tz-aware"],
)
def test_invalid_input_raises(bad: pd.Series) -> None:
    with pytest.raises(ValueError):
        align({"S": full(15), "B": full(15), "X": bad}, ["S"])


# --- returns ---


def test_simple_and_log_returns_hand_checked() -> None:
    p = pd.Series([100.0, 110.0, 99.0], index=D[:3])
    assert simple_returns(p).tolist() == pytest.approx([0.10, -0.10])
    assert log_returns(p).tolist() == pytest.approx([math.log(1.1), math.log(0.9)])
    assert simple_returns(p).index.tolist() == D[1:3].tolist()  # first row dropped


def test_log_returns_equal_log1p_of_simple_returns() -> None:
    p = pd.DataFrame(
        {"a": [100.0, 103.0, 101.0, 108.0], "b": [50.0, 49.0, 52.0, 52.0]}, index=D[:4]
    )
    pd.testing.assert_frame_equal(log_returns(p), np.log1p(simple_returns(p)))
    assert simple_returns(p).shape == (3, 2)


def test_simple_returns_aggregate_across_assets_log_returns_do_not() -> None:
    # Rs 1,000 portfolio: 6 units of A at 100 (60%), 8 units of B at 50 (40%).
    p = pd.DataFrame({"A": [100.0, 110.0], "B": [50.0, 45.0]}, index=D[:2])
    w = np.array([0.6, 0.4])
    portfolio_return = (6 * 110 + 8 * 45) / 1000 - 1  # 1020 / 1000 - 1 = 2%
    assert simple_returns(p).iloc[0].to_numpy() @ w == pytest.approx(portfolio_return)
    weighted_log = log_returns(p).iloc[0].to_numpy() @ w  # 0.6 ln1.1 + 0.4 ln0.9 = 0.01504
    assert weighted_log == pytest.approx(0.015042, abs=1e-6)
    assert weighted_log != pytest.approx(math.log(1 + portfolio_return))  # ln 1.02 = 0.01980


def test_returns_reject_unaligned_prices() -> None:
    with pytest.raises(ValueError, match="align"):
        simple_returns(pd.Series([100.0, np.nan, 101.0], index=D[:3]))


def test_summary_is_readable_when_every_asset_sets_the_end() -> None:
    out = align({"S": full(10), "F": full(10, base=10.0), "B": full(10)}, ["S"])
    assert out.report.end_set_by == ("S", "F", "B")
    assert "end set by all 3 assets" in out.report.summary()
