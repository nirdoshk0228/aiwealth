"""Tests for app.engine.stress.historical on tiny price series with hand-computed answers."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from app.engine.stress.historical import historical_replay, window_drawdown
from app.engine.stress.scenarios import HistoricalScenario

DAYS = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"])


@pytest.fixture
def prices() -> pd.DataFrame:
    # A: +10%, -10%, +22.2%  (121 / 100 = +21% over the window)
    # B: 0, +10%, +9.1%       (60 / 50 = +20%)
    # C: not in the portfolio (ignored)
    return pd.DataFrame(
        {"A": [100.0, 110.0, 99.0, 121.0], "B": [50.0, 50.0, 55.0, 60.0], "C": [1.0, 2, 3, 4]},
        index=DAYS,
    )


def scenario(start: str = "2024-01-01", end: str = "2024-01-04") -> HistoricalScenario:
    return HistoricalScenario(
        id="t", name="test", start=date.fromisoformat(start), end=date.fromisoformat(end)
    )


def by_id(result) -> dict:
    return {i.asset_id: i for i in result.impacts}


def test_replay_by_hand(prices: pd.DataFrame) -> None:
    r = historical_replay(prices, {"A": 0.5, "B": 0.5}, scenario(), portfolio_value=1_000_000)
    imp = by_id(r)
    # R_A = 121/100 - 1 = 0.21, R_B = 60/50 - 1 = 0.20
    assert imp["A"].asset_return == pytest.approx(0.21)
    assert imp["B"].asset_return == pytest.approx(0.20)
    assert imp["A"].contribution == pytest.approx(0.105)
    assert imp["B"].contribution == pytest.approx(0.100)
    assert r.pnl == pytest.approx(0.205)
    assert r.pnl_rupees == pytest.approx(205_000)
    assert imp["A"].contribution_rupees == pytest.approx(105_000)
    assert r.sessions == 3
    assert set(imp) == {"A", "B"}  # C ignored
    assert all(i.source == "historical" for i in r.impacts)
    assert any("buy-and-hold" in n for n in r.notes)


def test_in_window_drawdown_by_hand(prices: pd.DataFrame) -> None:
    # V = 0.5 A/100 + 0.5 B/50 = 1.000, 1.050, 1.045, 1.205
    # max drawdown = 1 - 1.045/1.050, from the peak on day 2 to the trough on day 3
    r = historical_replay(prices, {"A": 0.5, "B": 0.5}, scenario(), portfolio_value=1)
    assert r.drawdown is not None
    assert r.drawdown.max_drawdown == pytest.approx(1 - 1.045 / 1.050)
    assert r.drawdown.peak_date == date(2024, 1, 2)
    assert r.drawdown.trough_date == date(2024, 1, 3)


def test_sub_window_uses_its_own_base(prices: pd.DataFrame) -> None:
    # day 2 -> day 3: R_A = 99/110 - 1 = -0.1, R_B = 55/50 - 1 = 0.1
    r = historical_replay(
        prices, {"A": 0.7, "B": 0.3}, scenario("2024-01-02", "2024-01-03"), portfolio_value=100
    )
    assert r.pnl == pytest.approx(0.7 * -0.1 + 0.3 * 0.1)  # -0.04
    assert r.pnl_rupees == pytest.approx(-4.0)
    assert r.sessions == 1
    assert r.worst_contributors()[0].asset_id == "A"


def test_buy_and_hold_not_daily_rebalancing() -> None:
    # A doubles then halves, B is flat. Buy-and-hold: A ends where it started, P&L = 0.
    # Daily rebalancing to 50/50 would give 1.5 x 0.75 - 1 = +12.5%.
    p = pd.DataFrame({"A": [100.0, 200.0, 100.0], "B": [100.0, 100.0, 100.0]}, index=DAYS[:3])
    r = historical_replay(
        p, {"A": 0.5, "B": 0.5}, scenario("2024-01-01", "2024-01-03"), portfolio_value=1
    )
    assert r.pnl == pytest.approx(0.0)
    # V = 1.0, 1.5, 1.0: a drawdown of 1/3 from the day-2 peak
    assert r.drawdown.max_drawdown == pytest.approx(1 / 3)


def test_rising_window_has_zero_drawdown() -> None:
    p = pd.DataFrame({"A": [1.0, 1.1, 1.2, 1.3]}, index=DAYS)
    r = historical_replay(p, {"A": 1.0}, scenario(), portfolio_value=1)
    assert r.drawdown.max_drawdown == 0.0
    assert r.drawdown.peak_date is None and r.drawdown.trough_date is None
    assert r.worst_contributors() == ()


def test_fall_from_the_start_counts_the_start_as_peak() -> None:
    v = pd.Series([1.0, 0.8, 0.9], index=DAYS[:3])
    dd = window_drawdown(v)
    assert dd.max_drawdown == pytest.approx(0.2)
    assert dd.peak_date == date(2024, 1, 1)


def test_contributions_sum_to_total_on_random_prices() -> None:
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2020-01-01", periods=60)
    p = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.02, (60, 6)), axis=0)),
        index=idx,
        columns=list("abcdef"),
    )
    w = dict(zip(p.columns, rng.dirichlet(np.ones(6)), strict=True))
    s = HistoricalScenario(id="r", name="r", start=idx[5].date(), end=idx[50].date())
    r = historical_replay(p, w, s, portfolio_value=2_500_000)
    assert sum(i.contribution for i in r.impacts) == pytest.approx(r.pnl, abs=1e-15)
    assert sum(i.contribution_rupees for i in r.impacts) == pytest.approx(r.pnl_rupees)
    # buy-and-hold P&L equals the end of the value path
    path = (p.loc[idx[5] : idx[50]] / p.loc[idx[5]]) @ pd.Series(w)
    assert r.pnl == pytest.approx(path.iloc[-1] - 1)


# --- rejection ---


@pytest.mark.parametrize(
    ("start", "end", "match"),
    [
        ("2023-12-29", "2024-01-03", "outside the data"),
        ("2024-01-02", "2024-01-05", "outside the data"),
    ],
)
def test_window_outside_data_rejected(prices, start: str, end: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        historical_replay(prices, {"A": 1.0}, scenario(start, end), portfolio_value=1)


def test_non_trading_day_rejected_with_nearest_rows() -> None:
    idx = pd.to_datetime(["2024-01-05", "2024-01-08", "2024-01-09"])  # Fri, Mon, Tue
    p = pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=idx)
    with pytest.raises(
        ValueError, match="2024-01-06 is not a trading day.*2024-01-05 and 2024-01-08"
    ):
        historical_replay(p, {"A": 1.0}, scenario("2024-01-06", "2024-01-09"), portfolio_value=1)


def test_start_not_before_end_rejected() -> None:
    with pytest.raises(ValueError, match="must be before"):
        scenario("2024-01-03", "2024-01-03")


@pytest.mark.parametrize(
    ("weights", "match"),
    [
        ({"A": 0.5, "B": 0.4}, "sum to 1"),
        ({"A": 0.5, "Z": 0.5}, "not in prices"),
        ({}, "empty"),
        ({"A": float("nan"), "B": 1.0}, "finite"),
    ],
)
def test_invalid_weights_rejected(prices, weights: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        historical_replay(prices, weights, scenario(), portfolio_value=1)


@pytest.mark.parametrize("value", [0, -1.0, float("inf"), float("nan")])
def test_invalid_portfolio_value_rejected(prices, value: float) -> None:
    with pytest.raises(ValueError, match="portfolio_value"):
        historical_replay(prices, {"A": 1.0}, scenario(), portfolio_value=value)


def test_nan_inside_window_rejected(prices: pd.DataFrame) -> None:
    prices.loc[DAYS[2], "B"] = np.nan
    with pytest.raises(ValueError, match=r"missing inside the window for \['B'\]"):
        historical_replay(prices, {"A": 0.5, "B": 0.5}, scenario(), portfolio_value=1)


def test_nan_outside_window_is_fine(prices: pd.DataFrame) -> None:
    prices.loc[DAYS[0], "B"] = np.nan
    r = historical_replay(
        prices, {"A": 0.5, "B": 0.5}, scenario("2024-01-02", "2024-01-04"), portfolio_value=1
    )
    assert by_id(r)["B"].asset_return == pytest.approx(0.2)  # 60 / 50


def test_non_positive_value_path_rejected() -> None:
    # 2x long A, 1x short B: V = 2 A/100 - B/100 hits 2*40/100 - 100/100 < 0 on day 2
    p = pd.DataFrame({"A": [100.0, 40.0, 50.0], "B": [100.0, 100.0, 100.0]}, index=DAYS[:3])
    with pytest.raises(ValueError, match="value path"):
        historical_replay(
            p, {"A": 2.0, "B": -1.0}, scenario("2024-01-01", "2024-01-03"), portfolio_value=1
        )


def test_unsorted_prices_rejected(prices: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        historical_replay(prices.iloc[::-1], {"A": 1.0}, scenario(), portfolio_value=1)


def test_prices_must_be_a_frame(prices: pd.DataFrame) -> None:
    with pytest.raises(TypeError):
        historical_replay(prices["A"], {"A": 1.0}, scenario(), portfolio_value=1)  # type: ignore[arg-type]
