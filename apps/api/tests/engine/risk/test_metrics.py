"""Tests for app.engine.risk.metrics with analytic or hand-checkable answers.

Every expected number is derived by hand in the comments (not taken from the
implementation). The whole module runs with warnings as errors, so a divide-by-zero or
invalid-value RuntimeWarning anywhere fails the test that triggers it.
"""

import math

import numpy as np
import pandas as pd
import pytest

from app.engine.risk.metrics import (
    annualised_return,
    annualised_volatility,
    arithmetic_annual_return,
    asset_metrics,
    benchmark_stats,
    correlation_matrix,
    downside_deviation,
    drawdown_episodes,
    drawdown_series,
    drawdown_stats,
    max_drawdown,
    performance_summary,
    sharpe_ratio,
    sortino_ratio,
)
from app.engine.risk.var import portfolio_returns

pytestmark = pytest.mark.filterwarnings("error")

SQRT_252 = math.sqrt(252)
DATES = pd.bdate_range("2024-01-01", periods=400)


def series(values: list[float], start: str = "2024-01-02") -> pd.Series:
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


def from_prices(prices: list[float]) -> pd.Series:
    """Simple returns of a price path; the first price (the origin) gets no return."""
    p = pd.Series(prices, index=pd.bdate_range("2024-01-01", periods=len(prices)), dtype=float)
    return (p / p.shift(1) - 1).iloc[1:]


def rf_with_daily(daily: float) -> float:
    """Annual rate whose geometric daily equivalent is exactly `daily`."""
    return (1 + daily) ** 252 - 1


# Fixed 5-day array used for hand-computed Sharpe / Sortino / downside deviation:
#   r = [0.01, -0.02, 0.03, 0.00, 0.02],  sum = 0.04,  mean = 0.008
#   deviations from mean: 0.002, -0.028, 0.022, -0.008, 0.012
#   squares (x1e-6):      4,     784,    484,   64,     144     -> sum 1480e-6
#   sample variance = 1480e-6 / 4 = 370e-6,  sd = sqrt(370e-6) = 0.0192353841...
SMALL = [0.01, -0.02, 0.03, 0.00, 0.02]
SMALL_MEAN = 0.008
SMALL_SD = math.sqrt(370e-6)


# --- volatility and return ---


def test_volatility_hand_checked() -> None:
    # +-1% alternating, n = 4: mean 0, sum of squares 4e-4, var = 4e-4 / 3
    r = series([0.01, -0.01, 0.01, -0.01])
    assert annualised_volatility(r, min_obs=2) == pytest.approx(
        math.sqrt(4e-4 / 3) * SQRT_252, rel=1e-12
    )
    assert annualised_volatility(series(SMALL), min_obs=2) == pytest.approx(
        SMALL_SD * SQRT_252, rel=1e-12
    )


def test_constant_returns_zero_vol_and_exact_cagr() -> None:
    r = series([0.001] * 300)
    assert annualised_volatility(r) == 0.0
    # (1.001^300)^(252/300) = 1.001^252
    assert annualised_return(r) == pytest.approx(1.001**252 - 1, rel=1e-12)
    assert arithmetic_annual_return(r) == pytest.approx(0.252, rel=1e-12)


def test_cagr_below_arithmetic_mean_volatility_drag() -> None:
    # +10% / -10% x 126: arithmetic mean exactly 0, growth 0.99^126 over exactly one year
    r = series([0.10, -0.10] * 126)
    assert arithmetic_annual_return(r) == pytest.approx(0.0, abs=1e-15)
    assert annualised_return(r) == pytest.approx(0.99**126 - 1, rel=1e-12)
    assert annualised_return(r) < 0


def test_cagr_two_years_hand_checked() -> None:
    # +21% in total over 504 days = exactly 2 years -> 10% a year
    r = series([0.0] * 503 + [0.21])
    assert annualised_return(r) == pytest.approx(0.10, rel=1e-12)


def test_total_loss_gives_minus_100_percent() -> None:
    r = series([0.01] * 260 + [-1.0] + [0.0] * 5)
    assert annualised_return(r) == -1.0
    assert max_drawdown(r) == 1.0


# --- Sharpe, Sortino, downside deviation ---


def test_sharpe_hand_checked() -> None:
    r = series(SMALL)
    # rf = 0: 0.008 / 0.019235 * sqrt(252) = 6.6022...
    assert sharpe_ratio(r, risk_free_annual=0.0, min_obs=2) == pytest.approx(
        SMALL_MEAN / SMALL_SD * SQRT_252, rel=1e-12
    )
    assert sharpe_ratio(r, risk_free_annual=0.0, min_obs=2) == pytest.approx(6.6022, abs=1e-4)
    # rf with daily equivalent 0.001: numerator 0.007, denominator unchanged
    assert sharpe_ratio(r, risk_free_annual=rf_with_daily(0.001), min_obs=2) == pytest.approx(
        0.007 / SMALL_SD * SQRT_252, rel=1e-10
    )


def test_downside_deviation_and_sortino_hand_checked() -> None:
    r = series(SMALL)
    # MAR = 0: shortfalls [0, -0.02, 0, 0, 0] -> mean of squares 4e-4 / 5 = 8e-5 (all n days)
    dd_daily = math.sqrt(8e-5)
    assert downside_deviation(r, mar_annual=0.0, min_obs=2) == pytest.approx(
        dd_daily * SQRT_252, rel=1e-12
    )
    # Sortino = 0.008 * 252 / (sqrt(8e-5) sqrt(252)) = 0.008 / sqrt(8e-5) * sqrt(252) = 14.199...
    assert sortino_ratio(r, mar_annual=0.0, min_obs=2) == pytest.approx(
        SMALL_MEAN / dd_daily * SQRT_252, rel=1e-12
    )
    # MAR daily 0.01: shortfalls [0, -0.03, 0, -0.01, 0] -> (9e-4 + 1e-4) / 5 = 2e-4
    mar = rf_with_daily(0.01)
    assert downside_deviation(r, mar_annual=mar, min_obs=2) == pytest.approx(
        math.sqrt(2e-4) * SQRT_252, rel=1e-10
    )
    # numerator: mean(r - 0.01) = -0.002
    assert sortino_ratio(r, mar_annual=mar, min_obs=2) == pytest.approx(
        -0.002 / math.sqrt(2e-4) * SQRT_252, rel=1e-9
    )


def test_ratios_undefined_for_zero_dispersion() -> None:
    r = series([0.001] * 300)
    assert sharpe_ratio(r, risk_free_annual=0.05) is None
    assert downside_deviation(r, mar_annual=0.0) == 0.0  # never below 0
    assert sortino_ratio(r, mar_annual=0.0) is None
    # Same constant series but MAR above it: every day is a shortfall of 0.001 - m
    m = 0.002
    assert downside_deviation(r, mar_annual=rf_with_daily(m)) == pytest.approx(
        0.001 * SQRT_252, rel=1e-9
    )


def test_all_zero_returns() -> None:
    r = series([0.0] * 300)
    assert annualised_volatility(r) == 0.0
    assert annualised_return(r) == 0.0
    assert max_drawdown(r) == 0.0
    assert sharpe_ratio(r, risk_free_annual=0.0) is None


# --- drawdowns ---


def test_drawdown_known_peak_trough_recovery() -> None:
    # prices 100 110 88 99 121 110 -> W = 1.1, 0.88, 0.99, 1.21, 1.10
    # D = 0, -0.2, -0.1, 0, -1/11
    r = from_prices([100, 110, 88, 99, 121, 110])
    d = r.index
    assert drawdown_series(r, min_obs=2).to_numpy() == pytest.approx(
        [0, -0.2, -0.1, 0, -1 / 11], abs=1e-12
    )
    eps = drawdown_episodes(r, min_obs=2)
    assert len(eps) == 2
    first, second = eps
    assert first.depth == pytest.approx(0.2, rel=1e-12)
    assert (first.peak_date, first.trough_date, first.recovery_date) == (d[0], d[1], d[3])
    assert (first.decline_days, first.recovery_days, first.duration_days) == (1, 2, 3)
    assert first.recovered
    # second episode: peak 121 (d[3]), trough 110 (d[4]), still underwater at the end
    assert second.depth == pytest.approx(1 / 11, rel=1e-12)
    assert (second.peak_date, second.trough_date, second.recovery_date) == (d[3], d[4], None)
    assert (second.decline_days, second.recovery_days, second.duration_days) == (1, None, 1)
    assert not second.recovered

    stats = drawdown_stats(r, min_obs=2)
    assert stats.max_drawdown == pytest.approx(0.2, rel=1e-12)
    assert stats.max_drawdown_episode == first
    assert stats.longest_drawdown == first
    assert stats.n_episodes == 2 and stats.n_obs == 5


def test_longest_underwater_differs_from_deepest_and_day_one_fall() -> None:
    # prices (origin 100) 70 100 101 95 96 97 98 99 100 102
    # Episode 1: falls on day 1 from the *starting value* (peak_date None) by 30%, and is back
    #            at exactly 100 on day 2 (70 -> 100 is 1e-16 short in floating point).
    # Episode 2: peak 101 (pos 2), trough 95 (pos 3), recovery 102 (pos 9): 7 days, 6/101 deep.
    r = from_prices([100, 70, 100, 101, 95, 96, 97, 98, 99, 100, 102])
    d = r.index
    stats = drawdown_stats(r, min_obs=2)
    deep, long_ = stats.max_drawdown_episode, stats.longest_drawdown
    assert stats.n_episodes == 2
    assert stats.max_drawdown == pytest.approx(0.3, rel=1e-12)
    assert deep.peak_date is None
    assert (deep.trough_date, deep.recovery_date) == (d[0], d[1])
    assert (deep.decline_days, deep.recovery_days, deep.duration_days) == (1, 1, 2)
    assert long_.depth == pytest.approx(6 / 101, rel=1e-12)
    assert (long_.peak_date, long_.trough_date, long_.recovery_date) == (d[2], d[3], d[9])
    assert (long_.decline_days, long_.recovery_days, long_.duration_days) == (1, 6, 7)


def test_drawdown_never_recovers() -> None:
    # 100 120 90 100: peak 120 (pos 0), trough 90 (pos 1), still 1/6 below at the end (pos 2)
    r = from_prices([100, 120, 90, 100])
    ep = drawdown_stats(r, min_obs=2).max_drawdown_episode
    assert ep.depth == pytest.approx(0.25, rel=1e-12)
    assert ep.recovery_date is None and ep.recovery_days is None
    assert ep.duration_days == 2  # peak -> last observation (lower bound)


def test_drawdown_never_falls() -> None:
    r = series([0.01, 0.0, 0.02, 0.005])
    stats = drawdown_stats(r, min_obs=2)
    assert stats.max_drawdown == 0.0
    assert stats.max_drawdown_episode is None and stats.longest_drawdown is None
    assert stats.n_episodes == 0
    assert (drawdown_series(r, min_obs=2) == 0).all()


def test_deepest_episode_tie_goes_to_earliest() -> None:
    # 100 -> 80 -> 100 -> 80 -> 100: two 20% episodes of 2 days each
    r = from_prices([100, 80, 100, 80, 100])
    stats = drawdown_stats(r, min_obs=2)
    assert stats.n_episodes == 2
    assert stats.max_drawdown_episode.trough_date == r.index[0]
    assert stats.longest_drawdown.trough_date == r.index[0]


def test_drawdown_requires_datetime_index() -> None:
    r = pd.Series([0.01, -0.02, 0.01])
    with pytest.raises(ValueError, match="DatetimeIndex"):
        drawdown_stats(r, min_obs=2)


def test_return_below_minus_one_rejected() -> None:
    with pytest.raises(ValueError, match="-100%"):
        max_drawdown(series([0.01, -1.5]), min_obs=2)


# --- benchmark-relative ---


@pytest.fixture(scope="module")
def bench() -> pd.Series:
    rng = np.random.default_rng(7)
    return pd.Series(rng.normal(0.0005, 0.01, 400), index=DATES, name="b")


def test_benchmark_times_two(bench) -> None:
    p = 2 * bench
    rf_d = 0.0002
    st = benchmark_stats(p, bench, risk_free_annual=rf_with_daily(rf_d))
    assert st.beta == pytest.approx(2.0, rel=1e-12)
    assert st.correlation == pytest.approx(1.0, rel=1e-12)
    assert st.r_squared == pytest.approx(1.0, rel=1e-12)
    assert st.residual_volatility == 0.0
    # alpha = mean(2b - rf) - 2 mean(b - rf) = rf_d per day: free leverage earns rf
    assert st.alpha == pytest.approx(rf_d * 252, rel=1e-8)
    # TE vs the benchmark itself is the benchmark's own volatility, not zero
    sd_b = float(np.std(bench.to_numpy(), ddof=1))
    assert st.tracking_error == pytest.approx(sd_b * SQRT_252, rel=1e-12)
    assert st.active_return == pytest.approx(bench.mean() * 252, rel=1e-12)
    # ... and zero against 2x benchmark exposure
    st2 = benchmark_stats(p, 2 * bench, risk_free_annual=0.0)
    assert st2.tracking_error == 0.0
    assert st2.information_ratio is None
    assert st2.beta == pytest.approx(1.0, rel=1e-12)


def test_benchmark_vs_itself(bench) -> None:
    st = benchmark_stats(bench, bench, risk_free_annual=0.06)
    assert st.beta == pytest.approx(1.0, rel=1e-12)
    assert st.alpha == pytest.approx(0.0, abs=1e-12)
    assert st.tracking_error == 0.0 and st.active_return == 0.0
    assert st.information_ratio is None


def test_negative_beta(bench) -> None:
    st = benchmark_stats(-0.5 * bench, bench, risk_free_annual=0.0)
    assert st.beta == pytest.approx(-0.5, rel=1e-12)
    assert st.correlation == pytest.approx(-1.0, rel=1e-12)


def test_information_ratio_hand_checked() -> None:
    # p - b = [0.002, 0.000, 0.002, 0.000]: mean 0.001, sample var = 4 * 1e-6 / 3
    b = series([0.01, -0.01, 0.02, 0.0])
    p = b + np.array([0.002, 0.0, 0.002, 0.0])
    st = benchmark_stats(p, b, risk_free_annual=0.0, min_obs=2)
    te = math.sqrt(4e-6 / 3) * SQRT_252
    assert st.tracking_error == pytest.approx(te, rel=1e-9)
    assert st.active_return == pytest.approx(0.252, rel=1e-9)
    assert st.information_ratio == pytest.approx(0.252 / te, rel=1e-9)


def test_constant_benchmark_beta_undefined() -> None:
    p = series([0.01, -0.01, 0.02, 0.0])
    b = series([0.001] * 4)
    st = benchmark_stats(p, b, risk_free_annual=0.0, min_obs=2)
    assert st.beta is None and st.alpha is None and st.correlation is None
    assert st.r_squared is None and st.residual_volatility is None
    assert st.tracking_error > 0


def test_single_index_model_recovered() -> None:
    # r_p = 0.0002 + 1.3 r_b + e,  r_b ~ N(0.0005, 0.01^2),  e ~ N(0, 0.008^2), n = 200k
    # sd_p = sqrt(1.69e-4 + 0.64e-4) = 0.015264; rho = 0.013 / 0.015264 = 0.85166
    # SEs: beta 0.008 / (0.01 sqrt(n)) = 0.0018; alpha 252 * 0.008 / sqrt(n) = 0.0045
    rng = np.random.default_rng(11)
    n = 200_000
    idx = pd.RangeIndex(n)
    b = pd.Series(rng.normal(0.0005, 0.01, n), index=idx)
    p = 0.0002 + 1.3 * b + rng.normal(0.0, 0.008, n)
    st = benchmark_stats(p, b, risk_free_annual=0.0)
    assert st.beta == pytest.approx(1.3, abs=0.01)
    assert st.correlation == pytest.approx(0.013 / math.sqrt(2.33e-4), abs=0.005)
    assert st.residual_volatility == pytest.approx(0.008 * SQRT_252, rel=0.01)
    assert st.alpha == pytest.approx(0.0002 * 252, abs=0.02)
    # identities that hold exactly in-sample (all ddof = 1)
    sd_p, sd_b = float(np.std(p, ddof=1)), float(np.std(b, ddof=1))
    assert st.beta == pytest.approx(st.correlation * sd_p / sd_b, rel=1e-10)
    assert st.residual_volatility == pytest.approx(
        sd_p * math.sqrt(1 - st.correlation**2) * SQRT_252, rel=1e-8
    )


def test_benchmark_index_mismatch_rejected(bench) -> None:
    shifted = pd.Series(bench.to_numpy(), index=DATES + pd.Timedelta(days=1))
    with pytest.raises(ValueError, match="identical indexes"):
        benchmark_stats(bench, shifted, risk_free_annual=0.0)
    with pytest.raises(ValueError, match="identical indexes"):
        benchmark_stats(bench, bench.iloc[::-1], risk_free_annual=0.0)


# --- correlation matrix ---


def test_correlation_matrix_known_values() -> None:
    rng = np.random.default_rng(3)
    a = rng.normal(0, 0.01, 300)
    df = pd.DataFrame({"A": a, "A2": 3 * a + 0.001, "negA": -a, "const": 0.001}, index=DATES[:300])
    c = correlation_matrix(df)
    assert list(c.index) == list(c.columns) == ["A", "A2", "negA", "const"]
    assert c.loc["A", "A2"] == pytest.approx(1.0, rel=1e-12)
    assert c.loc["A", "negA"] == pytest.approx(-1.0, rel=1e-12)
    assert c.loc["A", "A"] == 1.0
    assert c["const"].isna().all() and c.loc["const"].isna().all()


def test_correlation_matrix_matches_pandas() -> None:
    rng = np.random.default_rng(5)
    df = pd.DataFrame(rng.normal(0, 0.01, (300, 4)), columns=list("WXYZ"), index=DATES[:300])
    pd.testing.assert_frame_equal(correlation_matrix(df), df.corr(), atol=1e-12, rtol=0)


def test_correlation_matrix_one_asset() -> None:
    df = pd.DataFrame({"A": SMALL}, index=DATES[:5])
    c = correlation_matrix(df, min_obs=2)
    assert c.shape == (1, 1) and c.loc["A", "A"] == 1.0


# --- summaries ---


def test_performance_summary_constant_series() -> None:
    r = series([0.001] * 300)
    s = performance_summary(r, risk_free_annual=0.05, benchmark=r)
    assert s.n_obs == 300 and s.start == r.index[0] and s.end == r.index[-1]
    assert s.annualised_volatility == 0.0 and s.sharpe is None
    assert s.mar_annual == s.risk_free_annual == 0.05
    assert s.drawdown.max_drawdown == 0.0
    assert s.benchmark is not None and s.benchmark.beta is None
    # mean 0.001 daily is above the 5% rf (0.000194 daily), so nothing is below the MAR
    assert s.downside_deviation == 0.0 and s.sortino is None


def test_asset_metrics_one_asset_matches_single_functions(bench) -> None:
    df = bench.to_frame("equity:X")
    table = asset_metrics(df, risk_free_annual=0.06, benchmark=bench)
    assert list(table.index) == ["equity:X"]
    row = table.loc["equity:X"]
    assert row["volatility"] == pytest.approx(annualised_volatility(bench), rel=1e-12)
    assert row["sharpe"] == pytest.approx(sharpe_ratio(bench, risk_free_annual=0.06), rel=1e-12)
    assert row["beta"] == pytest.approx(1.0, rel=1e-12)
    assert math.isnan(row["information_ratio"])  # TE = 0 vs itself -> undefined, NaN


def test_asset_metrics_without_benchmark_has_no_relative_columns(bench) -> None:
    table = asset_metrics(bench.to_frame("b"), risk_free_annual=0.0)
    assert list(table.columns) == ["cagr", "volatility", "sharpe", "sortino", "max_drawdown"]


def test_portfolio_series_from_var_module(bench) -> None:
    # equal-weight portfolio of b and 3b is 2b: beta 2 against b
    df = pd.DataFrame({"x": bench, "y": 3 * bench})
    p = portfolio_returns(df, {"x": 0.5, "y": 0.5})
    assert benchmark_stats(p, bench, risk_free_annual=0.0).beta == pytest.approx(2.0, rel=1e-12)


# --- validation ---


ALL_SERIES_FUNCS = [
    lambda r, **k: annualised_volatility(r, **k),
    lambda r, **k: annualised_return(r, **k),
    lambda r, **k: arithmetic_annual_return(r, **k),
    lambda r, **k: downside_deviation(r, mar_annual=0.0, **k),
    lambda r, **k: sharpe_ratio(r, risk_free_annual=0.0, **k),
    lambda r, **k: sortino_ratio(r, mar_annual=0.0, **k),
    lambda r, **k: drawdown_series(r, **k),
    lambda r, **k: drawdown_stats(r, **k),
    lambda r, **k: max_drawdown(r, **k),
    lambda r, **k: benchmark_stats(r, r, risk_free_annual=0.0, **k),
    lambda r, **k: performance_summary(r, risk_free_annual=0.0, **k),
]


@pytest.mark.parametrize("func", ALL_SERIES_FUNCS)
def test_short_history_rejected(func) -> None:
    with pytest.raises(ValueError, match="at least 252"):
        func(series([0.01] * 251))


@pytest.mark.parametrize("func", ALL_SERIES_FUNCS)
def test_nan_and_inf_rejected(func) -> None:
    with pytest.raises(ValueError, match="NaN"):
        func(series([0.01, np.nan, 0.02]), min_obs=2)
    with pytest.raises(ValueError, match="inf"):
        func(series([0.01, np.inf, 0.02]), min_obs=2)


@pytest.mark.parametrize("func", ALL_SERIES_FUNCS)
def test_empty_and_dataframe_rejected(func) -> None:
    with pytest.raises(ValueError, match="empty"):
        func(pd.Series([], dtype=float, index=pd.DatetimeIndex([])), min_obs=2)
    with pytest.raises(TypeError, match="portfolio_returns"):
        func(pd.DataFrame({"a": SMALL}), min_obs=2)


def test_min_obs_floor() -> None:
    with pytest.raises(ValueError, match="min_obs"):
        annualised_volatility(series([0.01]), min_obs=1)


@pytest.mark.parametrize("rate", [math.nan, math.inf, -1.0, -2.0])
def test_bad_rates_rejected(rate) -> None:
    r = series(SMALL)
    with pytest.raises(ValueError, match="risk_free_annual"):
        sharpe_ratio(r, risk_free_annual=rate, min_obs=2)
    with pytest.raises(ValueError, match="mar_annual"):
        sortino_ratio(r, mar_annual=rate, min_obs=2)


def test_frame_functions_validate() -> None:
    with pytest.raises(TypeError):
        correlation_matrix(series(SMALL), min_obs=2)
    with pytest.raises(ValueError, match="NaN"):
        correlation_matrix(pd.DataFrame({"a": [0.01, np.nan, 0.0]}), min_obs=2)
    with pytest.raises(ValueError, match="no columns"):
        asset_metrics(pd.DataFrame(index=DATES[:5]), risk_free_annual=0.0, min_obs=2)
