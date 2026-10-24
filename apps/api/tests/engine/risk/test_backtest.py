"""Tests for app.engine.risk.backtest with hand-computed or analytically known answers.

Reference values are computed by hand (arithmetic in the comments) or from closed forms,
never from the implementation. ln values: ln 0.99 = -0.01005034, ln 0.98 = -0.02020271,
ln 0.96 = -0.04082199, ln 0.01 = -4.60517019, ln 0.02 = -3.91202301, ln 0.04 = -3.21887582.
chi2(1) survival = erfc(sqrt(x/2)); chi2(2) survival = exp(-x/2).
"""

import math

import numpy as np
import pandas as pd
import pytest

from app.engine.risk.backtest import (
    BASEL_DAYS,
    basel_zone,
    christoffersen_test,
    kupiec_test,
    var_backtest,
)
from app.engine.risk.var import historical_var, parametric_var

METHODS = ["historical", "parametric", "monte_carlo"]


def frame(columns: dict[str, np.ndarray | list[float]]) -> pd.DataFrame:
    n = len(next(iter(columns.values())))
    return pd.DataFrame(columns, index=pd.bdate_range("2015-01-01", periods=n), dtype=float)


# --- Kupiec ---


def test_kupiec_hand_computed_250_days_5_breaches_at_99() -> None:
    # n = 250, x = 5, p = 0.01, pi_hat = 0.02
    # H0:  245 ln 0.99 + 5 ln 0.01 = -2.46233228 - 23.02585093 = -25.48818321
    # MLE: 245 ln 0.98 + 5 ln 0.02 = -4.94966329 - 19.56011503 = -24.50977832
    # LR = -2 (-25.48818321 + 24.50977832) = 1.95680979; p = erfc(sqrt(0.97840489)) = 0.16185
    k = kupiec_test(250, 5, 0.01)
    assert k.lr_uc == pytest.approx(1.95680979, abs=1e-7)
    assert k.p_value == pytest.approx(0.1618549, abs=1e-6)


def test_kupiec_hand_computed_rejection_10_breaches() -> None:
    # n = 250, x = 10, p = 0.01, pi_hat = 0.04
    # H0:  240 ln 0.99 + 10 ln 0.01 = -2.41208060 - 46.05170186 = -48.46378246
    # MLE: 240 ln 0.96 + 10 ln 0.04 = -9.79727868 - 32.18875825 = -41.98603693
    # LR = -2 (-48.46378246 + 41.98603693) = 12.95549106 -> p = 0.000319 (rejects at 1%)
    k = kupiec_test(250, 10, 0.01)
    assert k.lr_uc == pytest.approx(12.95549106, abs=1e-7)
    assert k.p_value == pytest.approx(3.1898e-4, rel=1e-3)


def test_kupiec_exact_rate_gives_zero_statistic() -> None:
    # n = 100, x = 5, p = 0.05: pi_hat = p, so both likelihoods are equal: LR = 0, p-value = 1
    k = kupiec_test(100, 5, 0.05)
    assert k.lr_uc == pytest.approx(0.0, abs=1e-12)
    assert k.p_value == pytest.approx(1.0)


def test_kupiec_zero_breaches_uses_0_log_0_convention() -> None:
    # x = 0: LR = -2 n ln(1 - p) = -2 * 250 * (-0.01005034) = 5.02516793; p = 0.02498
    k = kupiec_test(250, 0, 0.01)
    assert k.lr_uc == pytest.approx(-2 * 250 * math.log(0.99))
    assert k.lr_uc == pytest.approx(5.02516793, abs=1e-7)
    assert k.p_value == pytest.approx(0.0249815, abs=1e-6)


def test_kupiec_all_breaches_uses_0_log_0_convention() -> None:
    # x = n = 10, p = 0.05: LR = -2 n ln p = -20 * (-2.99573227) = 59.9146455
    k = kupiec_test(10, 10, 0.05)
    assert k.lr_uc == pytest.approx(59.9146455, abs=1e-6)
    assert k.p_value < 1e-13


@pytest.mark.parametrize(
    ("n", "x", "p"), [(0, 0, 0.01), (10, 11, 0.01), (10, -1, 0.01), (10, 1, 0.0), (10, 1, 1.0)]
)
def test_kupiec_rejects_invalid_input(n: int, x: int, p: float) -> None:
    with pytest.raises(ValueError):
        kupiec_test(n, x, p)


# --- Christoffersen ---

# 20 days, 4 breaches in both sequences, p = 0.2 so x / n = p and LR_uc = 0 (LR_cc = LR_ind).
CLUSTERED = [0] * 8 + [1, 1, 1, 1] + [0] * 8
SPREAD = [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]
# Both: 19 transitions, pi = (n01 + n11) / 19 = 4/19,
#   ln L0 = 15 ln(15/19) + 4 ln(4/19) = -3.54582 - 6.23259 = -9.77841


def test_christoffersen_clustered_hand_computed() -> None:
    # Pairs: 7 (0,0) in the leading zeros + 7 in the trailing zeros = n00 = 14,
    # one 0->1 (n01 = 1), three 1->1 (n11 = 3), one 1->0 (n10 = 1); total 19.
    # pi01 = 1/15, pi11 = 3/4
    # ln L1 = 14 ln(14/15) + ln(1/15) + ln(1/4) + 3 ln(3/4)
    #       = -0.96592 - 2.70805 - 1.38629 - 0.86305 = -5.92329
    # LR_ind = -2 (-9.77841 + 5.92329) = 7.71024 -> p = 0.00549
    c = christoffersen_test([bool(b) for b in CLUSTERED], 0.2)
    assert (c.n00, c.n01, c.n10, c.n11) == (14, 1, 1, 3)
    assert c.pi01 == pytest.approx(1 / 15)
    assert c.pi11 == pytest.approx(3 / 4)
    assert c.lr_ind == pytest.approx(7.710238, abs=1e-5)
    assert c.p_ind == pytest.approx(0.0054909, abs=1e-6)
    assert c.lr_cc == pytest.approx(c.lr_ind, abs=1e-12)  # LR_uc = 0 here
    assert c.p_cc == pytest.approx(math.exp(-7.710238 / 2), abs=1e-6)


def test_christoffersen_spread_hand_computed() -> None:
    # Breaches at 2, 7, 12, 17: four 0->1 and four 1->0, no 1->1; n00 = 19 - 8 = 11.
    # pi01 = 4/15, pi11 = 0/4 = 0
    # ln L1 = 11 ln(11/15) + 4 ln(4/15) + 4 ln(1) + 0 ln(0) = -3.41170 - 5.28703 = -8.69873
    # LR_ind = -2 (-9.77841 + 8.69873) = 2.15937 -> p = 0.1417 (no rejection)
    c = christoffersen_test([bool(b) for b in SPREAD], 0.2)
    assert (c.n00, c.n01, c.n10, c.n11) == (11, 4, 4, 0)
    assert c.pi11 == 0.0
    assert c.lr_ind == pytest.approx(2.159365, abs=1e-5)
    assert c.p_ind == pytest.approx(0.141703, abs=1e-5)


def test_christoffersen_clustered_more_significant_than_spread() -> None:
    clustered = christoffersen_test([bool(b) for b in CLUSTERED], 0.2)
    spread = christoffersen_test([bool(b) for b in SPREAD], 0.2)
    assert clustered.lr_ind > spread.lr_ind
    assert clustered.p_ind < 0.01 < spread.p_ind


def test_christoffersen_zero_breaches_is_undefined() -> None:
    c = christoffersen_test([False] * 50, 0.01)
    assert (c.n00, c.n01, c.n10, c.n11) == (49, 0, 0, 0)
    assert c.pi01 == 0.0 and c.pi11 is None
    assert c.lr_ind is None and c.p_ind is None and c.lr_cc is None and c.p_cc is None


def test_christoffersen_single_breach_on_last_day_is_undefined() -> None:
    # The breach is never followed by another day, so n10 + n11 = 0: pi11 undefined.
    c = christoffersen_test([False] * 9 + [True], 0.05)
    assert (c.n00, c.n01, c.n10, c.n11) == (8, 1, 0, 0)
    assert c.pi11 is None and c.lr_ind is None


def test_christoffersen_all_breaches_is_undefined() -> None:
    c = christoffersen_test([True] * 10, 0.05)
    assert c.pi01 is None and c.lr_ind is None and c.p_cc is None


def test_christoffersen_rejects_short_sequence() -> None:
    with pytest.raises(ValueError):
        christoffersen_test([True], 0.05)


# --- Basel ---


@pytest.mark.parametrize(
    ("x", "zone"),
    [(0, "green"), (4, "green"), (5, "yellow"), (9, "yellow"), (10, "red"), (30, "red")],
)
def test_basel_zone_boundaries(x: int, zone: str) -> None:
    assert basel_zone(x) == zone


def test_basel_zone_rejects_negative() -> None:
    with pytest.raises(ValueError):
        basel_zone(-1)


# --- the backtest on hand-built data ---


def test_breach_is_strictly_greater_and_var_uses_prior_window() -> None:
    # window 20 at 95%: k = ceil(20 * 0.05) = 1, so VaR_t = the largest loss of the 20 prior
    # days. The first 20 days have a worst loss of exactly 0.02.
    base = [0.01, -0.01] * 9 + [-0.02, 0.005]
    r = base + [-0.02, -0.03, 0.01]
    res = var_backtest(frame({"A": r}), {"A": 1.0}, confidence=0.95, window=20)
    assert res.n_test_days == 3
    # day 20: loss 0.02 == VaR 0.02 -> not a breach (strict >)
    # day 21: window = rows 1..20, worst loss 0.02 -> loss 0.03 > 0.02 -> breach
    # day 22: window = rows 2..21, which now include the 0.03 loss -> VaR 0.03; gain, no breach
    assert res.var == pytest.approx((0.02, 0.02, 0.03))
    assert res.n_breaches == 1
    assert res.breach_dates == (frame({"A": r}).index[21].date(),)
    assert res.to_frame()["breach"].tolist() == [False, True, False]


def test_window_and_index_handling() -> None:
    rng = np.random.default_rng(1)
    df = frame({"A": rng.normal(0, 0.01, 300), "B": rng.normal(0, 0.02, 300)})
    w = {"A": 0.6, "B": 0.4}
    res = var_backtest(df, w, method="historical", confidence=0.99, window=200)
    assert res.n_test_days == 100 == len(res.var) == len(res.dates) == len(res.realised)
    assert res.dates == tuple(d.date() for d in df.index[200:])
    # first forecast uses exactly rows 0..199, last uses rows 99..298
    assert res.var[0] == historical_var(df.iloc[:200], w, confidence=0.99, min_obs=200).var
    assert res.var[-1] == historical_var(df.iloc[99:299], w, confidence=0.99, min_obs=200).var
    assert res.realised[0] == pytest.approx(0.6 * df["A"].iloc[200] + 0.4 * df["B"].iloc[200])
    assert res.horizon_days == 1 and res.window == 200
    assert res.expected_breaches == pytest.approx(1.0)
    assert res.basel_zone_last_250 is None  # fewer than 250 test days


def test_parametric_forecast_matches_var_module() -> None:
    rng = np.random.default_rng(2)
    df = frame({"A": rng.normal(0, 0.01, 120)})
    res = var_backtest(df, {"A": 1.0}, method="parametric", window=100)
    assert res.var[7] == parametric_var(df.iloc[7:107], {"A": 1.0}, min_obs=100).var


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window": 1},
        {"window": 300},  # == len(returns): no test day
        {"window": 400},
        {"method": "garch"},
        {"confidence": 1.0},
        {"method": "monte_carlo", "n_sims": 10},
        {"window": 50, "confidence": 0.99},  # 50 * 1% < 1 observation in the tail
    ],
)
def test_backtest_rejects_invalid_input(kwargs: dict) -> None:
    df = frame({"A": np.random.default_rng(3).normal(0, 0.01, 300)})
    with pytest.raises(ValueError):
        var_backtest(df, {"A": 1.0}, **kwargs)


def test_backtest_rejects_bad_weights() -> None:
    df = frame({"A": np.random.default_rng(3).normal(0, 0.01, 300)})
    with pytest.raises(ValueError):
        var_backtest(df, {"A": 0.5}, window=100)
    with pytest.raises(ValueError):
        var_backtest(df, {"Z": 1.0}, window=100)


# --- no look-ahead ---


@pytest.mark.parametrize("method", METHODS)
def test_no_look_ahead(method: str) -> None:
    rng = np.random.default_rng(4)
    df = frame({"A": rng.normal(0, 0.01, 400), "B": rng.normal(0, 0.015, 400)})
    w = {"A": 0.5, "B": 0.5}
    window, t = 252, 320  # test-day position of row t is t - window
    base = var_backtest(df, w, method=method, confidence=0.99, window=window, n_sims=2_000)

    shocked = df.copy()
    shocked.iloc[t] = -0.30  # a -30% day on row t
    alt = var_backtest(shocked, w, method=method, confidence=0.99, window=window, n_sims=2_000)

    i = t - window
    # VaR for day t and every earlier day is bit-identical: row t is not in their windows
    assert alt.var[: i + 1] == base.var[: i + 1]
    # ... but row t is realised on day t (a breach) and enters the forecast for day t + 1
    assert alt.realised[i] == pytest.approx(-0.30)
    assert alt.dates[i] in alt.breach_dates
    assert alt.var[i + 1] > base.var[i + 1]

    # Truncating the data after row t gives the same forecasts up to day t
    trunc = var_backtest(
        df.iloc[: t + 1], w, method=method, confidence=0.99, window=window, n_sims=2_000
    )
    assert trunc.var == base.var[: i + 1]


# --- statistical behaviour on synthetic data ---

N_IID, WINDOW = 4504, 504


@pytest.fixture(scope="module")
def iid_normal() -> pd.DataFrame:
    rng = np.random.default_rng(2024)
    cov = np.array([[1.0, 0.4], [0.4, 1.0]]) * 0.01**2
    return frame(dict(zip(["A", "B"], rng.multivariate_normal([0, 0], cov, N_IID).T, strict=True)))


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("confidence", [0.95, 0.99])
def test_iid_normal_breach_rate_close_to_alpha(
    iid_normal: pd.DataFrame, method: str, confidence: float
) -> None:
    res = var_backtest(
        iid_normal, {"A": 0.5, "B": 0.5}, method=method, confidence=confidence, window=WINDOW
    )
    alpha = 1 - confidence
    assert res.n_test_days == 4000
    if method == "historical":
        # P(next loss > k-th worst of N) = k / (N + 1) for i.i.d. continuous data:
        # 95%: k = ceil(25.2) = 26 -> 26/505 = 5.149%; 99%: k = ceil(5.04) = 6 -> 6/505 = 1.188%
        expected = math.ceil(round(WINDOW * alpha, 9)) / (WINDOW + 1)
    else:
        expected = alpha
    # 4,000 days: binomial sd of the rate is sqrt(a(1-a)/4000) = 0.34% (95%) / 0.16% (99%)
    assert abs(res.breach_rate - expected) < 4 * math.sqrt(expected * (1 - expected) / 4000)
    assert res.kupiec.p_value > 0.01
    assert res.christoffersen.p_ind is not None and res.christoffersen.p_ind > 0.01


def test_regime_shift_breaches_cluster() -> None:
    # 1,000 calm days (sigma 1%) then 500 turbulent days (sigma 3%). Right after the shift
    # the window still holds calm data, so the 99% VaR is still ~2.326 * 1%, and a 3%-sigma
    # day exceeds it with probability 1 - Phi(2.326 / 3) = 1 - Phi(0.775) = 21.9%, versus 1%
    # in the calm period. Historical VaR (6th-worst of 504) adapts once ~6 big losses are in
    # the window; the sample sigma behind parametric VaR moves more slowly.
    rng = np.random.default_rng(7)
    r = np.concatenate([rng.normal(0, 0.01, 1000), rng.normal(0, 0.03, 500)])
    df = frame({"A": r})
    first_after = 1000 - WINDOW  # test-day position of the first turbulent day
    for method in ("historical", "parametric"):
        res = var_backtest(df, {"A": 1.0}, method=method, confidence=0.99, window=WINDOW)
        c = res.christoffersen
        breach = res.to_frame()["breach"].to_numpy()
        # 50 days at a true 1% rate: P(>= 5 breaches) = 0.0016; theory here says ~11 of 50
        assert breach[first_after : first_after + 50].mean() >= 0.10
        assert breach[:first_after].mean() < 0.03  # calm period roughly on target
        assert res.breach_rate > 0.01 and res.kupiec.p_value < 0.01
        assert c.pi11 is not None and c.pi01 is not None and c.pi11 > 2 * c.pi01
        assert c.p_ind is not None and c.p_ind < 0.05
        assert c.p_cc is not None and c.p_cc < 0.01


def test_basel_zone_on_last_250_days() -> None:
    rng = np.random.default_rng(5)
    df = frame({"A": rng.normal(0, 0.01, 1000)})
    res99 = var_backtest(df, {"A": 1.0}, method="parametric", confidence=0.99, window=WINDOW)
    expected_count = int(res99.to_frame()["breach"].iloc[-BASEL_DAYS:].sum())
    assert res99.basel_breaches_last_250 == expected_count
    assert res99.basel_zone_last_250 == basel_zone(expected_count)
    res95 = var_backtest(df, {"A": 1.0}, method="parametric", confidence=0.95, window=WINDOW)
    assert res95.basel_zone_last_250 is None and res95.basel_breaches_last_250 is None
