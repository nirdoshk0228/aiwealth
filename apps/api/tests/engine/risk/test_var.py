"""Tests for app.engine.risk.var with analytic or hand-checkable answers.

Normal quantiles are hard-coded from tables (not taken from the implementation):
z_0.95 = 1.6448536269514722, z_0.99 = 2.3263478740408408; phi(z) = exp(-z^2/2) / sqrt(2 pi).
"""

import math
import warnings

import numpy as np
import pandas as pd
import pytest

from app.engine.risk.var import (
    MAX_HORIZON_DAYS,
    empirical_var_cvar,
    historical_var,
    monte_carlo_var,
    parametric_var,
    portfolio_returns,
    tail_count,
)

Z = {0.95: 1.6448536269514722, 0.99: 2.3263478740408408}
METHODS = [historical_var, parametric_var, monte_carlo_var]


def phi(z: float) -> float:
    return math.exp(-z * z / 2) / math.sqrt(2 * math.pi)


def frame(columns: dict[str, list[float]]) -> pd.DataFrame:
    n = len(next(iter(columns.values())))
    return pd.DataFrame(columns, index=pd.bdate_range("2024-01-01", periods=n), dtype=float)


# --- synthetic normal returns with known parameters ---

N_SYNTH = 200_000
MU = np.array([0.0005, 0.0003, 0.0008])  # daily means
VOL = np.array([0.010, 0.015, 0.020])  # daily volatilities
CORR = np.array([[1.0, 0.3, 0.5], [0.3, 1.0, 0.2], [0.5, 0.2, 1.0]])
SIGMA = CORR * np.outer(VOL, VOL)
W = {"A": 0.5, "B": 0.3, "C": 0.2}
W_ARR = np.array([0.5, 0.3, 0.2])


@pytest.fixture(scope="module")
def synthetic() -> pd.DataFrame:
    rng = np.random.default_rng(2024)
    draws = rng.multivariate_normal(MU, SIGMA, size=N_SYNTH)
    return pd.DataFrame(draws, columns=["A", "B", "C"])


def analytic(confidence: float) -> tuple[float, float]:
    """True VaR/CVaR of R_p ~ N(w'mu, w'Sigma w): VaR = z sigma - mu, CVaR = sigma phi(z)/a - mu."""
    mu_p, sigma_p = W_ARR @ MU, math.sqrt(W_ARR @ SIGMA @ W_ARR)
    z = Z[confidence]
    return z * sigma_p - mu_p, sigma_p * phi(z) / (1 - confidence) - mu_p


@pytest.mark.parametrize("confidence", [0.95, 0.99])
@pytest.mark.parametrize("method", METHODS)
def test_methods_match_analytic_normal_var(synthetic, method, confidence) -> None:
    # Tolerance 2% relative. Standard error of the empirical 99% quantile with n = 200k is
    # sqrt(0.01*0.99/n)/phi(2.326) = 0.0083 sigma_p, i.e. ~0.4% of VaR (~1.5% incl. the
    # 100k-draw MC noise and the estimation error of mu/Sigma), so 2% is > 4 standard errors.
    true_var, true_cvar = analytic(confidence)
    r = method(synthetic, W, confidence=confidence)
    assert r.var == pytest.approx(true_var, rel=0.02)
    assert r.cvar == pytest.approx(true_cvar, rel=0.02)
    assert r.confidence == confidence and r.horizon_days == 1 and r.n_obs == N_SYNTH


@pytest.mark.parametrize("confidence", [0.95, 0.99])
def test_three_methods_agree_with_each_other(synthetic, confidence) -> None:
    results = [m(synthetic, W, confidence=confidence) for m in METHODS]
    vars_ = [r.var for r in results]
    assert max(vars_) / min(vars_) < 1.03


@pytest.mark.parametrize("confidence", [0.90, 0.95, 0.99])
@pytest.mark.parametrize("method", METHODS)
def test_cvar_at_least_var(synthetic, method, confidence) -> None:
    r = method(synthetic, W, confidence=confidence)
    assert r.cvar >= r.var > 0


# --- hand-checkable historical VaR ---

# 20 returns; sorted losses worst-first: 0.05, 0.03, 0.02, 0.01, then gains.
HAND = [0.01, -0.03, 0.02, 0.005, -0.01, 0.0, 0.015, -0.05, 0.01, 0.02,
        0.003, -0.02, 0.01, 0.004, 0.006, 0.012, 0.0, 0.007, 0.002, 0.001]  # fmt: skip


def test_historical_var_hand_checked() -> None:
    returns = frame({"A": HAND})
    # 90%: n*alpha = 20*0.1 = 2 -> VaR = 2nd worst loss, CVaR = mean(0.05, 0.03).
    r90 = historical_var(returns, {"A": 1.0}, confidence=0.90, min_obs=20)
    assert r90.var == pytest.approx(0.03, abs=1e-15)
    assert r90.cvar == pytest.approx(0.04, abs=1e-15)
    # 95%: n*alpha = 1 -> the single worst day for both.
    # (20 * (1 - 0.95) is 1.0000000000000009 in floating point; a naive ceil would give k = 2.)
    r95 = historical_var(returns, {"A": 1.0}, confidence=0.95, min_obs=20)
    assert r95.var == pytest.approx(0.05, abs=1e-15)
    assert r95.cvar == pytest.approx(0.05, abs=1e-15)
    # 85%: n*alpha = 3 -> 3rd worst = 0.02; CVaR = (0.05 + 0.03 + 0.02) / 3.
    r85 = historical_var(returns, {"A": 1.0}, confidence=0.85, min_obs=20)
    assert r85.var == pytest.approx(0.02, abs=1e-15)
    assert r85.cvar == pytest.approx(0.1 / 3, abs=1e-15)


def test_tail_count_is_rounding_safe() -> None:
    assert tail_count(100, 0.95) == 5  # 100 * (1 - 0.95) = 5.000000000000004
    assert tail_count(20, 0.90) == 2
    assert tail_count(252, 0.99) == 3  # 2.52 -> 3 tail observations
    assert tail_count(756, 0.99) == 8


def test_portfolio_returns_hand_checked_and_label_aligned() -> None:
    returns = frame({"A": [0.01, -0.02, 0.03], "B": [0.02, 0.01, -0.04], "X": [9.0, 9.0, 9.0]})
    expected = [0.6 * 0.01 + 0.4 * 0.02, 0.6 * -0.02 + 0.4 * 0.01, 0.6 * 0.03 + 0.4 * -0.04]
    # weights given in reverse column order; unweighted column X is ignored
    rp = portfolio_returns(returns, {"B": 0.4, "A": 0.6})
    np.testing.assert_allclose(rp.to_numpy(), expected, atol=1e-15)
    assert rp.index.equals(returns.index)


def test_two_asset_historical_hand_checked() -> None:
    # Portfolio 0.5 A + 0.5 B equals HAND when B = 2 * HAND - A.
    a = [0.02 * ((i % 3) - 1) for i in range(20)]
    b = [2 * h - x for h, x in zip(HAND, a, strict=True)]
    r = historical_var(frame({"A": a, "B": b}), {"A": 0.5, "B": 0.5}, confidence=0.90, min_obs=20)
    assert r.var == pytest.approx(0.03, abs=1e-12)
    assert r.cvar == pytest.approx(0.04, abs=1e-12)


def test_negative_weights_allowed() -> None:
    # 1.5 A - 0.5 B with B = A gives exactly A.
    returns = frame({"A": HAND, "B": HAND})
    r = historical_var(returns, {"A": 1.5, "B": -0.5}, confidence=0.90, min_obs=20)
    assert r.var == pytest.approx(0.03, abs=1e-12)


# --- hand-checkable parametric VaR ---


def test_parametric_var_hand_checked() -> None:
    # +1%, -1% alternating x 50: mean exactly 0, sample std (ddof=1) = 0.01 * sqrt(100/99).
    returns = frame({"A": [0.01, -0.01] * 50})
    sigma = 0.01 * math.sqrt(100 / 99)
    r = parametric_var(returns, {"A": 1.0}, confidence=0.95, min_obs=100)
    assert r.var == pytest.approx(Z[0.95] * sigma, rel=1e-12)
    assert r.cvar == pytest.approx(sigma * phi(Z[0.95]) / 0.05, rel=1e-12)


def test_negative_var_reported_as_is() -> None:
    # Always-positive returns: the portfolio gains even at the 95% quantile.
    returns = frame({"A": [0.010, 0.012] * 150})
    hist = historical_var(returns, {"A": 1.0})
    assert hist.var == pytest.approx(-0.010)
    par = parametric_var(returns, {"A": 1.0})
    assert par.var < 0
    assert par.var == pytest.approx(Z[0.95] * returns["A"].std() - 0.011, rel=1e-12)


# --- edge cases ---


@pytest.mark.parametrize("method", METHODS)
def test_one_asset_portfolio_equals_asset(synthetic, method) -> None:
    single = method(synthetic[["A"]], {"A": 1.0})
    from_frame = method(synthetic, {"A": 1.0})  # other columns ignored
    assert single == from_frame


@pytest.mark.parametrize("method", [historical_var, parametric_var])
def test_duplicated_asset_singular_covariance(synthetic, method) -> None:
    dup = pd.DataFrame({"A": synthetic["A"], "A2": synthetic["A"]})
    split = method(dup, {"A": 0.5, "A2": 0.5})
    single = method(dup, {"A": 1.0})
    assert split.var == pytest.approx(single.var, rel=1e-12)
    assert split.cvar == pytest.approx(single.cvar, rel=1e-12)


def test_monte_carlo_handles_singular_covariance(synthetic) -> None:
    dup = pd.DataFrame({"A": synthetic["A"], "A2": synthetic["A"]})
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no "not positive-semidefinite" warning either
        mc = monte_carlo_var(dup, {"A": 0.5, "A2": 0.5})
    par = parametric_var(dup, {"A": 1.0})
    assert mc.var == pytest.approx(par.var, rel=0.02)


@pytest.mark.parametrize("method", METHODS)
def test_short_history_rejected(method) -> None:
    returns = frame({"A": [0.01, -0.01] * 50})  # 100 obs < default min_obs 252
    with pytest.raises(ValueError, match="at least 252"):
        method(returns, {"A": 1.0})


def test_historical_needs_a_full_tail_observation() -> None:
    returns = frame({"A": [0.01, -0.01] * 25})  # 50 obs: 99% tail holds 0.5 observations
    with pytest.raises(ValueError, match="need at least 100"):
        historical_var(returns, {"A": 1.0}, confidence=0.99, min_obs=50)
    # parametric does not need tail observations
    parametric_var(returns, {"A": 1.0}, confidence=0.99, min_obs=50)


def test_empirical_needs_tail_observation_directly() -> None:
    with pytest.raises(ValueError, match="no full observation"):
        empirical_var_cvar(np.zeros(19), 0.95)
    assert empirical_var_cvar(np.array([-0.05] + [0.0] * 19), 0.95) == (0.05, 0.05)


# --- input validation ---


@pytest.mark.parametrize(
    ("weights", "match"),
    [
        ({"A": 0.5, "B": 0.4}, "sum to 1"),
        ({"A": 0.5, "Z": 0.5}, "not in returns"),
        ({"A": float("nan"), "B": 1.0}, "finite"),
        ({}, "empty"),
    ],
)
def test_bad_weights(weights, match) -> None:
    returns = frame({"A": HAND, "B": HAND})
    with pytest.raises(ValueError, match=match):
        historical_var(returns, weights, min_obs=20, confidence=0.9)


def test_duplicate_weight_ids_rejected() -> None:
    returns = frame({"A": HAND, "B": HAND})
    w = pd.Series([0.5, 0.5], index=["A", "A"])
    with pytest.raises(ValueError, match="duplicate"):
        parametric_var(returns, w, min_obs=20)


def test_nan_and_inf_returns_rejected() -> None:
    bad = HAND.copy()
    bad[3] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        parametric_var(frame({"A": bad}), {"A": 1.0}, min_obs=20)
    bad[3] = float("inf")
    with pytest.raises(ValueError, match="inf"):
        parametric_var(frame({"A": bad}), {"A": 1.0}, min_obs=20)


def test_non_dataframe_rejected() -> None:
    with pytest.raises(TypeError):
        parametric_var(pd.Series(HAND), {"A": 1.0}, min_obs=20)  # type: ignore[arg-type]


@pytest.mark.parametrize("confidence", [0.5, 1.0, 0.0, 95])
def test_bad_confidence(confidence) -> None:
    with pytest.raises(ValueError, match="confidence"):
        historical_var(frame({"A": HAND}), {"A": 1.0}, confidence=confidence, min_obs=20)


@pytest.mark.parametrize("horizon", [0, MAX_HORIZON_DAYS + 1])
def test_bad_horizon(horizon) -> None:
    with pytest.raises(ValueError, match="horizon_days"):
        parametric_var(frame({"A": HAND}), {"A": 1.0}, horizon_days=horizon, min_obs=20)


def test_too_few_simulations() -> None:
    with pytest.raises(ValueError, match="n_sims"):
        monte_carlo_var(frame({"A": HAND}), {"A": 1.0}, n_sims=10, min_obs=20)


# --- horizon scaling and reproducibility ---


@pytest.mark.parametrize("method", METHODS)
def test_sqrt_time_scaling(synthetic, method) -> None:
    one = method(synthetic, W, confidence=0.99)
    four = method(synthetic, W, confidence=0.99, horizon_days=4)
    assert four.var == pytest.approx(2 * one.var, rel=1e-12)
    assert four.cvar == pytest.approx(2 * one.cvar, rel=1e-12)
    assert four.horizon_days == 4


def test_monte_carlo_seed_reproducible(synthetic) -> None:
    a = monte_carlo_var(synthetic, W, confidence=0.99, seed=7)
    b = monte_carlo_var(synthetic, W, confidence=0.99, seed=7)
    c = monte_carlo_var(synthetic, W, confidence=0.99, seed=8)
    assert a == b  # bit-identical
    assert (a.n_sims, a.seed) == (100_000, 7)
    assert c.var != a.var
    assert c.var == pytest.approx(a.var, rel=0.02)
