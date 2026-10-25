"""Mean-variance: estimates, frontier shape, closed-form tangency, max-return greedy."""

import math

import numpy as np
import pandas as pd
import pytest

from app.engine.optimize.mean_variance import (
    Estimates,
    efficient_frontier,
    estimate,
    evaluate,
    max_return_weights,
    mean_variance,
    min_variance,
)


def est_from(mu: list[float], cov: np.ndarray) -> Estimates:
    ids = tuple(f"a{i}" for i in range(len(mu)))
    return Estimates(asset_ids=ids, mu=np.array(mu), cov=np.asarray(cov), n_obs=1000)


def test_estimate_annualises_mean_and_covariance() -> None:
    idx = pd.bdate_range("2020-01-01", periods=4)
    r = pd.DataFrame({"x": [0.01, -0.01, 0.02, 0.0], "y": [0.0, 0.01, 0.01, 0.02]}, index=idx)
    e = estimate(r, ["y", "x"], min_obs=4)
    assert e.asset_ids == ("y", "x")
    np.testing.assert_allclose(e.mu, [0.01 * 252, 0.005 * 252])
    np.testing.assert_allclose(e.cov, np.cov(r[["y", "x"]].to_numpy().T, ddof=1) * 252)


def test_estimate_validation() -> None:
    r = pd.DataFrame({"x": [0.0, 0.01]}, index=pd.bdate_range("2020-01-01", periods=2))
    with pytest.raises(ValueError, match="not in returns"):
        estimate(r, ["z"], min_obs=2)
    with pytest.raises(ValueError, match="at least 252"):
        estimate(r, ["x"])
    with pytest.raises(ValueError, match="unique"):
        estimate(r, ["x", "x"], min_obs=2)


def test_max_sharpe_equals_closed_form_tangency_when_interior() -> None:
    vols = np.array([0.12, 0.18, 0.25])
    corr = np.array([[1.0, 0.3, 0.2], [0.3, 1.0, 0.4], [0.2, 0.4, 1.0]])
    cov = np.outer(vols, vols) * corr
    mu = [0.08, 0.11, 0.14]
    rf = 0.05
    x = np.linalg.solve(cov, np.array(mu) - rf)
    tangency = x / x.sum()
    assert (tangency > 0).all()  # interior, so long-only does not bind
    res = efficient_frontier(est_from(mu, cov), risk_free_annual=rf)
    w = np.array(list(res.max_sharpe.weights.values()))
    np.testing.assert_allclose(w, tangency, atol=1e-6)
    t_sharpe = (tangency @ mu - rf) / math.sqrt(tangency @ cov @ tangency)
    assert res.max_sharpe.sharpe == pytest.approx(t_sharpe, rel=1e-10)


def test_frontier_is_efficient_and_runs_from_min_variance_to_max_return() -> None:
    rng = np.random.default_rng(7)
    a = rng.normal(size=(60, 6))
    cov = a.T @ a / 60 * 0.04 + 0.002 * np.eye(6)
    mu = [0.04, 0.07, 0.09, 0.12, 0.15, 0.06]
    res = efficient_frontier(est_from(mu, cov), risk_free_annual=0.05, max_weight=0.5)
    f = res.frontier
    assert f[0] == res.min_variance and len(f) > 10
    vols = [p.volatility for p in f]
    rets = [p.expected_return for p in f]
    assert all(np.diff(vols) >= -1e-12)
    assert all(np.diff(rets) >= -1e-10)  # upper (efficient) branch only
    assert rets[-1] == pytest.approx(res.max_return.expected_return)
    # Max return with a 50% cap: the two highest-mu assets at 50% each.
    assert res.max_return.weights == {"a0": 0, "a1": 0, "a2": 0, "a3": 0.5, "a4": 0.5, "a5": 0}
    assert all(max(p.weights.values()) <= 0.5 + 1e-12 for p in f)
    assert all(p.kkt_residual < 1e-9 for p in f)
    # Max Sharpe is at least as good as every frontier point.
    assert res.max_sharpe.sharpe >= max(p.sharpe for p in f) - 1e-9


def test_mean_variance_moves_from_min_variance_towards_max_return() -> None:
    cov = np.diag([0.01, 0.04, 0.09])
    e = est_from([0.05, 0.08, 0.12], cov)
    mv = min_variance(e)
    # Interior solution in closed form: w = w_mv + (1/gamma) Sigma^-1 (mu - mubar 1),
    # mubar = 1'Sigma^-1 mu / 1'Sigma^-1 1 (Lagrange condition gamma Sigma w = mu - nu 1).
    gamma = 50.0
    inv_cov = np.linalg.inv(cov)
    mu = e.mu
    mubar = inv_cov.sum(axis=0) @ mu / inv_cov.sum()
    w_mv = inv_cov.sum(axis=1) / inv_cov.sum()
    exact = w_mv + inv_cov @ (mu - mubar) / gamma
    assert (exact > 0).all()
    got = mean_variance(e, gamma)
    np.testing.assert_allclose(list(got.weights.values()), exact, atol=1e-14)
    np.testing.assert_allclose(list(mv.weights.values()), w_mv, atol=1e-14)
    assert mean_variance(e, 1e-3).weights["a2"] == 1.0
    # Diagonal min variance: w_i ~ 1/var_i.
    inv = 1 / np.diag(cov)
    np.testing.assert_allclose(list(mv.weights.values()), inv / inv.sum(), atol=1e-14)


def test_max_return_greedy() -> None:
    np.testing.assert_allclose(max_return_weights(np.array([0.1, 0.3, 0.2]), 0.4), [0.2, 0.4, 0.4])
    np.testing.assert_allclose(max_return_weights(np.array([0.1, 0.3, 0.2]), 1.0), [0, 1, 0])


def test_warning_when_nothing_beats_cash() -> None:
    res = efficient_frontier(est_from([0.01, 0.02], np.diag([0.01, 0.02])), risk_free_annual=0.06)
    assert res.max_sharpe.sharpe < 0
    assert any("beat the risk-free rate" in w for w in res.warnings)


def test_evaluate_matches_definitions() -> None:
    e = est_from([0.1, 0.2], np.array([[0.04, 0.01], [0.01, 0.09]]))
    p = evaluate(e, {"a0": 0.25, "a1": 0.75}, 0.05)
    assert p.expected_return == pytest.approx(0.175)
    var = 0.25**2 * 0.04 + 0.75**2 * 0.09 + 2 * 0.25 * 0.75 * 0.01
    assert p.volatility == pytest.approx(math.sqrt(var))
    assert p.sharpe == pytest.approx((0.175 - 0.05) / math.sqrt(var))
    with pytest.raises(ValueError, match="outside"):
        evaluate(e, {"zz": 1.0}, 0.0)
