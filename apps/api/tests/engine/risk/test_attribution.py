"""Tests for app.engine.risk.attribution with analytic or hand-checkable answers.

Exactly uncorrelated data: columns of a Hadamard matrix are +-1 sequences that are zero-mean
and mutually orthogonal. Scaling column j by s_j and tiling it to n rows gives a sample
covariance of exactly diag(s_j^2 * n/(n-1)) (ddof = 1), so every expected value below is
computed by hand from s_j and n, never from the implementation.
"""

import math

import numpy as np
import pandas as pd
import pytest

from app.engine.risk.attribution import (
    diversification_ratio,
    euler_contributions,
    parametric_var_attribution,
    risk_budget_report,
    volatility_attribution,
)
from app.engine.risk.metrics import annualised_volatility
from app.engine.risk.var import parametric_var, portfolio_returns

N = 256  # multiple of 8, >= the default min_obs of 252
F = N / (N - 1)  # sample-variance factor for a zero-mean +-1 sequence of length N
ROOT_T = math.sqrt(252)
Z95 = 1.6448536269514722


def hadamard_frame(vols: dict[str, float], n: int = N) -> pd.DataFrame:
    """Columns = s_j * (Hadamard column j+1) tiled to n rows: zero mean, exactly orthogonal."""
    h2 = np.array([[1.0, 1.0], [1.0, -1.0]])
    h8 = np.kron(h2, np.kron(h2, h2))
    assert len(vols) <= 7 and n % 8 == 0
    cols = {a: s * np.tile(h8[:, j + 1], n // 8) for j, (a, s) in enumerate(vols.items())}
    return pd.DataFrame(cols, index=pd.bdate_range("2024-01-01", periods=n))


def by_id(result) -> dict:
    return {c.asset_id: c for c in result.contributions}


@pytest.fixture(scope="module")
def correlated() -> pd.DataFrame:
    """Five correlated assets with non-zero means, for exact-sum and order checks."""
    rng = np.random.default_rng(7)
    vol = np.array([0.010, 0.015, 0.020, 0.008, 0.025])
    corr = np.full((5, 5), 0.4) + 0.6 * np.eye(5)
    corr[0, 3] = corr[3, 0] = -0.3
    draws = rng.multivariate_normal(np.full(5, 4e-4), corr * np.outer(vol, vol), size=1000)
    return pd.DataFrame(draws, columns=list("ABCDE"))


W5 = {"A": 0.3, "B": 0.25, "C": 0.2, "D": 0.15, "E": 0.1}


# --- analytic cases ---


def test_two_uncorrelated_assets_by_hand() -> None:
    # s = (1%, 2%) daily, w = (0.5, 0.5): var_p = F (0.25*1e-4 + 0.25*4e-4) = F * 1.25e-4
    # component_i = w_i^2 s_i^2 F / sigma_p -> shares 0.25e-4 : 1e-4 = 20% : 80%
    r = volatility_attribution(hadamard_frame({"A": 0.01, "B": 0.02}), {"A": 0.5, "B": 0.5})
    sigma_d = math.sqrt(F * 1.25e-4)
    c = by_id(r)
    assert r.portfolio_volatility == pytest.approx(sigma_d * ROOT_T, rel=1e-12)
    assert c["A"].component == pytest.approx(0.25 * 1e-4 * F / sigma_d * ROOT_T, rel=1e-12)
    assert c["B"].component == pytest.approx(0.25 * 4e-4 * F / sigma_d * ROOT_T, rel=1e-12)
    assert c["A"].marginal == pytest.approx(0.5 * 1e-4 * F / sigma_d * ROOT_T, rel=1e-12)
    assert c["B"].marginal == pytest.approx(0.5 * 4e-4 * F / sigma_d * ROOT_T, rel=1e-12)
    assert c["A"].percent == pytest.approx(0.2, rel=1e-12)
    assert c["B"].percent == pytest.approx(0.8, rel=1e-12)
    assert r.n_obs == N


def test_identical_assets_risk_share_equals_weight() -> None:
    # Same series twice: Sigma w = sigma^2 (w_A + w_B) for both, so share_i = w_i, marginal = sigma
    base = hadamard_frame({"A": 0.012})
    df = base.assign(B=base["A"])
    r = volatility_attribution(df, {"A": 0.3, "B": 0.7})
    sigma_ann = 0.012 * math.sqrt(F) * ROOT_T
    assert r.portfolio_volatility == pytest.approx(sigma_ann, rel=1e-12)
    for a, w in (("A", 0.3), ("B", 0.7)):
        assert by_id(r)[a].percent == pytest.approx(w, rel=1e-12)
        assert by_id(r)[a].marginal == pytest.approx(sigma_ann, rel=1e-12)
    assert diversification_ratio(df, {"A": 0.3, "B": 0.7}) == pytest.approx(1.0, rel=1e-12)


def test_perfect_hedge_has_negative_contribution() -> None:
    # B = -A (singular covariance), w = (0.75, 0.25): R_p = 0.5 A, sigma_p = 0.5 sigma.
    # Sigma w = sigma^2 (0.5, -0.5) -> components (0.75 sigma, -0.25 sigma), shares 150% / -50%.
    base = hadamard_frame({"A": 0.01})
    df = base.assign(B=-base["A"])
    w = {"A": 0.75, "B": 0.25}
    sigma_ann = 0.01 * math.sqrt(F) * ROOT_T
    r = volatility_attribution(df, w)
    c = by_id(r)
    assert r.portfolio_volatility == pytest.approx(0.5 * sigma_ann, rel=1e-12)
    assert c["A"].component == pytest.approx(0.75 * sigma_ann, rel=1e-12)
    assert c["B"].component == pytest.approx(-0.25 * sigma_ann, rel=1e-12)
    assert c["A"].percent == pytest.approx(1.5, rel=1e-12)
    assert c["B"].percent == pytest.approx(-0.5, rel=1e-12)
    # gross average vol / portfolio vol = (0.75 + 0.25) sigma / (0.5 sigma)
    assert diversification_ratio(df, w) == pytest.approx(2.0, rel=1e-12)
    rb = {row.asset_id: row for row in risk_budget_report(df, w).rows}
    assert rb["B"].ratio == pytest.approx(-2.0, rel=1e-12)  # a hedge: beta to portfolio < 0


def test_fully_hedged_portfolio_is_undefined_not_zero() -> None:
    base = hadamard_frame({"A": 0.01})
    df = base.assign(B=-base["A"])
    with pytest.raises(ValueError, match="volatility is zero"):
        volatility_attribution(df, {"A": 0.5, "B": 0.5})
    with pytest.raises(ValueError, match="volatility is zero"):
        parametric_var_attribution(df, {"A": 0.5, "B": 0.5})
    assert diversification_ratio(df, {"A": 0.5, "B": 0.5}) is None


def test_equal_risk_contribution_two_uncorrelated_assets() -> None:
    # Uncorrelated: component_i proportional to w_i^2 s_i^2, equal when w_i proportional to 1/s_i.
    # s = (1%, 2%) -> w = (2/3, 1/3): shares 50/50, ratios 0.5/(2/3) = 0.75 and 0.5/(1/3) = 1.5.
    df = hadamard_frame({"A": 0.01, "B": 0.02})
    rep = risk_budget_report(df, {"A": 2 / 3, "B": 1 / 3})
    rows = {r.asset_id: r for r in rep.rows}
    assert rows["A"].risk_share == pytest.approx(0.5, rel=1e-12)
    assert rows["B"].risk_share == pytest.approx(0.5, rel=1e-12)
    assert rows["A"].ratio == pytest.approx(0.75, rel=1e-12)
    assert rows["B"].ratio == pytest.approx(1.5, rel=1e-12)
    assert rep.max_abs_deviation == pytest.approx(1 / 6, rel=1e-12)


def test_equal_weights_equal_vol_uncorrelated_is_risk_parity() -> None:
    df = hadamard_frame({"A": 0.01, "B": 0.01, "C": 0.01, "D": 0.01})
    rep = risk_budget_report(df, dict.fromkeys("ABCD", 0.25))
    for row in rep.rows:
        assert row.risk_share == pytest.approx(0.25, rel=1e-12)
        assert row.ratio == pytest.approx(1.0, rel=1e-12)
    assert rep.max_abs_deviation == pytest.approx(0.0, abs=1e-12)


def test_diversification_ratio_sqrt_n() -> None:
    # n uncorrelated equal-vol assets, equal weights: DR = sigma / (sigma / sqrt(n)) = sqrt(n)
    for n_assets in (1, 4, 7):
        vols = {f"X{j}": 0.015 for j in range(n_assets)}
        w = dict.fromkeys(vols, 1 / n_assets)
        dr = diversification_ratio(hadamard_frame(vols), w)
        assert dr == pytest.approx(math.sqrt(n_assets), rel=1e-12)


def test_component_var_uncorrelated_zero_mean_by_hand() -> None:
    # Zero sample mean -> cVaR_i = z * w_i (Sigma w)_i / sigma_p (daily), shares as for vol.
    df = hadamard_frame({"A": 0.01, "B": 0.02})
    r = parametric_var_attribution(df, {"A": 0.5, "B": 0.5}, confidence=0.95)
    sigma_d = math.sqrt(F * 1.25e-4)
    c = {x.asset_id: x for x in r.contributions}
    assert c["A"].component_var == pytest.approx(Z95 * 0.25e-4 * F / sigma_d, rel=1e-9)
    assert c["B"].component_var == pytest.approx(Z95 * 1e-4 * F / sigma_d, rel=1e-9)
    assert c["A"].percent_var == pytest.approx(0.2, rel=1e-9)
    assert r.portfolio_var == pytest.approx(Z95 * sigma_d, rel=1e-9)
    phi = math.exp(-Z95 * Z95 / 2) / math.sqrt(2 * math.pi)
    assert r.portfolio_cvar == pytest.approx(phi / 0.05 * sigma_d, rel=1e-9)
    assert c["B"].percent_cvar == pytest.approx(0.8, rel=1e-9)


# --- identities on general correlated data ---


def test_volatility_components_sum_exactly_and_match_metrics(correlated) -> None:
    r = volatility_attribution(correlated, W5)
    assert sum(c.component for c in r.contributions) == pytest.approx(
        r.portfolio_volatility, rel=1e-12
    )
    assert sum(c.percent for c in r.contributions) == pytest.approx(1.0, rel=1e-12)
    # total must be the same number metrics.py reports for the portfolio return series
    port = portfolio_returns(correlated, W5)
    assert r.portfolio_volatility == pytest.approx(annualised_volatility(port), rel=1e-12)


@pytest.mark.parametrize("confidence", [0.95, 0.99])
@pytest.mark.parametrize("horizon_days", [1, 10])
def test_var_components_sum_to_parametric_var(correlated, confidence, horizon_days) -> None:
    r = parametric_var_attribution(correlated, W5, confidence=confidence, horizon_days=horizon_days)
    ref = parametric_var(correlated, W5, confidence=confidence, horizon_days=horizon_days)
    assert r.portfolio_var == pytest.approx(ref.var, rel=1e-12)
    assert r.portfolio_cvar == pytest.approx(ref.cvar, rel=1e-12)
    assert sum(c.component_var for c in r.contributions) == pytest.approx(ref.var, rel=1e-12)
    assert sum(c.component_cvar for c in r.contributions) == pytest.approx(ref.cvar, rel=1e-12)
    assert sum(c.percent_var for c in r.contributions) == pytest.approx(1.0, rel=1e-12)
    assert (r.confidence, r.horizon_days, r.n_obs) == (confidence, horizon_days, 1000)


def test_marginal_is_the_derivative_of_sigma_p(correlated) -> None:
    # Central finite difference of sigma_p(w) = sqrt(w' Sigma w) w.r.t. each w_i
    cov = correlated.cov().to_numpy()
    w = np.array(list(W5.values()))
    _, marginal, _ = euler_contributions(cov, w)
    h = 1e-6
    for i in range(len(w)):
        e = np.zeros_like(w)
        e[i] = h
        up, down = math.sqrt((w + e) @ cov @ (w + e)), math.sqrt((w - e) @ cov @ (w - e))
        assert marginal[i] == pytest.approx((up - down) / (2 * h), rel=1e-6)


def test_weight_and_column_order_independence(correlated) -> None:
    ref = by_id(volatility_attribution(correlated, W5))
    ref_var = {c.asset_id: c for c in parametric_var_attribution(correlated, W5).contributions}
    shuffled_w = dict(reversed(list(W5.items())))
    shuffled_df = correlated[["D", "B", "E", "A", "C"]]
    r = volatility_attribution(shuffled_df, shuffled_w)
    assert [c.asset_id for c in r.contributions] == list(shuffled_w)  # input order kept
    for a, c in by_id(r).items():
        assert c.component == pytest.approx(ref[a].component, rel=1e-12)
        assert c.marginal == pytest.approx(ref[a].marginal, rel=1e-12)
    for c in parametric_var_attribution(shuffled_df, shuffled_w).contributions:
        assert c.component_var == pytest.approx(ref_var[c.asset_id].component_var, rel=1e-12)
    assert diversification_ratio(shuffled_df, shuffled_w) == pytest.approx(
        diversification_ratio(correlated, W5), rel=1e-12
    )


def test_unweighted_columns_are_ignored(correlated) -> None:
    w = {"A": 0.6, "C": 0.4}
    full = volatility_attribution(correlated, w)
    sub = volatility_attribution(correlated[["A", "C"]], w)
    assert full.portfolio_volatility == pytest.approx(sub.portfolio_volatility, rel=1e-12)


# --- edge cases ---


def test_single_asset() -> None:
    df = hadamard_frame({"A": 0.01})
    sigma_ann = 0.01 * math.sqrt(F) * ROOT_T
    r = volatility_attribution(df, {"A": 1.0})
    (c,) = r.contributions
    assert c.component == pytest.approx(sigma_ann, rel=1e-12)
    assert c.marginal == pytest.approx(sigma_ann, rel=1e-12)
    assert c.percent == pytest.approx(1.0, rel=1e-12)
    assert diversification_ratio(df, {"A": 1.0}) == pytest.approx(1.0, rel=1e-12)
    (row,) = risk_budget_report(df, {"A": 1.0}).rows
    assert row.ratio == pytest.approx(1.0, rel=1e-12)


def test_zero_weight_asset_has_zero_component_but_a_marginal() -> None:
    # C has zero weight: component 0, ratio None; marginal = cov(C, R_p)/sigma_p = 0 here
    # because C is orthogonal to A and B.
    df = hadamard_frame({"A": 0.01, "B": 0.02, "C": 0.03})
    r = volatility_attribution(df, {"A": 0.5, "B": 0.5, "C": 0.0})
    c = by_id(r)["C"]
    assert c.component == 0.0 and c.percent == 0.0
    assert c.marginal == pytest.approx(0.0, abs=1e-15)
    rows = {x.asset_id: x for x in risk_budget_report(df, {"A": 0.5, "B": 0.5, "C": 0.0}).rows}
    assert rows["C"].ratio is None


def test_zero_weight_asset_marginal_by_hand() -> None:
    # B = A copy with zero weight: cov(B, R_p) = sigma^2, sigma_p = sigma -> marginal_B = sigma
    base = hadamard_frame({"A": 0.01})
    r = volatility_attribution(base.assign(B=base["A"]), {"A": 1.0, "B": 0.0})
    assert by_id(r)["B"].marginal == pytest.approx(0.01 * math.sqrt(F) * ROOT_T, rel=1e-12)


def test_short_history_raises() -> None:
    df = hadamard_frame({"A": 0.01, "B": 0.02}, n=248)
    for fn in (volatility_attribution, parametric_var_attribution, diversification_ratio):
        with pytest.raises(ValueError, match="at least 252"):
            fn(df, {"A": 0.5, "B": 0.5})
    # an explicit lower minimum is allowed, down to 2
    assert volatility_attribution(df, {"A": 0.5, "B": 0.5}, min_obs=200).n_obs == 248
    with pytest.raises(ValueError, match="min_obs"):
        volatility_attribution(df, {"A": 0.5, "B": 0.5}, min_obs=1)


def test_invalid_inputs_raise(correlated) -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        volatility_attribution(correlated, {"A": 0.5, "B": 0.4})
    with pytest.raises(ValueError, match="not in returns"):
        volatility_attribution(correlated, {"A": 0.5, "Z": 0.5})
    bad = correlated.copy()
    bad.iloc[3, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        volatility_attribution(bad, W5)
    with pytest.raises(ValueError, match="confidence"):
        parametric_var_attribution(correlated, W5, confidence=0.4)
    with pytest.raises(ValueError, match="horizon_days"):
        parametric_var_attribution(correlated, W5, horizon_days=30)
    with pytest.raises(ValueError, match="n x n"):
        euler_contributions(np.eye(3), np.ones(2))


def test_to_frame(correlated) -> None:
    df = volatility_attribution(correlated, W5).to_frame()
    assert list(df.index) == list(W5)
    assert list(df.columns) == ["weight", "marginal", "component", "percent"]
    rb = risk_budget_report(correlated, W5).to_frame()
    assert list(rb.columns) == ["weight", "risk_share", "ratio"]
