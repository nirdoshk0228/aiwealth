"""factor_regression: exact recovery, a hand-computed case, HAC references, validation."""

import math

import numpy as np
import pandas as pd
import pytest

from app.engine.factors.regression import (
    DEFAULT_FACTORS,
    FactorRegressionResult,
    factor_regression,
    factor_regression_by_asset,
    hac_covariance,
    newey_west_lags,
    ols_hac,
    scaled_condition_number,
)

SQRT252 = math.sqrt(252)


def days(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2020-01-01", periods=n, name="date")


def random_factors(n: int, seed: int = 0) -> pd.DataFrame:
    """Daily-sized, mildly correlated factors plus a time-varying RF."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, 4))
    mix = np.array(
        [[1.0, 0.0, 0.0, 0.0], [-0.3, 1.0, 0.0, 0.0], [0.2, 0.1, 1.0, 0.0], [0.1, -0.2, -0.3, 1.0]]
    )
    f = (z @ mix.T) * np.array([0.012, 0.008, 0.009, 0.010])
    rf = 0.0002 + 0.00005 * np.sin(np.arange(n) / 50)
    return pd.DataFrame(np.column_stack([f, rf]), index=days(n), columns=[*DEFAULT_FACTORS, "RF"])


# --- exact answers ---


def test_noise_free_data_recovers_known_alpha_and_betas() -> None:
    f = random_factors(600)
    alpha, betas = 0.0003, {"MKT_RF": 0.95, "SMB": -0.4, "HML": 0.25, "WML": 0.1}
    excess = alpha + sum(b * f[k] for k, b in betas.items())
    returns = (excess + f["RF"]).rename("p")

    res = factor_regression(returns, f)

    assert res.alpha_daily == pytest.approx(alpha, abs=1e-10)
    assert res.alpha_annual == 252 * res.alpha_daily
    for k, b in betas.items():
        assert res.betas[k] == pytest.approx(b, abs=1e-10)
    assert res.r_squared == pytest.approx(1.0, abs=1e-12)
    assert res.residual_vol_annual < 1e-12
    assert res.n_obs == 600 and res.hac_lags == newey_west_lags(600) == 5
    assert (res.start, res.end) == (f.index[0], f.index[-1])


def hand_case() -> tuple[pd.Series, pd.DataFrame]:
    """n = 4, regressors A, B orthogonal to each other and to the intercept, RF = 0.

        A = [1, -1, 1, -1],  B = [1, 1, -1, -1],  y = [3, 1, 2, -2]
    X'X = 4 I, so alpha = mean(y) = 1, beta_A = A.y/4 = 6/4 = 1.5, beta_B = B.y/4 = 4/4 = 1.
    Fitted [3.5, 0.5, 1.5, -1.5], residuals e = [-0.5, 0.5, 0.5, -0.5], RSS = 1.
    TSS = 2^2 + 0^2 + 1^2 + (-3)^2 = 14, R^2 = 13/14, adj R^2 = 1 - (1/14)(3/1) = 11/14.
    Classical OLS: s^2 = RSS/(n - p) = 1, Var = s^2 (X'X)^-1 = I/4, se = 0.5 for all three.
    """
    idx = days(4)
    factors = pd.DataFrame(
        {"A": [1.0, -1.0, 1.0, -1.0], "B": [1.0, 1.0, -1.0, -1.0], "RF": 0.0}, index=idx
    )
    return pd.Series([3.0, 1.0, 2.0, -2.0], index=idx), factors


def test_hand_computed_two_regressor_case() -> None:
    returns, factors = hand_case()
    res = factor_regression(returns, factors, factor_names=("A", "B"), hac_lags=0, min_obs=4)

    assert res.alpha_daily == pytest.approx(1.0, abs=1e-14)
    assert res.betas == pytest.approx({"A": 1.5, "B": 1.0}, abs=1e-14)
    assert res.r_squared == pytest.approx(13 / 14, abs=1e-14)
    assert res.adj_r_squared == pytest.approx(11 / 14, abs=1e-14)
    assert res.residual_vol_annual == pytest.approx(1.0 * SQRT252, abs=1e-12)
    # |e| = 0.5 for every row, so lag-0 HAC (HC1) equals classical OLS: se = 0.5.
    assert res.std_errors == pytest.approx({"alpha": 0.5, "A": 0.5, "B": 0.5}, abs=1e-14)
    assert res.t_stats == pytest.approx({"alpha": 2.0, "A": 3.0, "B": 2.0}, abs=1e-13)
    assert res.p_values["alpha"] == pytest.approx(math.erfc(2 / math.sqrt(2)), abs=1e-14)
    assert res.condition_number == pytest.approx(1.0, abs=1e-12)
    assert res.warnings == ()


def test_lag0_hac_equals_classical_ols_when_all_residuals_have_equal_size() -> None:
    # Replacement for "HAC(0) == OLS": that holds only in special cases. Construct one with
    # n = 50: residuals e = +-c (25 each, so they sum to 0), and regressors made orthogonal
    # to e, so the OLS residual is exactly e. Then sum e_t^2 x_t x_t' = c^2 X'X and
    # HC1 = n/(n-p) c^2 (X'X)^-1 = (RSS/(n-p)) (X'X)^-1 = classical OLS.
    rng = np.random.default_rng(7)
    n, c = 50, 0.01
    e = c * rng.permutation(np.r_[np.ones(25), -np.ones(25)])
    Z = rng.normal(size=(n, 2))
    Z -= np.outer(e, e @ Z) / (e @ e)
    X = np.column_stack([np.ones(n), Z])
    y = X @ np.array([0.001, 0.8, -0.3]) + e

    fit = ols_hac(y, X, lags=0)
    assert np.allclose(fit.resid, e, atol=1e-14)
    p = X.shape[1]
    classical = (fit.resid @ fit.resid) / (n - p) * np.linalg.inv(X.T @ X)
    np.testing.assert_allclose(fit.cov, classical, rtol=1e-12, atol=0)


def test_lag0_hac_equals_an_independent_hc1_sandwich() -> None:
    # Heteroskedastic errors: HC1 differs from classical OLS here, and must match a direct
    # textbook implementation: n/(n-p) (X'X)^-1 X' diag(e^2) X (X'X)^-1.
    rng = np.random.default_rng(3)
    n = 40
    X = np.column_stack([np.ones(n), rng.normal(size=(n, 2))])
    e = rng.normal(size=n) * (0.5 + np.abs(X[:, 1]))
    y = X @ np.array([0.1, 1.0, -0.5]) + e

    fit = ols_hac(y, X, lags=0)
    bread = np.linalg.inv(X.T @ X)
    meat = X.T @ np.diag(fit.resid**2) @ X
    hc1 = n / (n - 3) * bread @ meat @ bread
    np.testing.assert_allclose(fit.cov, hc1, rtol=1e-12, atol=0)
    classical = (fit.resid @ fit.resid) / (n - 3) * bread
    assert not np.allclose(fit.cov, classical, rtol=1e-3)


def reference_newey_west(X: np.ndarray, e: np.ndarray, lags: int) -> np.ndarray:
    """Slow loop-by-loop Newey-West with Bartlett weights and the n/(n-p) factor."""
    n, p = X.shape
    S = np.zeros((p, p))
    for t in range(n):
        S += e[t] ** 2 * np.outer(X[t], X[t])
    for lag in range(1, lags + 1):
        w = 1 - lag / (lags + 1)
        for t in range(lag, n):
            S += w * e[t] * e[t - lag] * (np.outer(X[t], X[t - lag]) + np.outer(X[t - lag], X[t]))
    bread = np.linalg.inv(X.T @ X)
    return n / (n - p) * bread @ S @ bread


@pytest.mark.parametrize("lags", [1, 2, 5])
def test_hac_matches_a_loop_reference(lags: int) -> None:
    rng = np.random.default_rng(11)
    n = 30
    X = np.column_stack([np.ones(n), rng.normal(size=(n, 2))])
    shocks = rng.normal(size=n)
    e = np.empty(n)
    e[0] = shocks[0]
    for t in range(1, n):  # AR(1) errors so the lag terms matter
        e[t] = 0.6 * e[t - 1] + shocks[t]
    np.testing.assert_allclose(
        hac_covariance(X, e, lags), reference_newey_west(X, e, lags), rtol=1e-12, atol=1e-15
    )


def test_newey_west_lag_rule() -> None:
    # floor(4 (n/100)^(2/9)): exactly 4 at n=100; 4 * 10^(2/9) = 6.67 at n=1000.
    assert newey_west_lags(100) == 4
    assert newey_west_lags(1000) == 6
    assert newey_west_lags(504) == 5
    assert newey_west_lags(2713) == 8
    with pytest.raises(ValueError):
        newey_west_lags(0)


def test_zero_standard_error_gives_nan_t_stat_not_inf() -> None:
    # Returns equal to RF: excess return 0 on every row, so lstsq returns exactly 0, the
    # residuals are exactly 0 and so is every standard error. (A merely "perfect" fit leaves
    # ~1e-16 residuals and a finite, huge t; that case is not special-cased.)
    _, factors = hand_case()
    factors = factors.assign(RF=0.0001)
    with np.errstate(all="raise"):
        res = factor_regression(
            factors["RF"], factors, factor_names=("A", "B"), hac_lags=0, min_obs=4
        )
    assert res.std_errors == {"alpha": 0.0, "A": 0.0, "B": 0.0}
    assert all(math.isnan(v) for v in res.t_stats.values())
    assert all(math.isnan(v) for v in res.p_values.values())
    assert math.isnan(res.r_squared)  # zero total variance


# --- collinearity ---


def test_exactly_collinear_factors_raise() -> None:
    f = random_factors(600)
    f["SMB"] = 2.0 * f["MKT_RF"]
    with pytest.raises(ValueError, match="collinear factors: scaled condition number"):
        factor_regression(f["MKT_RF"] + f["RF"], f)


def test_constant_factor_raises_with_a_specific_message() -> None:
    f = random_factors(600).assign(HML=0.001)
    with pytest.raises(ValueError, match="'HML' is constant"):
        factor_regression(f["MKT_RF"] + f["RF"], f)


def test_near_collinear_factors_warn_but_fit() -> None:
    f = random_factors(600)
    noise = np.random.default_rng(1).normal(size=600) * 0.0002
    f["WML"] = f["MKT_RF"] + noise  # correlation ~0.9999
    res = factor_regression(f["MKT_RF"] + f["RF"], f)
    assert 30 < res.condition_number < 1e6
    assert len(res.warnings) == 1 and "condition number" in res.warnings[0]


def test_scaled_condition_number_ignores_units() -> None:
    X = np.column_stack([np.ones(5), [1.0, 2.0, 3.0, 4.0, 6.0]])
    assert scaled_condition_number(X) == pytest.approx(scaled_condition_number(X * [1, 1e-4]))
    assert math.isinf(scaled_condition_number(np.column_stack([np.ones(3), np.zeros(3)])))


# --- per-asset ---


def test_by_asset_equals_single_regressions() -> None:
    f = random_factors(600)
    noise = np.random.default_rng(2).normal(size=(600, 2)) * 0.005
    returns = pd.DataFrame(
        {
            "a": f["RF"] + 0.9 * f["MKT_RF"] + noise[:, 0],
            "b": f["RF"] + 0.5 * f["MKT_RF"] - 0.3 * f["SMB"] + noise[:, 1],
        }
    )
    table = factor_regression_by_asset(returns, f)

    assert list(table.index) == ["a", "b"]
    for col in returns:
        single = factor_regression(returns[col], f)
        assert table.loc[col].to_dict() == pytest.approx(single.to_row(), rel=1e-15, abs=0)
    assert table.loc["a", "beta_MKT_RF"] == pytest.approx(0.9, abs=0.05)
    assert list(table.columns[:4]) == ["alpha_annual", "t_alpha", "beta_MKT_RF", "t_MKT_RF"]


def test_by_asset_validates_frame() -> None:
    f = random_factors(600)
    with pytest.raises(TypeError):
        factor_regression_by_asset(f["MKT_RF"], f)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="no columns"):
        factor_regression_by_asset(pd.DataFrame(index=f.index), f)


# --- validation and alignment ---


def base() -> tuple[pd.Series, pd.DataFrame]:
    f = random_factors(600)
    return (f["RF"] + 0.9 * f["MKT_RF"] + 0.001 * np.sin(np.arange(600))).rename("p"), f


def test_index_mismatch_raises_and_counts_dates() -> None:
    r, f = base()
    with pytest.raises(ValueError, match="1 return dates not in factors, 1 factor dates"):
        factor_regression(r.iloc[1:], f.iloc[:-1])
    with pytest.raises(ValueError, match="identical indexes"):
        factor_regression(r, f.iloc[::-1])


def test_nan_in_returns_or_factors_raises() -> None:
    r, f = base()
    with pytest.raises(ValueError, match="returns contain NaN"):
        factor_regression(r.where(r.index != r.index[5]), f)
    with pytest.raises(ValueError, match=r"factors contain NaN; select the rows with factor data"):
        factor_regression(r, f.assign(SMB=f["SMB"].where(f.index != f.index[9])))
    with pytest.raises(ValueError, match="RF contain NaN"):
        factor_regression(r, f.assign(RF=f["RF"].where(f.index != f.index[9])))
    with pytest.raises(ValueError, match="inf"):
        factor_regression(r.replace(r.iloc[3], np.inf), f)


def test_minimum_observations() -> None:
    r, f = base()
    factor_regression(r.iloc[:504], f.iloc[:504])  # default min_obs = 504 is enough
    with pytest.raises(ValueError, match="at least 504 observations, got 503"):
        factor_regression(r.iloc[:503], f.iloc[:503])
    with pytest.raises(ValueError, match="min_obs must be"):
        factor_regression(r, f, min_obs=5)  # 5 parameters need at least 6 rows


@pytest.mark.parametrize("lags", [-1, 600, 2.0, True])
def test_invalid_hac_lags_raise(lags: object) -> None:
    r, f = base()
    with pytest.raises(ValueError, match="hac_lags"):
        factor_regression(r, f, hac_lags=lags)  # type: ignore[arg-type]


def test_argument_and_column_checks() -> None:
    r, f = base()
    with pytest.raises(ValueError, match="missing columns \\['WML'\\]"):
        factor_regression(r, f.drop(columns="WML"))
    with pytest.raises(ValueError, match="risk-free column"):
        factor_regression(r, f, factor_names=("MKT_RF", "RF"))
    with pytest.raises(ValueError, match="duplicate"):
        factor_regression(r, f, factor_names=("MKT_RF", "MKT_RF"))
    with pytest.raises(ValueError, match="must not be empty"):
        factor_regression(r, f, factor_names=())
    with pytest.raises(TypeError):
        factor_regression(r.to_frame(), f)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="DatetimeIndex"):
        factor_regression(r.reset_index(drop=True), f.reset_index(drop=True))


def test_result_is_a_typed_record() -> None:
    r, f = base()
    res = factor_regression(r, f, factor_names=("MKT_RF",))
    assert isinstance(res, FactorRegressionResult)
    assert set(res.t_stats) == {"alpha", "MKT_RF"}
    # One regressor: slope = cov(y, x) / var(x), intercept = mean(y) - slope * mean(x).
    y, x = r - f["RF"], f["MKT_RF"]
    slope = np.cov(y, x)[0, 1] / np.var(x, ddof=1)
    assert res.betas["MKT_RF"] == pytest.approx(slope, rel=1e-12)
    assert res.alpha_daily == pytest.approx(y.mean() - slope * x.mean(), rel=1e-10)
