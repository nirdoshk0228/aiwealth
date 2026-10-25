"""Risk parity: inverse-volatility special cases and equal shares on random covariances."""

import numpy as np
import pytest

from app.engine.optimize.risk_parity import risk_parity_weights


def test_diagonal_covariance_gives_inverse_volatility() -> None:
    vols = np.array([0.05, 0.1, 0.2, 0.4])
    res = risk_parity_weights(np.diag(vols**2))
    inv = 1 / vols
    np.testing.assert_allclose(res.weights, inv / inv.sum(), atol=1e-12)
    np.testing.assert_allclose(res.risk_shares, 0.25, atol=1e-10)


@pytest.mark.parametrize("rho", [-0.6, 0.0, 0.5, 0.95])
def test_two_assets_are_inverse_volatility_for_any_correlation(rho: float) -> None:
    # With two assets, w1 s1 (w1 s1 + rho w2 s2) = w2 s2 (w2 s2 + rho w1 s1) <=> w1 s1 = w2 s2.
    s = np.array([0.15, 0.3])
    cov = np.outer(s, s) * np.array([[1, rho], [rho, 1]])
    res = risk_parity_weights(cov)
    np.testing.assert_allclose(res.weights, [2 / 3, 1 / 3], atol=1e-10)


@pytest.mark.parametrize("seed", range(5))
def test_random_covariance_equal_risk_shares(seed: int) -> None:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(50, 12))
    cov = a.T @ a / 50 + 0.05 * np.eye(12)
    res = risk_parity_weights(cov)
    assert res.weights.sum() == pytest.approx(1.0, abs=1e-14)
    assert (res.weights > 0).all()
    assert res.max_share_error < 1e-9
    np.testing.assert_allclose(res.risk_shares.sum(), 1.0, atol=1e-12)


def test_invalid_covariance() -> None:
    with pytest.raises(ValueError, match="positive variance"):
        risk_parity_weights(np.diag([0.01, 0.0]))
    with pytest.raises(ValueError, match="square"):
        risk_parity_weights(np.ones((2, 3)))
