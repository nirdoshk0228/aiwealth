"""Active-set QP on the capped simplex: closed forms, KKT certificates, brute force."""

import numpy as np
import pytest

from app.engine.optimize.qp import kkt_residual, solve_simplex_qp


def random_cov(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(3 * n, n))
    return a.T @ a / (3 * n) + 0.01 * np.eye(n)


def test_two_asset_min_variance_closed_form() -> None:
    # w1 = (s2^2 - s12) / (s1^2 + s2^2 - 2 s12) when it lies in [0, 1].
    s1, s2, rho = 0.2, 0.3, 0.25
    s12 = rho * s1 * s2
    cov = np.array([[s1**2, s12], [s12, s2**2]])
    res = solve_simplex_qp(cov, np.zeros(2))
    w1 = (s2**2 - s12) / (s1**2 + s2**2 - 2 * s12)
    np.testing.assert_allclose(res.w, [w1, 1 - w1], atol=1e-14)
    assert res.kkt_residual < 1e-12


def test_two_asset_corner_solution() -> None:
    # Highly correlated, very different vols: the unconstrained minimum shorts asset 2
    # (w1 = (0.09 - 0.057) / (0.04 + 0.09 - 0.114) > 1), so long-only puts 100% in asset 1.
    s12 = 0.95 * 0.2 * 0.3
    cov = np.array([[0.04, s12], [s12, 0.09]])
    assert (0.09 - s12) / (0.04 + 0.09 - 2 * s12) > 1
    res = solve_simplex_qp(cov, np.zeros(2))
    np.testing.assert_array_equal(res.w, [1.0, 0.0])
    assert res.kkt_residual < 1e-12


def test_interior_min_variance_matches_inverse_formula() -> None:
    # Equicorrelated assets with similar vols: the solution is interior, so it equals the
    # unconstrained Sigma^-1 1 / (1' Sigma^-1 1).
    vols = np.array([0.15, 0.18, 0.2, 0.22])
    corr = np.full((4, 4), 0.3) + 0.7 * np.eye(4)
    cov = np.outer(vols, vols) * corr
    x = np.linalg.solve(cov, np.ones(4))
    res = solve_simplex_qp(cov, np.zeros(4))
    np.testing.assert_allclose(res.w, x / x.sum(), atol=1e-13)
    assert (res.w > 0).all()


def test_cap_binds_and_kkt_holds() -> None:
    # One very low-volatility asset would take most of the weight; a 40% cap binds.
    cov = np.diag([0.0004, 0.04, 0.05, 0.06])
    free = solve_simplex_qp(cov, np.zeros(4))
    assert free.w[0] > 0.9
    capped = solve_simplex_qp(cov, np.zeros(4), cap=0.4)
    assert capped.w[0] == pytest.approx(0.4, abs=1e-15)
    assert capped.w.sum() == pytest.approx(1.0, abs=1e-15)
    assert capped.w.max() <= 0.4 + 1e-15
    assert capped.kkt_residual < 1e-12
    # Among the remaining three, the diagonal min-variance split is w_i ~ 1/var_i.
    inv = 1 / np.diag(cov)[1:]
    np.testing.assert_allclose(capped.w[1:], 0.6 * inv / inv.sum(), atol=1e-13)


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("cap", [1.0, 0.3])
def test_random_problems_beat_random_feasible_points(seed: int, cap: float) -> None:
    n = 8
    rng = np.random.default_rng(100 + seed)
    Q = random_cov(n, seed)
    c = rng.normal(0, 0.2, n)
    res = solve_simplex_qp(Q, c, cap=cap)
    assert res.kkt_residual < 1e-10
    assert res.w.sum() == pytest.approx(1.0, abs=1e-14)
    assert res.w.min() >= 0 and res.w.max() <= cap + 1e-15
    # No feasible random point does better (Dirichlet samples, kept if within the cap).
    pts = rng.dirichlet(np.ones(n), size=20_000)
    pts = pts[pts.max(axis=1) <= cap]
    obj = 0.5 * np.einsum("ij,jk,ik->i", pts, Q, pts) - pts @ c
    assert res.objective <= obj.min() + 1e-12


def test_warm_start_gives_the_same_answer() -> None:
    Q = random_cov(10, 3)
    c = np.linspace(-0.1, 0.1, 10)
    cold = solve_simplex_qp(Q, c, cap=0.25)
    warm = solve_simplex_qp(Q, c, cap=0.25, w0=np.full(10, 0.1))
    np.testing.assert_allclose(cold.w, warm.w, atol=1e-13)


def test_kkt_residual_detects_a_non_optimal_point() -> None:
    cov = np.diag([0.01, 0.04])
    assert kkt_residual(cov, np.zeros(2), np.array([0.5, 0.5]), 1.0) > 1e-3
    assert kkt_residual(cov, np.zeros(2), np.array([0.8, 0.2]), 1.0) < 1e-15


def test_infeasible_cap_and_bad_inputs() -> None:
    with pytest.raises(ValueError, match="cannot sum to 100%"):
        solve_simplex_qp(np.eye(3), np.zeros(3), cap=0.3)
    with pytest.raises(ValueError, match="must be 2 x 2"):
        solve_simplex_qp(np.eye(3), np.zeros(2))
    with pytest.raises(ValueError, match="finite"):
        solve_simplex_qp(np.array([[np.nan]]), np.zeros(1))


def test_singular_covariance_gives_a_clear_error() -> None:
    # Two identical assets: Q is singular on the free set.
    v = np.array([[0.04, 0.04], [0.04, 0.04]])
    with pytest.raises(ValueError, match="singular"):
        solve_simplex_qp(v, np.zeros(2))
