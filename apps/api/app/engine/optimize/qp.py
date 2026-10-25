"""Small convex quadratic programs on the capped simplex, solved exactly by a primal active set.

Problem (all portfolio problems in this package reduce to it):

    minimise    q(w) = 1/2 w'Qw - c'w
    subject to  sum_i w_i = 1,   0 <= w_i <= u      (budget, long-only, optional cap u)

with Q symmetric positive semi-definite (a covariance times a constant) and n <= ~50.

Method: primal active-set (Nocedal & Wright, Numerical Optimization, 2nd ed., Algorithm 16.3).
The working set W holds bounds treated as equalities (w_i = 0 or w_i = u); the free set F is
the rest. Each iteration solves the equality-constrained step on F,

    [Q_FF  1] [p_F]   [-g_F]          g = Qw - c (gradient at w)
    [1'    0] [ nu] = [  0 ],         p_W = 0,

then either moves along p until a bound blocks (adding it to W), or, when p = 0, checks the
bound multipliers from stationarity  g_i + nu - alpha_i + beta_i = 0:

    at lower bound (w_i = 0):  alpha_i = g_i + nu   must be >= 0
    at upper bound (w_i = u):  beta_i  = -(g_i + nu) must be >= 0

and drops the most violated bound. For a convex Q the first point where all multipliers have
the right sign is a global minimum (KKT conditions are sufficient). `kkt_residual` reports how
far a returned point is from satisfying them, so every result carries its own certificate.

Why not scipy: the engine has no scipy dependency (backtest p-values are closed-form too), and
an active set is exact and fast at this size; warm starts make a 100-point frontier cheap.
Limits: Q_FF must be non-singular on the free set together with the budget row (true for a
sample covariance of more rows than assets; a singular one raises a clear error).
"""

from dataclasses import dataclass

import numpy as np

TOL = 1e-12
MAX_ITER_FACTOR = 50  # iterations allowed per variable


@dataclass(frozen=True)
class QPResult:
    w: np.ndarray
    objective: float
    iterations: int
    kkt_residual: float  # max violation of the KKT conditions (stationarity and bound signs)


def check_feasible(n: int, cap: float) -> None:
    if n < 1:
        raise ValueError("need at least one asset")
    if not 0 < cap <= 1:
        raise ValueError(f"max weight must be in (0, 1], got {cap}")
    if n * cap < 1 - 1e-12:
        raise ValueError(
            f"infeasible: {n} assets with a {cap:.0%} cap cannot sum to 100% "
            f"(need at least {int(np.ceil(1 / cap))} assets)"
        )


def kkt_residual(Q: np.ndarray, c: np.ndarray, w: np.ndarray, cap: float) -> float:
    """Largest KKT violation at w (0 at an exact optimum). Feasibility is checked too.

    nu is chosen as the value that best satisfies stationarity on the free coordinates
    (their mean of -g); with no free coordinate, the midpoint of the admissible interval.
    """
    g = Q @ w - c
    lower = w <= TOL
    upper = w >= cap - TOL
    free = ~(lower | upper)
    if free.any():
        nu = float(-g[free].mean())
    else:
        lo = float(np.max(-g[lower])) if lower.any() else -np.inf
        hi = float(np.min(-g[upper])) if upper.any() else np.inf
        if np.isfinite(lo) and np.isfinite(hi):
            nu = 0.5 * (lo + hi)
        else:
            nu = lo if np.isfinite(lo) else hi
    r = g + nu
    viol = [
        abs(float(w.sum()) - 1.0),
        float(np.max(np.maximum(-w, 0.0))),
        float(np.max(np.maximum(w - cap, 0.0))),
        float(np.max(np.abs(r[free]))) if free.any() else 0.0,
        float(np.max(np.maximum(-r[lower], 0.0))) if lower.any() else 0.0,  # alpha >= 0
        float(np.max(np.maximum(r[upper], 0.0))) if upper.any() else 0.0,  # beta >= 0
    ]
    scale = max(1.0, float(np.max(np.abs(g))))
    return max(viol) / scale


def _equal_start(n: int, cap: float) -> np.ndarray:
    return np.full(n, 1.0 / n)  # feasible: 1/n <= cap because n * cap >= 1


def solve_simplex_qp(
    Q: np.ndarray,
    c: np.ndarray,
    *,
    cap: float = 1.0,
    w0: np.ndarray | None = None,
) -> QPResult:
    """Minimise 1/2 w'Qw - c'w on {sum w = 1, 0 <= w <= cap} (see the module docstring).

    `w0` warm-starts the search (any feasible point, e.g. the previous frontier solution).
    """
    Q = np.asarray(Q, dtype=float)
    c = np.asarray(c, dtype=float)
    n = c.shape[0]
    if Q.shape != (n, n):
        raise ValueError(f"Q must be {n} x {n}, got {Q.shape}")
    if not (np.isfinite(Q).all() and np.isfinite(c).all()):
        raise ValueError("Q and c must be finite")
    check_feasible(n, cap)
    Q = 0.5 * (Q + Q.T)

    w = _equal_start(n, cap) if w0 is None else np.clip(np.asarray(w0, dtype=float), 0.0, cap)
    if abs(w.sum() - 1.0) > 1e-9:
        w = _equal_start(n, cap)
    # state: 0 = free, -1 = at lower bound, +1 = at upper bound
    state = np.zeros(n, dtype=int)
    state[w <= TOL] = -1
    state[(w >= cap - TOL) & (state == 0)] = 1
    w[state == -1] = 0.0
    w[state == 1] = cap

    for it in range(1, MAX_ITER_FACTOR * n + 1):
        g = Q @ w - c
        free = np.flatnonzero(state == 0)
        if free.size == 0:
            nu = None
            p = np.zeros(n)
        else:
            k = free.size
            kkt = np.zeros((k + 1, k + 1))
            kkt[:k, :k] = Q[np.ix_(free, free)]
            kkt[:k, k] = 1.0
            kkt[k, :k] = 1.0
            rhs = np.concatenate([-g[free], [0.0]])
            try:
                sol = np.linalg.solve(kkt, rhs)
            except np.linalg.LinAlgError as exc:
                raise ValueError(
                    "covariance is singular on the free assets (e.g. two identical series); "
                    "remove the duplicate or use more history"
                ) from exc
            p = np.zeros(n)
            p[free] = sol[:k]
            nu = float(sol[k])

        if np.max(np.abs(p)) <= 1e-14 * max(1.0, float(np.max(np.abs(w)))):
            # Stationary on the working set: check bound multipliers.
            if nu is None:  # every asset at a bound: pick nu in the admissible interval
                lower, upper = state == -1, state == 1
                lo = float(np.max(-g[lower])) if lower.any() else -np.inf
                hi = float(np.min(-g[upper])) if upper.any() else np.inf
                if lo <= hi + 1e-15:
                    return _result(Q, c, w, cap, it)
                nu = lo if upper.any() else hi
            mult = np.where(state == -1, g + nu, np.where(state == 1, -(g + nu), np.inf))
            j = int(np.argmin(mult))
            if mult[j] >= -1e-13 * max(1.0, float(np.max(np.abs(g)))):
                return _result(Q, c, w, cap, it)
            state[j] = 0  # release the most violated bound
            continue

        # Step length: go as far as possible (up to 1) without leaving the box.
        alpha, block, block_state = 1.0, -1, 0
        for i in np.flatnonzero(p < 0):
            a = -w[i] / p[i]
            if a < alpha:
                alpha, block, block_state = a, int(i), -1
        for i in np.flatnonzero(p > 0):
            a = (cap - w[i]) / p[i]
            if a < alpha:
                alpha, block, block_state = a, int(i), 1
        w = w + alpha * p
        if block >= 0:
            state[block] = block_state
            w[block] = 0.0 if block_state == -1 else cap
        w[state == -1] = 0.0
        w[state == 1] = cap

    raise RuntimeError("active-set QP did not converge (should not happen for convex problems)")


def _result(Q: np.ndarray, c: np.ndarray, w: np.ndarray, cap: float, it: int) -> QPResult:
    w = np.clip(w, 0.0, cap)
    w = w / w.sum()  # remove last-bit drift; the budget holds to rounding already
    return QPResult(
        w=w,
        objective=float(0.5 * w @ Q @ w - c @ w),
        iterations=it,
        kkt_residual=kkt_residual(Q, c, w, cap),
    )
