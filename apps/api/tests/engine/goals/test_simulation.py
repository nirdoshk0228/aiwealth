"""SIP Monte Carlo: exact zero-volatility limits and the analytic lump-sum distribution."""

import math
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from app.engine.goals.simulation import (
    GbmParams,
    SipPlan,
    calibrate_gbm,
    contributions,
    deterministic_value,
    simulate_sip,
)


def test_calibration_uses_log_returns() -> None:
    idx = pd.bdate_range("2020-01-01", periods=300)
    r = pd.Series(0.001, index=idx)
    p = calibrate_gbm(r)
    assert p.m == pytest.approx(252 * math.log(1.001), rel=1e-14)
    assert p.s == pytest.approx(0.0, abs=1e-15)
    # Alternating +10% / -10%: the log drift is negative although the simple mean is 0.
    alt = pd.Series(np.tile([0.1, -0.1], 150), index=idx)
    q = calibrate_gbm(alt)
    assert q.m == pytest.approx(252 * 0.5 * (math.log(1.1) + math.log(0.9)), rel=1e-12)
    assert q.m < 0


def test_calibration_validation() -> None:
    idx = pd.bdate_range("2020-01-01", periods=300)
    with pytest.raises(ValueError, match="at least 252"):
        calibrate_gbm(pd.Series(0.0, index=idx[:10]))
    with pytest.raises(ValueError, match="-100%"):
        calibrate_gbm(pd.Series(-1.0, index=idx))


def test_monthly_sip_with_zero_volatility_is_a_growing_annuity() -> None:
    # Rs 1,000 a month for a year at 1% a month, invested at the start of each month:
    # V = 1000 * sum_{j=1..12} 1.01^j = 1000 * 1.01 * (1.01^12 - 1) / 0.01.
    m = 12 * math.log(1.01)
    plan = SipPlan(initial=0, monthly=1000, years=1, goal_today=10_000)
    expected = 1000 * 1.01 * (1.01**12 - 1) / 0.01
    assert deterministic_value(plan, m) == pytest.approx(expected, rel=1e-13)
    sim = simulate_sip(GbmParams(m=m, s=0.0, n_obs=1000), plan, n_paths=1000)
    for v in sim.final_percentiles.values():
        assert v == pytest.approx(expected, rel=1e-12)
    assert sim.probability == 1.0  # 12,809 > 10,000
    assert sim.total_invested == 12_000


def test_step_up_and_inflation() -> None:
    plan = SipPlan(
        initial=50_000,
        monthly=10_000,
        years=3,
        goal_today=1e6,
        inflation_annual=0.04,
        step_up_annual=0.1,
    )
    c = contributions(plan)
    assert (
        c[0] == c[11] == 10_000
        and c[12] == pytest.approx(11_000)
        and c[24] == pytest.approx(12_100)
    )
    sim = simulate_sip(GbmParams(m=0.0, s=0.0, n_obs=1000), plan, n_paths=1000)
    assert sim.total_invested == pytest.approx(50_000 + 12 * (10_000 + 11_000 + 12_100))
    assert sim.goal_nominal == pytest.approx(1e6 * 1.04**3)
    assert sim.final_percentiles[50] == pytest.approx(sim.total_invested)  # m = 0, s = 0
    assert sim.final_percentiles_real[50] == pytest.approx(sim.total_invested / 1.04**3)
    assert sim.invested_by_year == pytest.approx((50_000, 170_000, 302_000, 447_200))


def test_lump_sum_matches_the_lognormal_distribution() -> None:
    # No contributions: ln V_T ~ N(ln V0 + m T, s^2 T), so P(V_T >= G) and the median are known.
    m, s, years, v0, goal = 0.08, 0.18, 10, 100_000.0, 200_000.0
    plan = SipPlan(initial=v0, monthly=0, years=years, goal_today=goal)
    sim = simulate_sip(GbmParams(m=m, s=s, n_obs=2000), plan, n_paths=200_000, seed=1)
    exact_p = 1 - NormalDist().cdf((math.log(goal / v0) - m * years) / (s * math.sqrt(years)))
    assert abs(sim.probability - exact_p) < 4 * sim.probability_se
    exact_median = v0 * math.exp(m * years)
    median_se = exact_median * s * math.sqrt(years) * math.sqrt(math.pi / 2 / 200_000)
    assert abs(sim.final_percentiles[50] - exact_median) < 4 * median_se
    assert sim.fan[50][0] == v0 and len(sim.fan[50]) == years + 1


def test_same_seed_same_result() -> None:
    plan = SipPlan(initial=0, monthly=5000, years=5, goal_today=400_000)
    p = GbmParams(m=0.1, s=0.15, n_obs=1000)
    a, b = simulate_sip(p, plan, seed=3), simulate_sip(p, plan, seed=3)
    assert a.final_percentiles == b.final_percentiles and a.probability == b.probability


@pytest.mark.parametrize(
    ("kw", "match"),
    [
        ({"monthly": 0, "initial": 0}, "invest something"),
        ({"years": 0}, "years"),
        ({"goal_today": 0}, "goal"),
        ({"inflation_annual": 4.0}, "inflation"),
        ({"step_up_annual": -0.1}, "step-up"),
        ({"monthly": -1}, "monthly"),
    ],
)
def test_plan_validation(kw: dict[str, float], match: str) -> None:
    base = {"initial": 0.0, "monthly": 1000.0, "years": 5, "goal_today": 1e5}
    with pytest.raises(ValueError, match=match):
        simulate_sip(GbmParams(0.1, 0.1, 1000), SipPlan(**(base | kw)))  # type: ignore[arg-type]
