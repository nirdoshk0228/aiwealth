"""Optimizer and goal-planner endpoints on synthetic data."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.engine.goals.simulation import calibrate_gbm
from app.engine.optimize.mean_variance import estimate
from app.engine.risk.attribution import euler_contributions
from app.engine.risk.var import portfolio_returns
from app.services.market import MarketData

OPT = "/api/v1/optimize"
GOAL = "/api/v1/goals/simulate"
QUESTIONS = "/api/v1/goals/questionnaire"
PROFILE = "/api/v1/goals/profile"


def weights(p: dict[str, object]) -> dict[str, float]:
    return {h["asset_id"]: h["weight"] for h in p["holdings"]}  # type: ignore[index,union-attr]


def as_dict(p: dict[str, object]) -> dict[str, float]:
    return {w["asset_id"]: w["weight"] for w in p["weights"]}  # type: ignore[index,union-attr]


def test_optimize_portfolio_assets(
    api: TestClient, market: MarketData, fund_mix: dict[str, object]
) -> None:
    r = api.post(OPT, json={"portfolio": fund_mix, "max_weight": 0.4})
    assert r.status_code == 200
    out = r.json()
    for key in ("min_variance", "max_sharpe", "max_return", "risk_parity", "current"):
        p = out[key]
        assert sum(w["weight"] for w in p["weights"]) == pytest.approx(1.0, abs=1e-12)
        assert all(w["weight"] > 0 for w in p["weights"])
    for key in ("min_variance", "max_sharpe", "max_return"):
        assert max(w["weight"] for w in out[key]["weights"]) <= 0.4 + 1e-12
    # Ordering on the frontier: min variance has the lowest vol, max return the highest return.
    vols = [p["volatility"] for p in out["frontier"]]
    assert out["min_variance"]["volatility"] == pytest.approx(min(vols))
    assert (
        out["max_return"]["expected_return"]
        >= max(p["expected_return"] for p in out["frontier"]) - 1e-12
    )
    assert out["max_sharpe"]["sharpe"] >= out["current"]["sharpe"] - 1e-12
    assert out["max_sharpe"]["sharpe"] >= out["min_variance"]["sharpe"] - 1e-12
    # Current portfolio evaluated with the same estimates.
    r_win = market.returns.iloc[-1260:]
    est = estimate(r_win, list(weights(fund_mix)))
    w = np.array([weights(fund_mix)[a] for a in est.asset_ids])
    assert out["current"]["expected_return"] == pytest.approx(float(w @ est.mu), rel=1e-12)
    assert out["window"]["n_obs"] == 1260
    # Risk parity: equal shares, verified with the attribution engine.
    rp = as_dict(out["risk_parity"])
    wr = np.array([rp[a] for a in est.asset_ids])
    sigma, _, comp = euler_contributions(est.cov, wr)
    np.testing.assert_allclose(comp / sigma, 1 / len(wr), atol=1e-9)


def test_optimize_explicit_asset_list_and_validation(api: TestClient) -> None:
    ids = ["equity:TCS", "equity:INFY", "equity:HDFCBANK", "mutual_fund:119533"]
    out = api.post(OPT, json={"asset_ids": ids}).json()
    assert {a["asset_id"] for a in out["assets"]} == set(ids)
    assert out["current"] is None
    assert api.post(OPT, json={"asset_ids": ["equity:TCS"]}).status_code == 422  # one asset
    assert api.post(OPT, json={}).status_code == 422
    infeasible = api.post(OPT, json={"asset_ids": ids, "max_weight": 0.2})
    assert infeasible.status_code == 422 and "cannot sum to 100%" in infeasible.json()["detail"]
    unknown = api.post(OPT, json={"asset_ids": ["equity:TCS", "equity:NOPE"]})
    assert unknown.status_code == 422


def test_goal_simulation_calibrates_on_log_returns(
    api: TestClient, market: MarketData, fund_mix: dict[str, object]
) -> None:
    body = {
        "portfolio": fund_mix,
        "initial_amount": 100000,
        "monthly_sip": 10000,
        "years": 10,
        "goal_amount": 2500000,
        "inflation_annual": 0.04,
    }
    r = api.post(GOAL, json=body)
    assert r.status_code == 200
    out = r.json()
    ref = calibrate_gbm(portfolio_returns(market.returns, weights(fund_mix)))
    assert out["calibration"]["log_drift"] == pytest.approx(ref.m, rel=1e-12)
    assert out["calibration"]["volatility"] == pytest.approx(ref.s, rel=1e-12)
    assert out["goal_nominal"] == pytest.approx(2_500_000 * 1.04**10)
    assert out["total_invested"] == pytest.approx(100_000 + 120 * 10_000)
    assert 0 <= out["probability"] <= 1
    nominal = [p["nominal"] for p in out["percentiles"]]
    assert nominal == sorted(nominal)
    assert out["fan"]["years"] == list(range(11))
    assert out["fan"]["invested"][-1] == pytest.approx(out["total_invested"])
    # Same inputs and seed: identical output.
    assert api.post(GOAL, json=body).json() == out


def test_goal_validation(api: TestClient, fund_mix: dict[str, object]) -> None:
    base = {"portfolio": fund_mix, "monthly_sip": 0, "years": 5, "goal_amount": 1e6}
    assert api.post(GOAL, json=base).status_code == 422  # nothing invested
    assert api.post(GOAL, json=base | {"monthly_sip": 1000, "years": 60}).status_code == 422


def test_questionnaire_and_profile(api: TestClient, fund_mix: dict[str, object]) -> None:
    q = api.get(QUESTIONS).json()
    assert len(q["questions"]) == 5 and len(q["bands"]) == 5
    assert "not a suitability assessment" in q["disclaimer"]
    middle = {x["id"]: x["options"][2]["id"] for x in q["questions"]}  # 2 points each = 10
    out = api.post(PROFILE, json={"answers": middle, "portfolio": fund_mix}).json()
    assert out["band"]["id"] == "balanced" and out["score"] == 10
    mix = out["portfolio_mix"]
    # Sample: 55% equity funds, 10% balanced advantage, 25% debt, 10% gold.
    assert mix["equity"] == pytest.approx(0.55)
    assert mix["hybrid"] == pytest.approx(0.10)
    assert mix["debt"] == pytest.approx(0.25)
    assert mix["gold"] == pytest.approx(0.10)
    assert mix["equity_oriented"] == pytest.approx(0.60)
    assert out["within_band"] is True  # 60% is the band's upper edge
    no_portfolio = api.post(PROFILE, json={"answers": middle}).json()
    assert no_portfolio["portfolio_mix"] is None and no_portfolio["within_band"] is None
    bad = api.post(PROFILE, json={"answers": {"horizon": "gt10"}})
    assert bad.status_code == 422 and "unanswered" in bad.json()["detail"]


def test_liquid_fund_dominance_is_flagged(api: TestClient) -> None:
    # The synthetic liquid fund has 0.002% daily vol and a small drift over RF: near-cash.
    ids = ["mutual_fund:120389", "equity:TCS", "equity:INFY", "mutual_fund:122639"]
    out = api.post(OPT, json={"asset_ids": ids, "include_liquid": True}).json()
    assert as_dict(out["min_variance"])["mutual_fund:120389"] > 0.5
    assert any("liquid fund" in w and "minimum-variance" in w for w in out["warnings"])


def test_liquid_fund_is_excluded_as_cash_by_default(api: TestClient) -> None:
    ids = ["mutual_fund:120389", "equity:TCS", "equity:INFY", "mutual_fund:122639"]
    out = api.post(OPT, json={"asset_ids": ids}).json()
    assert {a["asset_id"] for a in out["assets"]} == set(ids) - {"mutual_fund:120389"}
    assert (
        out["notes"][0]
        == "Treated as cash and excluded: Axis Liquid Fund - Direct Plan - Growth Option."
    )
    assert not any("liquid fund" in w for w in out["warnings"])
    # Only the liquid fund and one other asset: too few left, with a hint.
    two = api.post(OPT, json={"asset_ids": ["mutual_fund:120389", "equity:TCS"]})
    assert two.status_code == 422 and "include_liquid" in two.json()["detail"]
    # A current portfolio holding the liquid fund is not compared, and says why.
    holding = {
        "holdings": [
            {"asset_id": "mutual_fund:120389", "weight": 0.5},
            {"asset_id": "equity:TCS", "weight": 0.5},
        ]
    }
    out2 = api.post(OPT, json={"asset_ids": ids, "portfolio": holding}).json()
    assert out2["current"] is None
    assert any("excluded as cash" in w for w in out2["warnings"])
