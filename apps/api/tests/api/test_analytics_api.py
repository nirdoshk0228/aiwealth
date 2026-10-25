"""Backtest, factor exposure and stress endpoints on the sample portfolios."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.engine.risk.backtest import var_backtest
from app.services import backtest as backtest_service
from app.services.market import MarketData
from tests.api.conftest import LOADINGS

BACKTEST = "/api/v1/backtest"
FACTORS = "/api/v1/factors/exposures"
SCENARIOS = "/api/v1/stress/scenarios"
RUN = "/api/v1/stress/run"


def weights(p: dict[str, object]) -> dict[str, float]:
    return {h["asset_id"]: h["weight"] for h in p["holdings"]}  # type: ignore[index,union-attr]


# --- backtest ---


def test_backtest_matches_engine_and_is_cached(
    api: TestClient, market: MarketData, stock_basket: dict[str, object]
) -> None:
    cache = backtest_service._CACHE
    cache.clear()
    hits0, misses0 = cache.hits, cache.misses  # counters are cumulative across tests
    body = {"portfolio": stock_basket, "method": "historical", "confidence": 0.99}
    r = api.post(BACKTEST, json=body)
    assert r.status_code == 200
    out = r.json()
    ref = var_backtest(market.returns, weights(stock_basket), confidence=0.99)
    assert out["n_breaches"] == ref.n_breaches
    assert out["kupiec"]["p_value"] == pytest.approx(ref.kupiec.p_value, rel=1e-12)
    assert out["test_period"]["n_obs"] == ref.n_test_days == len(market.returns) - 504
    assert len(out["dates"]) == len(out["var"]) == len(out["realised"]) == ref.n_test_days
    assert out["basel_zone_last_250"] in ("green", "yellow", "red")
    # Breach dates are exactly the days whose loss exceeds the forecast.
    rows = zip(out["dates"], out["var"], out["realised"], strict=True)
    breached = [d for d, v, x in rows if -x > v]
    assert breached == out["breach_dates"]
    assert any("k/(N+1) = 1.19%" in n for n in out["notes"])  # k = 6, N = 504

    assert (cache.hits - hits0, cache.misses - misses0) == (0, 1)
    api.post(BACKTEST, json=body)
    assert (cache.hits - hits0, cache.misses - misses0) == (1, 1)


def test_backtest_rejects_monte_carlo_on_request_path(
    api: TestClient, stock_basket: dict[str, object]
) -> None:
    r = api.post(BACKTEST, json={"portfolio": stock_basket, "method": "monte_carlo"})
    assert r.status_code == 422


def test_basel_zone_only_at_99(api: TestClient, stock_basket: dict[str, object]) -> None:
    out = api.post(BACKTEST, json={"portfolio": stock_basket, "confidence": 0.95}).json()
    assert out["basel_zone_last_250"] is None and out["basel_breaches_last_250"] is None


# --- factor exposures ---


def test_factor_betas_recover_synthetic_loadings(
    api: TestClient, market: MarketData, fund_mix: dict[str, object]
) -> None:
    out = api.post(FACTORS, json={"portfolio": fund_mix}).json()
    assert out["window"]["end"] == "2025-12-31"  # only rows with factor data
    assert out["source"].startswith("synthetic test factors, release synthetic")
    by_id = {a["asset_id"]: a for a in out["assets"]}
    small_cap = by_id["mutual_fund:118778"]

    # Exact: an independent least-squares fit on the same rows.
    f = market.factors.factors.dropna()
    y = market.returns.loc[f.index, "mutual_fund:118778"] - f["RF"]
    X = np.column_stack([np.ones(len(f)), f[["MKT_RF", "SMB", "HML", "WML"]].to_numpy()])
    coef = np.linalg.lstsq(X, y.to_numpy(), rcond=None)[0]
    assert small_cap["betas"]["MKT_RF"] == pytest.approx(coef[1], rel=1e-10)
    assert small_cap["betas"]["SMB"] == pytest.approx(coef[2], rel=1e-10)
    # Statistical: within 4 standard errors (se = beta / t) of the true synthetic loadings.
    b_m, b_s, _, _ = LOADINGS["small_cap"]
    for name, true in (("MKT_RF", b_m), ("SMB", b_s)):
        beta, t = small_cap["betas"][name], small_cap["t_stats"][name]
        assert abs(beta - true) < 4 * abs(beta / t)
    # Portfolio beta = weighted average of asset betas (OLS is linear in y, same X).
    w = weights(fund_mix)
    port_beta = {c["factor"]: c["beta"] for c in out["portfolio"]["coefficients"]}
    implied = sum(w[a] * by_id[a]["betas"]["MKT_RF"] for a in w)
    assert port_beta["MKT_RF"] == pytest.approx(implied, rel=1e-10)
    assert out["portfolio"]["hac_lags"] == 8  # floor(4 (n/100)^(2/9)) for n ~ 2,870


def test_factor_assets_can_be_skipped(api: TestClient, fund_mix: dict[str, object]) -> None:
    out = api.post(FACTORS, json={"portfolio": fund_mix, "include_assets": False}).json()
    assert out["assets"] == []


# --- stress ---


def test_scenario_list(api: TestClient, market: MarketData) -> None:
    out = api.get(SCENARIOS).json()
    assert len(out) == len(market.scenarios) == 10
    types = [s["type"] for s in out]
    assert types == sorted(types)  # historical before hypothetical
    covid = next(s for s in out if s["id"] == "covid_crash_2020")
    assert (covid["start"], covid["end"]) == ("2020-01-14", "2020-03-23")


def test_historical_replay_equals_buy_and_hold_return(
    api: TestClient, market: MarketData, stock_basket: dict[str, object]
) -> None:
    out = api.post(RUN, json={"portfolio": stock_basket, "scenario_id": "covid_crash_2020"})
    assert out.status_code == 200
    res = out.json()
    p = market.prices
    w = weights(stock_basket)
    expected = sum(w[a] * (p.loc["2020-03-23", a] / p.loc["2020-01-14", a] - 1) for a in w)
    assert res["pnl"] == pytest.approx(expected, rel=1e-12)
    assert res["pnl_rupees"] == pytest.approx(expected * 500_000, rel=1e-12)
    assert sum(i["contribution"] for i in res["impacts"]) == pytest.approx(res["pnl"], rel=1e-12)
    assert res["sessions"] == len(p.loc["2020-01-14":"2020-03-23"]) - 1
    assert res["max_drawdown"] is not None and res["max_drawdown"] >= 0


def test_hypothetical_factor_shock_uses_betas(
    api: TestClient, market: MarketData, fund_mix: dict[str, object]
) -> None:
    res = api.post(RUN, json={"portfolio": fund_mix, "scenario_id": "credit_event"}).json()
    by_id = {i["asset_id"]: i for i in res["impacts"]}
    # Direct category shocks override the factor-implied value.
    assert by_id["mutual_fund:119533"]["asset_return"] == -0.02
    assert by_id["mutual_fund:119533"]["source"] == "direct_category"
    # Equity funds get beta x (-5%).
    pf = by_id["mutual_fund:122639"]
    beta = market.exposures()["mutual_fund:122639"].betas["MKT_RF"]
    assert pf["source"] == "factor"
    assert pf["asset_return"] == pytest.approx(beta * -0.05, rel=1e-12)
    # The gold fund's factor fit has R^2 ~ 0, which must be flagged.
    assert any("mutual_fund:118663 (Nippon India Gold Savings Fund" in w for w in res["warnings"])


def test_custom_scenario(api: TestClient, stock_basket: dict[str, object]) -> None:
    scenario = {
        "id": "custom_it_and_market",
        "type": "hypothetical",
        "name": "IT -20%, market -5%",
        "description": "test",
        "rationale": "test",
        "factor_shocks": {"MKT_RF": -0.05},
        "direct_shocks": [
            {"target": {"kind": "sector", "sector": "Information Technology"}, "shock": -0.2}
        ],
    }
    res = api.post(RUN, json={"portfolio": stock_basket, "scenario": scenario})
    assert res.status_code == 200
    tcs = next(i for i in res.json()["impacts"] if i["asset_id"] == "equity:TCS")
    assert tcs["asset_return"] == -0.2 and tcs["source"] == "direct_sector"


def test_custom_scenario_with_unknown_sector_is_422(
    api: TestClient, stock_basket: dict[str, object]
) -> None:
    scenario = {
        "id": "typo",
        "type": "hypothetical",
        "name": "typo",
        "description": "test",
        "rationale": "test",
        "direct_shocks": [{"target": {"kind": "sector", "sector": "IT"}, "shock": -0.2}],
    }
    r = api.post(RUN, json={"portfolio": stock_basket, "scenario": scenario})
    assert r.status_code == 422
    assert "not in the universe" in r.json()["detail"]


def test_custom_scenario_shock_in_percent_is_rejected(
    api: TestClient, stock_basket: dict[str, object]
) -> None:
    scenario = {
        "id": "pct",
        "type": "hypothetical",
        "name": "pct",
        "description": "test",
        "rationale": "test",
        "factor_shocks": {"MKT_RF": -20},
    }
    assert api.post(RUN, json={"portfolio": stock_basket, "scenario": scenario}).status_code == 422


def test_unknown_scenario_is_404(api: TestClient, stock_basket: dict[str, object]) -> None:
    r = api.post(RUN, json={"portfolio": stock_basket, "scenario_id": "nope"})
    assert r.status_code == 404


def test_exactly_one_scenario_required(api: TestClient, stock_basket: dict[str, object]) -> None:
    assert api.post(RUN, json={"portfolio": stock_basket}).status_code == 422
