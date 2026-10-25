"""Risk endpoints on the sample portfolios: values equal the engine's, units and sums hold."""

import math

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.engine.risk.attribution import volatility_attribution
from app.engine.risk.metrics import max_drawdown, performance_summary
from app.engine.risk.var import historical_var, parametric_var, portfolio_returns
from app.services.market import MarketData
from tests.api.conftest import RF_DAILY

VAR = "/api/v1/risk/var"
METRICS = "/api/v1/risk/metrics"
ATTR = "/api/v1/risk/attribution"
CORR = "/api/v1/risk/correlation"


def weights(p: dict[str, object]) -> dict[str, float]:
    return {h["asset_id"]: h["weight"] for h in p["holdings"]}  # type: ignore[index,union-attr]


def test_var_matches_engine_and_converts_to_rupees(
    api: TestClient, market: MarketData, fund_mix: dict[str, object]
) -> None:
    r = api.post(VAR, json={"portfolio": fund_mix, "confidence": 0.99, "lookback_days": 756})
    assert r.status_code == 200
    body = r.json()
    est = {e["method"]: e for e in body["estimates"]}
    assert list(est) == ["historical", "parametric", "monte_carlo"]

    window = market.returns.iloc[-756:]
    h = historical_var(window, weights(fund_mix), confidence=0.99)
    p = parametric_var(window, weights(fund_mix), confidence=0.99)
    assert est["historical"]["var"] == pytest.approx(h.var, rel=1e-12)
    assert est["parametric"]["cvar"] == pytest.approx(p.cvar, rel=1e-12)
    assert est["monte_carlo"]["n_sims"] == 100_000 and est["monte_carlo"]["seed"] == 0
    for e in est.values():
        assert e["cvar"] >= e["var"] > 0
        assert e["var_rupees"] == pytest.approx(e["var"] * 1_000_000)
    assert body["window"] == {
        "start": str(window.index[0].date()),
        "end": str(window.index[-1].date()),
        "n_obs": 756,
    }


def test_var_horizon_scales_by_sqrt_time(api: TestClient, fund_mix: dict[str, object]) -> None:
    one = api.post(VAR, json={"portfolio": fund_mix}).json()["estimates"][1]["var"]
    ten = api.post(VAR, json={"portfolio": fund_mix, "horizon_days": 10}).json()
    assert ten["estimates"][1]["var"] == pytest.approx(math.sqrt(10) * one, rel=1e-12)
    assert any("square root" in n for n in ten["notes"])


def test_lookback_longer_than_history_is_clipped(
    api: TestClient, market: MarketData, stock_basket: dict[str, object]
) -> None:
    body = api.post(VAR, json={"portfolio": stock_basket, "lookback_days": 5000}).json()
    assert body["window"]["n_obs"] == len(market.returns)


def test_metrics_default_risk_free_is_the_data_t_bill_rate(
    api: TestClient, market: MarketData, fund_mix: dict[str, object]
) -> None:
    body = api.post(METRICS, json={"portfolio": fund_mix}).json()
    # Synthetic RF is a constant 0.00025 a day, so its geometric annual average is exact.
    assert body["risk_free"]["annual"] == pytest.approx((1 + RF_DAILY) ** 252 - 1, rel=1e-12)
    assert "91-day T-bill" in body["risk_free"]["source"]

    window = market.returns.iloc[-756:]
    port = portfolio_returns(window, weights(fund_mix))
    s = performance_summary(
        port, risk_free_annual=body["risk_free"]["annual"], benchmark=window["index:^NSEI"]
    )
    assert body["sharpe"] == pytest.approx(s.sharpe, rel=1e-12)
    assert body["annualised_volatility"] == pytest.approx(s.annualised_volatility, rel=1e-12)
    assert body["benchmark"]["beta"] == pytest.approx(s.benchmark.beta, rel=1e-12)
    assert body["max_drawdown"] == pytest.approx(max_drawdown(port), rel=1e-12)
    # The drawdown series is <= 0 and its minimum is minus the max drawdown.
    assert max(body["drawdown"]["values"]) <= 0
    assert min(body["drawdown"]["values"]) == pytest.approx(-body["max_drawdown"], rel=1e-12)
    assert body["growth"]["values"][-1] == pytest.approx(float(np.prod(1 + port.to_numpy())))
    assert len(body["growth"]["dates"]) == 756


def test_metrics_risk_free_override(api: TestClient, fund_mix: dict[str, object]) -> None:
    body = api.post(METRICS, json={"portfolio": fund_mix, "risk_free_annual": 0.0}).json()
    assert body["risk_free"] == {"annual": 0.0, "source": "supplied by the caller"}


def test_risk_free_falls_back_to_latest_year_when_window_has_no_factor_rows(
    api: TestClient, fund_mix: dict[str, object]
) -> None:
    # The synthetic factor file ends 2025-12-31: only 44 of the last 252 rows have factor data.
    body = api.post(METRICS, json={"portfolio": fund_mix, "lookback_days": 252}).json()
    assert "latest year is used" in body["risk_free"]["source"]
    assert body["risk_free"]["annual"] == pytest.approx((1 + RF_DAILY) ** 252 - 1, rel=1e-12)


def test_attribution_components_add_up(
    api: TestClient, market: MarketData, fund_mix: dict[str, object]
) -> None:
    body = api.post(ATTR, json={"portfolio": fund_mix, "confidence": 0.99}).json()
    assets = body["assets"]
    assert sum(a["component_volatility"] for a in assets) == pytest.approx(
        body["portfolio_volatility"], rel=1e-12
    )
    assert sum(a["volatility_share"] for a in assets) == pytest.approx(1.0, abs=1e-12)
    assert sum(a["component_var"] for a in assets) == pytest.approx(
        body["parametric_var"], rel=1e-12
    )
    window = market.returns.iloc[-756:]
    vol = volatility_attribution(window, weights(fund_mix))
    assert body["portfolio_volatility"] == pytest.approx(vol.portfolio_volatility, rel=1e-12)
    pv = parametric_var(window, weights(fund_mix), confidence=0.99)
    assert body["parametric_var"] == pytest.approx(pv.var, rel=1e-12)
    assert body["diversification_ratio"] >= 1
    assert assets[0]["name"].startswith("Parag Parikh Flexi Cap")


def test_correlation_matrix_is_symmetric_with_unit_diagonal(
    api: TestClient, stock_basket: dict[str, object]
) -> None:
    body = api.post(CORR, json={"portfolio": stock_basket}).json()
    m = np.array(body["matrix"], dtype=float)
    assert body["asset_ids"][-1] == "index:^NSEI"  # benchmark appended
    assert m.shape == (11, 11)
    np.testing.assert_allclose(m, m.T, atol=1e-15)
    np.testing.assert_allclose(np.diag(m), 1.0)
    no_bench = api.post(CORR, json={"portfolio": stock_basket, "include_benchmark": False})
    assert len(no_bench.json()["asset_ids"]) == 10


@pytest.mark.parametrize("path", [VAR, METRICS, ATTR, CORR])
def test_unknown_asset_is_422(api: TestClient, path: str) -> None:
    body = {"portfolio": {"holdings": [{"asset_id": "equity:NOTREAL", "weight": 1.0}]}}
    r = api.post(path, json=body)
    assert r.status_code == 422
    assert "equity:NOTREAL" in r.json()["detail"]


@pytest.mark.parametrize("path", [VAR, METRICS, ATTR, CORR])
def test_short_weights_are_422(api: TestClient, path: str) -> None:
    holdings = [
        {"asset_id": "equity:TCS", "weight": 1.5},
        {"asset_id": "equity:INFY", "weight": -0.5},
    ]
    assert api.post(path, json={"portfolio": {"holdings": holdings}}).status_code == 422


def test_out_of_range_parameters_are_422(api: TestClient, fund_mix: dict[str, object]) -> None:
    for extra in ({"confidence": 0.5}, {"horizon_days": 30}, {"lookback_days": 100}):
        assert api.post(VAR, json={"portfolio": fund_mix, **extra}).status_code == 422


def test_engine_error_is_422_not_500(api: TestClient) -> None:
    # A portfolio of only the liquid fund is fine; zero volatility would be undefined, but the
    # more realistic engine-level failure is a single asset with no tail at 99.9% on 252 rows:
    # 252 * 0.001 < 1 leaves no full observation in the tail.
    body = {
        "portfolio": {"holdings": [{"asset_id": "equity:TCS", "weight": 1.0}]},
        "confidence": 0.999,
        "lookback_days": 252,
    }
    r = api.post(VAR, json=body)
    assert r.status_code == 422
    assert "no full observation" in r.json()["detail"]
