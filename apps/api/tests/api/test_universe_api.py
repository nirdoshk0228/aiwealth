"""Universe, demo portfolios and market-data status (cold start, errors)."""

import threading

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.data.errors import NotInCacheError
from app.data.universe import load_universe
from app.main import create_app
from app.schemas.portfolio import SamplePortfolio
from app.services.market import MarketData, MarketDataStore


def test_universe_lists_all_assets_with_data_window(api: TestClient, market: MarketData) -> None:
    out = api.get("/api/v1/universe").json()
    assert len(out["assets"]) == 36
    assert all(a["available"] for a in out["assets"])
    assert out["benchmark_id"] == "index:^NSEI"
    assert out["data_end"] == "2026-10-19" and out["n_rows"] == len(market.prices)
    assert out["factor_data_end"] == "2025-12-31"
    tcs = next(a for a in out["assets"] if a["asset_id"] == "equity:TCS")
    assert tcs["industry"] == "Information Technology" and tcs["category"] is None


def test_demo_portfolios_use_universe_assets(samples: dict[str, SamplePortfolio]) -> None:
    ids = {a.id for a in load_universe().assets()}
    assert set(samples) == {"sample_fund_mix", "sample_stock_basket"}
    for s in samples.values():
        assert {h.asset_id for h in s.portfolio.holdings} <= ids
        assert "not a recommendation" in s.description.lower()


def test_demo_portfolios_run_through_the_risk_endpoint(api: TestClient) -> None:
    demos = api.get("/api/v1/demo/portfolios").json()
    assert len(demos) == 2
    for d in demos:
        r = api.post("/api/v1/risk/var", json={"portfolio": d["portfolio"]})
        assert r.status_code == 200, d["id"]


def test_status_reports_ready(api: TestClient) -> None:
    api.get("/api/v1/universe")  # forces the load
    assert api.get("/api/v1/status").json()["state"] == "ready"


def test_status_reports_loading_without_blocking(test_settings: Settings) -> None:
    release = threading.Event()

    def slow_loader() -> MarketData:
        release.wait(5)
        raise NotInCacheError("not cached: [...]; run scripts/seed_data.py")

    store = MarketDataStore(slow_loader, backend="parquet")
    thread = store.warm_in_background()
    assert store.state() == ("loading", None)
    release.set()
    thread.join(5)
    state, message = store.state()
    assert state == "error" and "run scripts/seed_data.py" in message


def test_missing_data_is_503_and_driver_errors_are_not_echoed(test_settings: Settings) -> None:
    def broken_db() -> MarketData:
        raise RuntimeError('connection to server at "ep-secret-host" failed')

    app = create_app(test_settings, market_loader=broken_db)
    with TestClient(app) as c:
        r = c.get("/api/v1/universe")
        assert r.status_code == 503
        assert "ep-secret-host" not in r.text
        assert "RuntimeError" in r.json()["detail"]
        status = c.get("/api/v1/status").json()
        assert status["state"] == "error" and "ep-secret-host" not in status["message"]


def test_failed_load_is_retried(market: MarketData) -> None:
    calls = {"n": 0}

    def flaky() -> MarketData:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient")
        return market

    store = MarketDataStore(flaky, backend="parquet")
    with pytest.raises(Exception, match="could not be loaded"):
        store.get()
    assert store.get() is market and store.state() == ("ready", None)
