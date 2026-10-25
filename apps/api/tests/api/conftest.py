"""Fixtures for API tests: synthetic market data for the real universe, SQLite database.

The data is synthetic (seeded RNG), so tests run in CI where data/raw/ does not exist, but it
goes through the real pipeline: Parquet repositories -> align_prices -> align_factors ->
MarketData. Every asset of data/universe.json gets a business-day series from 2015-01-01 to
2026-10-19, so every committed scenario window and sample portfolio is valid.

Returns are built from known factor loadings, r = RF + b_MKT MKT_RF + b_SMB SMB + noise, and
RF is a constant 0.00025 a day, so the default risk-free rate is exactly
1.00025^252 - 1 and the factor betas are known.
"""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import Settings
from app.data.factor_repository import ParquetFactorRepository
from app.data.models import DEFAULT_FACTOR_DATASET, FactorHistory, PriceHistory
from app.data.repository import ParquetRepository
from app.data.scenarios import load_scenarios
from app.data.universe import load_universe
from app.db.base import Base
from app.db.session import get_db
from app.main import create_app
from app.schemas.portfolio import SamplePortfolio
from app.services.market import MarketData, build_market_data
from app.services.universe import sample_portfolios

RF_DAILY = 0.00025
START, END, FACTOR_END = "2015-01-01", "2026-10-19", "2025-12-31"

# category -> (beta MKT, beta SMB, idiosyncratic daily vol, extra daily drift)
LOADINGS: dict[str, tuple[float, float, float, float]] = {
    "large_cap": (0.95, -0.15, 0.003, 0.0),
    "flexi_cap": (0.85, 0.05, 0.004, 0.0),
    "mid_cap": (1.00, 0.25, 0.005, 0.0),
    "small_cap": (1.05, 0.45, 0.006, 0.0),
    "hybrid_balanced_advantage": (0.50, 0.00, 0.002, 0.0),
    "debt_corporate_bond": (0.00, 0.00, 0.0006, 0.0001),
    "debt_gilt": (0.00, 0.00, 0.0010, 0.0001),
    "debt_liquid": (0.00, 0.00, 0.00002, 0.00002),
    "gold": (0.00, 0.00, 0.0090, 0.0002),
}
STOCK = (1.0, -0.20, 0.012, 0.0)
INDEX = (1.0, -0.20, 0.002, 0.0)


def _synthetic(root: Path) -> MarketData:
    universe = load_universe()
    idx = pd.bdate_range(START, END, name="date")
    n = len(idx)
    rng = np.random.default_rng(20261025)
    mkt = rng.normal(0.0003, 0.010, n)
    smb, hml, wml = rng.normal(0.0, 0.004, (3, n))

    repo = ParquetRepository(root)
    loadings = {f.asset().id: LOADINGS[f.category] for f in universe.funds}
    loadings |= {s.asset().id: STOCK for s in universe.stocks}
    loadings[universe.benchmark.asset().id] = INDEX
    for asset in universe.assets():
        b_m, b_s, vol, drift = loadings[asset.id]
        r = RF_DAILY + drift + b_m * mkt + b_s * smb + rng.normal(0.0, vol, n)
        prices = pd.Series(100.0 * np.cumprod(1 + r), index=idx)
        repo.save(PriceHistory(asset=asset, prices=prices, source="synthetic"))

    f = pd.DataFrame(
        {"MKT_RF": mkt, "SMB": smb, "HML": hml, "WML": wml, "RF": RF_DAILY}, index=idx
    ).loc[:FACTOR_END]
    frepo = ParquetFactorRepository(DEFAULT_FACTOR_DATASET, root)
    frepo.save(
        FactorHistory(
            factors=f,
            dataset=DEFAULT_FACTOR_DATASET,
            release="synthetic",
            source="synthetic test factors",
            source_url="https://example.invalid/factors.csv",
            source_file="synthetic.csv",
            raw_csv=b"synthetic",
            units_source="decimal",
            citation="synthetic test data",
        )
    )
    return build_market_data(repo, frepo, universe, load_scenarios(), backend="parquet")


@pytest.fixture(scope="session")
def market(tmp_path_factory: pytest.TempPathFactory) -> MarketData:
    return _synthetic(tmp_path_factory.mktemp("market"))


@pytest.fixture
def sqlite_engine() -> Iterator[Engine]:
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def test_settings() -> Settings:
    return Settings(
        app_env="test",
        warm_market_data=False,
        market_data_backend="parquet",
        jwt_secret="test-secret-not-for-production-use-0123456789",  # type: ignore[arg-type]
        jwt_expire_minutes=30,
    )


@pytest.fixture
def api(market: MarketData, sqlite_engine: Engine, test_settings: Settings) -> Iterator[TestClient]:
    """TestClient on synthetic market data and an in-memory SQLite database."""
    from app.core.config import get_settings

    app = create_app(test_settings, market_loader=lambda: market)
    factory = sessionmaker(bind=sqlite_engine, expire_on_commit=False)

    def _db() -> Iterator[Session]:
        with factory() as session:
            yield session

    app.dependency_overrides[get_db] = _db
    app.dependency_overrides[get_settings] = lambda: test_settings
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def samples() -> dict[str, SamplePortfolio]:
    return {s.id: s for s in sample_portfolios()}


@pytest.fixture(scope="session")
def fund_mix(samples: dict[str, SamplePortfolio]) -> dict[str, object]:
    return samples["sample_fund_mix"].portfolio.model_dump()


@pytest.fixture(scope="session")
def stock_basket(samples: dict[str, SamplePortfolio]) -> dict[str, object]:
    return samples["sample_stock_basket"].portfolio.model_dump()
