"""SQL repositories: same contract as the Parquet ones, on SQLite (unit) and Postgres (marked).

The Postgres tests run in a throwaway schema (see the `pg_engine` fixture) and are skipped
when DATABASE_URL is not set, e.g. in CI.
"""

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import Engine, create_engine, func, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.data.errors import NotInCacheError
from app.data.factor_repository import ParquetFactorRepository
from app.data.models import AssetRef, FactorHistory, PriceHistory
from app.data.repository import ParquetRepository
from app.data.sql_repository import SqlFactorRepository, SqlSeriesRepository, copy_parquet_to_sql
from app.db.base import Base
from app.db.models import FactorReturn, MarketPrice

NOW = datetime(2026, 10, 25, 9, 30, tzinfo=UTC)
TCS = AssetRef(kind="equity", symbol="TCS", name="Tata Consultancy Services")
MM = AssetRef(kind="equity", symbol="M&M", name="Mahindra & Mahindra")
NSEI = AssetRef(kind="index", symbol="^NSEI", name="Nifty 50")


def prices(n: int, start: float = 100.0) -> pd.Series:
    idx = pd.bdate_range("2025-01-06", periods=n, name="date")
    return pd.Series(start + np.arange(n) * 0.5, index=idx)


def factor_history() -> FactorHistory:
    idx = pd.DatetimeIndex(["2025-10-17", "2025-10-18", "2025-10-20"], name="date")
    frame = pd.DataFrame(
        {
            "MKT_RF": [0.0125, -0.0075, 0.005],
            "SMB": [0.0025, np.nan, -0.00125],
            "HML": [-0.005, 0.0025, 0.00125],
            "WML": [0.0075, -0.00375, 0.0025],
            "RF": [0.00015, 0.00015, 0.0003],
        },
        index=idx,
    )
    return FactorHistory(
        factors=frame,
        dataset="test_ff4",
        release="2025-12",
        source="test source",
        source_url="https://example.invalid/f.csv",
        source_file="f.csv",
        raw_csv=b"raw bytes",
        units_source="percent",
        citation="test citation",
        warnings=("NA kept as NaN",),
    )


@pytest.fixture
def sqlite() -> Iterator[Engine]:
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


def check_series_contract(engine: Engine) -> None:
    repo = SqlSeriesRepository(engine, clock=lambda: NOW)
    assert not repo.exists(TCS)
    with pytest.raises(NotInCacheError, match="equity:TCS"):
        repo.load(TCS)

    meta = repo.save(PriceHistory(TCS, prices(5), source="yfinance", warnings=("w1",)))
    assert meta.rows == 5 and meta.fetched_at == NOW
    repo.save(PriceHistory(MM, prices(3, 50.0), source="yfinance"))
    assert repo.exists(TCS)
    assert repo.metadata(TCS) == meta

    loaded = repo.load(TCS)
    pd.testing.assert_series_equal(loaded, prices(5).rename("equity:TCS"), check_freq=False)
    assert loaded.index.name == "date" and loaded.dtype == np.float64

    # A second save replaces the series completely (fewer rows, new values).
    repo.save(PriceHistory(TCS, prices(2, 200.0), source="yfinance"))
    assert repo.load(TCS).tolist() == [200.0, 200.5]
    with Session(engine) as s:
        n = s.scalar(select(func.count()).select_from(MarketPrice))
    assert n == 2 + 3

    many = repo.load_many([MM, TCS])
    assert list(many) == ["equity:M&M", "equity:TCS"]
    with pytest.raises(NotInCacheError, match=r"index:\^NSEI"):
        repo.load_many([TCS, NSEI])
    assert [m.asset.id for m in repo.list_cached()] == ["equity:M&M", "equity:TCS"]


def check_factor_contract(engine: Engine) -> None:
    repo = SqlFactorRepository(engine, "test_ff4", clock=lambda: NOW)
    assert not repo.exists()
    with pytest.raises(NotInCacheError):
        repo.load()
    meta = repo.save(factor_history())
    assert meta.raw_file.startswith("(not stored in the database)")
    assert meta.na_rows == 1 and meta.weekend_rows == 1
    out = repo.load()
    pd.testing.assert_frame_equal(out, factor_history().factors, check_freq=False)
    assert np.isnan(out.loc["2025-10-18", "SMB"])  # NA survives as NaN (NULL in SQL)
    assert repo.metadata() == meta
    with pytest.raises(ValueError, match="history is for"):
        SqlFactorRepository(engine, "other").save(factor_history())


def test_series_contract_sqlite(sqlite: Engine) -> None:
    check_series_contract(sqlite)


def test_factor_contract_sqlite(sqlite: Engine) -> None:
    check_factor_contract(sqlite)


def test_invalid_prices_are_rejected_before_any_write(sqlite: Engine) -> None:
    repo = SqlSeriesRepository(sqlite)
    bad = prices(3)
    bad.iloc[1] = -1.0
    with pytest.raises(ValueError, match="finite and > 0"):
        repo.save(PriceHistory(TCS, bad, source="x"))
    assert not repo.exists(TCS)


def test_copy_from_parquet_keeps_provenance(sqlite: Engine, tmp_path: Path) -> None:
    parquet = ParquetRepository(tmp_path, clock=lambda: NOW)
    parquet.save(PriceHistory(TCS, prices(4), source="yfinance", warnings=("phantom bar",)))
    parquet.save(PriceHistory(NSEI, prices(4, 22000.0), source="yfinance"))
    pfactors = ParquetFactorRepository("test_ff4", tmp_path, clock=lambda: NOW)
    pfactors.save(factor_history())

    target = SqlSeriesRepository(sqlite)
    ftarget = SqlFactorRepository(sqlite, "test_ff4")
    report = copy_parquet_to_sql(parquet, target, pfactors, ftarget)

    assert (report.assets, report.price_rows, report.factor_rows) == (2, 8, 3)
    assert target.metadata(TCS) == parquet.metadata(TCS)  # fetched_at, warnings unchanged
    pd.testing.assert_series_equal(target.load(NSEI), parquet.load(NSEI))
    pd.testing.assert_frame_equal(ftarget.load(), pfactors.load(), check_freq=False)
    fmeta = ftarget.metadata()
    assert fmeta.raw_sha256 == pfactors.metadata().raw_sha256
    assert fmeta.raw_file == "(not stored in the database) source/f.csv"
    with Session(sqlite) as s:
        assert s.scalar(select(func.count()).select_from(FactorReturn)) == 3


# --- the same contracts on real Postgres ---


@pytest.mark.postgres
def test_series_contract_postgres(pg_engine: Engine) -> None:
    check_series_contract(pg_engine)


@pytest.mark.postgres
def test_factor_contract_postgres(pg_engine: Engine) -> None:
    check_factor_contract(pg_engine)
