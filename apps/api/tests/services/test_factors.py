from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.data.errors import NotInCacheError
from app.data.factor_repository import ParquetFactorRepository
from app.data.models import FACTOR_COLUMNS, FactorHistory
from app.data.repository import ParquetRepository
from app.data.universe import load_universe
from app.engine.returns import simple_returns
from app.services.factors import load_aligned_factors
from app.services.market_data import load_aligned_prices


def seed_factors(repo: ParquetFactorRepository) -> None:
    idx = pd.bdate_range("2025-01-06", periods=10, name="date")
    frame = pd.DataFrame({c: 0.001 * np.arange(1, 11) for c in FACTOR_COLUMNS}, index=idx)
    repo.save(
        FactorHistory(
            factors=frame,
            dataset=repo.dataset,
            release="2025-12",
            source="t",
            source_url="https://example.invalid/f.csv",
            source_file="f.csv",
            raw_csv=b"raw",
            units_source="percent",
            citation="c",
        )
    )


def test_service_aligns_cached_factors_to_the_calendar(tmp_path: Path) -> None:
    repo = ParquetFactorRepository("ff", tmp_path)
    seed_factors(repo)
    calendar = pd.bdate_range("2025-01-08", periods=4, name="date")

    out = load_aligned_factors(repo, calendar)

    assert list(out.factors.index) == list(calendar[1:])
    assert out.factors.loc["2025-01-09", "SMB"] == pytest.approx(0.004)
    assert out.report.rows_with_factors == 3


def test_service_never_fetches_missing_factors(tmp_path: Path) -> None:
    with pytest.raises(NotInCacheError):
        load_aligned_factors(
            ParquetFactorRepository("ff", tmp_path), pd.bdate_range("2025-01-06", periods=3)
        )


# --- real caches (data/raw from scripts/seed_data.py); skipped when not seeded, e.g. in CI ---


def _real() -> tuple[pd.DatetimeIndex, ParquetFactorRepository] | None:
    try:
        universe = load_universe()
    except FileNotFoundError:
        return None
    prices, factors = ParquetRepository(), ParquetFactorRepository("iima_ff4_daily")
    if not factors.exists() or not all(prices.exists(a) for a in universe.assets()):
        return None
    return load_aligned_prices(prices, universe).prices.index, factors


REAL = _real()


@pytest.mark.skipif(REAL is None, reason="real caches not seeded (run scripts/seed_data.py)")
def test_real_factor_alignment() -> None:
    assert REAL is not None
    calendar, repo = REAL
    out = load_aligned_factors(repo, calendar)
    r = out.report
    print("\n" + r.summary())

    assert repo.metadata().release == "2025-12"
    assert r.factor_last == date(2025, 12, 31)
    assert r.rows_before_coverage == 0  # factors start in 1993, prices in 2015
    assert r.rows_with_factors > 2500
    prices = load_aligned_prices(ParquetRepository(), load_universe()).prices
    pd.testing.assert_index_equal(out.factors.index, simple_returns(prices).index)
    valid = out.factors.dropna()
    assert valid.abs().max().max() < 0.2
    assert (valid["RF"] > 0).all()
