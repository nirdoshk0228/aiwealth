from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.data.errors import NotInCacheError
from app.data.models import PriceHistory
from app.data.repository import ParquetRepository
from app.data.universe import Universe, load_universe
from app.engine.returns import simple_returns
from app.services.market_data import load_aligned_prices

SMALL_UNIVERSE = Universe.model_validate(
    {
        "as_of": "2026-10-21",
        "data_start": "2025-01-08",
        "benchmark": {"symbol": "^NSEI", "name": "Nifty 50"},
        "funds": [{"code": "122639", "name": "PPFAS", "category": "flexi_cap", "amc": "PPFAS"}],
        "stocks": [
            {"symbol": "TCS", "name": "TCS", "industry": "IT", "isin": "INE467B01029"},
            {"symbol": "INFY", "name": "Infosys", "industry": "IT", "isin": "INE009A01021"},
        ],
    }
)


def seed(repo: ParquetRepository, universe: Universe, n: int = 20) -> None:
    idx = pd.bdate_range("2025-01-06", periods=n, name="date")
    for i, asset in enumerate(universe.assets()):
        prices = pd.Series(100.0 + i + np.arange(n), index=idx)
        repo.save(PriceHistory(asset=asset, prices=prices, source="test"))


def test_service_wires_universe_into_engine(tmp_path: Path) -> None:
    repo = ParquetRepository(tmp_path)
    seed(repo, SMALL_UNIVERSE)

    out = load_aligned_prices(repo, SMALL_UNIVERSE, min_obs=5)

    assert list(out.prices.columns) == [
        "mutual_fund:122639",
        "equity:TCS",
        "equity:INFY",
        "index:^NSEI",
    ]
    assert out.report.calendar_ids == ("equity:TCS", "equity:INFY")
    assert out.report.benchmark_id == "index:^NSEI"
    assert out.prices.index[0] == pd.Timestamp("2025-01-08")  # universe.data_start
    assert out.prices.shape == (18, 4)


def test_service_never_fetches_missing_data(tmp_path: Path) -> None:
    with pytest.raises(NotInCacheError):
        load_aligned_prices(ParquetRepository(tmp_path), SMALL_UNIVERSE, min_obs=5)


# --- real cache (data/raw from scripts/seed_data.py); skipped when not seeded, e.g. in CI ---


def _real_cache() -> tuple[ParquetRepository, Universe] | None:
    try:
        universe = load_universe()
    except FileNotFoundError:
        return None
    repo = ParquetRepository()
    if not all(repo.exists(a) for a in universe.assets()):
        return None
    return repo, universe


REAL = _real_cache()


@pytest.mark.skipif(REAL is None, reason="real cache not seeded (run scripts/seed_data.py)")
def test_real_cache_alignment() -> None:
    assert REAL is not None
    repo, universe = REAL
    out = load_aligned_prices(repo, universe)
    r = out.report
    print("\n" + r.summary())

    assert not out.prices.isna().any().any()
    assert r.common_start >= date(2015, 1, 1)
    assert r.dropped_assets == ()
    assert out.prices.shape[1] == 36
    assert r.final_rows > 2500
    # Known Yahoo gaps (docs/data.md): a real session missing from ^NSEI, and a real session
    # where every stock has a phantom bar.
    assert date(2016, 1, 1) in r.calendar_only_days
    assert date(2025, 3, 18) in r.benchmark_only_days
    returns = simple_returns(out.prices)
    assert returns.abs().max().max() < 0.5  # no split/demerger-sized jumps left
