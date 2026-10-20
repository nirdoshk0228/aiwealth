from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest

from app.data.errors import NotInCacheError
from app.data.models import AssetRef, PriceHistory
from app.data.repository import ParquetRepository

FIXED_NOW = datetime(2026, 10, 21, 6, 30, tzinfo=UTC)
FUND = AssetRef(kind="mutual_fund", symbol="122639", name="PPFAS Flexi Cap")
NIFTY = AssetRef(kind="index", symbol="^NSEI", name="Nifty 50")
MM = AssetRef(kind="equity", symbol="M&M", name="Mahindra & Mahindra")


def prices(values: list[float], start: str = "2026-01-05") -> pd.Series:
    idx = pd.bdate_range(start, periods=len(values), name="date")
    return pd.Series(values, index=idx, dtype=float)


def history(asset: AssetRef, values: list[float], warnings: tuple[str, ...] = ()) -> PriceHistory:
    return PriceHistory(
        asset=asset, prices=prices(values), source="test", source_name="src", warnings=warnings
    )


@pytest.fixture
def repo(tmp_path: Path) -> ParquetRepository:
    return ParquetRepository(tmp_path, clock=lambda: FIXED_NOW)


def test_round_trip_is_exact(repo: ParquetRepository) -> None:
    original = prices([100.0, 101.25, 99.875, 102.0])
    repo.save(PriceHistory(asset=FUND, prices=original, source="test"))

    loaded = repo.load(FUND)

    pd.testing.assert_series_equal(loaded, original.rename(FUND.id), check_freq=False)
    assert loaded.dtype == "float64"
    assert loaded.index.tz is None


def test_metadata_records_source_range_and_warnings(repo: ParquetRepository) -> None:
    repo.save(history(FUND, [10.0, 11.0, 12.0], warnings=("dropped 1 row",)))
    meta = repo.metadata(FUND)
    assert meta.asset == FUND
    assert meta.source == "test" and meta.source_name == "src"
    assert meta.fetched_at == FIXED_NOW
    assert (meta.first_date, meta.last_date) == (date(2026, 1, 5), date(2026, 1, 7))
    assert meta.rows == 3
    assert meta.warnings == ["dropped 1 row"]


def test_save_overwrites_previous_version(repo: ParquetRepository) -> None:
    repo.save(history(FUND, [1.0, 2.0]))
    repo.save(history(FUND, [5.0, 6.0, 7.0]))
    assert repo.load(FUND).tolist() == [5.0, 6.0, 7.0]
    assert repo.metadata(FUND).rows == 3


def test_missing_asset_raises_not_in_cache(repo: ParquetRepository) -> None:
    assert not repo.exists(FUND)
    with pytest.raises(NotInCacheError, match="seed_data"):
        repo.load(FUND)
    with pytest.raises(NotInCacheError):
        repo.metadata(FUND)


def test_symbols_are_encoded_to_safe_unique_filenames(repo: ParquetRepository) -> None:
    repo.save(history(NIFTY, [1.0, 2.0]))
    repo.save(history(MM, [3.0, 4.0]))
    files = sorted(p.relative_to(repo.root).as_posix() for p in repo.root.rglob("*") if p.is_file())
    assert files == [
        "equity/M%26M.meta.json",
        "equity/M%26M.parquet",
        "index/%5ENSEI.meta.json",
        "index/%5ENSEI.parquet",
    ]
    assert repo.load(MM).tolist() == [3.0, 4.0]


def test_symbol_with_dot_is_not_truncated(repo: ParquetRepository) -> None:
    a = AssetRef(kind="equity", symbol="ABC.X", name="a")
    b = AssetRef(kind="equity", symbol="ABC.Y", name="b")
    repo.save(history(a, [1.0, 2.0]))
    repo.save(history(b, [3.0, 4.0]))
    assert repo.load(a).tolist() == [1.0, 2.0]
    assert repo.load(b).tolist() == [3.0, 4.0]


def test_no_temp_files_left_behind(repo: ParquetRepository) -> None:
    repo.save(history(FUND, [1.0, 2.0]))
    assert not list(repo.root.rglob("*.tmp"))


def test_interrupted_save_does_not_look_complete(repo: ParquetRepository) -> None:
    # Simulate a crash after the data file but before the metadata file was written.
    repo.save(history(FUND, [1.0, 2.0]))
    meta_file = repo.root / "mutual_fund" / "122639.meta.json"
    meta_file.unlink()
    assert not repo.exists(FUND)
    with pytest.raises(NotInCacheError):
        repo.load(FUND)


@pytest.mark.parametrize(
    "bad",
    [
        pd.Series([1.0, 0.0], index=pd.bdate_range("2026-01-05", periods=2)),
        pd.Series([1.0, float("nan")], index=pd.bdate_range("2026-01-05", periods=2)),
        pd.Series([1.0, 2.0], index=pd.DatetimeIndex(["2026-01-06", "2026-01-05"])),
        pd.Series([1.0, 2.0], index=pd.DatetimeIndex(["2026-01-05", "2026-01-05"])),
        pd.Series([1.0], index=pd.DatetimeIndex(["2026-01-05"], tz="Asia/Kolkata")),
        pd.Series([], dtype=float, index=pd.DatetimeIndex([])),
    ],
    ids=["zero", "nan", "unsorted", "duplicate", "tz-aware", "empty"],
)
def test_invalid_series_rejected_and_nothing_written(
    repo: ParquetRepository, bad: pd.Series
) -> None:
    with pytest.raises(ValueError):
        repo.save(PriceHistory(asset=FUND, prices=bad, source="test"))
    assert not repo.exists(FUND)


def test_load_many_names_every_missing_asset(repo: ParquetRepository) -> None:
    repo.save(history(FUND, [1.0, 2.0]))
    with pytest.raises(NotInCacheError) as exc:
        repo.load_many([FUND, NIFTY, MM])
    assert NIFTY.id in str(exc.value) and MM.id in str(exc.value)
    assert FUND.id not in str(exc.value)


def test_load_many_and_list_cached(repo: ParquetRepository) -> None:
    repo.save(history(FUND, [1.0, 2.0]))
    repo.save(history(NIFTY, [3.0, 4.0]))
    loaded = repo.load_many([FUND, NIFTY])
    assert set(loaded) == {FUND.id, NIFTY.id}
    assert {m.asset for m in repo.list_cached()} == {FUND, NIFTY}


def test_empty_cache_lists_nothing(repo: ParquetRepository) -> None:
    assert repo.list_cached() == []
