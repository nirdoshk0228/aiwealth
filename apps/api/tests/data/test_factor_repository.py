import hashlib
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.data.errors import NotInCacheError
from app.data.factor_repository import UNITS_STORED, ParquetFactorRepository
from app.data.models import FACTOR_COLUMNS, AssetRef, FactorHistory, PriceHistory
from app.data.repository import ParquetRepository
from app.engine.factors import align

FIXED_NOW = datetime(2026, 10, 25, 9, 0, tzinfo=UTC)
DATASET = "test_ff4"
RAW = b"Date,SMB,HML,WML,MF,RF\r\n(pretend this is the downloaded file)\r\n"


def factor_frame() -> pd.DataFrame:
    # Fri, Sat (special session), Mon, Tue; one NA cell kept as NaN.
    idx = pd.DatetimeIndex(["2025-10-17", "2025-10-18", "2025-10-20", "2025-10-21"], name="date")
    return pd.DataFrame(
        {
            "MKT_RF": [0.0125, -0.0075, 0.005, 0.015],
            "SMB": [0.0025, np.nan, -0.00125, 0.00375],
            "HML": [-0.005, 0.0025, 0.00125, -0.0025],
            "WML": [0.0075, -0.00375, 0.0025, 0.00125],
            "RF": [0.00015, 0.00015, 0.0003, 0.00015],
        },
        index=idx,
    )


def history(frame: pd.DataFrame | None = None, **overrides: object) -> FactorHistory:
    fields: dict[str, object] = {
        "factors": factor_frame() if frame is None else frame,
        "dataset": DATASET,
        "release": "2025-12",
        "source": "test source",
        "source_url": "https://example.invalid/iffm/DATA/file.csv",
        "source_file": "2025-12_test.csv",
        "raw_csv": RAW,
        "units_source": "percent",
        "citation": "Agarwalla, Jacob and Varma (2013)",
        "warnings": ("NA in source on 2025-10-18 (SMB); kept as NaN",),
    }
    fields.update(overrides)
    return FactorHistory(**fields)  # type: ignore[arg-type]


@pytest.fixture
def repo(tmp_path: Path) -> ParquetFactorRepository:
    return ParquetFactorRepository(DATASET, tmp_path, clock=lambda: FIXED_NOW)


def test_engine_and_data_layer_agree_on_factor_columns() -> None:
    assert align.FACTOR_COLUMNS == FACTOR_COLUMNS


def test_default_factor_dataset_matches_the_provider() -> None:
    from app.data.models import DEFAULT_FACTOR_DATASET
    from app.data.providers.iima_factors import DATASET as PROVIDER_DATASET

    assert DEFAULT_FACTOR_DATASET == PROVIDER_DATASET == "iima_ff4_daily"


def test_round_trip_is_exact_including_nan(repo: ParquetFactorRepository) -> None:
    repo.save(history())
    loaded = repo.load()
    pd.testing.assert_frame_equal(loaded, factor_frame(), check_freq=False)
    assert tuple(loaded.columns) == FACTOR_COLUMNS


def test_metadata_records_release_units_and_provenance(repo: ParquetFactorRepository) -> None:
    returned = repo.save(history())
    meta = repo.metadata()

    assert meta == returned
    assert meta.release == "2025-12"
    assert meta.downloaded_at == FIXED_NOW
    assert meta.units_source == "percent"
    assert meta.units_stored == UNITS_STORED
    assert meta.columns == list(FACTOR_COLUMNS)
    assert (meta.first_date, meta.last_date, meta.rows) == (
        date(2025, 10, 17),
        date(2025, 10, 21),
        4,
    )
    assert meta.weekend_rows == 1
    assert meta.na_rows == 1
    assert meta.raw_sha256 == hashlib.sha256(RAW).hexdigest()
    assert meta.raw_bytes == len(RAW)
    assert meta.citation.startswith("Agarwalla")
    assert meta.warnings == ["NA in source on 2025-10-18 (SMB); kept as NaN"]


def test_raw_file_is_kept_byte_for_byte(repo: ParquetFactorRepository) -> None:
    meta = repo.save(history())
    assert (repo.directory / meta.raw_file).read_bytes() == RAW
    assert meta.raw_file == "source/2025-12_test.csv"


def test_layout(repo: ParquetFactorRepository, tmp_path: Path) -> None:
    repo.save(history())
    base = tmp_path / "factors" / DATASET
    assert (base / "factors.parquet").is_file()
    assert (base / "meta.json").is_file()


def test_missing_cache_raises_not_in_cache(repo: ParquetFactorRepository) -> None:
    assert not repo.exists()
    with pytest.raises(NotInCacheError, match="--kind factors"):
        repo.load()
    with pytest.raises(NotInCacheError):
        repo.metadata()


def test_interrupted_save_does_not_look_complete(repo: ParquetFactorRepository) -> None:
    repo.save(history())
    (repo.directory / "meta.json").unlink()  # as if the process died before the last write
    assert not repo.exists()


def test_save_replaces_previous_release(repo: ParquetFactorRepository) -> None:
    repo.save(history())
    newer = factor_frame().iloc[:2]
    repo.save(history(newer, release="2026-06", warnings=()))
    assert repo.metadata().release == "2026-06"
    assert len(repo.load()) == 2


@pytest.mark.parametrize(
    ("frame", "match"),
    [
        (factor_frame()[["SMB", "MKT_RF", "HML", "WML", "RF"]], "columns must be"),
        (factor_frame().iloc[::-1], "strictly increasing"),
        (factor_frame().iloc[:0], "empty"),
        (factor_frame().replace(0.0125, np.inf), "finite"),
    ],
)
def test_save_validates_the_frame(
    repo: ParquetFactorRepository, frame: pd.DataFrame, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        repo.save(history(frame))
    assert not repo.exists()


def test_save_rejects_wrong_dataset_and_path_like_file_names(
    repo: ParquetFactorRepository,
) -> None:
    with pytest.raises(ValueError, match="repository is"):
        repo.save(history(dataset="other"))
    with pytest.raises(ValueError, match="bare file name"):
        repo.save(history(source_file="../escape.csv"))


def test_price_cache_listing_ignores_the_factor_cache(tmp_path: Path) -> None:
    # Both caches share data/raw; ParquetRepository.list_cached globs <root>/*/*.meta.json.
    ParquetFactorRepository(DATASET, tmp_path).save(history())
    prices = ParquetRepository(tmp_path)
    nifty = AssetRef(kind="index", symbol="^NSEI", name="Nifty 50")
    idx = pd.bdate_range("2026-01-05", periods=3, name="date")
    prices.save(PriceHistory(asset=nifty, prices=pd.Series([1.0, 2.0, 3.0], index=idx), source="t"))

    assert [m.asset for m in prices.list_cached()] == [nifty]
