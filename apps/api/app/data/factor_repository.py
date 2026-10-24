"""Factor-return cache: the only place request paths read factor data from.

Same rules as the price cache (app/data/repository.py): `FactorRepository` is the interface,
`ParquetFactorRepository` the Phase 2 implementation, reads never call a provider (a miss
raises NotInCacheError) and only the seed script writes.

Layout under `root` (default <DATA_DIR>/raw, gitignored), one directory per dataset:

    factors/<dataset>/factors.parquet       FACTOR_COLUMNS (float64, decimal), index "date"
    factors/<dataset>/meta.json             FactorCacheMeta (release, units, sha256, ...)
    factors/<dataset>/source/<file name>    the downloaded file, byte for byte

The files sit two levels below `root` and the metadata is not named *.meta.json, so the price
cache's `list_cached` (which globs <root>/*/*.meta.json) never picks them up.
"""

import hashlib
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
from pydantic import BaseModel

from app.core.config import get_settings
from app.data.errors import NotInCacheError
from app.data.models import FACTOR_COLUMNS, FactorHistory
from app.data.repository import _atomic_write

UNITS_STORED = "decimal simple returns per row (0.01 = 1%)"


class FactorCacheMeta(BaseModel):
    """Provenance of a cached factor file; readable without loading the data."""

    dataset: str
    source: str
    release: str
    source_url: str
    citation: str
    downloaded_at: datetime
    raw_file: str  # relative to the dataset directory
    raw_sha256: str
    raw_bytes: int
    units_source: str
    units_stored: str
    columns: list[str]
    first_date: date
    last_date: date
    rows: int
    weekend_rows: int
    na_rows: int
    warnings: list[str]


class FactorRepository(Protocol):
    def save(self, history: FactorHistory) -> FactorCacheMeta: ...
    def load(self) -> pd.DataFrame: ...
    def exists(self) -> bool: ...
    def metadata(self) -> FactorCacheMeta: ...


def validate_factors(frame: pd.DataFrame, label: str) -> None:
    """Enforce the FactorHistory.factors invariants before anything is persisted."""
    if tuple(frame.columns) != FACTOR_COLUMNS:
        raise ValueError(f"{label}: columns must be {FACTOR_COLUMNS}, got {tuple(frame.columns)}")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError(f"{label}: index must be a DatetimeIndex")
    if frame.index.tz is not None:
        raise ValueError(f"{label}: index must be tz-naive")
    if frame.empty:
        raise ValueError(f"{label}: frame is empty")
    if not frame.index.is_monotonic_increasing or not frame.index.is_unique:
        raise ValueError(f"{label}: dates must be strictly increasing")
    values = frame.to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError(f"{label}: values must be finite or NaN")


class ParquetFactorRepository:
    """FactorRepository for one dataset, stored as Parquet + JSON metadata + the raw file."""

    def __init__(
        self,
        dataset: str,
        root: Path | None = None,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.dataset = dataset
        self.root = root or get_settings().raw_data_dir
        self._clock = clock or (lambda: datetime.now(UTC))

    @property
    def directory(self) -> Path:
        return self.root / "factors" / self.dataset

    def _data_path(self) -> Path:
        return self.directory / "factors.parquet"

    def _meta_path(self) -> Path:
        return self.directory / "meta.json"

    # --- writes ---

    def save(self, history: FactorHistory) -> FactorCacheMeta:
        """Persist a factor file, replacing the cached version of this dataset.

        Order: raw file, then Parquet, then metadata. `exists` checks the metadata, so an
        interrupted save never leaves an entry that looks complete.
        """
        if history.dataset != self.dataset:
            raise ValueError(f"history is for {history.dataset!r}, repository is {self.dataset!r}")
        frame = history.factors
        validate_factors(frame, history.dataset)
        if Path(history.source_file).name != history.source_file:
            raise ValueError(f"source_file must be a bare file name, got {history.source_file!r}")

        raw_rel = f"source/{history.source_file}"
        out = frame.astype("float64")
        out.index = index = pd.DatetimeIndex(frame.index, name="date")
        meta = FactorCacheMeta(
            dataset=history.dataset,
            source=history.source,
            release=history.release,
            source_url=history.source_url,
            citation=history.citation,
            downloaded_at=self._clock(),
            raw_file=raw_rel,
            raw_sha256=hashlib.sha256(history.raw_csv).hexdigest(),
            raw_bytes=len(history.raw_csv),
            units_source=history.units_source,
            units_stored=UNITS_STORED,
            columns=list(FACTOR_COLUMNS),
            first_date=index[0].date(),
            last_date=index[-1].date(),
            rows=len(frame),
            weekend_rows=int((index.day_of_week >= 5).sum()),
            na_rows=int(frame.isna().any(axis=1).sum()),
            warnings=list(history.warnings),
        )
        _atomic_write(self.directory / raw_rel, lambda p: p.write_bytes(history.raw_csv))
        _atomic_write(self._data_path(), lambda p: out.to_parquet(p, engine="pyarrow"))
        _atomic_write(
            self._meta_path(),
            lambda p: p.write_text(meta.model_dump_json(indent=2), encoding="utf-8"),
        )
        return meta

    # --- reads (never touch the network) ---

    def exists(self) -> bool:
        return self._meta_path().is_file() and self._data_path().is_file()

    def _require(self) -> None:
        if not self.exists():
            raise NotInCacheError(
                f"factors {self.dataset!r} are not cached; run scripts/seed_data.py --kind factors"
            )

    def metadata(self) -> FactorCacheMeta:
        self._require()
        return FactorCacheMeta.model_validate_json(self._meta_path().read_text(encoding="utf-8"))

    def load(self) -> pd.DataFrame:
        """Cached factors: FACTOR_COLUMNS as decimals, tz-naive DatetimeIndex "date"."""
        self._require()
        frame = pd.read_parquet(self._data_path(), engine="pyarrow")
        frame.index = pd.DatetimeIndex(frame.index, name="date")
        return frame[list(FACTOR_COLUMNS)]
