"""Price-series cache: the only place request paths read market data from.

`SeriesRepository` is the interface; `ParquetRepository` is the Phase 2 implementation
(a Postgres-backed one replaces it in Phase 4 without touching callers). Repositories never
call providers: a cache miss raises NotInCacheError, so a user request can never trigger a
network fetch (CLAUDE.md rule 4). Only the seed/refresh script writes.

Parquet layout under `root` (default <DATA_DIR>/raw, gitignored):

    <kind>/<encoded symbol>.parquet    one column "price" (float64), DatetimeIndex "date"
    <kind>/<encoded symbol>.meta.json  CacheMeta (source, fetched_at, date range, warnings)

Symbols are percent-encoded so every symbol maps to a valid, unique Windows/Unix filename
("^NSEI" -> "%5ENSEI", "M&M" -> "M%26M"); the mapping is reversible (urllib.parse.unquote).
"""

import os
import tempfile
from collections.abc import Callable, Iterable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Protocol
from urllib.parse import quote

import numpy as np
import pandas as pd
from pydantic import BaseModel

from app.core.config import get_settings
from app.data.errors import NotInCacheError
from app.data.models import AssetRef, PriceHistory


class CacheMeta(BaseModel):
    """What was cached, from where and when; readable without loading the series."""

    asset: AssetRef
    source: str
    source_name: str | None
    fetched_at: datetime
    first_date: date
    last_date: date
    rows: int
    warnings: list[str]


class SeriesRepository(Protocol):
    def save(self, history: PriceHistory) -> CacheMeta: ...
    def load(self, asset: AssetRef) -> pd.Series: ...
    def load_many(self, assets: Iterable[AssetRef]) -> dict[str, pd.Series]: ...
    def exists(self, asset: AssetRef) -> bool: ...
    def metadata(self, asset: AssetRef) -> CacheMeta: ...
    def list_cached(self) -> list[CacheMeta]: ...


def validate_prices(prices: pd.Series, label: str) -> None:
    """Enforce the PriceHistory.prices invariants before anything is persisted."""
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError(f"{label}: index must be a DatetimeIndex")
    if prices.index.tz is not None:
        raise ValueError(f"{label}: index must be tz-naive")
    if prices.empty:
        raise ValueError(f"{label}: series is empty")
    if not prices.index.is_monotonic_increasing or not prices.index.is_unique:
        raise ValueError(f"{label}: dates must be strictly increasing")
    values = prices.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError(f"{label}: prices must be finite and > 0")


def _atomic_write(path: Path, write: Callable[[Path], None]) -> None:
    """Write via a temp file in the same directory, then os.replace (atomic on one volume)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


class ParquetRepository:
    """SeriesRepository backed by one Parquet file (+ JSON metadata) per asset."""

    def __init__(
        self, root: Path | None = None, *, clock: Callable[[], datetime] | None = None
    ) -> None:
        self.root = root or get_settings().raw_data_dir
        self._clock = clock or (lambda: datetime.now(UTC))

    # --- paths ---

    def _path(self, asset: AssetRef, suffix: str) -> Path:
        # String concat, not with_suffix(): a symbol containing "." must not be truncated.
        return self.root / asset.kind / (quote(asset.symbol, safe="") + suffix)

    def _data_path(self, asset: AssetRef) -> Path:
        return self._path(asset, ".parquet")

    def _meta_path(self, asset: AssetRef) -> Path:
        return self._path(asset, ".meta.json")

    # --- writes ---

    def save(self, history: PriceHistory) -> CacheMeta:
        """Persist a history, replacing any previous version of the same asset.

        Data is written before metadata, and `exists` checks the metadata file, so an
        interrupted save never leaves an entry that looks complete.
        """
        asset, prices = history.asset, history.prices
        validate_prices(prices, asset.id)
        frame = pd.DataFrame({"price": prices.to_numpy(dtype="float64")}, index=prices.index)
        frame.index.name = "date"
        meta = CacheMeta(
            asset=asset,
            source=history.source,
            source_name=history.source_name,
            fetched_at=self._clock(),
            first_date=prices.index[0].date(),
            last_date=prices.index[-1].date(),
            rows=len(prices),
            warnings=list(history.warnings),
        )
        _atomic_write(self._data_path(asset), lambda p: frame.to_parquet(p, engine="pyarrow"))
        _atomic_write(
            self._meta_path(asset),
            lambda p: p.write_text(meta.model_dump_json(indent=2), encoding="utf-8"),
        )
        return meta

    # --- reads (never touch the network) ---

    def exists(self, asset: AssetRef) -> bool:
        return self._meta_path(asset).is_file() and self._data_path(asset).is_file()

    def metadata(self, asset: AssetRef) -> CacheMeta:
        if not self.exists(asset):
            raise NotInCacheError(f"{asset.id} is not cached; run scripts/seed_data.py")
        return CacheMeta.model_validate_json(self._meta_path(asset).read_text(encoding="utf-8"))

    def load(self, asset: AssetRef) -> pd.Series:
        """Cached prices as a float Series named `asset.id`, indexed by tz-naive dates."""
        if not self.exists(asset):
            raise NotInCacheError(f"{asset.id} is not cached; run scripts/seed_data.py")
        frame = pd.read_parquet(self._data_path(asset), engine="pyarrow")
        series = frame["price"].rename(asset.id)
        series.index = pd.DatetimeIndex(series.index, name="date")
        return series

    def load_many(self, assets: Iterable[AssetRef]) -> dict[str, pd.Series]:
        """Load several assets; if any are missing, raise once naming all of them."""
        assets = list(assets)
        missing = [a.id for a in assets if not self.exists(a)]
        if missing:
            raise NotInCacheError(f"not cached: {missing}; run scripts/seed_data.py")
        return {a.id: self.load(a) for a in assets}

    def list_cached(self) -> list[CacheMeta]:
        metas = []
        for meta_file in sorted(self.root.glob("*/*.meta.json")):
            meta = CacheMeta.model_validate_json(meta_file.read_text(encoding="utf-8"))
            if self.exists(meta.asset):
                metas.append(meta)
        return metas
