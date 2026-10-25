"""SQL (Postgres) implementations of SeriesRepository and FactorRepository.

Same contract as the Parquet versions: reads never call a provider (a miss raises
NotInCacheError), and a save replaces the whole series in one transaction, so a reader never
sees half of a series. Written against generic SQLAlchemy, so unit tests run them on SQLite in
memory and integration tests on Neon Postgres.

The Parquet cache stays the seed source: `copy_parquet_to_sql` (used by scripts/seed_db.py)
copies every cached series and the factor returns, keeping the original `fetched_at` and
provenance. The raw IIMA file is never copied (no stated licence; docs/data.md).
"""

import hashlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from sqlalchemy import Engine, delete, insert, select
from sqlalchemy.orm import Session

from app.data.errors import NotInCacheError
from app.data.factor_repository import (
    UNITS_STORED,
    FactorCacheMeta,
    ParquetFactorRepository,
    validate_factors,
)
from app.data.models import FACTOR_COLUMNS, AssetRef, FactorHistory, PriceHistory
from app.data.repository import CacheMeta, ParquetRepository, validate_prices
from app.db.models import FactorDataset, FactorReturn, MarketAsset, MarketPrice

_FACTOR_ATTR = {"MKT_RF": "mkt_rf", "SMB": "smb", "HML": "hml", "WML": "wml", "RF": "rf"}
_BATCH = 5_000


def _date_index(dates: pd.Series) -> pd.DatetimeIndex:
    """SQL DATE values -> tz-naive DatetimeIndex "date" in microseconds (pandas' default unit;
    the unit carries no meaning, the Parquet cache itself mixes ms and us)."""
    return pd.DatetimeIndex(pd.to_datetime(dates), name="date").as_unit("us")


def _batches[T](rows: list[T], size: int = _BATCH) -> Iterable[list[T]]:
    for i in range(0, len(rows), size):
        yield rows[i : i + size]


class SqlSeriesRepository:
    """SeriesRepository on the market_assets / market_prices tables."""

    def __init__(self, engine: Engine, *, clock: Callable[[], datetime] | None = None) -> None:
        self._engine = engine
        self._clock = clock or (lambda: datetime.now(UTC))

    # --- writes ---

    def save(self, history: PriceHistory) -> CacheMeta:
        validate_prices(history.prices, history.asset.id)
        prices = history.prices
        meta = CacheMeta(
            asset=history.asset,
            source=history.source,
            source_name=history.source_name,
            fetched_at=self._clock(),
            first_date=prices.index[0].date(),
            last_date=prices.index[-1].date(),
            rows=len(prices),
            warnings=list(history.warnings),
        )
        self.import_series(meta, prices)
        return meta

    def import_series(self, meta: CacheMeta, prices: pd.Series) -> None:
        """Replace one asset's metadata and prices atomically (used by the Parquet copy)."""
        asset = meta.asset
        validate_prices(prices, asset.id)
        if len(prices) != meta.rows:
            raise ValueError(f"{asset.id}: metadata says {meta.rows} rows, got {len(prices)}")
        rows = [
            {"asset_id": asset.id, "date": d.date(), "price": float(p)}
            for d, p in zip(prices.index, prices.to_numpy(dtype=float), strict=True)
        ]
        fetched_utc = meta.fetched_at.astimezone(UTC)
        with Session(self._engine) as session, session.begin():
            session.execute(delete(MarketPrice).where(MarketPrice.asset_id == asset.id))
            session.execute(delete(MarketAsset).where(MarketAsset.asset_id == asset.id))
            session.add(
                MarketAsset(
                    asset_id=asset.id,
                    kind=asset.kind,
                    symbol=asset.symbol,
                    name=asset.name,
                    source=meta.source,
                    source_name=meta.source_name,
                    fetched_at=fetched_utc,
                    first_date=meta.first_date,
                    last_date=meta.last_date,
                    rows=meta.rows,
                    warnings=list(meta.warnings),
                )
            )
            session.flush()
            for batch in _batches(rows):
                session.execute(insert(MarketPrice), batch)

    # --- reads (never touch the network) ---

    @staticmethod
    def _to_meta(row: MarketAsset) -> CacheMeta:
        fetched = row.fetched_at
        if fetched.tzinfo is None:  # SQLite drops the zone; values are always stored as UTC
            fetched = fetched.replace(tzinfo=UTC)
        return CacheMeta(
            asset=AssetRef(kind=row.kind, symbol=row.symbol, name=row.name),  # type: ignore[arg-type]
            source=row.source,
            source_name=row.source_name,
            fetched_at=fetched,
            first_date=row.first_date,
            last_date=row.last_date,
            rows=row.rows,
            warnings=list(row.warnings),
        )

    def exists(self, asset: AssetRef) -> bool:
        with Session(self._engine) as session:
            return session.get(MarketAsset, asset.id) is not None

    def metadata(self, asset: AssetRef) -> CacheMeta:
        with Session(self._engine) as session:
            row = session.get(MarketAsset, asset.id)
            if row is None:
                raise NotInCacheError(f"{asset.id} is not in the database; run scripts/seed_db.py")
            return self._to_meta(row)

    def load(self, asset: AssetRef) -> pd.Series:
        return self.load_many([asset])[asset.id]

    def load_many(self, assets: Iterable[AssetRef]) -> dict[str, pd.Series]:
        """One query for all assets; if any are missing, raise once naming all of them."""
        assets = list(assets)
        ids = [a.id for a in assets]
        with Session(self._engine) as session:
            present = set(
                session.scalars(select(MarketAsset.asset_id).where(MarketAsset.asset_id.in_(ids)))
            )
            missing = [i for i in ids if i not in present]
            if missing:
                raise NotInCacheError(f"not in the database: {missing}; run scripts/seed_db.py")
            result = session.execute(
                select(MarketPrice.asset_id, MarketPrice.date, MarketPrice.price)
                .where(MarketPrice.asset_id.in_(ids))
                .order_by(MarketPrice.asset_id, MarketPrice.date)
            ).all()
        frame = pd.DataFrame(result, columns=["asset_id", "date", "price"])
        out: dict[str, pd.Series] = {}
        for asset_id in ids:
            part = frame[frame["asset_id"] == asset_id]
            out[asset_id] = pd.Series(
                part["price"].to_numpy(dtype="float64"),
                index=_date_index(part["date"]),
                name=asset_id,
            )
        return out

    def list_cached(self) -> list[CacheMeta]:
        with Session(self._engine) as session:
            rows = session.scalars(select(MarketAsset).order_by(MarketAsset.asset_id)).all()
            return [self._to_meta(r) for r in rows]


class SqlFactorRepository:
    """FactorRepository for one dataset on the factor_datasets / factor_returns tables."""

    def __init__(
        self, engine: Engine, dataset: str, *, clock: Callable[[], datetime] | None = None
    ) -> None:
        self._engine = engine
        self.dataset = dataset
        self._clock = clock or (lambda: datetime.now(UTC))

    def save(self, history: FactorHistory) -> FactorCacheMeta:
        """Store factor returns and provenance. `raw_file` is recorded as not stored."""
        if history.dataset != self.dataset:
            raise ValueError(f"history is for {history.dataset!r}, repository is {self.dataset!r}")
        frame = history.factors
        validate_factors(frame, history.dataset)
        index = pd.DatetimeIndex(frame.index)
        meta = FactorCacheMeta(
            dataset=history.dataset,
            source=history.source,
            release=history.release,
            source_url=history.source_url,
            citation=history.citation,
            downloaded_at=self._clock(),
            raw_file=f"(not stored in the database) {history.source_file}",
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
        self.import_factors(meta, frame)
        return meta

    def import_factors(self, meta: FactorCacheMeta, frame: pd.DataFrame) -> None:
        """Replace the dataset's provenance and returns atomically."""
        if meta.dataset != self.dataset:
            raise ValueError(f"meta is for {meta.dataset!r}, repository is {self.dataset!r}")
        validate_factors(frame, self.dataset)
        values = frame[list(FACTOR_COLUMNS)].to_numpy(dtype=float)
        rows = []
        for d, vals in zip(pd.DatetimeIndex(frame.index), values, strict=True):
            row: dict[str, object] = {"dataset": self.dataset, "date": d.date()}
            for col, v in zip(FACTOR_COLUMNS, vals, strict=True):
                row[_FACTOR_ATTR[col]] = None if np.isnan(v) else float(v)
            rows.append(row)
        with Session(self._engine) as session, session.begin():
            session.execute(delete(FactorReturn).where(FactorReturn.dataset == self.dataset))
            session.execute(delete(FactorDataset).where(FactorDataset.dataset == self.dataset))
            session.add(FactorDataset(dataset=self.dataset, meta=meta.model_dump(mode="json")))
            session.flush()
            for batch in _batches(rows):
                session.execute(insert(FactorReturn), batch)

    def exists(self) -> bool:
        with Session(self._engine) as session:
            return session.get(FactorDataset, self.dataset) is not None

    def metadata(self) -> FactorCacheMeta:
        with Session(self._engine) as session:
            row = session.get(FactorDataset, self.dataset)
            if row is None:
                raise NotInCacheError(
                    f"factors {self.dataset!r} are not in the database; run scripts/seed_db.py"
                )
            return FactorCacheMeta.model_validate(row.meta)

    def load(self) -> pd.DataFrame:
        """FACTOR_COLUMNS as decimals (NaN where the source had NA), DatetimeIndex "date"."""
        if not self.exists():
            raise NotInCacheError(
                f"factors {self.dataset!r} are not in the database; run scripts/seed_db.py"
            )
        cols = [getattr(FactorReturn, _FACTOR_ATTR[c]) for c in FACTOR_COLUMNS]
        with Session(self._engine) as session:
            result = session.execute(
                select(FactorReturn.date, *cols)
                .where(FactorReturn.dataset == self.dataset)
                .order_by(FactorReturn.date)
            ).all()
        frame = pd.DataFrame(result, columns=["date", *FACTOR_COLUMNS])
        out = frame[list(FACTOR_COLUMNS)].astype("float64")
        out.index = _date_index(frame["date"])
        return out


# --- seeding from the Parquet cache ---


@dataclass(frozen=True)
class CopyReport:
    assets: int
    price_rows: int
    factor_rows: int


def copy_parquet_to_sql(
    source: ParquetRepository,
    target: SqlSeriesRepository,
    factor_source: ParquetFactorRepository | None = None,
    factor_target: SqlFactorRepository | None = None,
    *,
    progress: Callable[[str], None] | None = None,
) -> CopyReport:
    """Copy every cached series (and optionally the factor returns) into SQL, unchanged.

    Metadata (fetched_at, warnings, provenance) is copied as-is, so the database states when
    and where the data was originally downloaded, not when it was copied.
    """
    say = progress or (lambda _msg: None)
    n_rows = 0
    metas = source.list_cached()
    for meta in metas:
        series = source.load(meta.asset)
        target.import_series(meta, series)
        n_rows += len(series)
        say(f"{meta.asset.id}: {len(series)} rows")
    factor_rows = 0
    if factor_source is not None and factor_target is not None and factor_source.exists():
        fmeta = factor_source.metadata()
        frame = factor_source.load()
        db_meta = fmeta.model_copy(
            update={"raw_file": f"(not stored in the database) {fmeta.raw_file}"}
        )
        factor_target.import_factors(db_meta, frame)
        factor_rows = len(frame)
        say(f"factors {fmeta.dataset} (release {fmeta.release}): {factor_rows} rows")
    return CopyReport(assets=len(metas), price_rows=n_rows, factor_rows=factor_rows)
