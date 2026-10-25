"""Market data for request paths: loaded once per process from the cache, then shared.

`MarketData` bundles everything the analytics services need: the universe, aligned prices,
simple returns, aligned factors (if seeded) and the scenario files. It is built by
`build_market_data` from a SeriesRepository and a FactorRepository, so the same code runs on
the Parquet cache (local, tests) and on Postgres (deployed). Nothing here calls a provider
(CLAUDE.md rule 4): a missing series raises NotInCacheError.

`MarketDataStore` holds one MarketData per process. It loads lazily under a lock (or in a
background thread at startup) and reports its state, so the UI can show a cold-start message
instead of a timeout.
"""

import logging
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Literal

import pandas as pd

from app.core.config import Settings, get_settings
from app.data.errors import NotInCacheError
from app.data.factor_repository import FactorCacheMeta, FactorRepository, ParquetFactorRepository
from app.data.models import DEFAULT_FACTOR_DATASET
from app.data.repository import ParquetRepository, SeriesRepository
from app.data.scenarios import (
    HistoricalScenarioSpec,
    HypotheticalScenarioSpec,
    load_scenarios,
)
from app.data.universe import Universe, load_universe
from app.engine.factors.align import AlignedFactors
from app.engine.returns import AlignmentReport, simple_returns
from app.engine.stress.scenarios import AssetExposure, AssetGroups
from app.services.errors import AnalyticsInputError, MarketDataUnavailable
from app.services.factors import load_aligned_factors
from app.services.market_data import load_aligned_prices
from app.services.stress import asset_groups, estimate_exposures

logger = logging.getLogger(__name__)

Backend = Literal["parquet", "postgres"]
ScenarioSpec = HistoricalScenarioSpec | HypotheticalScenarioSpec


@dataclass
class MarketData:
    universe: Universe
    prices: pd.DataFrame  # aligned, NaN-free, columns = asset ids
    alignment: AlignmentReport
    factors: AlignedFactors | None  # None if the factor cache is not seeded
    factor_meta: FactorCacheMeta | None
    scenarios: Mapping[str, ScenarioSpec]
    backend: Backend
    loaded_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    returns: pd.DataFrame = field(init=False)
    names: dict[str, str] = field(init=False)  # asset id -> name
    groups: dict[str, AssetGroups] = field(init=False)  # asset id -> sector/category
    _exposures: dict[str, AssetExposure] | None = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        self.returns = simple_returns(self.prices)
        self.names = {a.id: a.name for a in self.universe.assets()}
        self.groups = asset_groups(self.universe)

    # --- convenience ---

    @property
    def benchmark_id(self) -> str:
        return self.universe.benchmark.asset().id

    @property
    def data_end(self) -> date:
        return self.prices.index[-1].date()

    @property
    def cache_key(self) -> str:
        """Identifies this load; part of every result-cache key."""
        return self.loaded_at.isoformat()

    def name(self, asset_id: str) -> str:
        return self.names.get(asset_id, asset_id)

    def check_assets(self, asset_ids: list[str]) -> None:
        """Every id must be a column of the aligned data (in the universe and not dropped)."""
        unknown = [a for a in asset_ids if a not in self.prices.columns]
        if unknown:
            raise AnalyticsInputError(
                f"assets not available for analytics: {unknown}; see GET /api/v1/universe"
            )

    def window(self, lookback_days: int) -> pd.DataFrame:
        """The last `lookback_days` return rows (clipped to the available history)."""
        return self.returns.iloc[-lookback_days:]

    def require_factors(self) -> AlignedFactors:
        if self.factors is None:
            raise MarketDataUnavailable(
                "factor data is not loaded; seed it with scripts/seed_data.py --kind factors"
            )
        return self.factors

    def exposures(self) -> dict[str, AssetExposure]:
        """Carhart betas and R^2 of every asset, computed once (36 regressions) and kept."""
        with self._lock:
            if self._exposures is None:
                factors = self.require_factors().factors
                self._exposures = estimate_exposures(self.returns, factors)
            return self._exposures


def build_market_data(
    series_repo: SeriesRepository,
    factor_repo: FactorRepository | None,
    universe: Universe,
    scenarios: Mapping[str, ScenarioSpec],
    *,
    backend: Backend,
) -> MarketData:
    """Load and align everything from the repositories (no network)."""
    aligned = load_aligned_prices(series_repo, universe)
    factors = meta = None
    if factor_repo is not None and factor_repo.exists():
        factors = load_aligned_factors(factor_repo, aligned.prices.index)
        meta = factor_repo.metadata()
    else:
        logger.warning("factor data not found; factor exposures and factor stress are disabled")
    return MarketData(
        universe=universe,
        prices=aligned.prices,
        alignment=aligned.report,
        factors=factors,
        factor_meta=meta,
        scenarios=scenarios,
        backend=backend,
    )


def load_market_data(settings: Settings | None = None) -> MarketData:
    """Build MarketData from the configured backend (Parquet cache or Postgres)."""
    settings = settings or get_settings()
    backend = settings.resolved_market_data_backend()
    universe = load_universe(settings.universe_path)
    scenarios = load_scenarios(settings.scenarios_dir)
    if backend == "postgres":
        from app.data.sql_repository import SqlFactorRepository, SqlSeriesRepository
        from app.db.session import get_engine

        engine = get_engine()
        series_repo: SeriesRepository = SqlSeriesRepository(engine)
        factor_repo: FactorRepository = SqlFactorRepository(engine, DEFAULT_FACTOR_DATASET)
    else:
        series_repo = ParquetRepository(settings.raw_data_dir)
        factor_repo = ParquetFactorRepository(DEFAULT_FACTOR_DATASET, settings.raw_data_dir)
    return build_market_data(series_repo, factor_repo, universe, scenarios, backend=backend)


State = Literal["loading", "ready", "error"]


def _public_reason(exc: Exception) -> str:
    """A reason safe to show a client: our own cache messages, otherwise only the error type
    (a driver error can contain the database host; the full error is in the server log)."""
    if isinstance(exc, NotInCacheError):
        return str(exc)
    return f"market data could not be loaded ({type(exc).__name__}); see the server log"


class MarketDataStore:
    """One MarketData per process, loaded lazily and thread-safely.

    `get()` blocks until the data is loaded (or raises MarketDataUnavailable with the reason);
    a failed load is retried on the next call, so a transient database error does not stick.
    """

    def __init__(self, loader: Callable[[], MarketData], backend: Backend) -> None:
        self._loader = loader
        self.backend = backend
        self._data: MarketData | None = None
        self._error: str | None = None
        self._loading = False
        self._lock = threading.Lock()

    def get(self) -> MarketData:
        if self._data is not None:
            return self._data
        with self._lock:
            if self._data is None:
                self._loading = True
                try:
                    self._data = self._loader()
                    self._error = None
                except Exception as exc:  # reported to the client as 503, logged in full
                    logger.exception("market data load failed")
                    self._error = _public_reason(exc)
                    raise MarketDataUnavailable(self._error) from exc
                finally:
                    self._loading = False
            return self._data

    def set(self, data: MarketData) -> None:
        """Install already-built data (tests, or a reload)."""
        with self._lock:
            self._data, self._error = data, None

    def warm_in_background(self) -> threading.Thread:
        def run() -> None:
            try:
                self.get()
            except MarketDataUnavailable:
                pass  # already logged; state() reports it

        thread = threading.Thread(target=run, name="market-data-warmup", daemon=True)
        self._loading = True
        thread.start()
        return thread

    def state(self) -> tuple[State, str | None]:
        if self._data is not None:
            return "ready", None
        if self._loading:
            return "loading", None
        if self._error is not None:
            return "error", self._error
        return "loading", None

    @property
    def data(self) -> MarketData | None:
        return self._data
