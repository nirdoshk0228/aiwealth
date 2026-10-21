"""Cached market data -> aligned prices for the engine.

Reads only from the repository (never from providers), then hands pandas objects to the
pure engine. This keeps app/engine free of any I/O or app.data imports.
"""

from app.data.repository import SeriesRepository
from app.data.universe import Universe
from app.engine.returns import (
    DEFAULT_MAX_FFILL_DAYS,
    DEFAULT_MIN_CALENDAR_COVERAGE,
    DEFAULT_MIN_OBS,
    AlignedPrices,
    align_prices,
)


def load_aligned_prices(
    repo: SeriesRepository,
    universe: Universe,
    *,
    max_ffill_days: int = DEFAULT_MAX_FFILL_DAYS,
    min_obs: int = DEFAULT_MIN_OBS,
    min_coverage: float = DEFAULT_MIN_CALENDAR_COVERAGE,
) -> AlignedPrices:
    """Load every universe asset from the cache and align it.

    Calendar assets = the universe's stocks; benchmark = universe.benchmark (^NSEI);
    window starts at universe.data_start. Columns are asset ids ("mutual_fund:120586",
    "equity:TCS", "index:^NSEI"). Raises NotInCacheError if anything is not seeded.
    """
    prices = repo.load_many(universe.assets())
    return align_prices(
        prices,
        calendar_ids=[s.asset().id for s in universe.stocks],
        benchmark_id=universe.benchmark.asset().id,
        start=universe.data_start,
        max_ffill_days=max_ffill_days,
        min_obs=min_obs,
        min_coverage=min_coverage,
    )
