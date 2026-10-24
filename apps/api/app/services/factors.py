"""Cached factor returns -> factor returns on the master calendar, for the engine.

Reads only from the factor repository (never from a provider) and hands pandas objects to the
pure engine, like app/services/market_data.py does for prices.
"""

import pandas as pd

from app.data.factor_repository import FactorRepository
from app.engine.factors.align import AlignedFactors, align_factors


def load_aligned_factors(repo: FactorRepository, calendar: pd.DatetimeIndex) -> AlignedFactors:
    """Load the cached factors and align them to a price calendar.

    `calendar` is the price calendar, normally `load_aligned_prices(...).prices.index`; the
    result's rows match `simple_returns(prices)`. Raises NotInCacheError if the factors are
    not seeded (run scripts/seed_data.py --kind factors).
    """
    return align_factors(repo.load(), calendar)
