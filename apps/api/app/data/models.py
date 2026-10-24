"""Core data-layer types shared by providers, the cache and the seed script."""

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict

AssetKind = Literal["mutual_fund", "equity", "index"]


class AssetRef(BaseModel):
    """Identifies one asset in the universe.

    `symbol` meaning depends on `kind`:
      - mutual_fund: AMFI scheme code as used by mfapi.in, e.g. "122639"
      - equity:      NSE symbol without suffix, e.g. "RELIANCE" (".NS" is added for Yahoo)
      - index:       Yahoo index symbol, e.g. "^NSEI"
    """

    model_config = ConfigDict(frozen=True)

    kind: AssetKind
    symbol: str
    name: str

    @property
    def id(self) -> str:
        """Stable unique key, e.g. "mutual_fund:122639" or "index:^NSEI"."""
        return f"{self.kind}:{self.symbol}"


@dataclass(frozen=True)
class PriceHistory:
    """A cleaned price series as returned by a provider.

    `prices` invariants (enforced by each provider's parser):
      - DatetimeIndex of tz-naive calendar dates (midnight), ascending, no duplicates
      - float values, all finite and > 0
      - for mutual funds: NAV per unit; for equities/indices: split- and
        dividend-adjusted close (yfinance auto_adjust=True)
    """

    asset: AssetRef
    prices: pd.Series
    source: str
    # Name as reported by the source (e.g. mfapi scheme_name); used to verify the universe.
    source_name: str | None = None
    # Non-fatal data-quality notes (dropped rows, duplicates) for the seed script to report.
    warnings: tuple[str, ...] = field(default_factory=tuple)


# Daily factor returns, in this column order everywhere in the data layer. The engine keeps
# its own copy (it may not import app.data); tests/data/test_factor_repository.py checks both
# are equal.
FACTOR_COLUMNS = ("MKT_RF", "SMB", "HML", "WML", "RF")


@dataclass(frozen=True)
class FactorHistory:
    """A parsed daily factor file as returned by a factor provider.

    `factors` invariants (enforced by the provider's parser):
      - DatetimeIndex "date" of tz-naive calendar dates, strictly increasing
      - columns exactly FACTOR_COLUMNS, float64, **decimal simple returns** (0.01 = 1%)
      - each row's return covers the period since the previous row of the file (so a
        Monday RF covers Saturday and Sunday too)
      - NaN only where the source file has "NA"; such rows are kept, not dropped, so that
        compounding across them propagates NaN instead of silently skipping a day
    """

    factors: pd.DataFrame
    dataset: str  # cache key, e.g. "iima_ff4_daily"
    release: str  # source release, e.g. "2025-12"
    source: str
    source_url: str  # final URL after redirects
    source_file: str  # original file name, used for the raw copy in the cache
    raw_csv: bytes  # the downloaded file, byte for byte
    units_source: str
    citation: str
    warnings: tuple[str, ...] = field(default_factory=tuple)
