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
