"""The investable universe: which funds, stocks and benchmark the platform covers.

Stored in data/universe.json (committed). Every entry was verified against its source before
being added (see docs/data.md): fund codes via mfapi.in search + full history, stock symbols
against the official Nifty 50 constituent list.
"""

from datetime import date
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.core.config import get_settings
from app.data.models import AssetRef

FundCategory = Literal[
    "large_cap",
    "flexi_cap",
    "mid_cap",
    "small_cap",
    "debt_corporate_bond",
    "debt_gilt",
    "debt_liquid",
    "gold",
    "hybrid_balanced_advantage",
]


class _Strict(BaseModel):
    # Typos in universe.json (e.g. "sybmol") must fail loudly, not be ignored.
    model_config = ConfigDict(extra="forbid", frozen=True)


class FundEntry(_Strict):
    code: str = Field(pattern=r"^\d+$", description="AMFI scheme code as used by mfapi.in")
    name: str = Field(description="Scheme name exactly as mfapi.in returns it")
    category: FundCategory
    amc: str

    def asset(self) -> AssetRef:
        return AssetRef(kind="mutual_fund", symbol=self.code, name=self.name)


class StockEntry(_Strict):
    symbol: str = Field(pattern=r"^[A-Z0-9&-]+$", description="NSE symbol without .NS")
    name: str = Field(description="Company name from the official Nifty 50 list")
    industry: str = Field(description="NSE industry classification")
    isin: str = Field(pattern=r"^INE[A-Z0-9]{9}$")

    def asset(self) -> AssetRef:
        return AssetRef(kind="equity", symbol=self.symbol, name=self.name)


class BenchmarkEntry(_Strict):
    symbol: str = Field(pattern=r"^\^")
    name: str

    def asset(self) -> AssetRef:
        return AssetRef(kind="index", symbol=self.symbol, name=self.name)


class Universe(_Strict):
    as_of: date = Field(description="Date the universe was selected and verified")
    data_start: date = Field(description="Earliest date used for analytics (cache keeps more)")
    benchmark: BenchmarkEntry
    funds: list[FundEntry]
    stocks: list[StockEntry]
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique(self) -> Self:
        for label, keys in (
            ("fund code", [f.code for f in self.funds]),
            ("stock symbol", [s.symbol for s in self.stocks]),
        ):
            dups = sorted({k for k in keys if keys.count(k) > 1})
            if dups:
                raise ValueError(f"duplicate {label}(s): {dups}")
        return self

    def assets(self) -> list[AssetRef]:
        """All assets to seed: funds, then stocks, then the benchmark."""
        return (
            [f.asset() for f in self.funds]
            + [s.asset() for s in self.stocks]
            + [self.benchmark.asset()]
        )

    def find(self, key: str) -> AssetRef:
        """Look up an asset by scheme code or symbol (case-insensitive for symbols)."""
        for a in self.assets():
            if a.symbol.upper() == key.upper():
                return a
        raise KeyError(f"{key!r} is not in the universe")


def load_universe(path: Path | None = None) -> Universe:
    """Load and validate universe.json (default: <DATA_DIR>/universe.json)."""
    path = path or get_settings().universe_path
    return Universe.model_validate_json(path.read_text(encoding="utf-8"))
