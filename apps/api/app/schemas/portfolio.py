"""Portfolio input: the one shape every analytics endpoint (and the AI tools) accepts.

Product rules enforced here, not in the engine (the engine accepts shorts, see
docs/methodology.md):
- long-only: every weight in [0, 1];
- each asset at most once;
- weights sum to 1 within WEIGHT_INPUT_TOL, then are renormalised so they sum to 1 exactly
  (the engine's own tolerance is 1e-8; a UI converting rupee amounts to weights lands well
  inside 1e-6, while a typo such as 0.6 + 0.6 does not).
Whether each asset id exists in the loaded market data is checked by the service, because
the schema cannot know the universe.
"""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

WEIGHT_INPUT_TOL = 1e-6
MAX_HOLDINGS = 40
DEFAULT_PORTFOLIO_VALUE = 1_000_000.0  # rupees, Rs 10 lakh

AssetId = str


class Holding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset_id: AssetId = Field(
        pattern=r"^(mutual_fund|equity|index):\S+$",
        description='Universe asset id, e.g. "mutual_fund:122639" or "equity:TCS".',
        examples=["mutual_fund:122639"],
    )
    weight: float = Field(
        ge=0.0, le=1.0, description="Share of portfolio value, decimal (0.25 = 25%). Long-only."
    )


class PortfolioIn(BaseModel):
    """Holdings as weights plus the portfolio's rupee value (for rupee P&L and VaR)."""

    model_config = ConfigDict(extra="forbid")

    holdings: list[Holding] = Field(min_length=1, max_length=MAX_HOLDINGS)
    portfolio_value: float = Field(
        default=DEFAULT_PORTFOLIO_VALUE,
        gt=0,
        le=1e12,
        description="Current portfolio value in rupees.",
    )

    @model_validator(mode="after")
    def _weights_valid(self) -> Self:
        ids = [h.asset_id for h in self.holdings]
        dups = sorted({i for i in ids if ids.count(i) > 1})
        if dups:
            raise ValueError(f"each asset may appear once; duplicated: {dups}")
        total = sum(h.weight for h in self.holdings)
        if abs(total - 1.0) > WEIGHT_INPUT_TOL:
            raise ValueError(f"weights must sum to 1 (100%), got {total:.6f}")
        # Renormalise so the engine's 1e-8 check always passes.
        self.holdings = [
            Holding(asset_id=h.asset_id, weight=h.weight / total) for h in self.holdings
        ]
        return self

    def weights(self) -> dict[str, float]:
        """asset id -> weight, in input order."""
        return {h.asset_id: h.weight for h in self.holdings}


class SamplePortfolio(BaseModel):
    """A sample portfolio shipped in data/portfolios/ for the no-login demo and tests."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9_]+$")
    name: str
    description: str
    portfolio: PortfolioIn
