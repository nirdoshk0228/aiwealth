"""Stress endpoints. P&L is **signed** (-0.15 = lose 15%), unlike VaR (positive loss).

Custom scenarios reuse `HypotheticalScenarioSpec` from app/data/scenarios.py, the same schema
as the committed scenario files and the AI scenario parser, so there is one definition.
"""

from datetime import date
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.data.scenarios import DirectShockSpec, HypotheticalScenarioSpec
from app.schemas.portfolio import PortfolioIn


class ScenarioOut(BaseModel):
    id: str
    name: str
    type: Literal["historical", "hypothetical"]
    description: str
    rationale: str
    caveats: list[str]
    start: date | None = None
    end: date | None = None
    factor_shocks: dict[str, float] = Field(default_factory=dict)
    direct_shocks: list[DirectShockSpec] = Field(default_factory=list)


class StressRequest(BaseModel):
    """Exactly one of `scenario_id` (a committed scenario) or `scenario` (a custom one)."""

    model_config = ConfigDict(extra="forbid")

    portfolio: PortfolioIn
    scenario_id: str | None = None
    scenario: HypotheticalScenarioSpec | None = None

    @model_validator(mode="after")
    def _one_scenario(self) -> Self:
        if (self.scenario_id is None) == (self.scenario is None):
            raise ValueError("give exactly one of scenario_id or scenario")
        return self


class AssetImpactOut(BaseModel):
    asset_id: str
    name: str
    weight: float
    asset_return: float = Field(description="Simple return applied to the asset.")
    contribution: float = Field(description="weight x asset_return; sums to pnl.")
    contribution_rupees: float
    source: str = Field(description="historical, direct_*, factor or unshocked.")
    factor_implied: float | None = None
    r_squared: float | None = None


class StressResponse(BaseModel):
    scenario_id: str
    scenario_name: str
    scenario_type: Literal["historical", "hypothetical"]
    portfolio_value: float
    pnl: float = Field(description="Signed fraction of portfolio value.")
    pnl_rupees: float
    impacts: list[AssetImpactOut]
    worst_contributors: list[str] = Field(description="Asset ids, most negative first.")
    notes: list[str]
    warnings: list[str]
    start: date | None = None
    end: date | None = None
    sessions: int | None = None
    max_drawdown: float | None = Field(default=None, description="In-window, positive.")
    drawdown_peak_date: date | None = None
    drawdown_trough_date: date | None = None
