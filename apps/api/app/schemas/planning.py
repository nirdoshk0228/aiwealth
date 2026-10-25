"""Optimizer and goal-planner schemas. Decimals for rates and weights, rupees for amounts."""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.common import AnalysisWindow, RiskFreeRate
from app.schemas.portfolio import AssetId, PortfolioIn

# --- optimizer ---


class OptimizeRequest(BaseModel):
    """Assets to allocate across: `asset_ids`, or the holdings of `portfolio` if omitted.
    `portfolio` (optional) is also evaluated under the same estimates for comparison."""

    model_config = ConfigDict(extra="forbid")

    asset_ids: list[AssetId] | None = Field(default=None, min_length=2, max_length=40)
    portfolio: PortfolioIn | None = None
    max_weight: float = Field(default=1.0, ge=0.05, le=1.0, description="Cap per asset.")
    lookback_days: int = Field(default=1260, ge=252, le=5000, description="Default 5 years.")
    risk_free_annual: float | None = Field(default=None, gt=-0.5, lt=0.5)
    include_liquid: bool = Field(
        default=False,
        description="Include liquid funds. Off by default: they are treated as cash, because "
        "their near-zero volatility makes them dominate minimum-variance, maximum-Sharpe and "
        "risk-parity portfolios.",
    )

    @model_validator(mode="after")
    def _assets_given(self) -> Self:
        if self.asset_ids is None and self.portfolio is None:
            raise ValueError("give asset_ids or a portfolio")
        if self.asset_ids is not None and len(set(self.asset_ids)) != len(self.asset_ids):
            raise ValueError("asset_ids must be unique")
        return self


class WeightOut(BaseModel):
    asset_id: str
    name: str
    weight: float


class OptimizedPortfolio(BaseModel):
    label: str
    expected_return: float = Field(description="Annual, historical arithmetic mean of the window.")
    volatility: float = Field(description="Annual.")
    sharpe: float | None
    weights: list[WeightOut] = Field(description="Non-zero weights, largest first.")
    risk_shares: list[float] | None = Field(
        default=None, description="Risk parity only: each weight's share of volatility."
    )


class FrontierPoint(BaseModel):
    expected_return: float
    volatility: float
    sharpe: float | None


class AssetEstimate(BaseModel):
    asset_id: str
    name: str
    expected_return: float
    volatility: float


class OptimizeResponse(BaseModel):
    window: AnalysisWindow
    risk_free: RiskFreeRate
    max_weight: float
    assets: list[AssetEstimate]
    frontier: list[FrontierPoint]
    min_variance: OptimizedPortfolio
    max_sharpe: OptimizedPortfolio
    max_return: OptimizedPortfolio
    risk_parity: OptimizedPortfolio
    current: OptimizedPortfolio | None
    warnings: list[str]
    notes: list[str]


# --- goal planner ---


class GoalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    portfolio: PortfolioIn = Field(description="Allocation whose history calibrates the model.")
    lookback_days: int = Field(default=5000, ge=252, le=5000, description="Default: all data.")
    initial_amount: float = Field(default=0.0, ge=0, le=1e11, description="Rupees today.")
    monthly_sip: float = Field(ge=0, le=1e9, description="Rupees per month in year 1.")
    step_up_annual: float = Field(default=0.0, ge=0, le=0.5)
    years: int = Field(ge=1, le=40)
    goal_amount: float = Field(gt=0, le=1e12, description="Target in today's rupees.")
    inflation_annual: float = Field(
        default=0.04,
        gt=-0.05,
        lt=0.5,
        description="Default 4%, the RBI's CPI inflation target; change it to your assumption.",
    )
    n_paths: int = Field(default=10_000, ge=1_000, le=50_000)
    seed: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _invests(self) -> Self:
        if self.initial_amount == 0 and self.monthly_sip == 0:
            raise ValueError("enter an initial amount or a monthly SIP")
        return self


class GbmCalibration(BaseModel):
    window: AnalysisWindow
    log_drift: float = Field(description="m: annual mean of daily log returns x 252.")
    volatility: float = Field(description="s: annual sd of daily log returns.")
    median_growth: float = Field(description="e^m - 1: annual growth of the median outcome.")
    mean_growth: float = Field(description="e^(m + s^2/2) - 1: annual expected growth.")


class PercentileValue(BaseModel):
    percentile: int
    nominal: float
    real: float = Field(description="In today's rupees (deflated by the inflation assumption).")


class GoalFan(BaseModel):
    years: list[int]
    p5: list[float]
    p25: list[float]
    p50: list[float]
    p75: list[float]
    p95: list[float]
    invested: list[float]


class GoalResponse(BaseModel):
    calibration: GbmCalibration
    goal_nominal: float
    total_invested: float
    probability: float = Field(description="Share of simulated paths reaching the goal.")
    probability_se: float
    percentiles: list[PercentileValue]
    fan: GoalFan
    n_paths: int
    seed: int
    notes: list[str]


class OptionOut(BaseModel):
    id: str
    label: str


class QuestionOut(BaseModel):
    id: str
    text: str
    options: list[OptionOut]


class BandOut(BaseModel):
    id: str
    label: str
    equity_min: float
    equity_max: float
    description: str


class QuestionnaireOut(BaseModel):
    questions: list[QuestionOut]
    bands: list[BandOut]
    disclaimer: str


class ProfileRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answers: dict[str, str] = Field(description="Question id -> option id.")
    portfolio: PortfolioIn | None = None


class AllocationMix(BaseModel):
    equity: float
    hybrid: float
    debt: float
    gold: float
    equity_oriented: float = Field(description="equity + half of hybrid (stated assumption).")


class ProfileResponse(BaseModel):
    score: int
    max_score: int
    band: BandOut
    capped_by_horizon: bool
    portfolio_mix: AllocationMix | None
    within_band: bool | None
    disclaimer: str
    notes: list[str]
