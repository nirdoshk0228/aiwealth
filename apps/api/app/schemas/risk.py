"""Risk endpoints: VaR/CVaR, performance metrics, attribution, correlation.

Conventions (docs/methodology.md): simple daily returns, 252-day annualisation, fixed weights
rebalanced daily. VaR and CVaR are **positive loss fractions** with confidence and horizon
stated; max drawdown is positive too.
"""

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.common import AnalysisWindow, RiskFreeRate, TimeSeries
from app.schemas.portfolio import PortfolioIn

DEFAULT_LOOKBACK_DAYS = 756  # about three years of trading days
MIN_LOOKBACK_DAYS = 252
MAX_LOOKBACK_DAYS = 5_000  # more than the whole cache (about 2,900 rows); clipped to the data

VaRMethod = Literal["historical", "parametric", "monte_carlo"]


class _Request(BaseModel):
    model_config = ConfigDict(extra="forbid")

    portfolio: PortfolioIn
    lookback_days: int = Field(
        default=DEFAULT_LOOKBACK_DAYS,
        ge=MIN_LOOKBACK_DAYS,
        le=MAX_LOOKBACK_DAYS,
        description="Most recent daily return rows to use; clipped to the available history.",
    )


class VaRRequest(_Request):
    confidence: float = Field(default=0.95, ge=0.9, le=0.999)
    horizon_days: int = Field(default=1, ge=1, le=20, description="Square-root-of-time scaled.")


class VaREstimate(BaseModel):
    method: VaRMethod
    var: float = Field(description="Positive loss fraction of portfolio value.")
    cvar: float = Field(description="Expected loss beyond VaR, same units; >= var.")
    var_rupees: float
    cvar_rupees: float
    n_sims: int | None = None
    seed: int | None = None


class VaRResponse(BaseModel):
    confidence: float
    horizon_days: int
    portfolio_value: float
    window: AnalysisWindow
    estimates: list[VaREstimate]
    notes: list[str]


class MetricsRequest(_Request):
    risk_free_annual: float | None = Field(
        default=None,
        gt=-0.5,
        lt=0.5,
        description="Annual decimal. Default: average 91-day T-bill (IIMA RF) over the window.",
    )


class DrawdownEpisodeOut(BaseModel):
    depth: float = Field(description="Positive: 0.2 = 20% below the peak at the trough.")
    peak_date: date | None
    trough_date: date
    recovery_date: date | None
    decline_days: int
    recovery_days: int | None
    duration_days: int
    recovered: bool


class BenchmarkStatsOut(BaseModel):
    benchmark_id: str
    beta: float | None
    alpha: float | None = Field(description="Jensen's alpha, annual, arithmetic.")
    correlation: float | None
    r_squared: float | None
    residual_volatility: float | None
    tracking_error: float
    active_return: float
    information_ratio: float | None


class MetricsResponse(BaseModel):
    window: AnalysisWindow
    risk_free: RiskFreeRate
    annualised_return: float = Field(description="CAGR (geometric).")
    arithmetic_annual_return: float
    annualised_volatility: float
    downside_deviation: float
    sharpe: float | None
    sortino: float | None
    max_drawdown: float
    max_drawdown_episode: DrawdownEpisodeOut | None
    longest_drawdown: DrawdownEpisodeOut | None
    n_drawdown_episodes: int
    benchmark: BenchmarkStatsOut | None
    growth: TimeSeries = Field(description="Value of 1 rupee invested at the window start.")
    benchmark_growth: TimeSeries | None
    drawdown: TimeSeries = Field(description="Drawdown from the running peak, <= 0.")
    notes: list[str]


class AttributionRequest(_Request):
    confidence: float = Field(default=0.95, ge=0.9, le=0.999)
    horizon_days: int = Field(default=1, ge=1, le=20)


class AssetAttribution(BaseModel):
    asset_id: str
    name: str
    weight: float
    marginal_volatility: float = Field(description="d sigma_p / d w_i, annualised.")
    component_volatility: float = Field(description="Sums to portfolio_volatility.")
    volatility_share: float = Field(description="Sums to 1; can be < 0 for a hedge.")
    component_var: float = Field(description="Parametric; sums to parametric_var.")
    component_cvar: float
    var_share: float | None
    risk_to_weight: float | None = Field(description="volatility_share / weight; None if w = 0.")


class AttributionResponse(BaseModel):
    window: AnalysisWindow
    confidence: float
    horizon_days: int
    portfolio_volatility: float = Field(description="Annualised.")
    parametric_var: float
    parametric_cvar: float
    diversification_ratio: float | None
    assets: list[AssetAttribution]
    notes: list[str]


class CorrelationRequest(_Request):
    include_benchmark: bool = True


class CorrelationResponse(BaseModel):
    window: AnalysisWindow
    asset_ids: list[str]
    names: list[str]
    matrix: list[list[float | None]] = Field(description="Pearson, daily simple returns.")
