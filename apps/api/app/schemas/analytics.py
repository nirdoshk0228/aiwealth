"""Backtest, factor exposure, universe and market-data status schemas."""

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.common import AnalysisWindow
from app.schemas.portfolio import PortfolioIn

# --- universe and status ---


class AssetOut(BaseModel):
    asset_id: str
    kind: Literal["mutual_fund", "equity", "index"]
    symbol: str
    name: str
    category: str | None = Field(default=None, description="Fund category (funds only).")
    amc: str | None = None
    industry: str | None = Field(default=None, description="NSE industry (stocks only).")
    available: bool = Field(description="Present in the aligned analytics data.")


class UniverseResponse(BaseModel):
    as_of: date = Field(description="Date the universe was selected and verified.")
    benchmark_id: str
    data_start: date
    data_end: date
    n_rows: int = Field(description="Aligned daily price rows.")
    factor_data_end: date | None
    assets: list[AssetOut]


class MarketDataStatus(BaseModel):
    state: Literal["loading", "ready", "error"]
    backend: Literal["parquet", "postgres"]
    loaded_at: datetime | None = None
    data_end: date | None = None
    message: str | None = None


# --- backtest ---


class BacktestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    portfolio: PortfolioIn
    method: Literal["historical", "parametric"] = Field(
        default="historical",
        description="Monte Carlo is excluded on the request path (re-estimated every test day "
        "it takes too long); it is available in the engine and the eval report.",
    )
    confidence: float = Field(default=0.99, ge=0.9, le=0.999)
    window: int = Field(default=504, ge=252, le=1_260, description="Rolling estimation rows.")


class KupiecOut(BaseModel):
    lr_uc: float
    p_value: float


class ChristoffersenOut(BaseModel):
    n00: int
    n01: int
    n10: int
    n11: int
    lr_ind: float | None
    p_ind: float | None
    lr_cc: float | None
    p_cc: float | None


class BacktestResponse(BaseModel):
    method: Literal["historical", "parametric"]
    confidence: float
    window: int
    horizon_days: int
    test_period: AnalysisWindow
    n_breaches: int
    expected_breaches: float
    breach_rate: float
    expected_rate: float
    kupiec: KupiecOut
    christoffersen: ChristoffersenOut
    basel_breaches_last_250: int | None
    basel_zone_last_250: Literal["green", "yellow", "red"] | None
    dates: list[date]
    var: list[float] = Field(description="VaR forecast per test day (positive loss fraction).")
    realised: list[float] = Field(description="Realised portfolio simple return per test day.")
    breach_dates: list[date]
    notes: list[str]


# --- factor exposures ---


class FactorRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    portfolio: PortfolioIn
    include_assets: bool = Field(default=True, description="Also regress each holding.")


class FactorCoefficient(BaseModel):
    factor: str
    beta: float
    std_error: float
    t_stat: float | None
    p_value: float | None


class FactorFit(BaseModel):
    alpha_annual: float = Field(description="252 x daily intercept, arithmetic.")
    alpha_t_stat: float | None
    coefficients: list[FactorCoefficient]
    r_squared: float
    adj_r_squared: float
    residual_vol_annual: float
    n_obs: int
    hac_lags: int
    warnings: list[str]


class AssetFactorRow(BaseModel):
    asset_id: str
    name: str
    alpha_annual: float
    betas: dict[str, float]
    t_stats: dict[str, float | None]
    r_squared: float


class FactorResponse(BaseModel):
    window: AnalysisWindow
    model: str
    source: str = Field(description="Factor data source and citation.")
    portfolio: FactorFit
    assets: list[AssetFactorRow]
    notes: list[str]
