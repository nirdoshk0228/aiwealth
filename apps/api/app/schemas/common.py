"""Response building blocks shared by the analytics schemas.

Units: every return, volatility, VaR and weight is a **decimal** (0.0234 = 2.34%); rupee
figures say so in their name (`*_rupees`). Undefined values (a ratio with a zero
denominator) are null, never NaN or infinity, so the JSON is always valid.
"""

from datetime import date

from pydantic import BaseModel, Field


class AnalysisWindow(BaseModel):
    """The return rows an estimate used. Dates are trading days (end inclusive)."""

    start: date
    end: date
    n_obs: int = Field(description="Number of daily return rows.")


class RiskFreeRate(BaseModel):
    annual: float = Field(description="Annual decimal, e.g. 0.065 = 6.5%.")
    source: str = Field(description="Where the rate comes from and the period it averages.")


class TimeSeries(BaseModel):
    dates: list[date]
    values: list[float]


class AssetLabel(BaseModel):
    asset_id: str
    name: str
