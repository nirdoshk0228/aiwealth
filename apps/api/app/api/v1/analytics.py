"""Analytics routes: risk, backtest, factor exposures, stress. Public (no login): the
portfolio travels in the request body, which is what makes the no-login demo work."""

from fastapi import APIRouter

from app.api.deps import MarketDataDep
from app.schemas.analytics import BacktestRequest, BacktestResponse, FactorRequest, FactorResponse
from app.schemas.risk import (
    AttributionRequest,
    AttributionResponse,
    CorrelationRequest,
    CorrelationResponse,
    MetricsRequest,
    MetricsResponse,
    VaRRequest,
    VaRResponse,
)
from app.schemas.stress import ScenarioOut, StressRequest, StressResponse
from app.services import backtest, factor_report, risk, stress_report

router = APIRouter()


@router.post("/risk/var", response_model=VaRResponse, tags=["risk"])
def value_at_risk(body: VaRRequest, md: MarketDataDep) -> VaRResponse:
    """Historical, parametric and Monte Carlo VaR/CVaR (positive loss fractions and rupees)."""
    return risk.var_report(md, body)


@router.post("/risk/metrics", response_model=MetricsResponse, tags=["risk"])
def metrics(body: MetricsRequest, md: MarketDataDep) -> MetricsResponse:
    """Return, volatility, Sharpe/Sortino, drawdowns and Nifty 50 relative statistics."""
    return risk.metrics_report(md, body)


@router.post("/risk/attribution", response_model=AttributionResponse, tags=["risk"])
def attribution(body: AttributionRequest, md: MarketDataDep) -> AttributionResponse:
    """Each holding's contribution to volatility and parametric VaR/CVaR (Euler)."""
    return risk.attribution_report(md, body)


@router.post("/risk/correlation", response_model=CorrelationResponse, tags=["risk"])
def correlation(body: CorrelationRequest, md: MarketDataDep) -> CorrelationResponse:
    """Correlation matrix of the holdings' daily returns (plus the benchmark)."""
    return risk.correlation_report(md, body)


@router.post("/backtest", response_model=BacktestResponse, tags=["risk"])
def var_backtest(body: BacktestRequest, md: MarketDataDep) -> BacktestResponse:
    """Rolling 1-day VaR backtest with Kupiec, Christoffersen and Basel traffic light."""
    return backtest.backtest_report(md, body)


@router.post("/factors/exposures", response_model=FactorResponse, tags=["factors"])
def factor_exposures(body: FactorRequest, md: MarketDataDep) -> FactorResponse:
    """Carhart four-factor betas with Newey-West t-statistics."""
    return factor_report.factor_report(md, body)


@router.get("/stress/scenarios", response_model=list[ScenarioOut], tags=["stress"])
def scenarios(md: MarketDataDep) -> list[ScenarioOut]:
    """Committed historical replays and hypothetical shocks."""
    return stress_report.list_scenarios(md)


@router.post("/stress/run", response_model=StressResponse, tags=["stress"])
def run_stress(body: StressRequest, md: MarketDataDep) -> StressResponse:
    """Run a committed scenario (scenario_id) or a custom hypothetical one (scenario)."""
    return stress_report.run_stress(md, body)
