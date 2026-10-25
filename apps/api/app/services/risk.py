"""Risk service: portfolio + window -> engine (var, metrics, attribution) -> response schemas.

Every function takes the loaded MarketData and a validated request; nothing here does I/O.
The window is the last `lookback_days` rows of the aligned simple returns.
"""

import pandas as pd

from app.engine.risk.attribution import (
    diversification_ratio,
    parametric_var_attribution,
    volatility_attribution,
)
from app.engine.risk.metrics import (
    DrawdownEpisode,
    correlation_matrix,
    drawdown_series,
    performance_summary,
)
from app.engine.risk.var import (
    historical_var,
    monte_carlo_var,
    parametric_var,
    portfolio_returns,
)
from app.schemas.risk import (
    AssetAttribution,
    AttributionRequest,
    AttributionResponse,
    BenchmarkStatsOut,
    CorrelationRequest,
    CorrelationResponse,
    DrawdownEpisodeOut,
    MetricsRequest,
    MetricsResponse,
    VaREstimate,
    VaRRequest,
    VaRResponse,
)
from app.services._common import engine_call, num, risk_free_for, series, weights_for, window_of
from app.services.market import MarketData

NOTE_FIXED_WEIGHTS = (
    "Weights are held fixed (rebalanced daily) and applied to the whole window, so this "
    "describes today's portfolio over past market conditions, not a record of past holdings."
)
NOTE_MC_SAME_MODEL = (
    "Monte Carlo draws from the same normal model as the parametric method, so for this "
    "linear portfolio it differs from parametric only by sampling noise."
)
NOTE_PRICE_INDEX = (
    "The benchmark (Nifty 50, ^NSEI) is a price index without dividends, while stocks and "
    "Growth NAVs include them, so alpha and active return are biased upward by roughly the "
    "index dividend yield."
)


def var_report(md: MarketData, req: VaRRequest) -> VaRResponse:
    """Historical, parametric and Monte Carlo VaR/CVaR on the same window and weights."""
    w = weights_for(md, req.portfolio)
    r = md.window(req.lookback_days)
    kw = {"confidence": req.confidence, "horizon_days": req.horizon_days}
    results = engine_call(
        lambda: [
            historical_var(r, w, **kw),
            parametric_var(r, w, **kw),
            monte_carlo_var(r, w, **kw),
        ]
    )
    value = req.portfolio.portfolio_value
    notes = [
        f"VaR and CVaR are positive loss fractions of portfolio value over {req.horizon_days} "
        f"trading day(s) at {req.confidence:.1%} confidence; rupee figures = fraction x "
        f"portfolio value.",
        NOTE_FIXED_WEIGHTS,
        NOTE_MC_SAME_MODEL,
    ]
    if req.horizon_days > 1:
        notes.append(
            "Multi-day figures are 1-day figures scaled by the square root of the horizon, "
            "which assumes independent, identically distributed daily returns."
        )
    return VaRResponse(
        confidence=req.confidence,
        horizon_days=req.horizon_days,
        portfolio_value=value,
        window=window_of(r.index),
        estimates=[
            VaREstimate(
                method=x.method,
                var=x.var,
                cvar=x.cvar,
                var_rupees=x.var * value,
                cvar_rupees=x.cvar * value,
                n_sims=x.n_sims,
                seed=x.seed,
            )
            for x in results
        ],
        notes=notes,
    )


def _episode(ep: DrawdownEpisode | None) -> DrawdownEpisodeOut | None:
    if ep is None:
        return None
    return DrawdownEpisodeOut(
        depth=ep.depth,
        peak_date=ep.peak_date.date() if ep.peak_date is not None else None,
        trough_date=ep.trough_date.date(),
        recovery_date=ep.recovery_date.date() if ep.recovery_date is not None else None,
        decline_days=ep.decline_days,
        recovery_days=ep.recovery_days,
        duration_days=ep.duration_days,
        recovered=ep.recovered,
    )


def metrics_report(md: MarketData, req: MetricsRequest) -> MetricsResponse:
    """Return, volatility, Sharpe/Sortino, drawdowns and benchmark statistics."""
    w = weights_for(md, req.portfolio)
    r = md.window(req.lookback_days)
    rf = risk_free_for(md, r.index, req.risk_free_annual)
    port = engine_call(lambda: portfolio_returns(r, w))
    bench = r[md.benchmark_id]
    s = engine_call(lambda: performance_summary(port, risk_free_annual=rf.annual, benchmark=bench))
    b = s.benchmark
    return MetricsResponse(
        window=window_of(r.index),
        risk_free=rf,
        annualised_return=s.annualised_return,
        arithmetic_annual_return=s.arithmetic_annual_return,
        annualised_volatility=s.annualised_volatility,
        downside_deviation=s.downside_deviation,
        sharpe=num(s.sharpe),
        sortino=num(s.sortino),
        max_drawdown=s.drawdown.max_drawdown,
        max_drawdown_episode=_episode(s.drawdown.max_drawdown_episode),
        longest_drawdown=_episode(s.drawdown.longest_drawdown),
        n_drawdown_episodes=s.drawdown.n_episodes,
        benchmark=(
            BenchmarkStatsOut(
                benchmark_id=md.benchmark_id,
                beta=num(b.beta),
                alpha=num(b.alpha),
                correlation=num(b.correlation),
                r_squared=num(b.r_squared),
                residual_volatility=num(b.residual_volatility),
                tracking_error=b.tracking_error,
                active_return=b.active_return,
                information_ratio=num(b.information_ratio),
            )
            if b is not None
            else None
        ),
        growth=series((1 + port).cumprod()),
        benchmark_growth=series((1 + bench).cumprod()),
        drawdown=series(drawdown_series(port)),
        notes=[
            "Annualised with 252 trading days; CAGR compounds, the arithmetic return does not.",
            "Sharpe and Sortino use the arithmetic mean excess return; Sortino's minimum "
            "acceptable return is the risk-free rate.",
            NOTE_FIXED_WEIGHTS,
            NOTE_PRICE_INDEX,
        ],
    )


def attribution_report(md: MarketData, req: AttributionRequest) -> AttributionResponse:
    """Euler decomposition of volatility and parametric VaR/CVaR per holding."""
    w = weights_for(md, req.portfolio)
    r = md.window(req.lookback_days)
    vol = engine_call(lambda: volatility_attribution(r, w))
    var = engine_call(
        lambda: parametric_var_attribution(
            r, w, confidence=req.confidence, horizon_days=req.horizon_days
        )
    )
    dr = engine_call(lambda: diversification_ratio(r, w))
    var_by_id = {c.asset_id: c for c in var.contributions}
    assets = []
    for c in vol.contributions:
        v = var_by_id[c.asset_id]
        assets.append(
            AssetAttribution(
                asset_id=c.asset_id,
                name=md.name(c.asset_id),
                weight=c.weight,
                marginal_volatility=c.marginal,
                component_volatility=c.component,
                volatility_share=c.percent,
                component_var=v.component_var,
                component_cvar=v.component_cvar,
                var_share=num(v.percent_var),
                risk_to_weight=None if c.weight == 0 else c.percent / c.weight,
            )
        )
    return AttributionResponse(
        window=window_of(r.index),
        confidence=req.confidence,
        horizon_days=req.horizon_days,
        portfolio_volatility=vol.portfolio_volatility,
        parametric_var=var.portfolio_var,
        parametric_cvar=var.portfolio_cvar,
        diversification_ratio=num(dr),
        assets=assets,
        notes=[
            "Components add up exactly to the portfolio figure (Euler decomposition) and are "
            "estimates from the sample covariance of the window.",
            "A negative share means the holding lowered portfolio risk over the window (it "
            "moved against the rest); shares can exceed 100% for other holdings.",
            "Only parametric (normal) VaR decomposes in closed form; historical VaR is not "
            "decomposed.",
        ],
    )


def correlation_report(md: MarketData, req: CorrelationRequest) -> CorrelationResponse:
    w = weights_for(md, req.portfolio)
    ids = list(w)
    if req.include_benchmark and md.benchmark_id not in ids:
        ids.append(md.benchmark_id)
    r = md.window(req.lookback_days)
    corr: pd.DataFrame = engine_call(lambda: correlation_matrix(r[ids]))
    return CorrelationResponse(
        window=window_of(r.index),
        asset_ids=ids,
        names=[md.name(a) for a in ids],
        matrix=[[num(x) for x in row] for row in corr.to_numpy()],
    )
