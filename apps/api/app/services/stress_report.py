"""Stress lab service: list scenarios and run one (committed or custom) on a portfolio."""

from app.data.scenarios import (
    HistoricalScenarioSpec,
    HypotheticalScenarioSpec,
    check_against_universe,
)
from app.engine.stress.result import StressResult
from app.schemas.stress import AssetImpactOut, ScenarioOut, StressRequest, StressResponse
from app.services._common import engine_call, weights_for
from app.services.errors import AnalyticsInputError, NotFound
from app.services.market import MarketData
from app.services.stress import run_scenario

WORST_N = 5


def scenario_out(spec: HistoricalScenarioSpec | HypotheticalScenarioSpec) -> ScenarioOut:
    common = {
        "id": spec.id,
        "name": spec.name,
        "type": spec.type,
        "description": spec.description,
        "rationale": spec.rationale,
        "caveats": list(spec.caveats),
    }
    if isinstance(spec, HistoricalScenarioSpec):
        return ScenarioOut(**common, start=spec.start, end=spec.end)
    return ScenarioOut(
        **common,
        factor_shocks=dict(spec.factor_shocks),
        direct_shocks=list(spec.direct_shocks),
    )


def list_scenarios(md: MarketData) -> list[ScenarioOut]:
    """Historical replays first (by start date), then hypothetical shocks (by id)."""
    specs = list(md.scenarios.values())
    hist = sorted(
        (s for s in specs if isinstance(s, HistoricalScenarioSpec)), key=lambda s: s.start
    )
    hypo = sorted((s for s in specs if isinstance(s, HypotheticalScenarioSpec)), key=lambda s: s.id)
    return [scenario_out(s) for s in [*hist, *hypo]]


def _with_names(md: MarketData, text: str, res: StressResult) -> str:
    """Engine warnings name assets by id; add the readable name after each id."""
    for i in res.impacts:
        if i.asset_id in text:
            text = text.replace(i.asset_id, f"{i.asset_id} ({md.name(i.asset_id)})", 1)
    return text


def _response(md: MarketData, res: StressResult) -> StressResponse:
    dd = res.drawdown
    return StressResponse(
        scenario_id=res.scenario_id,
        scenario_name=res.scenario_name,
        scenario_type=res.scenario_type,
        portfolio_value=res.portfolio_value,
        pnl=res.pnl,
        pnl_rupees=res.pnl_rupees,
        impacts=[
            AssetImpactOut(
                asset_id=i.asset_id,
                name=md.name(i.asset_id),
                weight=i.weight,
                asset_return=i.asset_return,
                contribution=i.contribution,
                contribution_rupees=i.contribution_rupees,
                source=i.source,
                factor_implied=i.factor_implied,
                r_squared=i.r_squared,
            )
            for i in res.impacts
        ],
        worst_contributors=[i.asset_id for i in res.worst_contributors(WORST_N)],
        notes=list(res.notes),
        warnings=[_with_names(md, w, res) for w in res.warnings],
        start=res.start,
        end=res.end,
        sessions=res.sessions,
        max_drawdown=dd.max_drawdown if dd is not None else None,
        drawdown_peak_date=dd.peak_date if dd is not None else None,
        drawdown_trough_date=dd.trough_date if dd is not None else None,
    )


def run_stress(md: MarketData, req: StressRequest) -> StressResponse:
    w = weights_for(md, req.portfolio)
    if req.scenario_id is not None:
        spec = md.scenarios.get(req.scenario_id)
        if spec is None:
            raise NotFound(
                f"unknown scenario {req.scenario_id!r}; see GET /api/v1/stress/scenarios"
            )
    else:
        assert req.scenario is not None  # guaranteed by StressRequest's validator
        spec = req.scenario
        try:
            check_against_universe(spec, md.universe)
        except ValueError as exc:
            raise AnalyticsInputError(str(exc)) from exc
    exposures = None
    if isinstance(spec, HypotheticalScenarioSpec) and spec.factor_shocks:
        exposures = md.exposures()
    res = engine_call(
        lambda: run_scenario(
            spec,
            w,
            portfolio_value=req.portfolio.portfolio_value,
            prices=md.prices,
            exposures=exposures,
            groups=md.groups,
        )
    )
    return _response(md, res)
