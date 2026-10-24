"""Stress testing service: scenario files + cached data -> engine inputs -> StressResult.

Reads only from the repositories (never from providers). Converts the Pydantic scenario specs
of app/data/scenarios.py into the engine's dataclasses, builds each asset's sector/category
from the universe and its factor betas from the regression engine, then calls the pure engine.
"""

from collections.abc import Mapping

import pandas as pd

from app.data.scenarios import (
    HistoricalScenarioSpec,
    HypotheticalScenarioSpec,
    target_key,
)
from app.data.universe import Universe
from app.engine.factors.regression import DEFAULT_FACTORS, factor_regression_by_asset
from app.engine.stress.historical import historical_replay
from app.engine.stress.hypothetical import hypothetical_shock
from app.engine.stress.result import StressResult
from app.engine.stress.scenarios import (
    AssetExposure,
    AssetGroups,
    DirectShock,
    HistoricalScenario,
    HypotheticalScenario,
)


def to_engine(
    spec: HistoricalScenarioSpec | HypotheticalScenarioSpec,
) -> HistoricalScenario | HypotheticalScenario:
    """Pydantic spec -> plain engine dataclass (validated again by the engine)."""
    if isinstance(spec, HistoricalScenarioSpec):
        return HistoricalScenario(id=spec.id, name=spec.name, start=spec.start, end=spec.end)
    shocks = []
    for d in spec.direct_shocks:
        kind, target = target_key(d.target)
        shocks.append(DirectShock(kind=kind, target=target, shock=d.shock))
    return HypotheticalScenario(
        id=spec.id,
        name=spec.name,
        factor_shocks=dict(spec.factor_shocks),
        direct_shocks=tuple(shocks),
    )


def asset_groups(universe: Universe) -> dict[str, AssetGroups]:
    """Asset id -> category (funds) or sector = NSE industry (stocks); the index has neither."""
    groups = {f.asset().id: AssetGroups(category=f.category) for f in universe.funds}
    groups |= {s.asset().id: AssetGroups(sector=s.industry) for s in universe.stocks}
    groups[universe.benchmark.asset().id] = AssetGroups()
    return groups


def estimate_exposures(returns: pd.DataFrame, factors: pd.DataFrame) -> dict[str, AssetExposure]:
    """Carhart betas and R^2 per asset from factor_regression_by_asset.

    `returns` = simple_returns(aligned prices); `factors` = load_aligned_factors(...).factors on
    the same index. Only rows with factor data are used (with release 2025-12:
    2015-01-05 -> 2025-12-31), exactly as in the factor regression module.
    """
    valid = factors.dropna()
    table = factor_regression_by_asset(returns.loc[valid.index], valid)
    return {
        str(a): AssetExposure(
            betas={f: float(row[f"beta_{f}"]) for f in DEFAULT_FACTORS},
            r_squared=float(row["r_squared"]),
        )
        for a, row in table.iterrows()
    }


def run_scenario(
    spec: HistoricalScenarioSpec | HypotheticalScenarioSpec,
    weights: Mapping[str, float],
    *,
    portfolio_value: float,
    prices: pd.DataFrame | None = None,
    exposures: Mapping[str, AssetExposure] | None = None,
    groups: Mapping[str, AssetGroups] | None = None,
) -> StressResult:
    """Run one scenario. Historical scenarios need `prices` (aligned); hypothetical ones need
    `exposures` when they shock factors and `groups` when they shock sectors or categories."""
    scenario = to_engine(spec)
    if isinstance(scenario, HistoricalScenario):
        if prices is None:
            raise ValueError(f"{spec.id}: a historical replay needs prices")
        return historical_replay(prices, weights, scenario, portfolio_value=portfolio_value)
    return hypothetical_shock(
        weights, scenario, portfolio_value=portfolio_value, exposures=exposures, groups=groups
    )
