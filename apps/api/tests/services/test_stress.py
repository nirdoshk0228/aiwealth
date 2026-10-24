"""Tests for app/services/stress.py: spec conversion, groups, exposures, and the real caches."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from app.core.config import REPO_ROOT
from app.data.factor_repository import ParquetFactorRepository
from app.data.repository import ParquetRepository
from app.data.scenarios import HistoricalScenarioSpec, load_scenarios, parse_scenario
from app.data.universe import load_universe
from app.engine.returns import simple_returns
from app.engine.stress.scenarios import AssetGroups, DirectShock, HistoricalScenario
from app.services.factors import load_aligned_factors
from app.services.market_data import load_aligned_prices
from app.services.stress import asset_groups, estimate_exposures, run_scenario, to_engine

SCENARIOS_DIR = REPO_ROOT / "data" / "scenarios"


def test_to_engine_converts_both_types() -> None:
    specs = load_scenarios(SCENARIOS_DIR)
    h = to_engine(specs["covid_crash_2020"])
    assert h == HistoricalScenario(
        id="covid_crash_2020",
        name=specs["covid_crash_2020"].name,
        start=date(2020, 1, 14),
        end=date(2020, 3, 23),
    )
    x = to_engine(specs["credit_event"])
    assert x.factor_shocks == {"MKT_RF": -0.05}
    assert DirectShock("category", "debt_corporate_bond", -0.02) in x.direct_shocks
    s = to_engine(specs["it_sector_shock"])
    assert s.direct_shocks == (DirectShock("sector", "Information Technology", -0.15),)


def test_asset_groups_from_universe() -> None:
    g = asset_groups(load_universe())
    assert g["mutual_fund:118663"] == AssetGroups(category="gold")
    assert g["equity:TCS"] == AssetGroups(sector="Information Technology")
    assert g["index:^NSEI"] == AssetGroups()
    assert len(g) == 36


def test_estimate_exposures_recovers_known_betas() -> None:
    # Returns built exactly as RF + 1.5 MKT_RF - 0.5 SMB + small noise: betas come back.
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2020-01-01", periods=600)
    f = pd.DataFrame(
        rng.normal(0, 0.01, (600, 4)), index=idx, columns=["MKT_RF", "SMB", "HML", "WML"]
    )
    f["RF"] = 0.0002
    r = pd.DataFrame(
        {"a": f["RF"] + 1.5 * f["MKT_RF"] - 0.5 * f["SMB"] + rng.normal(0, 1e-4, 600)}, index=idx
    )
    f.iloc[:10] = np.nan  # rows without factor data are dropped, not fed to the regression
    exp = estimate_exposures(r, f)["a"]
    assert exp.betas["MKT_RF"] == pytest.approx(1.5, abs=0.01)
    assert exp.betas["SMB"] == pytest.approx(-0.5, abs=0.01)
    assert exp.r_squared > 0.99


def test_run_scenario_historical_needs_prices() -> None:
    spec = load_scenarios(SCENARIOS_DIR)["covid_crash_2020"]
    with pytest.raises(ValueError, match="needs prices"):
        run_scenario(spec, {"equity:TCS": 1.0}, portfolio_value=1)


def test_run_scenario_hypothetical_end_to_end() -> None:
    spec = parse_scenario(
        '{"id": "g", "type": "hypothetical", "name": "g", "description": "d", "rationale": "r",'
        ' "direct_shocks": [{"target": {"kind": "category", "category": "gold"}, "shock": 0.1}]}'
    )
    r = run_scenario(
        spec,
        {"mutual_fund:118663": 0.5, "equity:TCS": 0.5},
        portfolio_value=1_000_000,
        groups=asset_groups(load_universe()),
    )
    assert r.pnl == pytest.approx(0.05)
    assert r.pnl_rupees == pytest.approx(50_000)


# --- real caches (data/raw from scripts/seed_data.py); skipped when not seeded, e.g. in CI ---


def _real():
    try:
        universe = load_universe()
    except FileNotFoundError:
        return None
    prices, factors = ParquetRepository(), ParquetFactorRepository("iima_ff4_daily")
    if not factors.exists() or not all(prices.exists(a) for a in universe.assets()):
        return None
    return universe, load_aligned_prices(prices, universe).prices, factors


REAL = _real()
real = pytest.mark.skipif(REAL is None, reason="real caches not seeded (run scripts/seed_data.py)")


@real
def test_historical_windows_match_the_cache() -> None:
    # The Nifty closes recorded in each scenario file must still be what the cache says, so a
    # re-seed that moves a peak or trough fails here instead of silently changing a scenario.
    _, prices, _ = REAL
    nifty = prices["index:^NSEI"]
    for spec in load_scenarios(SCENARIOS_DIR).values():
        if not isinstance(spec, HistoricalScenarioSpec):
            continue
        bc = spec.benchmark_check
        start, end = pd.Timestamp(spec.start), pd.Timestamp(spec.end)
        assert nifty[start] == pytest.approx(bc.start_close, abs=0.01), spec.id
        assert nifty[end] == pytest.approx(bc.end_close, abs=0.01), spec.id
        window = nifty[start:end]
        if len(window) > 2:  # drawdown windows: start is the peak, end the trough
            assert window.idxmax() == start, spec.id
            assert window.idxmin() == end, spec.id


@real
def test_all_scenarios_run_on_equal_weight_portfolios() -> None:
    universe, prices, factor_repo = REAL
    factors = load_aligned_factors(factor_repo, prices.index).factors
    exposures = estimate_exposures(simple_returns(prices), factors)
    groups = asset_groups(universe)
    portfolios = {
        "stocks": [s.asset().id for s in universe.stocks],
        "funds": [f.asset().id for f in universe.funds],
    }
    for ids in portfolios.values():
        w = dict.fromkeys(ids, 1 / len(ids))
        for spec in load_scenarios(SCENARIOS_DIR).values():
            r = run_scenario(
                spec,
                w,
                portfolio_value=1_000_000,
                prices=prices,
                exposures=exposures,
                groups=groups,
            )
            assert sum(i.contribution for i in r.impacts) == pytest.approx(r.pnl, abs=1e-12)
            assert -1 < r.pnl < 1
