"""Tests for the scenario schema and loader (app/data/scenarios.py) and the committed files."""

import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.config import REPO_ROOT
from app.data.scenarios import (
    HistoricalScenarioSpec,
    HypotheticalScenarioSpec,
    check_against_universe,
    load_scenarios,
    parse_scenario,
)
from app.data.universe import load_universe

SCENARIOS_DIR = REPO_ROOT / "data" / "scenarios"

HISTORICAL = {
    "id": "h",
    "type": "historical",
    "name": "H",
    "description": "d",
    "rationale": "r",
    "start": "2020-01-14",
    "end": "2020-03-23",
    "benchmark_check": {"start_close": 12362.3, "end_close": 7610.25, "change": -0.3844},
}
HYPOTHETICAL = {
    "id": "x",
    "type": "hypothetical",
    "name": "X",
    "description": "d",
    "rationale": "r",
    "factor_shocks": {"MKT_RF": -0.2},
    "direct_shocks": [
        {"target": {"kind": "category", "category": "gold"}, "shock": 0.1},
        {"target": {"kind": "sector", "sector": "Information Technology"}, "shock": -0.1},
        {"target": {"kind": "asset", "asset_id": "equity:TCS"}, "shock": -0.2},
    ],
}


def parse(d: dict) -> HistoricalScenarioSpec | HypotheticalScenarioSpec:
    return parse_scenario(json.dumps(d))


def test_committed_scenarios_load_and_match_the_universe() -> None:
    specs = load_scenarios(SCENARIOS_DIR)
    universe = load_universe()
    historical = {k for k, s in specs.items() if isinstance(s, HistoricalScenarioSpec)}
    assert historical == {
        "demonetisation_2016",
        "ilfs_2018_equity_drawdown",
        "covid_crash_2020",
        "election_results_2024",
        "drawdown_2024_2025",
        "selloff_2026",
    }
    assert {k for k in specs if k not in historical} == {
        "equity_bear_factor",
        "equity_selloff_gold_rally",
        "credit_event",
        "it_sector_shock",
    }
    for spec in specs.values():
        check_against_universe(spec, universe)


def test_valid_examples_parse() -> None:
    h = parse(HISTORICAL)
    assert isinstance(h, HistoricalScenarioSpec)
    x = parse(HYPOTHETICAL)
    assert isinstance(x, HypotheticalScenarioSpec)
    assert x.direct_shocks[2].target.kind == "asset"


def mutated(base: dict, fn) -> dict:
    d = copy.deepcopy(base)
    fn(d)
    return d


@pytest.mark.parametrize(
    "bad",
    [
        mutated(HISTORICAL, lambda d: d.update(end="2020-01-14")),  # end == start
        mutated(HISTORICAL, lambda d: d.update(start="2020-02-30")),  # not a date
        mutated(HISTORICAL, lambda d: d.update(colour="red")),  # unknown key
        mutated(HISTORICAL, lambda d: d.pop("benchmark_check")),
        mutated(HISTORICAL, lambda d: d["benchmark_check"].update(change=-0.30)),  # inconsistent
        mutated(HISTORICAL, lambda d: d.update(type="replay")),
        mutated(HISTORICAL, lambda d: d.pop("type")),
        mutated(HISTORICAL, lambda d: d.update(id="Bad Id")),
        mutated(HYPOTHETICAL, lambda d: d.update(factor_shocks={}, direct_shocks=[])),
        mutated(HYPOTHETICAL, lambda d: d.update(factor_shocks={"RF": -0.01})),
        mutated(HYPOTHETICAL, lambda d: d.update(factor_shocks={"MKT_RF": -20})),  # percent
        mutated(HYPOTHETICAL, lambda d: d.update(factor_shocks={"MKT_RF": -1.0})),
        mutated(HYPOTHETICAL, lambda d: d["direct_shocks"][0].update(shock=1.5)),
        mutated(
            HYPOTHETICAL, lambda d: d["direct_shocks"][0]["target"].update(category="thematic")
        ),
        mutated(HYPOTHETICAL, lambda d: d["direct_shocks"][0]["target"].update(kind="fund")),
        mutated(HYPOTHETICAL, lambda d: d["direct_shocks"][2]["target"].update(asset_id="TCS")),
        mutated(
            HYPOTHETICAL, lambda d: d["direct_shocks"].append(copy.deepcopy(d["direct_shocks"][0]))
        ),
    ],
    ids=[
        "end-not-after-start",
        "invalid-date",
        "unknown-key",
        "no-benchmark-check",
        "inconsistent-change",
        "unknown-type",
        "no-type",
        "bad-id",
        "no-shocks",
        "unknown-factor",
        "percent-not-decimal",
        "minus-100pct",
        "shock-above-100pct",
        "unknown-category",
        "unknown-target-kind",
        "asset-id-without-kind",
        "duplicate-target",
    ],
)
def test_invalid_scenarios_rejected(bad: dict) -> None:
    with pytest.raises(ValidationError):
        parse(bad)


def test_targets_outside_the_universe_rejected() -> None:
    universe = load_universe()
    for target in (
        {"kind": "asset", "asset_id": "equity:ITC"},
        {"kind": "sector", "sector": "Information Tech"},
    ):
        d = mutated(
            HYPOTHETICAL, lambda d, t=target: d.update(direct_shocks=[{"target": t, "shock": -0.1}])
        )
        with pytest.raises(ValueError, match="not in the universe"):
            check_against_universe(parse(d), universe)
    check_against_universe(parse(HYPOTHETICAL), universe)  # the valid one passes


def test_loader_requires_id_to_match_file_name(tmp_path: Path) -> None:
    (tmp_path / "other_name.json").write_text(json.dumps(HISTORICAL), encoding="utf-8")
    with pytest.raises(ValueError, match="must equal the file name"):
        load_scenarios(tmp_path)


def test_loader_rejects_an_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no scenario files"):
        load_scenarios(tmp_path)


def test_loader_keys_by_id(tmp_path: Path) -> None:
    (tmp_path / "h.json").write_text(json.dumps(HISTORICAL), encoding="utf-8")
    (tmp_path / "x.json").write_text(json.dumps(HYPOTHETICAL), encoding="utf-8")
    assert sorted(load_scenarios(tmp_path)) == ["h", "x"]


def test_json_schema_is_usable_for_structured_output() -> None:
    # Phase 7: the AI scenario parser is given this schema. Closed enums must survive.
    schema = json.dumps(HypotheticalScenarioSpec.model_json_schema())
    for token in ('"MKT_RF"', '"debt_gilt"', '"sector"', '"exclusiveMinimum": -1.0'):
        assert token in schema
