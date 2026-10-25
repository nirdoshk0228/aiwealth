"""Agent tools on synthetic data: schemas, error handling, outputs equal to the services."""

import json

import pytest

from app.ai.agent.display import inr, pct, signed_pct
from app.ai.agent.prompts import SYSTEM_PROMPT
from app.ai.agent.tools import TOOLS, ToolContext, execute
from app.data.universe import load_universe
from app.schemas.portfolio import PortfolioIn
from app.schemas.risk import MetricsRequest, VaRRequest
from app.schemas.stress import StressRequest
from app.services import risk, stress_report
from app.services.market import MarketData


@pytest.fixture
def ctx(market: MarketData, fund_mix: dict[str, object]) -> ToolContext:
    return ToolContext(market=lambda: market, portfolio=PortfolioIn.model_validate(fund_mix))


def test_tool_schemas_are_valid_and_do_not_take_the_portfolio() -> None:
    names = [t.name for t in TOOLS]
    assert len(names) == len(set(names))
    for t in TOOLS:
        s = t.schema()
        assert s["type"] == "function" and s["function"]["name"] == t.name
        params = s["function"]["parameters"]
        assert params["type"] == "object"
        assert "portfolio" not in params.get("properties", {})
        json.dumps(s)  # serialisable
    # Compact: the whole tool list stays well under the free tier's per-minute token budget.
    assert len(json.dumps([t.schema() for t in TOOLS])) < 9_000


def test_sector_names_are_listed_in_the_custom_stress_schema() -> None:
    schema = next(t for t in TOOLS if t.name == "run_custom_stress").schema()
    target = schema["function"]["parameters"]["$defs"]["FlatShock"]["properties"]["target"]
    sectors = sorted({s.industry for s in load_universe().stocks})
    assert sectors and all(s in target["description"] for s in sectors)
    assert "Information Technology" in target["description"]


def test_stress_tools_prefer_the_users_own_numbers() -> None:
    by_name = {t.name: t.description for t in TOOLS}
    assert "use run_custom_stress" in by_name["run_stress_scenario"]
    assert "user's own numbers" in by_name["run_custom_stress"]
    assert "use exactly those numbers in run_custom_stress" in SYSTEM_PROMPT


def test_risk_summary_quotes_service_values(ctx: ToolContext, market: MarketData) -> None:
    out = execute("get_risk_summary", '{"confidence": 0.99}', ctx)
    p = ctx.portfolio
    v = risk.var_report(market, VaRRequest(portfolio=p, confidence=0.99))
    m = risk.metrics_report(market, MetricsRequest(portfolio=p))
    hist = next(e for e in v.estimates if e.method == "historical")
    assert out["var"]["historical"]["var"] == pct(hist.var)
    assert out["var"]["historical"]["var_rupees"] == inr(hist.var_rupees)
    assert out["annualised_volatility"] == pct(m.annualised_volatility)
    assert out["confidence"] == "99%" and out["horizon"] == "1 trading day(s)"


def test_custom_stress_runs_the_engine(ctx: ToolContext, market: MarketData) -> None:
    args = {"name": "Market -20%", "factor_shocks": {"MKT_RF": -0.2}}
    out = execute("run_custom_stress", json.dumps(args), ctx)
    assert "error" not in out
    from app.data.scenarios import HypotheticalScenarioSpec

    spec = HypotheticalScenarioSpec(
        id="custom",
        type="hypothetical",
        name="Market -20%",
        description="x",
        rationale="x",
        factor_shocks={"MKT_RF": -0.2},
    )
    ref = stress_report.run_stress(market, StressRequest(portfolio=ctx.portfolio, scenario=spec))
    assert out["portfolio_pnl"] == signed_pct(ref.pnl)
    assert out["shocks_applied"]["factor_shocks"] == {"MKT_RF": -0.2}


def test_errors_are_returned_to_the_model_not_raised(ctx: ToolContext) -> None:
    assert "unknown tool" in execute("delete_everything", "{}", ctx)["error"]
    assert execute("get_risk_summary", "{not json", ctx) == {
        "error": "arguments are not valid JSON"
    }
    bad = execute("get_risk_summary", '{"confidence": 0.5}', ctx)["error"]
    assert bad.startswith("invalid arguments: confidence")
    extra = execute("get_factor_exposures", '{"portfolio": []}', ctx)["error"]
    assert "Extra inputs" in extra
    pct_mixup = execute("run_custom_stress", '{"name": "x", "factor_shocks": {"MKT_RF": -20}}', ctx)
    assert "error" in pct_mixup  # -20 is not a decimal shock in (-1, 1]
    unknown = execute("run_stress_scenario", '{"scenario_id": "mars_landing"}', ctx)
    assert "unknown scenario" in unknown["error"]


def test_tools_need_a_portfolio_except_listing(market: MarketData) -> None:
    empty = ToolContext(market=lambda: market, portfolio=None)
    assert "no portfolio is loaded" in execute("get_risk_summary", "{}", empty)["error"]
    listing = execute("list_stress_scenarios", "{}", empty)
    assert len(listing["scenarios"]) == 10


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("get_risk_contributions", {}),
        ("get_factor_exposures", {}),
        ("backtest_var", {"method": "parametric", "confidence": 0.95}),
        ("run_stress_scenario", {"scenario_id": "covid_crash_2020"}),
        ("optimize_portfolio", {"scope": "holdings", "max_weight": 0.4}),
        ("simulate_goal", {"monthly_sip": 10000, "years": 10, "goal_amount": 2000000}),
    ],
)
def test_every_tool_runs_and_stays_compact(ctx: ToolContext, name: str, args: dict) -> None:
    out = execute(name, json.dumps(args), ctx)
    assert "error" not in out, out
    assert len(json.dumps(out)) < 2_500  # tool results are re-sent to the model


def test_display_formats() -> None:
    assert pct(0.01384) == "1.38%" and pct(None) == "n/a"
    assert signed_pct(-0.3416) == "-34.16%" and signed_pct(0.05) == "+5.00%"
    assert inr(1_000_000) == "Rs 10,00,000" and inr(-50719.4) == "-Rs 50,719"
    assert inr(999) == "Rs 999" and inr(12_345_678.9) == "Rs 1,23,45,679"


def test_flat_direct_shocks_are_converted_and_checked(ctx: ToolContext) -> None:
    args = {
        "name": "IT -15%, market -8%",
        "factor_shocks": {"MKT_RF": -0.08},
        "direct_shocks": [{"kind": "sector", "target": "Information Technology", "shock": -0.15}],
    }
    out = execute("run_custom_stress", json.dumps(args), ctx)
    applied = out["shocks_applied"]["direct_shocks"][0]
    assert applied["target"] == {"kind": "sector", "sector": "Information Technology"}
    bad = dict(args, direct_shocks=[{"kind": "sector", "target": "IT", "shock": -0.15}])
    err = execute("run_custom_stress", json.dumps(bad), ctx)["error"]
    assert err.startswith("unknown sector 'IT'; use one of") and "Information Technology" in err
    gold = dict(args, direct_shocks=[{"kind": "category", "target": "gold", "shock": 0.1}])
    assert "error" not in execute("run_custom_stress", json.dumps(gold), ctx)
