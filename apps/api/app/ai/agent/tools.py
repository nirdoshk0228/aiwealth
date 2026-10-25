"""Agent tools: the ONLY bridge from the AI layer to the risk engine (CLAUDE.md rule 2).

Each tool validates its arguments with a Pydantic model (whose JSON schema is what the model
sees), calls an app.services function (never the engine directly), and returns a compact dict:
the numbers as display strings the model must quote verbatim, plus the window and the key
assumption. Outputs are kept small because the free LLM tier allows ~8,000 tokens a minute.

The user's portfolio is not a tool argument: it travels with the chat request (ToolContext), so
the model cannot invent or alter holdings.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.ai.agent.display import inr, pct, ratio, signed_pct
from app.ai.guardrails import sanitize_passage
from app.ai.rag.citations import NOT_FOUND, passages
from app.ai.rag.retrieve import Mode, Retriever
from app.data.scenarios import DirectShockSpec, FactorName, HypotheticalScenarioSpec
from app.data.universe import load_universe
from app.schemas.analytics import BacktestRequest, FactorRequest
from app.schemas.planning import GoalRequest, OptimizeRequest
from app.schemas.portfolio import PortfolioIn
from app.schemas.risk import AttributionRequest, MetricsRequest, VaRRequest
from app.schemas.stress import StressRequest
from app.services import backtest, factor_report, planning, risk, stress_report
from app.services.errors import AnalyticsInputError, MarketDataUnavailable, NotFound
from app.services.market import MarketData

MAX_ROWS = 6  # per-asset rows returned to the model


@dataclass
class ToolContext:
    """What tools may use besides their arguments: market data, the user's portfolio and the
    knowledge base. `retrieved` collects every passage shown to the model in this turn, so
    citations and grounding can be checked against exactly what it saw."""

    market: Callable[[], MarketData]
    portfolio: PortfolioIn | None
    knowledge: Callable[[], Retriever] | None = None
    rag_mode: Mode = "hybrid"
    retrieved: list[dict[str, Any]] = field(default_factory=list)
    user_texts: list[str] = field(default_factory=list)  # the user's messages (grounding)
    context_note: str = ""  # portfolio context shown to the model (grounding)
    sanitized_passages: int = 0  # passages with instruction-like text removed


class ToolInputError(ValueError):
    """Arguments the model sent were invalid; the message goes back to the model to fix."""


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args: type[BaseModel]
    run: Callable[[ToolContext, Any], dict[str, Any]]
    needs_portfolio: bool = True

    def schema(self) -> dict[str, Any]:
        params = self.args.model_json_schema()
        params.pop("title", None)
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": params},
        }


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _need_portfolio(ctx: ToolContext) -> PortfolioIn:
    if ctx.portfolio is None:
        raise ToolInputError("no portfolio is loaded; ask the user to enter holdings first")
    return ctx.portfolio


def _window(w: Any) -> str:
    return f"{w.start} to {w.end} ({w.n_obs} trading days)"


def _name(md: MarketData, asset_id: str) -> str:
    return md.name(asset_id).split(" - ")[0]


# --- risk ---


class RiskArgs(_Args):
    confidence: Literal[0.95, 0.99] = Field(default=0.95, description="VaR confidence level.")
    horizon_days: int = Field(default=1, ge=1, le=20, description="VaR horizon in trading days.")
    lookback_days: Literal[252, 756, 1260] = Field(
        default=756, description="History window: 252 = 1 year, 756 = 3 years, 1260 = 5 years."
    )


def _risk_summary(ctx: ToolContext, a: RiskArgs) -> dict[str, Any]:
    p, md = _need_portfolio(ctx), ctx.market()
    v = risk.var_report(
        md,
        VaRRequest(
            portfolio=p,
            confidence=a.confidence,
            horizon_days=a.horizon_days,
            lookback_days=a.lookback_days,
        ),
    )
    m = risk.metrics_report(md, MetricsRequest(portfolio=p, lookback_days=a.lookback_days))
    return {
        "portfolio_value": inr(v.portfolio_value),
        "var": {
            e.method: {
                "var": pct(e.var),
                "var_rupees": inr(e.var_rupees),
                "cvar": pct(e.cvar),
                "cvar_rupees": inr(e.cvar_rupees),
            }
            for e in v.estimates
        },
        "confidence": pct(a.confidence, 0),
        "horizon": f"{a.horizon_days} trading day(s)",
        "annualised_return_cagr": pct(m.annualised_return),
        "annualised_volatility": pct(m.annualised_volatility),
        "sharpe_ratio": ratio(m.sharpe),
        "max_drawdown": pct(m.max_drawdown),
        "max_drawdown_trough": (
            str(m.max_drawdown_episode.trough_date) if m.max_drawdown_episode else "n/a"
        ),
        "beta_to_nifty50": ratio(m.benchmark.beta if m.benchmark else None),
        "risk_free_rate": pct(m.risk_free.annual),
        "window": _window(v.window),
        "conventions": "VaR/CVaR are losses as % of portfolio value; historical = empirical "
        "quantile, parametric/monte_carlo = normal model.",
    }


class AttributionArgs(_Args):
    lookback_days: Literal[252, 756, 1260] = Field(default=756)


def _attribution(ctx: ToolContext, a: AttributionArgs) -> dict[str, Any]:
    p, md = _need_portfolio(ctx), ctx.market()
    r = risk.attribution_report(md, AttributionRequest(portfolio=p, lookback_days=a.lookback_days))
    rows = sorted(r.assets, key=lambda x: -x.volatility_share)[:MAX_ROWS]
    return {
        "portfolio_volatility": pct(r.portfolio_volatility),
        "diversification_ratio": ratio(r.diversification_ratio),
        "largest_risk_contributors": [
            {
                "asset": _name(md, x.asset_id),
                "weight": pct(x.weight, 1),
                "share_of_volatility": pct(x.volatility_share, 1),
            }
            for x in rows
        ],
        "window": _window(r.window),
        "conventions": "Euler decomposition of volatility; shares sum to 100%.",
    }


class NoArgs(_Args):
    pass


def _factors(ctx: ToolContext, a: NoArgs) -> dict[str, Any]:
    p, md = _need_portfolio(ctx), ctx.market()
    r = factor_report.factor_report(md, FactorRequest(portfolio=p, include_assets=False))
    return {
        "betas": {
            c.factor: {"beta": ratio(c.beta, 3), "t_stat": ratio(c.t_stat, 1)}
            for c in r.portfolio.coefficients
        },
        "alpha_annual": pct(r.portfolio.alpha_annual),
        "r_squared": ratio(r.portfolio.r_squared, 3),
        "window": _window(r.window),
        "conventions": "Carhart 4-factor on IIM Ahmedabad Indian factors (MKT_RF = market, SMB = "
        "size, HML = value, WML = momentum); Newey-West t-stats. Alpha is not evidence of skill "
        "(survivorship bias in the universe).",
    }


class BacktestArgs(_Args):
    method: Literal["historical", "parametric"] = "historical"
    confidence: Literal[0.95, 0.99] = 0.99


def _backtest(ctx: ToolContext, a: BacktestArgs) -> dict[str, Any]:
    p, md = _need_portfolio(ctx), ctx.market()
    r = backtest.backtest_report(
        md, BacktestRequest(portfolio=p, method=a.method, confidence=a.confidence)
    )
    return {
        "method": a.method,
        "confidence": pct(a.confidence, 0),
        "breaches": str(r.n_breaches),
        "expected_breaches": ratio(r.expected_breaches, 1),
        "test_days": str(r.test_period.n_obs),
        "breach_rate": pct(r.breach_rate),
        "kupiec_p_value": ratio(r.kupiec.p_value, 3),
        "christoffersen_independence_p_value": ratio(r.christoffersen.p_ind, 3),
        "basel_zone_last_250_days": r.basel_zone_last_250 or "n/a (99% only)",
        "test_period": _window(r.test_period),
        "conventions": "Rolling 504-day window, 1-day VaR, each forecast uses only earlier data. "
        "A high p-value means the model is not rejected, not that it is proven right.",
    }


# --- stress ---


def _list_scenarios(ctx: ToolContext, a: NoArgs) -> dict[str, Any]:
    md = ctx.market()
    return {
        "scenarios": [
            {"id": s.id, "type": s.type, "name": s.name} for s in stress_report.list_scenarios(md)
        ]
    }


class StressArgs(_Args):
    scenario_id: str = Field(description="Id from list_stress_scenarios, e.g. covid_crash_2020.")


def _stress_out(md: MarketData, r: Any) -> dict[str, Any]:
    worst = [i for i in sorted(r.impacts, key=lambda i: i.contribution) if i.contribution < 0]
    out: dict[str, Any] = {
        "scenario": r.scenario_name,
        "type": r.scenario_type,
        "portfolio_pnl": signed_pct(r.pnl),
        "portfolio_pnl_rupees": inr(r.pnl_rupees),
        "worst_contributors": [
            {
                "asset": _name(md, i.asset_id),
                "asset_return": signed_pct(i.asset_return),
                "contribution": signed_pct(i.contribution),
            }
            for i in worst[:4]
        ],
        "warnings": r.warnings[:3],
        "conventions": "P&L is signed (negative = loss). A scenario is one path, not a forecast.",
    }
    if r.scenario_type == "historical":
        out["window"] = f"{r.start} to {r.end} ({r.sessions} sessions)"
        out["max_drawdown_in_window"] = pct(r.max_drawdown)
    return out


def _stress(ctx: ToolContext, a: StressArgs) -> dict[str, Any]:
    p, md = _need_portfolio(ctx), ctx.market()
    r = stress_report.run_stress(md, StressRequest(portfolio=p, scenario_id=a.scenario_id))
    return _stress_out(md, r)


# Sector names as the universe spells them, listed in the schema: in the tool-call evaluation the
# model guessed "IT" and "Banking" when only an example was given. Validation stays in _direct.
SECTOR_NAMES = sorted({s.industry for s in load_universe().stocks})


class FlatShock(_Args):
    """A direct shock in a flat shape that models fill reliably; converted to DirectShockSpec."""

    kind: Literal["sector", "category", "asset"]
    target: str = Field(
        description="sector: exactly one of " + "; ".join(SECTOR_NAMES) + ". category: fund "
        "category, e.g. 'gold' or 'small_cap'; asset: id, e.g. 'equity:TCS'."
    )
    shock: float = Field(description="Decimal: -0.15 means -15%.")


class CustomStressArgs(_Args):
    name: str = Field(description="Short scenario name, e.g. 'Market -20%'.")
    factor_shocks: dict[FactorName, float] = Field(
        default_factory=dict,
        description="Factor shocks as DECIMALS: -0.2 means -20%. MKT_RF = broad market.",
    )
    direct_shocks: list[FlatShock] = Field(
        default_factory=list,
        description="Shocks to a sector, fund category or asset; override factor-implied returns.",
    )


def _direct(md: MarketData, f: FlatShock) -> DirectShockSpec:
    sectors = sorted({s.industry for s in md.universe.stocks})
    categories = sorted({x.category for x in md.universe.funds})
    if f.kind == "sector" and f.target not in sectors:
        raise ToolInputError(f"unknown sector {f.target!r}; use one of {sectors}")
    if f.kind == "category" and f.target not in categories:
        raise ToolInputError(f"unknown category {f.target!r}; use one of {categories}")
    key = {"sector": "sector", "category": "category", "asset": "asset_id"}[f.kind]
    try:
        return DirectShockSpec.model_validate(
            {"target": {"kind": f.kind, key: f.target}, "shock": f.shock}
        )
    except ValidationError as exc:
        raise ToolInputError(_brief(exc)) from exc


def _custom_stress(ctx: ToolContext, a: CustomStressArgs) -> dict[str, Any]:
    p, md = _need_portfolio(ctx), ctx.market()
    direct = [_direct(md, f) for f in a.direct_shocks]
    try:
        spec = HypotheticalScenarioSpec(
            id="custom",
            type="hypothetical",
            name=a.name,
            description="Built by the assistant from the user's description",
            rationale="User-described scenario",
            factor_shocks=a.factor_shocks,
            direct_shocks=direct,
        )
    except ValidationError as exc:
        raise ToolInputError(_brief(exc)) from exc
    r = stress_report.run_stress(md, StressRequest(portfolio=p, scenario=spec))
    return _stress_out(md, r) | {
        "shocks_applied": spec.model_dump(include={"factor_shocks", "direct_shocks"})
    }


# --- planning ---


class OptimizeArgs(_Args):
    scope: Literal["holdings", "funds", "stocks", "all"] = Field(
        default="holdings", description="Assets to allocate across."
    )
    max_weight: float = Field(default=1.0, ge=0.05, le=1.0, description="Cap per asset, decimal.")


def _optimize(ctx: ToolContext, a: OptimizeArgs) -> dict[str, Any]:
    p, md = _need_portfolio(ctx), ctx.market()
    ids = None
    if a.scope != "holdings":
        kinds = {"funds": {"mutual_fund"}, "stocks": {"equity"}, "all": {"mutual_fund", "equity"}}
        ids = [x.id for x in md.universe.assets() if x.kind in kinds[a.scope]]
    r = planning.optimize(md, OptimizeRequest(asset_ids=ids, portfolio=p, max_weight=a.max_weight))

    def brief(x: Any) -> dict[str, Any]:
        return {
            "expected_return": pct(x.expected_return),
            "volatility": pct(x.volatility),
            "sharpe": ratio(x.sharpe),
            "top_weights": {w.name.split(" - ")[0]: pct(w.weight, 1) for w in x.weights[:4]},
        }

    out = {
        "minimum_variance": brief(r.min_variance),
        "maximum_sharpe": brief(r.max_sharpe),
        "risk_parity": brief(r.risk_parity),
        "window": _window(r.window),
        "warnings": r.warnings[:2],
        "conventions": "Expected returns = historical averages of the window (very uncertain). "
        "These describe past trade-offs; they are not recommendations.",
    }
    if r.current is not None:
        out["current_portfolio"] = brief(r.current)
    return out


class GoalArgs(_Args):
    monthly_sip: float = Field(ge=0, description="Rupees per month.")
    years: int = Field(ge=1, le=40)
    goal_amount: float = Field(gt=0, description="Target in today's rupees.")
    initial_amount: float = Field(default=0, ge=0)
    inflation_annual: float = Field(default=0.04, gt=-0.05, lt=0.5, description="Decimal.")
    step_up_annual: float = Field(default=0.0, ge=0, le=0.5, description="Decimal.")


def _goal(ctx: ToolContext, a: GoalArgs) -> dict[str, Any]:
    p, md = _need_portfolio(ctx), ctx.market()
    try:
        req = GoalRequest(portfolio=p, **a.model_dump())
    except ValidationError as exc:
        raise ToolInputError(_brief(exc)) from exc
    r = planning.simulate_goal(md, req)
    by_q = {x.percentile: x for x in r.percentiles}
    return {
        "probability_of_reaching_goal": pct(r.probability, 0),
        "goal_at_horizon_nominal": inr(r.goal_nominal),
        "total_invested": inr(r.total_invested),
        "median_final_value": inr(by_q[50].nominal),
        "pessimistic_10th_percentile": inr(by_q[10].nominal),
        "optimistic_90th_percentile": inr(by_q[90].nominal),
        "inflation_assumption": pct(a.inflation_annual, 1),
        "calibration": f"median growth {pct(r.calibration.median_growth)} a year, volatility "
        f"{pct(r.calibration.volatility)}, from {_window(r.calibration.window)}",
        "conventions": "Monte Carlo GBM on the portfolio's history (10,000 paths); a model "
        "output, not a promise.",
    }


# --- knowledge base ---

KB_HITS = 4


class SearchArgs(_Args):
    query: str = Field(min_length=3, max_length=300, description="What to look up.")
    fund: str | None = Field(
        default=None,
        max_length=100,
        description="Fund name to restrict the search to its documents, e.g. 'Axis Liquid "
        "Fund'. Omit for general rules (SEBI circulars, AMFI guide) or several funds.",
    )


def _compact(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


def _fund_code(name: str) -> str | None:
    """Universe fund whose name contains every word of `name`, ignoring case, spaces and
    punctuation inside words ("Flexicap" matches "Flexi Cap", "midcap" matches "Mid Cap")."""
    words = [_compact(w) for w in name.lower().replace("fund", " ").split()]
    words = [w for w in words if len(w) > 1]
    for f in load_universe().funds:
        hay = _compact(f.name)
        if words and all(w in hay for w in words):
            return f.code
    return None


def _search_kb(ctx: ToolContext, a: SearchArgs) -> dict[str, Any]:
    if ctx.knowledge is None:
        raise ToolInputError("the document knowledge base is not available")
    try:
        retriever = ctx.knowledge()
    except RuntimeError as exc:
        raise ToolInputError(str(exc)) from exc
    code = _fund_code(a.fund) if a.fund else None
    hits = retriever.search(a.query, k=KB_HITS, mode=ctx.rag_mode, fund_code=code)
    found = passages(hits)
    for p in found:  # retrieved text is untrusted data: strip instruction-like sentences
        clean, removed = sanitize_passage(p["text"])
        if removed:
            p["text"] = clean
            ctx.sanitized_passages += 1
    ctx.retrieved.extend(found)
    shown = [{k: p[k] for k in ("ref", "document", "page", "text")} for p in found]
    out: dict[str, Any] = {
        "passages": shown,
        "instructions": "Answer only from these passages and cite each claim as [ref], e.g. "
        f"[{shown[0]['ref'] if shown else 'doc_id p.3'}]. If they do not contain the answer, "
        f"say '{NOT_FOUND}'.",
    }
    if a.fund and code is None:
        out["note"] = f"no universe fund matched {a.fund!r}; searched all documents"
    if not shown:
        out["result"] = NOT_FOUND
    return out


def _brief(exc: ValidationError) -> str:
    return "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:4])


TOOLS: tuple[Tool, ...] = (
    Tool(
        "get_risk_summary",
        "VaR and CVaR (historical, parametric, Monte Carlo), return, volatility, Sharpe, max "
        "drawdown and beta of the user's portfolio.",
        RiskArgs,
        _risk_summary,
    ),
    Tool(
        "get_risk_contributions",
        "Which holdings contribute most to portfolio volatility, and the diversification ratio.",
        AttributionArgs,
        _attribution,
    ),
    Tool(
        "get_factor_exposures",
        "Carhart four-factor betas (market, size, value, momentum) and alpha of the portfolio.",
        NoArgs,
        _factors,
    ),
    Tool(
        "backtest_var",
        "How often the portfolio's past 1-day VaR forecasts were breached (Kupiec, Basel zone).",
        BacktestArgs,
        _backtest,
    ),
    Tool(
        "list_stress_scenarios",
        "Ids and names of the available historical and hypothetical stress scenarios.",
        NoArgs,
        _list_scenarios,
        needs_portfolio=False,
    ),
    Tool(
        "run_stress_scenario",
        "Portfolio P&L in one available scenario (by id), with the worst contributors. "
        "If the user gives shock sizes that differ from the scenario's, use run_custom_stress.",
        StressArgs,
        _stress,
    ),
    Tool(
        "run_custom_stress",
        "Portfolio P&L for a scenario described by the user, as factor and/or direct shocks "
        "in decimals. Use the user's own numbers, even if a similar available scenario exists.",
        CustomStressArgs,
        _custom_stress,
    ),
    Tool(
        "optimize_portfolio",
        "Minimum-variance, maximum-Sharpe and risk-parity portfolios over historical estimates.",
        OptimizeArgs,
        _optimize,
    ),
    Tool(
        "search_knowledge_base",
        "Search fund documents (scheme information documents, factsheets) and SEBI/AMFI rules: "
        "fees, exit loads, benchmarks, objectives, categories, riskometer, regulations.",
        SearchArgs,
        _search_kb,
        needs_portfolio=False,
    ),
    Tool(
        "simulate_goal",
        "Monte Carlo SIP simulation: probability of reaching a goal with the current mix.",
        GoalArgs,
        _goal,
    ),
)

TOOLS_BY_NAME = {t.name: t for t in TOOLS}


def execute(name: str, arguments: str, ctx: ToolContext) -> dict[str, Any]:
    """Run one tool call. Every failure becomes {"error": ...} for the model to read; nothing
    raises out of here, so a bad call never breaks the conversation."""
    tool = TOOLS_BY_NAME.get(name)
    if tool is None:
        return {"error": f"unknown tool {name!r}; available: {sorted(TOOLS_BY_NAME)}"}
    try:
        raw = json.loads(arguments or "{}")
        args = tool.args.model_validate(raw)
        return tool.run(ctx, args)
    except json.JSONDecodeError:
        return {"error": "arguments are not valid JSON"}
    except ValidationError as exc:
        return {"error": f"invalid arguments: {_brief(exc)}"}
    except (ToolInputError, AnalyticsInputError, NotFound) as exc:
        return {"error": str(exc)}
    except MarketDataUnavailable as exc:
        return {"error": f"market data unavailable: {exc}"}
    except ValueError as exc:  # e.g. a retrieval mode that needs an unloaded reranker
        return {"error": str(exc)}
