"""Optimizer and goal-planner services: request -> engine -> response schemas."""

from app.engine.goals.profile import BANDS, DISCLAIMER, QUESTIONS, Band, score_answers
from app.engine.goals.simulation import SipPlan, calibrate_gbm, simulate_sip
from app.engine.optimize.mean_variance import (
    PortfolioPoint,
    efficient_frontier,
    estimate,
    evaluate,
)
from app.engine.optimize.risk_parity import risk_parity_weights
from app.engine.risk.var import portfolio_returns
from app.schemas.planning import (
    AllocationMix,
    AssetEstimate,
    BandOut,
    FrontierPoint,
    GbmCalibration,
    GoalFan,
    GoalRequest,
    GoalResponse,
    OptimizedPortfolio,
    OptimizeRequest,
    OptimizeResponse,
    OptionOut,
    PercentileValue,
    ProfileRequest,
    ProfileResponse,
    QuestionnaireOut,
    QuestionOut,
    WeightOut,
)
from app.schemas.portfolio import PortfolioIn
from app.services._common import engine_call, num, risk_free_for, weights_for, window_of
from app.services.errors import AnalyticsInputError
from app.services.market import MarketData

MAX_FRONTIER_POINTS = 60

NOTE_EXPECTED_RETURNS = (
    "Expected returns are the historical average returns of the window. Mean-variance weights "
    "are very sensitive to them: a few strong years for one asset can dominate the result. "
    "These portfolios describe the trade-off in past data, not a forecast."
)
NOTE_RISK_PARITY = (
    "Risk parity uses only the covariance (no return estimates): each holding contributes the "
    "same share of portfolio volatility."
)


def _portfolio_out(
    md: MarketData, label: str, p: PortfolioPoint, risk_shares: list[float] | None = None
) -> OptimizedPortfolio:
    order = sorted(p.weights.items(), key=lambda kv: -kv[1])
    shares_by_id = dict(zip(p.weights, risk_shares, strict=True)) if risk_shares else None
    kept = [(a, w) for a, w in order if w > 0]
    return OptimizedPortfolio(
        label=label,
        expected_return=p.expected_return,
        volatility=p.volatility,
        sharpe=num(p.sharpe),
        weights=[WeightOut(asset_id=a, name=md.name(a), weight=w) for a, w in kept],
        risk_shares=[shares_by_id[a] for a, _ in kept] if shares_by_id else None,
    )


def _thin(points: list[PortfolioPoint], k: int) -> list[PortfolioPoint]:
    if len(points) <= k:
        return points
    step = (len(points) - 1) / (k - 1)
    return [points[round(i * step)] for i in range(k)]


def _is_liquid(md: MarketData, asset_id: str) -> bool:
    g = md.groups.get(asset_id)
    return g is not None and g.category == "debt_liquid"


def optimize(md: MarketData, req: OptimizeRequest) -> OptimizeResponse:
    current_w = weights_for(md, req.portfolio) if req.portfolio is not None else None
    ids = list(req.asset_ids) if req.asset_ids is not None else list(current_w or {})
    md.check_assets(ids)
    # Liquid funds are cash-like (volatility ~0.3% a year): left in, they take over the
    # minimum-variance, maximum-Sharpe and risk-parity portfolios. Excluded unless requested.
    excluded = [] if req.include_liquid else [a for a in ids if _is_liquid(md, a)]
    ids = [a for a in ids if a not in excluded]
    if len(ids) < 2:
        raise AnalyticsInputError(
            "select at least two assets to optimise across"
            + (
                " (liquid funds are excluded as cash; set include_liquid to use them)"
                if excluded
                else ""
            )
        )
    r = md.window(req.lookback_days)
    rf = risk_free_for(md, r.index, req.risk_free_annual)
    est = engine_call(lambda: estimate(r, ids))
    res = engine_call(
        lambda: efficient_frontier(est, risk_free_annual=rf.annual, max_weight=req.max_weight)
    )
    rp = engine_call(lambda: risk_parity_weights(est.cov))
    rp_point = evaluate(est, dict(zip(est.asset_ids, rp.weights, strict=True)), rf.annual)

    current = None
    warnings = list(res.warnings)
    if current_w is not None:
        outside = [a for a in current_w if a not in est.asset_ids]
        if outside and all(a in excluded for a in outside):
            warnings.append(
                "The current portfolio holds a liquid fund, which is excluded as cash, so it is "
                "not compared here; include liquid funds to compare it."
            )
        elif outside:
            warnings.append(f"The current portfolio holds assets outside the selection: {outside}")
        else:
            current = _portfolio_out(md, "Current portfolio", evaluate(est, current_w, rf.annual))
    liquid = [a for a in est.asset_ids if _is_liquid(md, a)]
    for label, point in (
        ("minimum-variance", res.min_variance),
        ("maximum-Sharpe", res.max_sharpe),
    ):
        share = sum(point.weights.get(a, 0.0) for a in liquid)
        if share > 0.5:
            warnings.append(
                f"The {label} portfolio is {share:.0%} in a liquid fund. Liquid funds behave "
                "almost like cash: their tiny volatility makes the small return spread over the "
                "T-bill rate look like a high Sharpe ratio. Exclude them to compare risky assets."
            )
    if req.max_weight < 1 and rp.weights.max() > req.max_weight + 1e-12:
        warnings.append(
            f"Risk parity puts {rp.weights.max():.0%} in one asset, above the "
            f"{req.max_weight:.0%} cap (the cap applies to the mean-variance portfolios only)."
        )
    sd = est.cov.diagonal() ** 0.5
    return OptimizeResponse(
        window=window_of(r.index),
        risk_free=rf,
        max_weight=req.max_weight,
        assets=[
            AssetEstimate(
                asset_id=a, name=md.name(a), expected_return=float(m), volatility=float(s)
            )
            for a, m, s in zip(est.asset_ids, est.mu, sd, strict=True)
        ],
        frontier=[
            FrontierPoint(
                expected_return=p.expected_return, volatility=p.volatility, sharpe=num(p.sharpe)
            )
            for p in _thin(list(res.frontier), MAX_FRONTIER_POINTS)
        ],
        min_variance=_portfolio_out(md, "Minimum variance", res.min_variance),
        max_sharpe=_portfolio_out(md, "Maximum Sharpe ratio", res.max_sharpe),
        max_return=_portfolio_out(md, "Maximum return", res.max_return),
        risk_parity=_portfolio_out(
            md,
            "Risk parity (equal risk contribution)",
            rp_point,
            [float(x) for x in rp.risk_shares],
        ),
        current=current,
        warnings=warnings,
        notes=[
            *(
                [f"Treated as cash and excluded: {', '.join(md.name(a) for a in excluded)}."]
                if excluded
                else []
            ),
            NOTE_EXPECTED_RETURNS,
            NOTE_RISK_PARITY,
            "Long-only, fully invested; annual figures from daily returns (x 252, x sqrt 252); "
            "weights held fixed and rebalanced daily.",
            "This shows what the historical estimates imply. It is not a recommendation to hold "
            "any of these portfolios.",
        ],
    )


# --- goal planner ---


def simulate_goal(md: MarketData, req: GoalRequest) -> GoalResponse:
    w = weights_for(md, req.portfolio)
    r = md.window(req.lookback_days)
    port = engine_call(lambda: portfolio_returns(r, w))
    params = engine_call(lambda: calibrate_gbm(port))
    plan = SipPlan(
        initial=req.initial_amount,
        monthly=req.monthly_sip,
        years=req.years,
        goal_today=req.goal_amount,
        inflation_annual=req.inflation_annual,
        step_up_annual=req.step_up_annual,
    )
    sim = engine_call(lambda: simulate_sip(params, plan, n_paths=req.n_paths, seed=req.seed))
    return GoalResponse(
        calibration=GbmCalibration(
            window=window_of(r.index),
            log_drift=params.m,
            volatility=params.s,
            median_growth=params.median_growth,
            mean_growth=params.arithmetic_mean,
        ),
        goal_nominal=sim.goal_nominal,
        total_invested=sim.total_invested,
        probability=sim.probability,
        probability_se=sim.probability_se,
        percentiles=[
            PercentileValue(percentile=q, nominal=v, real=sim.final_percentiles_real[q])
            for q, v in sim.final_percentiles.items()
        ],
        fan=GoalFan(
            years=list(sim.years),
            p5=list(sim.fan[5]),
            p25=list(sim.fan[25]),
            p50=list(sim.fan[50]),
            p75=list(sim.fan[75]),
            p95=list(sim.fan[95]),
            invested=list(sim.invested_by_year),
        ),
        n_paths=sim.n_paths,
        seed=sim.seed,
        notes=[
            "Simulated with geometric Brownian motion calibrated on the portfolio's daily log "
            "returns over the window: the future is assumed to look like that history, with "
            "normally distributed log returns (real markets have fatter tails).",
            f"Contributions are invested at the start of each month; the goal is inflated at "
            f"{req.inflation_annual:.1%} a year to {sim.goal_nominal:,.0f} rupees at the horizon.",
            "No taxes, exit loads or missed contributions are modelled; fund NAVs are already net "
            "of expense ratios.",
            f"The probability is the share of {sim.n_paths:,} simulated paths that reach the goal "
            f"(Monte Carlo standard error {sim.probability_se:.1%}). It is a model output, not a "
            "promise or a forecast.",
        ],
    )


def _band_out(b: Band) -> BandOut:
    return BandOut(
        id=b.id,
        label=b.label,
        equity_min=b.equity_min,
        equity_max=b.equity_max,
        description=b.description,
    )


def questionnaire() -> QuestionnaireOut:
    return QuestionnaireOut(
        questions=[
            QuestionOut(
                id=q.id, text=q.text, options=[OptionOut(id=o.id, label=o.label) for o in q.options]
            )
            for q in QUESTIONS
        ],
        bands=[_band_out(b) for b in BANDS],
        disclaimer=DISCLAIMER,
    )


EQUITY_CATEGORIES = {"large_cap", "flexi_cap", "mid_cap", "small_cap"}
DEBT_CATEGORIES = {"debt_corporate_bond", "debt_gilt", "debt_liquid"}


def allocation_mix(md: MarketData, portfolio: PortfolioIn) -> AllocationMix:
    """Weights by broad asset type from the universe categories. Stocks and the index count as
    equity; balanced-advantage funds as hybrid (counted half in `equity_oriented`)."""
    w = weights_for(md, portfolio)
    mix = {"equity": 0.0, "hybrid": 0.0, "debt": 0.0, "gold": 0.0}
    for a, x in w.items():
        g = md.groups.get(a)
        cat = g.category if g is not None else None
        if cat is None or cat in EQUITY_CATEGORIES:
            mix["equity"] += x
        elif cat == "hybrid_balanced_advantage":
            mix["hybrid"] += x
        elif cat in DEBT_CATEGORIES:
            mix["debt"] += x
        elif cat == "gold":
            mix["gold"] += x
        else:
            raise AnalyticsInputError(f"no asset type mapping for category {cat!r}")
    return AllocationMix(**mix, equity_oriented=mix["equity"] + 0.5 * mix["hybrid"])


def profile(md: MarketData | None, req: ProfileRequest) -> ProfileResponse:
    try:
        res = score_answers(req.answers)
    except ValueError as exc:
        raise AnalyticsInputError(str(exc)) from exc
    mix = within = None
    notes = [
        "Bands are this tool's illustrative choice of equity-oriented share for each profile.",
    ]
    if req.portfolio is not None and md is not None:
        mix = allocation_mix(md, req.portfolio)
        within = res.band.equity_min - 1e-9 <= mix.equity_oriented <= res.band.equity_max + 1e-9
        notes.append(
            "Equity-oriented share = equity funds, stocks and the index, plus half of balanced-"
            "advantage funds (their equity share changes over time with each fund's model; half is "
            "a simplifying assumption, not the funds' actual allocation)."
        )
    if res.capped_by_horizon:
        notes.append("A horizon under three years caps the band at moderately conservative.")
    return ProfileResponse(
        score=res.score,
        max_score=res.max_score,
        band=_band_out(res.band),
        capped_by_horizon=res.capped_by_horizon,
        portfolio_mix=mix,
        within_band=within,
        disclaimer=res.disclaimer,
        notes=notes,
    )
