"""VaR backtest service: rolling out-of-sample backtest over the full aligned history.

The backtest re-estimates VaR on every test day (about 2,400 for a 504-row window), so results
are memoised in an in-process LRU cache keyed by the market-data load, the weights and the
parameters. Monte Carlo is not offered on the request path (see BacktestRequest).
"""

from app.core.cache import LRUCache
from app.engine.risk.backtest import BacktestResult, var_backtest
from app.engine.risk.var import tail_count
from app.schemas.analytics import (
    BacktestRequest,
    BacktestResponse,
    ChristoffersenOut,
    KupiecOut,
)
from app.services._common import engine_call, weights_for, window_of
from app.services.market import MarketData

_CACHE: LRUCache[BacktestResult] = LRUCache(maxsize=64)


def _run(md: MarketData, req: BacktestRequest, w: dict[str, float]) -> BacktestResult:
    key = (
        md.cache_key,
        tuple(sorted(w.items())),
        req.method,
        req.confidence,
        req.window,
    )
    return _CACHE.get_or_compute(
        key,
        lambda: engine_call(
            lambda: var_backtest(
                md.returns, w, method=req.method, confidence=req.confidence, window=req.window
            )
        ),
    )


def backtest_report(md: MarketData, req: BacktestRequest) -> BacktestResponse:
    w = weights_for(md, req.portfolio)
    res = _run(md, req, w)
    c = res.christoffersen
    notes = [
        "For each test day, VaR is estimated only from the preceding window, so the forecast "
        "never sees the day it is tested on. A breach is a loss strictly greater than VaR.",
        "Kupiec tests the breach rate; Christoffersen tests whether breaches cluster on "
        "consecutive days. A high p-value means the data cannot reject the model, which is "
        "not proof that the model is right (the tests have low power at 99%).",
        "Today's weights are applied to the whole history.",
    ]
    if req.method == "historical":
        k = tail_count(req.window, req.confidence)
        notes.append(
            f"Historical VaR uses the k-th worst loss of the window (k = {k}); with independent "
            f"returns its expected breach rate is k/(N+1) = {k / (req.window + 1):.2%} "
            f"(nominal {1 - req.confidence:.2%})."
        )
    if res.basel_zone_last_250 is not None:
        notes.append(
            "The Basel traffic light (green 0-4, yellow 5-9, red 10+ exceptions) is a "
            "regulatory convention for 250 days of 1-day 99% VaR only."
        )
    return BacktestResponse(
        method=req.method,
        confidence=res.confidence,
        window=res.window,
        horizon_days=res.horizon_days,
        test_period=window_of(res.dates),
        n_breaches=res.n_breaches,
        expected_breaches=res.expected_breaches,
        breach_rate=res.breach_rate,
        expected_rate=res.expected_rate,
        kupiec=KupiecOut(lr_uc=res.kupiec.lr_uc, p_value=res.kupiec.p_value),
        christoffersen=ChristoffersenOut(
            n00=c.n00,
            n01=c.n01,
            n10=c.n10,
            n11=c.n11,
            lr_ind=c.lr_ind,
            p_ind=c.p_ind,
            lr_cc=c.lr_cc,
            p_cc=c.p_cc,
        ),
        basel_breaches_last_250=res.basel_breaches_last_250,
        basel_zone_last_250=res.basel_zone_last_250,
        dates=list(res.dates),
        var=list(res.var),
        realised=list(res.realised),
        breach_dates=list(res.breach_dates),
        notes=notes,
    )
