# Methodology

Formulas and conventions used by the deterministic risk engine (`apps/api/app/engine`).
Data handling (alignment, gaps, sources) is described in [data.md](data.md).

## Returns

- Simple daily returns, `r_t = P_t / P_{t-1} - 1`, on the aligned trading calendar
  (`app/engine/returns.py`). Log returns are used only for time aggregation / GBM calibration.
- Annualisation uses 252 trading days.

## Value-at-Risk and CVaR (`app/engine/risk/var.py`)

### Conventions

| Item | Convention |
|---|---|
| Portfolio return | `R_p,t = Σ_i w_i r_i,t`, fixed weights (rebalanced daily) |
| Weights | keyed by asset id, finite, sum to 1 (negative = short; long-only is enforced at the API layer) |
| Loss | `L = -R_p` |
| VaR | positive loss fraction of portfolio value, with confidence level and horizon stated, e.g. "1-day 99% VaR = 2.3%" |
| CVaR | `E[L \| L ≥ VaR]`, same units; always ≥ VaR |
| Sign | not clipped: a negative VaR means a gain even at the chosen quantile |
| Horizon | 1 trading day base; `h`-day = `√h ×` 1-day, `h ≤ 20` |
| Minimum history | 252 observations by default |

`α = 1 − confidence`.

### Historical

Sort the `n` portfolio losses worst-first, `L_(1) ≥ … ≥ L_(n)`, and let `k = ⌈n·α⌉`
(with `n·α` rounded to 9 decimals first, because e.g. `100 × (1 − 0.95) = 5.000000000000004`
in floating point).

    VaR  = L_(k)
    CVaR = (1/k) Σ_{j=1..k} L_(j)

This is the generalised inverse of the empirical CDF (`np.quantile(R, α, method="inverted_cdf")`).
`np.quantile`'s default (`"linear"`) and Excel's `PERCENTILE.INC` interpolate between
neighbouring order statistics and give slightly different values. Requires `n·α ≥ 1`.
With 252 observations a 99% VaR rests on only `k = 3` tail points.

### Parametric (variance-covariance)

    μ_p = wᵀμ̂,   σ_p = √(wᵀΣ̂w)        (sample mean, sample covariance with ddof = 1)
    z   = Φ⁻¹(confidence)              (1.6449 at 95%, 2.3263 at 99%)
    VaR  = z·σ_p − μ_p
    CVaR = σ_p·φ(z)/α − μ_p

Φ⁻¹ and φ come from Python's `statistics.NormalDist`. Assumes normal portfolio returns.

### Monte Carlo

Draw `n_sims` (default 100,000) vectors `X_s ~ N(μ̂, Σ̂)` with
`np.random.default_rng(seed)` (default seed 0, SVD factorisation so singular Σ̂ works),
form `R_s = wᵀX_s`, then apply the historical order-statistic estimator.

For a fixed-weight portfolio `wᵀX` is exactly `N(μ_p, σ_p²)`, so this is the same model as
the parametric method plus sampling noise. It is the base for later non-normal
(e.g. Student-t) or non-linear simulation.

### Horizon scaling

`VaR_h = √h · VaR_1` (same for CVaR) for every method. Exact only for i.i.d. zero-mean normal
returns; it ignores drift (which scales with `h`), volatility clustering, autocorrelation,
the changing shape of fat tails under aggregation, and compounding.

## Risk and performance metrics (`app/engine/risk/metrics.py`)

### Conventions

| Item | Convention |
|---|---|
| Input | one Series of simple daily returns (a portfolio from `var.portfolio_returns`, or one asset); `asset_metrics` / `correlation_matrix` take the DataFrame |
| Benchmark | must have the **identical index**; nothing is re-aligned silently |
| Annualisation | `n` = number of return rows; volatility × √252, means × 252, compounded growth ^(252/n) |
| Risk-free rate / MAR | **annual decimals, required, no default**; daily rate `(1 + r)^(1/252) − 1` |
| Signs | max drawdown is a positive loss fraction (like VaR); the drawdown series is ≤ 0 |
| Undefined ratios | denominator ≤ 1e-12 (daily) → `None` (NaN in tables); never ±inf, never raises |
| Minimum history | 252 observations by default (floor 2) |

### Formulas

    σ          = sd(r, ddof=1) · √252
    CAGR       = (∏(1 + r_t))^(252/n) − 1
    arithmetic = mean(r) · 252
    Sharpe     = mean(r − rf_d) / sd(r) · √252
    DD (MAR m) = √( (1/n) Σ min(r_t − m_d, 0)² ) · √252        over all n days
    Sortino    = mean(r − m_d) · 252 / DD
    D_t        = W_t / max_{s≤t} W_s − 1,  W_0 = 1,  W_t = ∏_{s≤t}(1 + r_s)
    MDD        = −min_t D_t
    β          = cov(p, b) / var(b),   ρ = corr(p, b),   R² = ρ²
    α (Jensen) = [mean(p − rf_d) − β · mean(b − rf_d)] · 252
    resid vol  = sd(p − β b) · √252 = σ_p √(1 − ρ²)
    TE         = sd(p − b) · √252,   active = mean(p − b) · 252,   IR = active / TE

**CAGR vs arithmetic mean.** On the same continuously compounded scale,
`ln(1 + CAGR) ≈ mean(r)·252 − σ²/2` (volatility drag; on the real cache this holds to 0.01 pp).
CAGR compounds but `mean × 252` does not, so CAGR can be the larger number for low-volatility,
high-return series (constant 0.1% a day: 28.6% vs 25.2%). Factsheet-style `(CAGR − rf)/σ` can
therefore be above or below the arithmetic Sharpe.

**Drawdowns.** The starting value counts as a peak (a first-day loss is a drawdown;
`peak_date = None`). A return to within 1e-12 of the peak counts as recovered. Two episodes
are reported: the **deepest** and the **longest underwater** (they can differ). An unrecovered
episode's length runs to the last observation (a lower bound). Ties go to the earliest episode.

**Tracking error vs residual volatility.** A 2× benchmark portfolio has β = 2, residual
volatility 0 and TE = σ_b (it is not tracking the benchmark one-for-one). With free leverage it
also shows α = rf exactly, because no borrowing cost is charged.

### Risk-free rate for India

Not set in the engine. Candidates: the 91-day T-bill yield (RBI auction cut-off, FBIL
benchmark, or the OECD/FRED monthly series `INDIR3TIB01STM`) or FBIL Overnight MIBOR (often
quoted on AMC factsheets, check the factsheet). Use the **average over the measurement window**,
not today's rate. A time-varying rf series is a possible later extension.

### Data limitations that affect these metrics

- **Price-index benchmark.** `^NSEI` excludes dividends; stocks and Growth NAVs include them.
  Alpha, active return and IR are biased **upward** by about the index dividend yield
  (~1–1.5% a year; × β for alpha). β, ρ and TE are barely affected.
- **Filled benchmark days.** The 7 sessions missing from `^NSEI` (data.md) are 0% benchmark
  returns while assets move, followed by a two-day return. This slightly distorts β, ρ and TE
  over windows that contain them (none fall in the last 756 rows as of 2026-10-19). Separately,
  `^NSEI` closes unchanged on 2024-05-08 (22,302.50 on both days), while the median stock rose
  0.23%. This is either a genuinely flat close or a stale Yahoo bar. **Not yet verified** against
  niftyindices.com historical data.
- **Selection / survivorship bias.** The stocks are today's Nifty constituents and the funds
  were picked in 2026, so historical α and IR of the equal-weight baskets are biased upward.
- **Benchmark fit.** β, α, TE and IR against the Nifty are meaningless for debt, liquid and gold
  funds (R² near 0). Report them only for equity-like assets.

## Risk attribution (`app/engine/risk/attribution.py`)

### Conventions

| Item | Convention |
|---|---|
| Input | simple daily returns DataFrame + weights (as `var.py`); fixed weights, `R_p = w'r` |
| Covariance | sample covariance Σ̂ of daily returns (ddof = 1), the same estimate as `parametric_var` |
| Units | volatility figures annualised (× √252); VaR/CVaR components are positive loss fractions with confidence and horizon, like `VaRResult` |
| Order | per-asset results follow the order of `weights`; values do not depend on any ordering |
| Zero portfolio volatility | contributions raise `ValueError` (undefined, not zero); `diversification_ratio` returns `None` |

### Formulas (Euler decomposition)

A risk measure with ρ(k·w) = k·ρ(w) satisfies ρ(w) = Σᵢ wᵢ ∂ρ/∂wᵢ, so components add up exactly.

    σ_p        = √(w'Σ̂w)
    marginalᵢ  = (Σ̂w)ᵢ / σ_p                  = ∂σ_p/∂wᵢ = cov(rᵢ, R_p)/σ_p
    componentᵢ = wᵢ · marginalᵢ                Σᵢ componentᵢ = σ_p
    percentᵢ   = componentᵢ / σ_p              Σᵢ percentᵢ = 1  (may be < 0 or > 1)
    cVaRᵢ      = √h · wᵢ (z (Σ̂w)ᵢ/σ_p − μ̂ᵢ)               Σ = parametric VaR
    cCVaRᵢ     = √h · wᵢ (φ(z)/α (Σ̂w)ᵢ/σ_p − μ̂ᵢ)          Σ = parametric CVaR
    DR         = Σᵢ |wᵢ| σᵢ / σ_p              (≥ 1; Choueifaty's definition for long-only)
    ratioᵢ     = percentᵢ / wᵢ = (Σ̂w)ᵢ / (w'Σ̂w) = β of asset i to the portfolio

`risk_budget_report` lists weight vs `percent` (the risk share) per asset. An equal-risk-
contribution portfolio has all shares equal to 1/n; that is the Phase 6 risk-parity target.
`euler_contributions(cov, w)` is the array-level core the optimiser will call.

### What the numbers mean, and what they do not

- **Estimates, not facts.** They inherit every error in Σ̂. A different window, frequency or
  covariance estimator (EWMA, shrinkage) gives different contributions.
- **Marginal = local derivative.** It holds the other weights fixed and is valid only for small
  changes. A real trade also sells something else, and its effect is the difference of two marginals.
- **Negative components** belong to hedges (negative covariance with the portfolio). Shares can
  exceed 100% for other assets. The sum is still exact.
- **A component is not "risk removed by selling".** That is incremental risk, which needs
  renormalised weights and differs from the component except for infinitesimal changes.
- **Only parametric VaR/CVaR decompose in closed form.** For historical or Monte Carlo VaR the
  Euler component is E[−wᵢrᵢ | L_p = VaR], estimated from the few observations near the
  quantile (noisy, needs a kernel/window choice). It is not provided.
- **The mean term matters for low-volatility assets.** A liquid fund's cVaR can be *negative*
  even though its volatility component is positive: its expected return (−μ̂ᵢ) outweighs its tiny
  volatility contribution. On the real cache (equal-weight 15 funds, 756 days to 2026-10-19), its
  volatility share is +0.01% and its 99% VaR share is −0.13%.
