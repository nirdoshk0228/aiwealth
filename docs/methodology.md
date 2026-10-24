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
not today's rate. A time-varying rf series is a possible later extension. The factor
regression already uses one: the daily T-bill series from the IIMA factor file (see
[Factor regression](#factor-regression-appenginefactorsregressionpy)), so its alpha and
the Sharpe/Sortino ratios here are measured against different risk-free rates.

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

## VaR backtesting (`app/engine/risk/backtest.py`)

### Conventions

| Item | Convention |
|---|---|
| Portfolio | fixed weights (rebalanced daily), today's weights applied to the whole history |
| Forecast | for test day t, `VaR_t` is estimated from rows `t−N … t−1` only (rolling window, default `N = 504`); row t never enters its own forecast |
| Methods | `historical`, `parametric`, `monte_carlo` from `var.py`, re-estimated every test day |
| Horizon | 1 trading day only (overlapping multi-day returns would make breaches autocorrelated by construction) |
| Breach | `I_t = 1` if `−R_p,t > VaR_t`, strictly; expected rate `p = α = 1 − confidence` |
| Monte Carlo | 10,000 simulations per day with the **same seed every day**; results depend on `(n_sims, seed)`: a different seed shifts each VaR by about `√(α(1−α)/n_sims)/φ(z)·σ_p` (≈2.7% of σ_p at 99%), which can flip a borderline day |

### Tests

    Kupiec:   LR_uc = −2[(n−x)ln(1−p) + x ln p − (n−x)ln(1−x/n) − x ln(x/n)]      ~ χ²(1)
              x = 0 → −2n ln(1−p);   x = n → −2n ln p        (0·ln 0 = 0)
    Christoffersen:  n_ij = #days with I_{t−1}=i, I_t=j;  π01 = n01/(n00+n01), π11 = n11/(n10+n11)
              LR_ind = −2[ln L(π) − ln L(π01, π11)]                                  ~ χ²(1)
              LR_cc  = LR_uc + LR_ind                                                 ~ χ²(2)

p-values without scipy: χ²(1) survival = `erfc(√(x/2))`, χ²(2) survival = `exp(−x/2)`.
`LR_ind`/`LR_cc` are `None` when π01 or π11 is undefined (zero breaches, a single breach on the
last day, or breaches on every day).

**Basel traffic light** (green 0–4, yellow 5–9, red 10+ exceptions) is the regulatory convention
for 250 days of 1-day 99% VaR only. It is reported for the last 250 test days at 99% and is `None`
otherwise.

### Reading the results

- **Low power.** At 99% with ~2,400 test days (~24 expected breaches) Kupiec at the 5% level rejects a
  model whose true breach rate is 1.5% only 59% of the time, and with 250 days a true 2% rate only
  24% of the time (exact binomial). "Not rejected" means the data cannot tell the model is wrong;
  it is not proof that the model is correct.
- **Historical VaR's expected rate is `k/(N+1)`, not α.** With `k = ⌈Nα⌉` and i.i.d. continuous
  returns, the chance that the next loss exceeds the k-th largest of N is exactly `k/(N+1)`:
  26/505 = 5.15% at 95% and 6/505 = 1.19% at 99% for N = 504.
- **Christoffersen sees only consecutive-day clustering.** Two breaches a week apart in a volatile
  month do not count as clustered.
- **A rolling equal-weight window reacts slowly.** After a volatility jump the forecast keeps
  using calm data. Historical VaR at 99% catches up once ~k big losses enter the window, while the
  sample σ behind parametric VaR moves more slowly.
- **Survivorship bias.** The 20 stocks are *today's* Nifty constituents, applied back to 2015.
  Companies that were dropped (often after large falls) are missing, so the realised losses of this
  basket are probably milder than those of the investable index portfolio at each past date.

### Real-cache results (2015-01-05 → 2026-10-19 returns, N = 504, 2,405 test days from 2017-01-20)

| Portfolio | Conf | Method | Expected | Breaches | Rate | Kupiec p | p_ind | p_cc | Basel (last 250) |
|---|---|---|---|---|---|---|---|---|---|
| ^NSEI | 95% | historical | 120.3 | 126 | 5.24% | 0.593 | 0.0009 | 0.0034 | – |
| ^NSEI | 95% | parametric | 120.3 | 115 | 4.78% | 0.621 | 0.0012 | 0.0045 | – |
| ^NSEI | 95% | monte_carlo | 120.3 | 117 | 4.86% | 0.760 | 0.0017 | 0.0068 | – |
| ^NSEI | 99% | historical | 24.1 | 30 | 1.25% | 0.240 | 0.0004 | 0.0009 | 5 yellow |
| ^NSEI | 99% | parametric | 24.1 | 41 | 1.70% | 0.0016 | 0.0005 | 0.00002 | 6 yellow |
| ^NSEI | 99% | monte_carlo | 24.1 | 39 | 1.62% | 0.0049 | 0.0003 | 0.00003 | 6 yellow |
| EW20 | 95% | historical | 120.3 | 128 | 5.32% | 0.473 | 0.0036 | 0.0113 | – |
| EW20 | 95% | parametric | 120.3 | 119 | 4.95% | 0.907 | 0.0024 | 0.0098 | – |
| EW20 | 95% | monte_carlo | 120.3 | 118 | 4.91% | 0.833 | 0.0020 | 0.0082 | – |
| EW20 | 99% | historical | 24.1 | 26 | 1.08% | 0.693 | 0.0001 | 0.0005 | 4 green |
| EW20 | 99% | parametric | 24.1 | 36 | 1.50% | 0.023 | 0.118 | 0.022 | 4 green |
| EW20 | 99% | monte_carlo | 24.1 | 36 | 1.50% | 0.023 | 0.118 | 0.022 | 4 green |

EW20 = equal-weight 20 stocks. The rate is right at 95% for every method; normal-based VaR
under-covers at 99% (fat tails), while historical VaR covers. Independence is rejected almost
everywhere, driven by March 2020 (7–8 breaches at 99%) and March 2026 (4–5). All 99% runs land in
yellow or green over the last 250 days, which include March 2026.

## Factor regression (`app/engine/factors/regression.py`)

Carhart four-factor regression of a portfolio's or an asset's excess return on the IIMA Indian
factors (source, units and calendar alignment: docs/data.md, "Factor data"). Pure numpy/pandas.
`factor_regression` takes one return Series; `factor_regression_by_asset` takes a DataFrame and
returns one row per column.

### Conventions

| Item | Convention |
|---|---|
| Model | `r_t − RF_t = α + β_M MKT_RF_t + β_S SMB_t + β_H HML_t + β_W WML_t + ε_t`, daily simple returns in decimals |
| Data window | **only the rows with factor data**: 2,713 rows, 2015-01-05 → 2025-12-31 for release 2025-12. The 196 return rows from 2026-01-01 have no factors yet and are excluded. The caller passes `aligned.factors.dropna()` and the matching return rows |
| Alignment | returns and factors must have **identical indexes**, no NaN, no inf. Nothing is re-aligned or dropped silently |
| Risk-free rate | **the daily 91-day T-bill series from the factor file (`RF`), row by row.** Sharpe and Sortino in `metrics.py` use a **constant** annual rate supplied by the caller. The two can differ (RF averaged 5.8% a year over the regression window, while the constant is whatever the caller passes) |
| Alpha | daily intercept, and `α_annual = 252 · α` (arithmetic, like Jensen's alpha in `metrics.py`; not compounded) |
| Standard errors | Newey–West HAC, Bartlett kernel, lag `L = ⌊4 (n/100)^(2/9)⌋` (Newey & West 1994; **L = 8** for n = 2,713), small-sample factor `n/(n − p)` |
| p-values | two-sided, standard normal (HAC inference is asymptotic). The t-stat and p-value are NaN if a standard error is exactly 0 |
| Minimum history | **504 rows** (two years). Five parameters plus a long-run (HAC) variance with ~8 lags need far more rows than parameters. With one year, the HML and WML betas are unstable |
| Collinearity | condition number of the design matrix after scaling each column to unit length (Belsley, Kuh & Welsch 1980): **raise above 1e6, warn above 30**. A constant factor raises with its own message. The real factors give 1.61 |

### Formulas

    X = [1, MKT_RF, SMB, HML, WML],  p = 5,  y = r − RF
    b       = argmin ‖y − Xb‖²                      (np.linalg.lstsq)
    e       = y − Xb
    R²      = 1 − e'e / Σ(y − ȳ)²,     adj R² = 1 − (1 − R²)(n − 1)/(n − p)
    resid σ = √(e'e/(n − p)) · √252
    u_t     = e_t x_t
    S       = Σ_t u_t u_t' + Σ_{l=1..L} (1 − l/(L+1)) Σ_{t>l} (u_t u_{t−l}' + u_{t−l} u_t')
    V       = n/(n − p) · (X'X)⁻¹ S (X'X)⁻¹,    se = √diag V,    t = b / se

**Why HAC.** Daily returns are heteroskedastic (volatility clusters) and fund NAVs are slightly
autocorrelated. On the real data, the HAC standard errors of the betas are **1.4–3.4× the
classical OLS ones** (EW 20 stocks: β_M 2.2×; EW 15 funds: β_M 3.4×). The alpha standard error
barely changes (1.0–1.1×). The residuals' lag-1 autocorrelation is −0.002 (stocks) and 0.09
(funds), so most of the correction comes from heteroskedasticity. Classical t-stats would
overstate the betas' significance by that factor.

**Lag 0 is White, not classical OLS.** With L = 0, V is White's HC1 estimator. It equals the
classical `s²(X'X)⁻¹` only when `Σ e_t² x_t x_t' = (e'e/n) X'X`, e.g. when every residual has the
same absolute value. The tests check both: lag-0 HAC against an independent HC1 sandwich on
heteroskedastic data, and HC1 = classical on a constructed design with |e_t| constant.

### Verification

- **Unit tests** (`tests/engine/factors/test_regression.py`):
  - noise-free data from known α and β is recovered to 1e-10;
  - a 4-row, 2-regressor case worked out by hand (coefficients, R², adj R², residual
    volatility, standard errors, t-stats);
  - HAC at lags 1, 2 and 5 against a slow loop-by-loop reference;
  - the lag-0 tests above, the lag rule, collinearity and all validation paths.
- **Cross-check** (`scripts/crosscheck_factor_regression.py`): compared with statsmodels 0.15.0,
  `OLS(...).fit(cov_type="HAC", cov_kwds={"maxlags": L, "use_correction": True})`, run in a
  scratch virtualenv (statsmodels is not a project dependency). On the real EW-20 regression
  and on synthetic heteroskedastic AR(1) data, the coefficients, standard errors, t-stats,
  p-values, R² and residual volatility all agree to a relative 1e-13 or better.
  - **statsmodels' default** HAC (`use_correction=False`) omits the n/(n − p) factor. Its
    variances are exactly (n − p)/n times ours (0.99816 at n = 2,713).

### Real-data results (release 2025-12, 2,713 rows 2015-01-05 → 2025-12-31, L = 8)

| Portfolio | β MKT (t) | β SMB (t) | β HML (t) | β WML (t) | α annual (t) | R² | resid σ |
|---|---|---|---|---|---|---|---|
| `^NSEI` (price index) | 1.008 (48.1) | −0.211 (−15.4) | −0.108 (−7.6) | −0.068 (−4.5) | −1.2% (−0.67) | 0.862 | 6.0% |
| EW 20 stocks | 0.991 (55.5) | −0.223 (−14.9) | −0.011 (−0.8) | −0.073 (−4.4) | 5.4% (3.05) | 0.874 | 5.8% |
| EW 15 funds | 0.619 (44.1) | 0.049 (7.3) | 0.035 (4.5) | −0.023 (−3.2) | 3.0% (2.89) | 0.913 | 3.0% |
| EW 9 equity funds | 0.887 (65.0) | 0.099 (9.1) | 0.046 (5.1) | −0.027 (−2.8) | 3.1% (2.22) | 0.922 | 4.1% |

Equal-weight portfolios are rebalanced daily (`var.portfolio_returns`).

### What the numbers mean, and what they do not

- **MKT is a broad market, not the Nifty.** It is IIMA's value-weighted portfolio of all stocks
  used to build the factors (mid and small caps included). So Nifty's β is ≈ 1 but its R² is
  0.86, and large-cap portfolios load **negatively on SMB** (Nifty −0.21). Small-cap funds load
  +0.41 to +0.47, mid-cap funds +0.20, large-cap funds −0.10 to −0.16.
- **Price-index alpha.** `^NSEI` excludes dividends while the factors are total returns, so its
  alpha is biased **down** by about the dividend yield (~1.2–1.5% a year). The −1.2% (t −0.67)
  is consistent with zero once that is allowed for.
- **The positive alphas are not evidence of skill.** EW 20 stocks show 5.4% (t 3.05), but the
  stocks are today's Nifty constituents chosen in 2026 (look-ahead/survivorship bias,
  docs/data.md). The funds are popular survivors, Direct plans, net of expenses.
- **Non-equity assets.** For debt, liquid and gold funds, R² ≈ 0 and the "alpha" is simply
  their return over the T-bill. The liquid fund's t-stat of 11.8 reflects a tiny but very
  steady spread over RF, not skill. Report factor exposures only for equity-like assets.
- **Factor correlations.** MKT and HML correlate 0.42 on daily data over the window (other pairs
  |ρ| < 0.1). The condition number (1.6) shows this is not a numerical problem, but β_HML is
  conditional on β_MKT, as in any multiple regression.
- **Windows and stability.** The betas are averages over 2015–2025. Rolling regressions are a
  possible extension. Two return rows (2019-02-14 and 2019-04-01) span two sessions, with the
  factors compounded to match (docs/data.md).
