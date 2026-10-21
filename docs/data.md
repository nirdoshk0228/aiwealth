# Data: sources, conventions and limitations

## Sources

| Asset type | Source | Price used |
|---|---|---|
| Mutual funds (Direct, Growth) | [mfapi.in](https://www.mfapi.in/) by AMFI scheme code | NAV per unit (INR) |
| NSE stocks | Yahoo Finance via `yfinance`, `.NS` suffix | Split- and dividend-adjusted close (`auto_adjust=True`) |
| Nifty 50 | Yahoo Finance via `yfinance`, `^NSEI` | Index level (price index, not total return) |

## Universe

Defined in [`data/universe.json`](../data/universe.json) and validated by
`app/data/universe.py` (Pydantic, unknown keys rejected). It contains 15 mutual funds, 20 stocks
and the Nifty 50 benchmark, selected and verified on 2026-10-16 to 2026-10-21:

- **Funds**: Direct Plan, Growth option only, across large, flexi, mid and small cap, debt
  (corporate bond, gilt, liquid), gold FoF and balanced advantage. Every scheme code was found
  through the mfapi.in search (never typed from memory), and its full history was downloaded and
  checked: all 15 start in 2013. A fund was not used if its history would shorten the common
  window (Parag Parikh Liquid starts 2018-05; Axis Liquid used instead). The stored name must
  match the name mfapi.in returns. The seed script warns if it ever stops matching (a mistyped
  code or a renamed scheme).
- **Stocks**: 20 current Nifty 50 constituents spread across 12 NSE industries (at most 4 per
  industry), not the top 20 by weight. Name, industry and ISIN match the official niftyindices.com
  list exactly. Big daily moves were checked against known events. TITAN +18.8% on 2017-11-06
  (on 10x normal volume) is believed to be real but has not been verified against the NSE
  bhavcopy.
- **ITC → NESTLEIND swap**: ITC was in the approved list but was removed because Yahoo does not
  adjust its 2025-01-06 ITC Hotels demerger (evidence in
  [Are prices dividend-adjusted?](#are-prices-dividend-adjusted)). NESTLEIND keeps FMCG at 2 stocks.
- **Gold FoF check**: Nippon India Gold Savings fell 9.9% on 2026-02-02 (vs. 2026-01-30). HDFC
  Gold ETF FoF (−10.8%) and SBI Gold (−8.5%) fell the same day, so the move is real and the fund
  was kept.

## Usable data window: 2015 onwards

The usable window for analytics starts **no earlier than 1 January 2015**, even when a source
returns older history (Yahoo returns RELIANCE.NS from 1996, for example).

Why: older Yahoo prices for NSE stocks can be unreliable. Known problem types include missing or
mis-applied split and bonus adjustments, stale prices repeated across days, and gaps. These are
hard to detect automatically and would distort volatility, correlations and VaR backtests. A
2015+ window still gives more than 10 years (about 2,900 trading days) of daily data, which is
enough for risk estimation and backtesting. Pre-2015 Yahoo data is also **not stable between
downloads**: phantom-bar totals for the same stock change from one download to the next
(NESTLEIND 1,719 / 833 / 1,719; TCS 309 / 310 / 22). The 2015+ part was identical in every
download (2,910–2,911 rows per stock, the same phantom dates).

The cache stores the full history as downloaded. The 2015 cut-off is applied when building
the analytics dataset, so it can be revisited without re-downloading.

## Are prices dividend-adjusted?

| Series | Dividends | Splits/bonus | Notes |
|---|---|---|---|
| Stocks (`.NS`) | **Yes**: back-adjusted | Yes | Total-return style, pre-tax |
| Nifty 50 (`^NSEI`) | **No**: price index | n/a | Excludes the index dividend yield |
| Mutual funds (Growth NAV) | **Yes**: retained inside the NAV | n/a | Net of expense ratio (TER) |

**Stocks.** `Ticker.history(auto_adjust=True)` multiplies every price *before* an ex-date by
`1 − D / P_prev`, where `D` is the dividend and `P_prev` the previous close. This was verified
on ITC's 2026-05-27 ex-date (D = ₹8.00, P_prev = ₹301.65), before ITC was dropped for the
demerger issue below: the adjustment ratio is 0.97348 (= 1 − 8/301.65), and the ex-date return
is −0.58% adjusted against −3.22% raw. The adjusted
close therefore behaves like a total-return price with dividends reinvested at the ex-date close,
before tax.

**Demergers are not reliably adjusted.** Yahoo adjusts for dividends, splits and bonuses (e.g.
TATASTEEL 1:10 split 2022-07-28, NESTLEIND 1:10 split 2024-01-05 and 1:1 bonus 2025-08-08), but
spin-offs are inconsistent. The check below compares the stock's return on the demerger date
with the Nifty 50 and its sector peers. Because the market moved on those days too, an
unadjusted demerger appears as a residual drop, not as the raw return:

| Demerger | Date | Stock | Nifty 50 | Peers | Residual | Verdict |
|---|---|---|---|---|---|---|
| RELIANCE → Jio Financial | 2023-07-20 | −0.12% | +0.74% | – | −0.9 pp | Adjusted: kept |
| ITC → ITC Hotels | 2025-01-06 | −4.55% | −1.62% | FMCG mean −0.87% (HINDUNILVR, NESTLEIND, TATACONSUM, BRITANNIA) | −2.9 pp vs Nifty, −3.7 pp vs FMCG | **Unadjusted: ITC removed** |

ITC was replaced by NESTLEIND (FMCG, official Nifty 50 list). If kept, ITC's unadjusted demerger
would add a fake loss of about 3–4% on one day, which is enough to distort its VaR, drawdown and
beta. NESTLEIND was checked the same way before it was added: 2,911 rows since 2015, its 1:10 split
and 1:1 bonus adjusted, no demerger, and only the 2 market-wide phantom dates since 2015. Its
largest moves since 2015 match events (COVID 2020, the Maggi ban on 2015-06-03). Its −7.8% on
2026-10-15 opened at the prior close and fell during the session, so it is a real move and not a
missed corporate action. Only demergers known in advance were checked. An unadjusted spin-off in another stock would
show up as a large one-day residual against the Nifty with no corporate action recorded, so
this remains a known limitation.

**Benchmark mismatch.** `^NSEI` is the *price* index. Comparing total-return stocks and funds
against it makes them look better by roughly the index dividend yield (about 1–1.5% a year).
This biases alpha and tracking statistics upward. A total-return index (Nifty 50 TRI) would fix
this, but it has not been verified to be available from the current sources. This is a known
limitation for now.

**Mutual funds.** Growth-option NAVs never pay out: income received by the scheme stays in the
NAV, and the expense ratio is deducted daily. No adjustment is needed or applied.

## Survivorship and selection bias

The universe was chosen in October 2026 using information that was not available historically:

- **Stocks are current Nifty 50 constituents** (checked against the official niftyindices.com
  list, 2026-10-16). Companies that dropped out of the index, often after poor performance,
  are absent. Backtests over 2015–2026 will therefore overstate returns and may understate
  drawdowns compared with what an investor holding the *then-current* index would have seen.
- **Funds are popular surviving schemes with 5+ years of history.** Merged, closed or renamed
  underperformers are excluded, and today's popularity partly reflects past performance.
- **Recent listings are excluded** (e.g. JIOFIN, ETERNAL, BSE) because they lack history, and
  TMPV is excluded because the 2025 Tata Motors demerger breaks its price history.

What this means for results: the risk and return figures describe *this* universe, not the
Indian market as it was investable at each date. Point-in-time constituents would remove the
stock bias, but they are out of scope for now.

## Cache and seeding

Request paths never call mfapi.in or Yahoo. `scripts/seed_data.py` (via `app/data/seed.py`, the
only module allowed to import providers; this is enforced by `tests/test_architecture.py`)
downloads the full history of every asset into a Parquet cache. Everything else reads that cache
through the `SeriesRepository` interface (`app/data/repository.py`), so a Postgres
implementation can replace it in Phase 4.

```
data/raw/                      (gitignored, about 3 MB for 36 assets)
  mutual_fund/120586.parquet   column "price" (float64), index "date" (tz-naive)
  mutual_fund/120586.meta.json source, fetched_at (UTC), first/last date, rows, warnings
  equity/M%26M.parquet         symbols are percent-encoded: M&M -> M%26M, ^NSEI -> %5ENSEI
  index/%5ENSEI.parquet
```

- A missing asset raises `NotInCacheError`. There is no fallback to the network.
- Writes are atomic (temp file then `os.replace`). Metadata is written last and `exists()`
  requires it, so an interrupted seed never leaves an entry that looks complete.
- The seed run continues past failures, prints a per-asset summary and exits 1 if anything
  failed. Run it with `python scripts/seed_data.py` (or `--only CODE_OR_SYMBOL`, `--kind`).

## Yahoo quirks handled when data is downloaded

| Quirk | Example | Handling |
|---|---|---|
| Live intraday bar for today | 2026-10-15 bar fetched during market hours | Bars dated today (IST) or later are dropped |
| Latest session incomplete for hours after close | RELIANCE 2026-10-15: volume present, Close NaN; ^NSEI row missing | NaN dropped; next seed picks it up |
| Phantom bars: Volume 0 and Open = High = Low = Close = previous close | All 20 stocks on 2026-01-15 (no MF NAVs, so a holiday) **and on 2025-03-18, a real trading day** (Nifty +1.45%); single stocks on MARUTI 2015-08-12 and M&M 2025-09-08 | Dropped for equities only (index volume is unreliable), listed in the cache warnings |
| Unadjusted demergers | ITC / ITC Hotels 2025-01-06 | ITC excluded (see above) |

Why phantom bars are dropped rather than kept: on 2025-03-18 keeping them would give every
stock a 0% return on a +1.45% market day, which would bias betas and correlations down. With
them dropped, that day's move goes into the next day's return. Phantom bars are mostly pre-2015
(roughly 10–1,700 per stock, and the count changes between downloads). Since 2015 there are 4
dates in total: 2025-03-18 and 2026-01-15 for all 20 stocks, plus MARUTI 2015-08-12 and M&M
2025-09-08. That is 2 per stock (3 for MARUTI and M&M), and the count is the same in every
download.

## Calendar and alignment

`app/engine/returns.py::align_prices` turns the cached series into one NaN-free price table.
It is pure pandas (no I/O, no `app.data` imports, enforced by `tests/test_architecture.py`).
`app/services/market_data.py::load_aligned_prices` loads the cache and calls it. Every step is
recorded in an `AlignmentReport`, and `report.summary()` prints it.

1. **Start:** everything before `universe.data_start` (2015-01-01) is cut.
2. **Master calendar:** dates where **at least 50% of the 20 stocks** have a bar (after phantom
   bars were dropped at download). The denominator is fixed at 20, so an incomplete latest
   session cannot pass. `^NSEI` is an ordinary column and never adds or removes days.
   Weekend sessions that pass the rule (Muhurat, Saturday budget day) are kept and listed.
3. **Off-calendar prices are discarded and counted**, e.g. liquid-fund weekend NAVs and
   31 March NAVs on holidays. The next calendar price already includes the accrual.
4. **Assets with fewer than 756 real prices** (about 3 years) on the calendar are dropped before
   the window is computed, so a dropped asset cannot shrink the data. The benchmark is mandatory.
5. **Common window:** from the latest first price to the earliest last price, with the asset(s)
   that set each bound reported.
6. **Forward-fill:** gaps of **at most 3 consecutive calendar days** are filled with the last
   real price, which gives a 0% return followed by a catch-up return. Longer gaps are not filled
   at all, and nothing is filled past an asset's last real price.
7. **Rows still containing a gap** (> 3 days in some asset) are dropped for every asset, so the
   next return spans two sessions for all of them. If one asset causes more than 1% of rows to
   be dropped, the report shows a warning.

### Result on the real cache (seeded 2026-10-20)

| | |
|---|---|
| Common window | 2015-01-02 → 2026-10-19, **2,910 rows × 36 assets**, no asset dropped, no gap rows |
| Start set by | `^NSEI` (no bar on 2015-01-01, a real session; nothing is filled before a first price) |
| End set by | all 36 assets (all end 2026-10-19) |
| On our calendar, not in `^NSEI` (7) | 2015-04-15, 2016-01-01, 2016-08-12, 2018-01-01, 2019-01-01, 2019-10-27 (Sun), 2020-11-14 (Sat) |
| In `^NSEI`, not on our calendar (1) | 2025-03-18 (every stock has a phantom bar; see Yahoo quirks) |
| Weekend sessions kept (3) | 2019-10-27, 2020-11-14 (Muhurat), 2025-02-01 (budget Saturday) |
| Forward-filled values | debt funds 64 each, liquid fund 44, equity/hybrid/gold funds 11–12, `^NSEI` 7, MARUTI 1, M&M 1, other stocks 0 |
| Off-calendar prices discarded | liquid fund 952 (weekends and holidays), other funds 6–15, `^NSEI` 1 (2025-03-18) |

Where the fills come from: debt funds miss bank and RBI holidays when stocks trade. Equity funds
publish no NAV on Muhurat sessions and the 3 weekend sessions, and also not on 2023-09-28 and
2025-09-29 (stocks traded on both days; the reason is not verified). The liquid fund's 44 fills
are isolated missing days in the mfapi.in data (2018–2025), each worth about 0.02% of accrual.

### Limitation: filled benchmark days

`^NSEI` is missing 7 real sessions (table above). On each one, the benchmark gets a filled
**0% return while stocks and funds move**, and the next day carries a **two-day return**. That
is 7 days out of 2,910 (0.24%). It slightly distorts benchmark-relative statistics (beta,
correlation, tracking error, relative VaR) on those days, and it can make a single day's
relative performance look large. Total returns over longer periods are unaffected. Fixing it
would need a second source for the index.

Similarly, a fund's filled day on a Muhurat or weekend session is a 0% return and the move
lands on the next NAV. Funds publish no NAV on those days, so there is no real price to use.

## Return conventions

- **Simple returns** everywhere: `r_t = P_t / P_{t-1} − 1` (decimal). `simple_returns` drops
  the first row and refuses NaN or non-positive prices. Annualisation uses 252 trading days.
- **Why simple returns:** they aggregate exactly *across assets*. A portfolio with weights `w_i`
  at the start of the day returns `R_p = Σ w_i r_i`, so portfolio P&L (`value × R_p`) and VaR as
  a rupee loss follow directly. Log returns do not: `ln(1 + R_p) ≠ Σ w_i ln(1 + r_i)` (a test
  checks this with numbers: Rs 1,000 portfolio, simple 2.00% vs weighted log 1.50%).
- **`log_returns`** (`ln(P_t / P_{t-1})`) is provided only where adding up returns *over time*
  matters, e.g. calibrating a GBM for Monte Carlo goal planning. It is not for portfolios.
- A "day" is a calendar row. Where a session was skipped (e.g. 2025-03-18), the return covers
  two sessions, identically for all assets.

## Other known limitations

- **NAV timing:** the three gold FoFs correlate only about 0.89 with each other on daily
  returns, although they hold the same asset. The likely cause is differences in when the
  underlying ETF prices are struck (not verified). Daily correlations involving FoFs are
  therefore understated, and multi-day horizons are less affected.
- **Fund NAV vs. stock close:** fund NAVs are struck at the end of the day from closing
  prices, so they line up with stock closes for domestic equity funds. Gold and overseas
  holdings may not.
