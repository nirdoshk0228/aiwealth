# Data: sources, conventions and limitations

## Sources

| Asset type | Source | Price used |
|---|---|---|
| Mutual funds (Direct, Growth) | [mfapi.in](https://www.mfapi.in/) by AMFI scheme code | NAV per unit (INR) |
| NSE stocks | Yahoo Finance via `yfinance`, `.NS` suffix | Split- and dividend-adjusted close (`auto_adjust=True`) |
| Nifty 50 | Yahoo Finance via `yfinance`, `^NSEI` | Index level (price index, not total return) |
| Fama-French-Carhart factors | IIM Ahmedabad, daily survivorship-bias-adjusted file (see [Factor data](#factor-data-iim-ahmedabad)) | Daily MKT−RF, SMB, HML, WML, RF |

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
- **^NSEI flat close check**: ^NSEI closed at exactly 22,302.50 on both 2024-05-07 and
  2024-05-08. Verified against the niftyindices.com historical data (different opens, highs,
  lows and volumes on the two days), so it is a genuine coincidence, not a stale Yahoo bar.

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
  factors/iima_ff4_daily/factors.parquet   factor returns as decimals (see Factor data)
  factors/iima_ff4_daily/meta.json         release, source URL, sha256, units, citation
  factors/iima_ff4_daily/source/<file>.csv the downloaded file, byte for byte
```

- A missing asset raises `NotInCacheError`. There is no fallback to the network.
- Writes are atomic (temp file then `os.replace`). Metadata is written last and `exists()`
  requires it, so an interrupted seed never leaves an entry that looks complete.
- The seed run continues past failures, prints a per-asset summary and exits 1 if anything
  failed. Run it with `python scripts/seed_data.py` (or `--only CODE_OR_SYMBOL`, `--kind`).
  Factors are seeded too; `--kind factors` downloads only them and `--factor-release YYYY-MM`
  picks the release.

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

## Factor data (IIM Ahmedabad)

**Source.** Indian Fama-French and momentum factors published by IIM Ahmedabad
(<https://faculty.iima.ac.in/~iffm/Indian-Fama-French-Momentum/>), built from CMIE Prowess.
Cite as:

> Agarwalla, S. K., Jacob, J. and Varma, J. R. (2013). *Four factor model in Indian equities
> market.* Working Paper No. 2013-09-05, Indian Institute of Management Ahmedabad.

**Licence.** The page and its FAQ state no licence or terms of use. The file is therefore never
committed: `scripts/seed_data.py` downloads it into the gitignored `data/raw/`, and the test
fixture (`tests/data/fixtures/iima_ff4_daily_synthetic.csv`) copies the file's format with
made-up values.

**File used.** Release `2025-12` (server Last-Modified: 2026-01-22),
daily, survivorship-bias adjusted, which the page labels "Recommended":
`DATA/2025-12_FourFactors_and_Market_Returns_Daily_SurvivorshipBiasAdjusted.csv`. The server
redirects `/~iffm/` to `/iffm/` (HTTP 301); the final URL is stored in the cache metadata.

| Item | What the file contains | How it was established |
|---|---|---|
| Header | `Date,SMB,HML,WML,MF,RF`, CRLF line endings, missing values are `NA` | Read from the file. The parser rejects any other header |
| Rows | 8,004, from 1993-10-01 to 2025-12-31, 40 weekend sessions | Read from the file |
| Units | **Percent**, simple (holding-period) returns per row | Daily RF sums to 5.6% over 2025, the 91-day T-bill level. The release's monthly file equals `∏(1 + daily) − 1` to about 1e-15 in every column; summing the daily values (reading them as log returns) is off by up to 0.7 percentage points a month |
| `MF` | Market risk premium Rm − Rf. The market is "the value-weighted portfolio of all the stocks involved in the estimation of SMB, HML, and WML portfolios", and Rf "computed using the 91-days T-bill rate, is deducted from the return of the market portfolio" | Working paper, section 5.3. It is a broad BSE/Prowess market, **not the Nifty 50**: its daily correlation with Nifty 50 minus RF is 0.94 (2007-09 to 2025-12) |
| `RF` | 91-day T-bill: "annualized yields of new issuances of 91-day Treasury Bills … provided by the Reserve Bank of India" | FAQ. Each row covers the time since the previous row, so a Monday's RF is about three weekdays' worth |
| NA | 1993-10-01 (MF, RF) and 1994-11-28 (SMB, HML, WML) | Read from the file. Both are before our 2015 window |

The working paper (section 5.1) defines stock returns with `ln(·)`, yet the files are simple
returns as shown above. We follow the files and the page's statement that values are holding
period returns.

**Conversion.** `app/data/providers/iima_factors.py` divides every value by 100 exactly once,
at parse time, and renames `MF` to `MKT_RF`. The cache holds decimal simple returns
(`0.01` = 1%) only, and its metadata says so (`units_source`, `units_stored`). To catch a units
change in a later release, the parser rejects the file if any |value| exceeds 0.5 after
conversion, if RF is outside [0, 0.01], or if the standard deviation of MKT_RF is outside
[0.002, 0.05] (the real value is 0.0136).

**NA rows are kept as NaN, not dropped.** If a dropped row fell on a day that gets compounded
into the next calendar row (below), that day's return would silently disappear. As NaN it
blanks the affected row instead. Both NA rows are listed in the metadata warnings.

**Cache metadata** (`meta.json`): dataset, source, release, final source URL, citation,
download time (UTC), raw file path, sha256 and size, source and stored units, columns,
first/last date, rows, weekend rows, NA rows and warnings. IIMA revises earlier years between
releases (FAQ: "CMIE makes changes in the data between different releases"), so record the
release and sha256 alongside any published result.

### Alignment to the master calendar

`app/engine/factors/align.py::align_factors` (pure) puts the factors on the rows of
`simple_returns(prices)`. `app/services/factors.py::load_aligned_factors` loads the cache and
calls it with the price calendar. Row c_i covers (c_{i−1}, c_i]:

- **Row gets factor data** if both c_{i−1} and c_i are factor dates. Every factor row dated in
  (c_{i−1}, c_i] is compounded: `∏(1 + x) − 1` for SMB, HML, WML and RF, and for MKT_RF
  `∏(1 + MKT_RF + RF) − ∏(1 + RF)` (total market over the period minus RF over the period).
  IIMA's monthly file compounds MF directly. The difference is about MKT_RF × RF, around 1e-6
  for a two-day fold. Compounding the long-short SMB, HML and WML is the usual approximation.
- **Factor day not on our calendar** (special sessions): compounded into the next calendar row
  and listed in the report.
- **Calendar day missing from the factor file:** the factor return around it spans two of our
  rows, so **both** rows are left NaN rather than given a mismatched period.
- **Outside the factor coverage, or an NA being compounded:** NaN, counted in the report.

### Result on the real data (release 2025-12, calendar from the cache seeded 2026-10-20)

| | |
|---|---|
| Return rows | 2,909 (2015-01-05 → 2026-10-19), **2,713 with factor data** |
| After factor coverage | **196 rows from 2026-01-01**: the file ends 2025-12-31. Regressions stop there until IIMA publishes a newer release |
| Calendar days missing from the factor file | 0 |
| Factor days compounded into the next row (11) | 2015-02-28, 2020-02-01 (budget Saturdays); 2015-11-11, 2016-10-30, 2023-11-12 (Muhurat); 2024-01-20, 2024-03-02, 2024-05-18 (Saturday sessions); 2025-03-18 (Yahoo phantom-bar day, see Yahoo quirks); **2019-02-13 and 2019-03-29** (below). The session labels come from the weekday and the known exchange calendar and were not checked against NSE circulars |

**2019-02-13 (Wed) and 2019-03-29 (Fri) are missing from our master calendar.** Yahoo has no
bar for any of the 20 stocks or `^NSEI` on either day. All 15 funds published a NAV and IIMA
has a row, so both were very likely trading sessions (not yet checked against the NSE
bhavcopy). Our 2019-02-14 and 2019-04-01 returns therefore each span two sessions for every
asset. The factor alignment compounds the factors over the same span, so the regression
inputs stay consistent. Fixing the calendar would need a second source for stock prices.

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

## Stress scenario windows

Historical replay windows are in `data/scenarios/*.json` (schema: `app/data/scenarios.py`). Every
window was read off our aligned cache (Nifty 50 closes), not typed from memory. Each file records
the Nifty closes at both ends in `benchmark_check`. `tests/services/test_stress.py` re-reads the
cache and checks that the closes still match and that, for drawdown windows, the start is the
window's highest close and the end its lowest. A re-seed that moves a peak therefore fails a test
instead of silently changing a scenario.

| Scenario | Window (closes) | Sessions | Nifty | How the dates were chosen |
|---|---|---|---|---|
| Demonetisation | 2016-11-08 → 2016-12-26 | 33 | 8,543.55 → 7,908.25 (−7.44%) | announced after the 8 Nov close; end = lowest close after it |
| 2018 drawdown (IL&FS period) | 2018-08-28 → 2018-10-26 | 39 | 11,738.50 → 10,030.00 (−14.55%) | 2018 peak to trough |
| COVID crash | 2020-01-14 → 2020-03-23 | 47 | 12,362.30 → 7,610.25 (−38.44%) | 2020 peak to trough |
| Election results day | 2024-06-03 → 2024-06-04 | 1 | 23,263.90 → 21,884.50 (−5.93%) | the counting day, close to close |
| Sep 2024 – Mar 2025 | 2024-09-26 → 2025-03-04 | 109 | 26,216.05 → 22,082.65 (−15.77%) | peak to trough |
| 2026 sell-off | 2026-01-02 → 2026-03-30 | 57 | 26,328.55 → 22,331.40 (−15.18%) | peak to trough; most of it in March (−11.31% from 2026-02-27) |

Caveats recorded in the files:

- **Demonetisation:** the US election result (9 Nov 2016) falls inside the window. The gold fund's
  −9.8% over the window reflects gold's post-election fall, not demonetisation.
- **2026 sell-off:** the cause is marked "verify" (not documented here). The gold fund's −9.9% day
  on 2026-02-02 is inside the window and is genuine (see "Gold FoF check" above).
- **Sep 2024 – Mar 2025:** the commonly cited causes (foreign outflows, weaker earnings) are
  marked "verify".

### Why the 2018 replay shows no credit stress

The 2018 window is labelled an **equity** drawdown covering the IL&FS period. It is not a debt stress
test. Over the window our three debt funds all *gained*: ABSL Corporate Bond +1.0%, SBI Gilt +1.0%,
Axis Liquid +1.2%. Their worst single day in Sep–Oct 2018 was −0.36% (SBI Gilt). The reason:

- **None of our three debt funds held IL&FS or DHFL paper.** The losses of 2018 were concentrated in
  the schemes that did, which marked those holdings down.
- **Our universe excludes those schemes by construction.** It consists of popular funds that still exist
  today with long histories (see "Survivorship and selection bias"). Schemes hit by the 2018
  defaults were often wound up, merged or fell out of favour, so they are not in a 2026 selection of
  popular funds.

The replay is therefore correct for *these* funds but must not be read as "debt funds were safe in
2018". For a credit shock, use the hypothetical `credit_event` scenario. It applies direct shocks
(corporate bond −2%, liquid −0.5%, gilt 0%) that are stated assumptions, because our cache has no
credit loss to calibrate them from. Its equity leg (MKT_RF −5%) matches the Nifty fall over the
fortnight of the DHFL crash (2018-09-14 → 2018-09-28: −5.08%).
