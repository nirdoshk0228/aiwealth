# Data: sources, conventions and limitations

> Work in progress (Phase 2). The TODO section is completed with the returns-cleaning step.

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

## TODO (step d)
- Calendar and alignment: master calendar = dates where at least 50% of stocks have a valid bar
  (^NSEI kept as benchmark only; report days that differ from it, e.g. 2025-03-18 and the
  missing 1 January sessions), forward-fill limit (3 days), minimum history (756 obs), common
  start/end trimming with the asset that sets each bound. Covers holidays, Muhurat sessions and
  debt-market holidays.
- NAV timing vs. market close (gold FoFs correlate only about 0.89 with each other)
