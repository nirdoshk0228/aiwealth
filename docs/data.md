# Data: sources, conventions and limitations

> Work in progress (Phase 2). Sections marked TODO are completed with the cache/seed
> script and returns cleaning steps.

## Sources

| Asset type | Source | Price used |
|---|---|---|
| Mutual funds (Direct, Growth) | [mfapi.in](https://www.mfapi.in/) by AMFI scheme code | NAV per unit (INR) |
| NSE stocks | Yahoo Finance via `yfinance`, `.NS` suffix | Split- and dividend-adjusted close (`auto_adjust=True`) |
| Nifty 50 | Yahoo Finance via `yfinance`, `^NSEI` | Index level (price index, not total return) |

## Usable data window: 2015 onwards

The usable window for analytics starts **no earlier than 1 January 2015**, even when a source
returns older history (Yahoo returns RELIANCE.NS from 1996, for example).

Why: older Yahoo prices for NSE stocks can be unreliable. Known problem types include missing or
mis-applied split and bonus adjustments, stale prices repeated across days, and gaps. These are
hard to detect automatically and would distort volatility, correlations and VaR backtests. A
2015+ window still gives more than 10 years (about 2,500 trading days) of daily data, which is
enough for risk estimation and backtesting.

The cache stores the full history as downloaded. The 2015 cut-off is applied when building
the analytics dataset, so it can be revisited without re-downloading.

## TODO
- Universe selection criteria and the confirmed list (`data/universe.json`)
- Cache layout (`data/raw/`) and the seed script
- Calendar alignment (Nifty 50 trading days), forward-fill limit (3 days), minimum history (756 obs)
- Limitations: survivorship bias, current Nifty constituents applied historically, holidays and
  Muhurat sessions, Nifty 50 as a price index, NAV timing vs. market close
