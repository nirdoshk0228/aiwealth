# API

FastAPI app in `apps/api`. Interactive docs at `/docs` (Swagger) and the OpenAPI spec at
`/openapi.json` while the API runs (`uvicorn app.main:app --reload`). The web app's client is
generated from that spec; types are never hand-written.

## Conventions

| Item | Convention |
|---|---|
| Units | Returns, volatilities, VaR, weights: **decimals** (0.0234 = 2.34%). Rupee values say so (`*_rupees`, `portfolio_value`) |
| VaR / CVaR | Positive loss fractions, with `confidence` and `horizon_days` in the response |
| Stress P&L | **Signed** (−0.15 = lose 15%), because a scenario can be a gain |
| Undefined | `null` (a ratio with a zero denominator), never NaN or infinity |
| Dates | ISO `YYYY-MM-DD`, trading days, ends inclusive |
| Errors | `{"detail": "..."}`: 401 not authenticated, 404 not found (or not yours), 409 email taken, 422 invalid input (schema, unknown asset, too little data), 503 market data or database unavailable |

### Portfolio input

Every analytics endpoint takes the same portfolio object, so the no-login demo, the saved
portfolios and the AI tools all use one shape:

```json
{
  "portfolio_value": 1000000,
  "holdings": [
    { "asset_id": "mutual_fund:122639", "weight": 0.6 },
    { "asset_id": "equity:TCS", "weight": 0.4 }
  ]
}
```

Rules (`app/schemas/portfolio.py`), enforced at the API and not in the engine:

- **long-only**: each weight in [0, 1];
- each asset at most once, at most 40 holdings;
- weights sum to 1 within 1e-6, then renormalised to sum to 1 exactly;
- every `asset_id` must be in the aligned data (`GET /api/v1/universe`), else 422.

## Endpoints

All under `/api/v1` (plus a bare `/health` for probes).

### Public

| Method | Path | What it returns |
|---|---|---|
| GET | `/health` | liveness: `{status, app, version, env}` |
| GET | `/status` | market-data state (`loading` / `ready` / `error`), backend, data end; never blocks (cold-start UI) |
| GET | `/universe` | 15 funds, 20 stocks, Nifty 50 with category/industry, the aligned data window and the factor data end |
| GET | `/demo/portfolios` | sample portfolios from `data/portfolios/` (illustrative, not recommendations) |
| POST | `/risk/var` | historical, parametric and Monte Carlo VaR/CVaR (fraction and rupees) on the last `lookback_days` rows (default 756) |
| POST | `/risk/metrics` | CAGR, arithmetic return, volatility, downside deviation, Sharpe, Sortino, drawdown episodes, Nifty 50 beta/alpha/TE/IR, growth and drawdown series |
| POST | `/risk/attribution` | per-holding marginal/component volatility, volatility share, component parametric VaR/CVaR, diversification ratio |
| POST | `/risk/correlation` | correlation matrix of the holdings (plus the benchmark) |
| POST | `/backtest` | rolling 1-day VaR backtest over the full history: breaches, Kupiec, Christoffersen, Basel zone (99% only), daily series for the chart |
| POST | `/factors/exposures` | Carhart four-factor regression (Newey-West t-stats) of the portfolio and each holding |
| GET | `/stress/scenarios` | committed historical replays and hypothetical shocks with rationale and caveats |
| POST | `/stress/run` | one scenario by `scenario_id`, or a custom hypothetical `scenario` (same schema as `data/scenarios/*.json`) |
| POST | `/optimize` | long-only frontier, minimum variance, maximum Sharpe, maximum return, risk parity, and the current portfolio under the same estimates (`asset_ids` or `portfolio`, optional `max_weight`) |
| POST | `/goals/simulate` | Monte Carlo SIP: probability of reaching an inflated goal, percentiles (nominal and in today's rupees), yearly fan |
| GET | `/goals/questionnaire` | five risk-profile questions and the illustrative bands |
| POST | `/goals/profile` | score answers; with a portfolio, its equity-oriented share vs the band |

### Assistant (rate-limited: 8 requests a minute per client IP, 300 a day per instance)

| Method | Path | What it returns |
|---|---|---|
| GET | `/ai/status` | whether an LLM key is configured, and the model id |
| POST | `/ai/chat` | `{messages, portfolio}` -> **server-sent events**, each `data:` line a `ChatEventEnvelope`: `tool_call`, `tool_result`, `answer` (content, citations, grounding result, disclaimer), `error`, then `done` (trace id, steps, tokens) |
| POST | `/ai/parse-scenario` | `{text}` -> a validated `HypotheticalScenarioSpec` (not run; post it to `/stress/run`) |

Document questions go through the `search_knowledge_base` tool; the `answer` event's
`citations` list the documents and pages cited (see docs/methodology.md, Knowledge base).

429 = rate limited (with `Retry-After`), 503 = no LLM key or free quota exhausted, 502 = provider
error, 422 = a scenario description that could not be turned into valid shocks after one retry.
The answer is sent whole after the guardrails, not token by token (see docs/architecture.md).

### Auth and saved portfolios

| Method | Path | Auth | Notes |
|---|---|---|---|
| POST | `/auth/register` | – | `{email, password}` (8–128 chars); 201 with the user; 409 if the email exists |
| POST | `/auth/login` | – | `{email, password}` → `{access_token, token_type: "bearer", expires_in}` |
| GET | `/auth/me` | Bearer | the current user |
| GET / POST | `/portfolios` | Bearer | list / create `{name, portfolio}` |
| GET / PUT / DELETE | `/portfolios/{id}` | Bearer | another user's portfolio answers 404, like a missing one |

## Request parameters and defaults

| Endpoint | Parameter | Default | Range | Why |
|---|---|---|---|---|
| risk/* | `lookback_days` | 756 (3 years) | 252–5,000, clipped to the data | enough tail points for 99% historical VaR (k = 8) |
| var, attribution | `confidence` | 0.95 | 0.90–0.999 | |
| var, attribution | `horizon_days` | 1 | 1–20 | square-root-of-time scaling degrades beyond a month |
| metrics | `risk_free_annual` | IIMA 91-day T-bill average over the window | −0.5–0.5 | see below |
| backtest | `method` | historical | historical, parametric | Monte Carlo re-estimated ~2,400 times is too slow for a request |
| backtest | `confidence`, `window` | 0.99, 504 | 0.90–0.999, 252–1,260 | Basel convention is 99% |
| optimize | `lookback_days`, `max_weight` | 1,260 (5 years), 1.0 | 252–5,000, 0.05–1 | longer window for return estimates |
| optimize | `include_liquid` | false | true/false | liquid funds are treated as cash and left out (methodology: optimisation) |
| goals/simulate | `lookback_days`, `inflation_annual`, `n_paths` | all data, 0.04, 10,000 | 252–5,000, −0.05–0.5, 1,000–50,000 | 4% = RBI CPI target |

**Default risk-free rate.** The geometric average of the daily 91-day T-bill return (`RF`) in
the IIMA factor file over the window's rows that have factor data, annualised over 252 rows.
If factor data covers less than half of the window (the factor file ends 2025-12-31), the
latest 252 rows of factor data are used instead. The response states the rate, its period and
which rule applied (`risk_free.source`).

## Data flow on a request

```
router (app/api/v1)  ->  service (app/services)  ->  engine (app/engine, pure)
                              |
                              +-> MarketData (in memory, loaded once per process from
                                  Postgres or the Parquet cache; never from a provider)
```

- Market data is loaded once per process (`MarketDataStore`), in a background thread at
  startup; `/status` reports progress, and requests wait for the load or get 503 with the
  reason (driver errors are reported by type only, so no host names leak).
- Backtest results are memoised in an in-process LRU cache (64 entries), keyed by the data
  load, the weights and the parameters. Redis is not used.
- Engine `ValueError` for bad input becomes `AnalyticsInputError` in the service and 422 in the
  response; other exceptions stay 500.

## Database

SQLAlchemy 2 models in `app/db/models.py`, migrations in `apps/api/alembic/`.

| Table | Contents |
|---|---|
| `users` | email (unique, lower-cased), scrypt password hash |
| `portfolios`, `portfolio_holdings` | saved portfolios (value, ordered holdings) |
| `market_assets`, `market_prices` | cached price series and their metadata (copied from Parquet) |
| `factor_datasets`, `factor_returns` | IIMA factor returns and provenance (**not** the raw file) |

```
cd apps/api && alembic upgrade head      # apply migrations (DATABASE_URL_DIRECT or DATABASE_URL)
python scripts/seed_db.py                # copy data/raw/ into Postgres (from the repo root)
```

## Auth

- Passwords: stdlib `hashlib.scrypt`, N = 2^14, r = 8, p = 5 (an OWASP-listed parameter set),
  16-byte random salt, parameters stored with the hash, constant-time comparison. A login for
  an unknown email also runs one scrypt, so response time does not reveal which emails exist.
- Tokens: HS256 JWT (PyJWT) with `sub`, `iat`, `exp`, `type = "access"`; all required on
  decode; `JWT_SECRET` must be at least 32 bytes and is mandatory in production.
