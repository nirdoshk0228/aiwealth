# AIWealth

**An Aladdin-inspired portfolio risk and goal-planning platform for Indian investors, with an AI assistant.**

> Status: in development. This README is updated as each phase is completed.

## What it will do

- **Risk analytics:** historical, parametric and Monte Carlo VaR/CVaR, volatility, drawdown, factor exposures
- **Stress testing:** historical scenarios (e.g. COVID March 2020) and custom shocks
- **Portfolio optimization:** mean-variance, risk parity, Black-Litterman, efficient frontier
- **Goal-based planning:** Monte Carlo SIP simulations giving the probability of reaching a goal
- **AI assistant:** an LLM that calls the risk engine as tools and answers questions from fund and SEBI documents (RAG), with cited sources

## Design principles

1. **Numbers are deterministic.** The risk engine never calls an LLM, and the LLM never computes numbers. It only explains tool outputs.
2. **Everything is validated.** VaR is backtested (Kupiec test) and the AI layer is evaluated on a golden question set.
3. **Market data is cached.** No external API calls on a user request path.

## Inspiration and scope

Inspired by the idea behind BlackRock's Aladdin Wealth: one consistent view of risk across a client's holdings, plus goal tracking for advisors. This is an independent student project. It covers a small slice (analytics for mutual funds and large-cap stocks), not trading, operations or institutional scale. It is not affiliated with BlackRock.

## Tech stack

Python, FastAPI, PostgreSQL (pgvector), Redis, Next.js, TypeScript, Docker, GitHub Actions

## Roadmap

- [ ] Phase 1: repo scaffold, API health check, CI
- [ ] Phase 2: market data layer (mutual fund NAVs, stocks, indices)
- [ ] Phase 3: risk engine (VaR/CVaR, metrics, factors, stress tests, backtests)
- [ ] Phase 4: API layer and auth
- [ ] Phase 5: web frontend and dashboard
- [ ] Phase 6: optimizer, goal planner, fund X-ray
- [ ] Phase 7: LLM assistant (tool-calling)
- [ ] Phase 8: RAG over fund and regulatory documents
- [ ] Phase 9: guardrails and evaluation
- [ ] Phase 10: deployment and demo



## Disclaimer

For educational and informational purposes only. Not investment advice.

## Author

Nirdosh Kumar 
