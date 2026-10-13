# Architecture

> Stub. Filled in as each phase lands.

## Components

- **apps/web**: Next.js frontend (Phase 5)
- **apps/api**: FastAPI backend
  - `app/api`: HTTP routers (thin). `/health` and `/api/v1/*`
  - `app/engine`: deterministic risk engine (no LLM, no network)
  - `app/ai`: LLM agent, RAG pipeline, guardrails
  - `app/data`: market data providers and caching
- **Postgres + pgvector**: portfolios, cached market data, document embeddings
- **Redis**: hot cache, rate limiting

## Dependency rules

1. `app.engine` never imports `app.ai`.
2. `app.ai.agent.tools` is the only bridge from the AI layer to the engine.
3. Request flow: router -> service -> engine/data.
4. No external API calls on a user request path. Data is fetched ahead of time and read from the cache.

## Open items

- Request flow diagram
- Data refresh schedule
- Deployment topology
