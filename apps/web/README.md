# apps/web

Next.js 16 (App Router) + TypeScript (strict) + Tailwind 4 + shadcn/ui (Base UI) + Plotly + Zustand.

```
npm install
cp .env.example .env.local     # NEXT_PUBLIC_API_URL, default http://localhost:8000
npm run dev                    # http://localhost:3000 (start the API first)
npm run lint && npm run typecheck && npm test && npm run build
```

## Typed API client

Types in `src/lib/api/schema.ts` are **generated** from the FastAPI OpenAPI spec, never written by
hand. After changing an API schema or route:

```
python scripts/export_openapi.py      # from the repo root: writes apps/web/openapi.json
cd apps/web && npm run gen:api        # openapi.json -> src/lib/api/schema.ts
```

CI fails if `openapi.json` is out of date. `src/lib/api/endpoints.ts` wraps each endpoint
(openapi-fetch); `src/lib/api/use-query.ts` gives every data view loading / error / success states.

## Layout

- `src/app/`: landing (`/`, "Try the demo"), `/portfolio` (search, holdings, CSV upload, samples),
  `/dashboard` (risk dashboard)
- `src/components/dashboard/cards.tsx`: one card per API query, each with its own loading and error state
- `src/components/charts/`: Plotly wrapper (the cartesian bundle is loaded on first use) and charts
- `src/lib/`: portfolio logic (amounts -> weights, CSV), formatting, Zustand store (persisted in
  localStorage), unit-tested with vitest
