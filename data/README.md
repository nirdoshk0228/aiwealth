# data

Inputs for the platform (committed unless stated otherwise):

- `universe.json`: the 15 funds, 20 stocks and the Nifty 50 benchmark (see docs/data.md)
- `scenarios/`: stress-test scenario definitions (JSON, schema in `app/data/scenarios.py`)
- `portfolios/`: sample portfolios for the no-login demo and API tests. Illustrative only, not
  recommendations
- `raw/` (**gitignored**): the Parquet market-data cache written by `scripts/seed_data.py`,
  including the IIM Ahmedabad factor file (no stated licence, never committed; see
  docs/data.md, "Factor data")
- `docs/` (**gitignored**, Phase 8): fund and regulatory PDFs downloaded by the ingest script
  from the URLs in its manifest

Large PDFs and datasets are not committed. They are fetched by download/ingest scripts.
