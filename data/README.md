# data

Inputs for the platform:

- `pdfs/`: fund factsheets, scheme documents and SEBI circulars for RAG (Phase 8)
- `portfolios/`: sample portfolios for demos and API tests
- `scenarios/`: stress-test scenario definitions (JSON)

Large PDFs and datasets are not committed. They are fetched by download/ingest scripts instead,
and the downloaded raw files go in `data/raw/`, which is gitignored.
