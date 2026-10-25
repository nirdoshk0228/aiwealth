# AIWealth API image. Build from the repo root:
#   docker build -f apps/api/Dockerfile -t aiwealth-api .
# Market data and the document index are read from Postgres (DATABASE_URL) at runtime; the image
# holds the code, the committed config data (universe, scenarios, sample portfolios) and the
# embedding model, downloaded at build time so no request ever fetches a model.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_ENV=production \
    MARKET_DATA_BACKEND=postgres \
    RAG_BACKEND=postgres \
    RAG_MODE=hybrid \
    RAG_MODEL_DIR=/app/.cache/models \
    DATA_DIR=/app/data \
    AI_TRACE_DIR=/tmp/aiwealth-logs

WORKDIR /app/apps/api

COPY apps/api/pyproject.toml ./
COPY apps/api/app ./app
RUN pip install .

# Query embedding model (BAAI/bge-small-en-v1.5, ~67 MB) baked into the image.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5', cache_dir='/app/.cache/models')"

COPY apps/api/alembic.ini ./
COPY apps/api/alembic ./alembic
COPY data/universe.json /app/data/universe.json
COPY data/scenarios /app/data/scenarios
COPY data/portfolios /app/data/portfolios

RUN useradd --create-home --uid 10001 app && chown -R app /app
USER app

EXPOSE 8000
# Apply migrations, then serve. PORT is set by the host (Render); 8000 locally.
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips='*'"]
