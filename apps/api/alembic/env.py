"""Alembic environment: migrations run against DATABASE_URL_DIRECT, else DATABASE_URL.

The URL is read from settings (environment / .env) and never written to alembic.ini or logs.
Neon's pooled endpoint works for these migrations (plain DDL in one transaction); a direct
(non-pooled) URL is preferred when available because PgBouncer can drop session features.
"""

from alembic import context

import app.db.models  # noqa: F401  (registers the tables on Base.metadata)
from app.ai.rag.store import RagBase
from app.core.config import get_settings
from app.db.base import Base
from app.db.session import make_engine

# Two metadata objects: the core tables, and the RAG tables (pgvector types, Postgres only).
target_metadata = [Base.metadata, RagBase.metadata]


def _url() -> str:
    settings = get_settings()
    url = settings.database_url_direct or settings.database_url
    if url is None:
        raise RuntimeError("set DATABASE_URL (or DATABASE_URL_DIRECT) to run migrations")
    return url.get_secret_value()


def run_migrations_offline() -> None:
    context.configure(url=_url(), target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = make_engine(_url())
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
