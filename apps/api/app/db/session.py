"""Engine and session factory. The URL comes from DATABASE_URL and is never logged.

Neon's pooled endpoint runs PgBouncer in transaction mode, so:
- psycopg's automatic server-side prepared statements are disabled (prepare_threshold=None);
- nothing depends on per-connection session state (no SET search_path).
`pool_pre_ping` and a short `pool_recycle` cope with Neon suspending idle computes.
"""

from collections.abc import Iterator
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings, sqlalchemy_url


class DatabaseNotConfigured(RuntimeError):
    """DATABASE_URL is not set; endpoints that need the database answer 503."""


def make_engine(url: str) -> Engine:
    url = sqlalchemy_url(url)
    if url.startswith("postgresql+psycopg://"):
        return create_engine(
            url,
            pool_pre_ping=True,
            pool_recycle=300,
            pool_size=5,
            max_overflow=5,
            connect_args={"prepare_threshold": None, "connect_timeout": 15},
        )
    return create_engine(url, pool_pre_ping=True)


@lru_cache
def get_engine() -> Engine:
    """The process-wide engine for DATABASE_URL (raises DatabaseNotConfigured if unset)."""
    url = get_settings().database_url
    if url is None:
        raise DatabaseNotConfigured("DATABASE_URL is not set")
    return make_engine(url.get_secret_value())


@lru_cache
def get_session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), expire_on_commit=False)


def get_db() -> Iterator[Session]:
    """FastAPI dependency: one session per request, closed afterwards."""
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()
