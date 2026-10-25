import os

# Before any app import: tests never warm market data in the background and never fall back
# to the developer's real configuration for anything the test does not set explicitly.
os.environ.setdefault("WARM_MARKET_DATA", "false")
os.environ.setdefault("APP_ENV", "test")

import ipaddress  # noqa: E402
import socket  # noqa: E402
import uuid  # noqa: E402
from collections.abc import Iterator  # noqa: E402

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import Engine, text  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.main import create_app  # noqa: E402

_real_connect = socket.socket.connect


def _guarded_connect(self: socket.socket, address: object) -> None:
    """Allow loopback only (asyncio on Windows uses a localhost socketpair internally)."""
    host = address[0] if isinstance(address, tuple) else None
    if host is not None:
        try:
            loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            loopback = host == "localhost"
        if not loopback:
            raise RuntimeError(f"Network access is disabled in tests (tried {address!r})")
    _real_connect(self, address)


def _blocked_yfinance(*args: object, **kwargs: object) -> None:
    raise RuntimeError("yfinance network access is disabled in tests; inject a fetcher")


@pytest.fixture(autouse=True)
def _no_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any test that tries to reach the internet.

    Python-level sockets (httpx) are blocked at connect(). yfinance uses libcurl via
    curl_cffi, which bypasses Python sockets, so its entry point is patched directly.
    Tests marked `postgres` may reach the database (and nothing else is exempt).
    """
    if request.node.get_closest_marker("postgres") is None:
        monkeypatch.setattr(socket.socket, "connect", _guarded_connect)
    import yfinance

    monkeypatch.setattr(yfinance, "Ticker", _blocked_yfinance)
    monkeypatch.setattr(yfinance, "download", _blocked_yfinance)


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture
def pg_engine() -> Iterator[Engine]:
    """A Postgres engine bound to a throwaway schema; skipped without DATABASE_URL.

    Tables are created in schema test_<random> through SQLAlchemy's schema_translate_map (no
    SET search_path, which is unsafe behind Neon's transaction-mode pooler) and the schema is
    dropped afterwards, so the developer's real tables are never touched.
    """
    url = get_settings().database_url
    if url is None:
        pytest.skip("DATABASE_URL is not set")
    from app.db.base import Base
    from app.db.session import make_engine

    schema = f"test_{uuid.uuid4().hex[:12]}"
    base_engine = make_engine(url.get_secret_value())
    with base_engine.begin() as conn:
        conn.execute(text(f'CREATE SCHEMA "{schema}"'))
    engine = base_engine.execution_options(schema_translate_map={None: schema})
    try:
        Base.metadata.create_all(engine)
        yield engine
    finally:
        with base_engine.begin() as conn:
            conn.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        base_engine.dispose()
