import ipaddress
import socket
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.main import create_app

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
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any test that tries to reach the internet.

    Python-level sockets (httpx) are blocked at connect(). yfinance uses libcurl via
    curl_cffi, which bypasses Python sockets, so its entry point is patched directly.
    """
    monkeypatch.setattr(socket.socket, "connect", _guarded_connect)
    import yfinance

    monkeypatch.setattr(yfinance, "Ticker", _blocked_yfinance)
    monkeypatch.setattr(yfinance, "download", _blocked_yfinance)


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(create_app()) as test_client:
        yield test_client
