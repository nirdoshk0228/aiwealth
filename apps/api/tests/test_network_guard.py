"""The autouse guard in conftest.py must block real network access."""

import httpx
import pytest
import yfinance


def test_external_http_is_blocked() -> None:
    with pytest.raises(RuntimeError, match="Network access is disabled"):
        httpx.get("http://93.184.215.14/", timeout=1)


def test_yfinance_is_blocked() -> None:
    with pytest.raises(RuntimeError, match="disabled in tests"):
        yfinance.Ticker("RELIANCE.NS")
