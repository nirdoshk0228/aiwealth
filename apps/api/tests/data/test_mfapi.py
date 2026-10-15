"""MfapiProvider tests. All HTTP is served by httpx.MockTransport from saved real responses
(tests/data/fixtures/mfapi_*.json, captured 2026-10-15 and trimmed to 10 rows)."""

import json
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pytest

from app.data.errors import BadResponseError, NotFoundError, ProviderError
from app.data.models import AssetRef
from app.data.providers.mfapi import (
    BASE_URL,
    MfapiProvider,
    parse_nav_response,
    parse_search_response,
)

FIXTURES = Path(__file__).parent / "fixtures"
PPFAS = 122639  # Parag Parikh Flexi Cap Fund - Direct Plan - Growth


def load(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text())


def make_provider(
    handler: Callable[[httpx.Request], httpx.Response],
) -> tuple[MfapiProvider, list[float]]:
    sleeps: list[float] = []
    client = httpx.Client(base_url=BASE_URL, transport=httpx.MockTransport(handler))
    return MfapiProvider(client, max_attempts=3, base_delay=1.0, sleep=sleeps.append), sleeps


# --- parsing ---


def test_parse_nav_sorts_ascending_and_converts_to_float() -> None:
    meta, s, warnings = parse_nav_response(load("mfapi_nav_122639.json"), PPFAS)

    assert meta.scheme_name == "Parag Parikh Flexi Cap Fund - Direct Plan - Growth"
    assert meta.scheme_category == "Equity Scheme - Flexi Cap Fund"
    assert len(s) == 10
    assert s.index.is_monotonic_increasing and s.index.is_unique
    assert s.index.tz is None
    # Source is newest-first; after sorting the oldest row (30-09-2026) comes first.
    assert s.index[0] == pd.Timestamp("2026-09-30")
    assert s.iloc[0] == pytest.approx(88.8355)
    assert s.index[-1] == pd.Timestamp("2026-10-14")
    assert s.iloc[-1] == pytest.approx(89.3598)
    assert s.dtype == float
    assert warnings == []


def test_parse_nav_dates_are_day_first() -> None:
    # "09-10-2026" must be 9 October, not 10 September.
    _, s, _ = parse_nav_response(load("mfapi_nav_122639.json"), PPFAS)
    assert pd.Timestamp("2026-10-09") in s.index
    assert pd.Timestamp("2026-09-10") not in s.index


def test_unknown_scheme_code_is_not_found() -> None:
    # Real response for a non-existent code: HTTP 200, status SUCCESS, empty data.
    with pytest.raises(NotFoundError):
        parse_nav_response(load("mfapi_nav_unknown_code.json"), 999999999)


def test_bad_navs_and_duplicates_are_dropped_with_warnings() -> None:
    payload = deepcopy(load("mfapi_nav_122639.json"))
    payload["data"][1]["nav"] = "N.A."  # 13-10-2026
    payload["data"][2]["nav"] = "0.00000"  # 12-10-2026
    payload["data"].append(dict(payload["data"][0], nav="1.0"))  # dup of 14-10, appears later

    _, s, warnings = parse_nav_response(payload, PPFAS)

    assert len(s) == 8
    assert pd.Timestamp("2026-10-13") not in s.index
    assert pd.Timestamp("2026-10-12") not in s.index
    assert s.loc["2026-10-14"] == pytest.approx(89.3598)  # first occurrence kept
    assert len(warnings) == 2


def test_all_navs_invalid_raises() -> None:
    payload = deepcopy(load("mfapi_nav_122639.json"))
    for row in payload["data"]:
        row["nav"] = "0"
    with pytest.raises(BadResponseError, match="no valid prices"):
        parse_nav_response(payload, PPFAS)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.pop("meta"),
        lambda p: p.update(status="FAIL"),
        lambda p: p["data"][0].update(date="2026/10/14"),
        lambda p: p["meta"].update(scheme_code=111111),  # response for a different scheme
    ],
    ids=["missing-meta", "status-not-success", "bad-date-format", "wrong-scheme"],
)
def test_malformed_payloads_raise_bad_response(mutate: Callable[[dict], object]) -> None:
    payload = deepcopy(load("mfapi_nav_122639.json"))
    mutate(payload)
    with pytest.raises(BadResponseError):
        parse_nav_response(payload, PPFAS)


def test_parse_search() -> None:
    results = parse_search_response(load("mfapi_search_parag_parikh.json"))
    assert len(results) == 4
    direct_growth = [r for r in results if r.scheme_name.endswith("Direct Plan - Growth")]
    assert [r.scheme_code for r in direct_growth] == [PPFAS]


def test_parse_search_rejects_non_list() -> None:
    with pytest.raises(BadResponseError):
        parse_search_response({"error": "x"})


# --- HTTP behaviour ---


def test_fetch_history_hits_expected_url_and_returns_price_history() -> None:
    seen: list[httpx.URL] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url)
        return httpx.Response(200, json=load("mfapi_nav_122639.json"))

    provider, sleeps = make_provider(handler)
    asset = AssetRef(kind="mutual_fund", symbol=str(PPFAS), name="PPFAS Flexi Cap")
    hist = provider.fetch_history(asset)

    assert str(seen[0]) == f"{BASE_URL}/mf/{PPFAS}"
    assert hist.asset == asset
    assert hist.prices.name == "mutual_fund:122639"
    assert hist.source == "mfapi.in"
    assert hist.source_name == "Parag Parikh Flexi Cap Fund - Direct Plan - Growth"
    assert len(hist.prices) == 10
    assert sleeps == []


def test_start_date_filters_client_side() -> None:
    provider, _ = make_provider(lambda r: httpx.Response(200, json=load("mfapi_nav_122639.json")))
    _, s, _ = provider.fetch_nav_history(PPFAS, start=pd.Timestamp("2026-10-12").date())
    assert list(s.index) == [pd.Timestamp(d) for d in ("2026-10-12", "2026-10-13", "2026-10-14")]


def test_search_sends_query_param() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/mf/search"
        assert request.url.params["q"] == "parag parikh flexi cap"
        return httpx.Response(200, json=load("mfapi_search_parag_parikh.json"))

    provider, _ = make_provider(handler)
    assert len(provider.search("parag parikh flexi cap")) == 4


def test_empty_search_query_rejected() -> None:
    provider, _ = make_provider(lambda r: httpx.Response(200, json=[]))
    with pytest.raises(ValueError):
        provider.search("  ")


def test_timeout_is_retried_with_exponential_backoff_then_succeeds() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            raise httpx.ReadTimeout("slow", request=request)
        return httpx.Response(200, json=load("mfapi_nav_122639.json"))

    provider, sleeps = make_provider(handler)
    _, s, _ = provider.fetch_nav_history(PPFAS)
    assert calls["n"] == 3
    assert sleeps == [1.0, 2.0]
    assert len(s) == 10


def test_persistent_timeout_raises_provider_error_after_max_attempts() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ConnectTimeout("down", request=request)

    provider, sleeps = make_provider(handler)
    with pytest.raises(ProviderError, match="after 3 attempts"):
        provider.fetch_nav_history(PPFAS)
    assert calls["n"] == 3
    assert sleeps == [1.0, 2.0]


@pytest.mark.parametrize("status", [429, 500, 503])
def test_transient_status_codes_are_retried(status: int) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(status)
        return httpx.Response(200, json=load("mfapi_nav_122639.json"))

    provider, sleeps = make_provider(handler)
    provider.fetch_nav_history(PPFAS)
    assert calls["n"] == 2
    assert sleeps == [1.0]


def test_404_fails_fast_as_not_found() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(404)

    provider, sleeps = make_provider(handler)
    with pytest.raises(NotFoundError):
        provider.fetch_nav_history(PPFAS)
    assert calls["n"] == 1
    assert sleeps == []


def test_other_4xx_fails_fast() -> None:
    provider, sleeps = make_provider(lambda r: httpx.Response(400))
    with pytest.raises(ProviderError, match="HTTP 400"):
        provider.fetch_nav_history(PPFAS)
    assert sleeps == []


def test_non_json_body_is_bad_response() -> None:
    provider, _ = make_provider(lambda r: httpx.Response(200, text="<html>maintenance</html>"))
    with pytest.raises(BadResponseError, match="not JSON"):
        provider.fetch_nav_history(PPFAS)


def test_rejects_non_mutual_fund_asset() -> None:
    provider, _ = make_provider(lambda r: httpx.Response(200, json={}))
    with pytest.raises(ValueError):
        provider.fetch_history(AssetRef(kind="equity", symbol="TCS", name="TCS"))
