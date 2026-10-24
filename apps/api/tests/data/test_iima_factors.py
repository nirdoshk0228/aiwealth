"""IimaFactorProvider tests. HTTP is served by httpx.MockTransport.

The fixture tests/data/fixtures/iima_ff4_daily_synthetic.csv has the exact format of the real
file (header, CRLF line endings, "NA" cells, a weekend session, Monday RF covering the weekend)
but **made-up values**: the source states no licence, so no real rows are committed.
"""

from collections.abc import Callable
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest

from app.data.errors import BadResponseError, NotFoundError, ProviderError
from app.data.models import FACTOR_COLUMNS
from app.data.providers.iima_factors import (
    BASE_URL,
    DATASET,
    IimaFactorProvider,
    daily_file_name,
    parse_factor_csv,
)

FIXTURE = Path(__file__).parent / "fixtures" / "iima_ff4_daily_synthetic.csv"
FILE_2025_12 = "2025-12_FourFactors_and_Market_Returns_Daily_SurvivorshipBiasAdjusted.csv"


def raw() -> bytes:
    # Normalise to CRLF like the real file, whatever git did to the fixture's line endings.
    return FIXTURE.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")


def with_row(old: bytes, new: bytes) -> bytes:
    content = raw()
    assert old in content
    return content.replace(old, new)


# --- file name ---


def test_daily_file_name_matches_the_published_name() -> None:
    assert daily_file_name("2025-12") == FILE_2025_12


@pytest.mark.parametrize("bad", ["2025-13", "2025-1", "25-12", "latest", "2025-12/../x"])
def test_daily_file_name_rejects_malformed_releases(bad: str) -> None:
    with pytest.raises(ValueError, match="YYYY-MM"):
        daily_file_name(bad)


# --- parsing ---


def test_parse_renames_mf_and_converts_percent_to_decimal() -> None:
    frame, _ = parse_factor_csv(raw())

    assert tuple(frame.columns) == FACTOR_COLUMNS
    assert frame.index.name == "date"
    assert frame.index[0] == pd.Timestamp("2025-10-16")
    assert frame.index[-1] == pd.Timestamp("2025-10-24")
    assert len(frame) == 8
    assert frame.dtypes.eq("float64").all()
    # 2025-10-16: SMB 0.25 %, HML -0.5 %, WML 0.75 %, MF 1.25 %, RF 0.015 %
    first = frame.loc["2025-10-16"]
    assert first["MKT_RF"] == pytest.approx(0.0125, abs=1e-15)
    assert first["SMB"] == pytest.approx(0.0025, abs=1e-15)
    assert first["HML"] == pytest.approx(-0.005, abs=1e-15)
    assert first["WML"] == pytest.approx(0.0075, abs=1e-15)
    assert first["RF"] == pytest.approx(0.00015, abs=1e-15)


def test_parse_keeps_na_rows_as_nan_and_warns() -> None:
    frame, warnings = parse_factor_csv(raw(), label="t")

    na_row = frame.loc["2025-10-21"]
    assert np.isnan(na_row[["SMB", "HML", "WML"]]).all()
    assert na_row["MKT_RF"] == pytest.approx(0.0025)
    assert warnings == ["t: NA in source on 2025-10-21 (SMB, HML, WML); kept as NaN"]


def test_parse_keeps_weekend_sessions() -> None:
    frame, _ = parse_factor_csv(raw())
    assert pd.Timestamp("2025-10-18") in frame.index  # a Saturday


def test_parse_is_line_ending_and_bom_agnostic() -> None:
    crlf, _ = parse_factor_csv(raw())
    lf, _ = parse_factor_csv(raw().replace(b"\r\n", b"\n"))
    bom, _ = parse_factor_csv(b"\xef\xbb\xbf" + raw())
    pd.testing.assert_frame_equal(crlf, lf)
    pd.testing.assert_frame_equal(crlf, bom)


@pytest.mark.parametrize(
    "header",
    [
        b"Date,SMB,HML,WML,MKT,RF",  # renamed column
        b"Date,SMB,HML,MF,WML,RF",  # reordered
        b"Date,SMB,HML,WML,MF",  # missing column
        b"Date,SMB,HML,WML,MF,RF,RMW",  # extra column
    ],
)
def test_parse_rejects_any_other_header(header: bytes) -> None:
    content = raw().replace(b"Date,SMB,HML,WML,MF,RF", header, 1)
    with pytest.raises(BadResponseError, match="unexpected header"):
        parse_factor_csv(content)


def test_parse_rejects_an_html_page() -> None:
    # e.g. an error or redirect page served with status 200
    html = b"<!DOCTYPE HTML PUBLIC><html><head><title>301 Moved Permanently</title></head></html>"
    with pytest.raises(BadResponseError, match="unexpected header"):
        parse_factor_csv(html)


def test_parse_rejects_header_only_file() -> None:
    with pytest.raises(BadResponseError, match="no data rows"):
        parse_factor_csv(b"Date,SMB,HML,WML,MF,RF\r\n")


@pytest.mark.parametrize(
    ("old", "new", "match"),
    [
        (b"2025-10-17,", b"17-10-2025,", "unparseable date"),
        (b"-0.125,0.25,-0.375", b"-0.125,abc,-0.375", "non-numeric value in HML"),
        (b"2025-10-22,", b"2025-10-21,", "duplicate dates"),
        (b"2025-10-17,", b"2025-10-25,", "not in increasing order"),
    ],
)
def test_parse_rejects_bad_rows(old: bytes, new: bytes, match: str) -> None:
    with pytest.raises(BadResponseError, match=match):
        parse_factor_csv(with_row(old, new))


def test_parse_rejects_a_file_already_in_decimals() -> None:
    # Same numbers / 100: after our /100 the market std would be ~1e-4, far below any real
    # daily equity market. A silent units change between releases must fail the seed.
    frame, _ = parse_factor_csv(raw())
    lines = ["Date,SMB,HML,WML,MF,RF"]
    for day, r in frame.iterrows():
        vals = [r["SMB"], r["HML"], r["WML"], r["MKT_RF"], r["RF"]]
        cells = ["NA" if np.isnan(v) else repr(float(v)) for v in vals]
        lines.append(f"{day.date()}," + ",".join(cells))
    with pytest.raises(BadResponseError, match="std of MKT_RF"):
        parse_factor_csv("\r\n".join(lines).encode())


def test_parse_rejects_values_too_large_for_percent() -> None:
    with pytest.raises(BadResponseError, match="no longer looks like percent"):
        parse_factor_csv(with_row(b"0.125,0.5,0.625,2,", b"0.125,0.5,62.5,2,"))


def test_parse_rejects_negative_rf() -> None:
    with pytest.raises(BadResponseError, match="RF outside"):
        parse_factor_csv(with_row(b"-1.25,0.015", b"-1.25,-0.015"))


# --- provider (HTTP via MockTransport) ---


def make_provider(
    handler: Callable[[httpx.Request], httpx.Response], release: str = "2025-12"
) -> tuple[IimaFactorProvider, list[float]]:
    sleeps: list[float] = []
    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = IimaFactorProvider(
        client, release=release, max_attempts=3, base_delay=1.0, sleep=sleeps.append
    )
    return provider, sleeps


def test_fetch_follows_the_tilde_redirect_and_records_provenance() -> None:
    # The real server answers /~iffm/... with 301 -> /iffm/... (observed 2026-10-25).
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        requested.append(url)
        if "/~iffm/" in url:
            return httpx.Response(301, headers={"Location": url.replace("/~iffm/", "/iffm/")})
        return httpx.Response(200, content=raw(), headers={"Content-Type": "text/csv"})

    provider, _ = make_provider(handler)
    history = provider.fetch()

    assert requested[0] == f"{BASE_URL}/{FILE_2025_12}"
    assert history.source_url == requested[0].replace("/~iffm/", "/iffm/")
    assert history.source_file == FILE_2025_12
    assert history.dataset == DATASET
    assert history.release == "2025-12"
    assert history.raw_csv == raw()
    assert "percent" in history.units_source
    assert "Agarwalla" in history.citation and "2013" in history.citation
    assert len(history.factors) == 8
    assert len(history.warnings) == 1


def test_fetch_maps_404_to_not_found() -> None:
    provider, sleeps = make_provider(lambda r: httpx.Response(404), release="2030-01")
    with pytest.raises(NotFoundError, match="release 2030-01"):
        provider.fetch()
    assert sleeps == []  # a 404 is not retried


def test_fetch_retries_server_errors_then_gives_up() -> None:
    provider, sleeps = make_provider(lambda r: httpx.Response(503))
    with pytest.raises(ProviderError, match="HTTP 503"):
        provider.fetch()
    assert sleeps == [1.0, 2.0]


def test_fetch_recovers_from_a_transient_error() -> None:
    calls = iter([httpx.Response(500), httpx.Response(200, content=raw())])
    provider, sleeps = make_provider(lambda r: next(calls))
    assert len(provider.fetch().factors) == 8
    assert sleeps == [1.0]


def test_fetch_maps_connection_errors_to_provider_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("unreachable", request=request)

    provider, _ = make_provider(handler)
    with pytest.raises(ProviderError, match="ConnectError after 3 attempts"):
        provider.fetch()


def test_provider_rejects_a_malformed_release_before_any_request() -> None:
    with pytest.raises(ValueError, match="YYYY-MM"):
        make_provider(lambda r: httpx.Response(200), release="latest")
