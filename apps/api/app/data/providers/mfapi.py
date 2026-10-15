"""Mutual fund NAV history from mfapi.in (free mirror of AMFI daily NAVs).

Endpoints used (verified against the live API, 2026-10):
  GET /mf/{scheme_code}  -> {"meta": {...},
                             "data": [{"date": "dd-mm-yyyy", "nav": "89.35980"}, ...],
                             "status": "SUCCESS"}   (data is newest-first)
  GET /mf/search?q=...   -> [{"schemeCode": 122639, "schemeName": "..."}, ...]

Quirk: an unknown scheme code returns HTTP 200, status "SUCCESS", empty `data` and
meta.scheme_code == 0. We map that to NotFoundError rather than trusting the status.
"""

import logging
import time
from collections.abc import Callable
from datetime import date
from types import TracebackType
from typing import Any, Self

import httpx
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.data.errors import BadResponseError, NotFoundError, ProviderError
from app.data.models import AssetRef, PriceHistory
from app.data.providers._clean import clean_price_series
from app.data.providers._retry import call_with_retry

logger = logging.getLogger(__name__)

BASE_URL = "https://api.mfapi.in"
SOURCE = "mfapi.in"


# --- Response schemas (only the fields we use; extra fields are ignored) ---


class SchemeMeta(BaseModel):
    fund_house: str
    scheme_category: str
    scheme_code: int
    scheme_name: str
    isin_growth: str | None = None


class _NavRow(BaseModel):
    date: str
    nav: str


class _NavResponse(BaseModel):
    meta: SchemeMeta
    data: list[_NavRow]
    status: str


class SchemeSearchResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    scheme_code: int = Field(alias="schemeCode")
    scheme_name: str = Field(alias="schemeName")


# --- Pure parsing (unit-tested directly with saved responses) ---


def parse_nav_response(payload: Any, scheme_code: int) -> tuple[SchemeMeta, pd.Series, list[str]]:
    """Validate an /mf/{code} payload and return (meta, NAV series, warnings).

    NAV is per unit in INR, published once per business day by the AMC. For Growth
    plans it already reflects expenses and reinvested income, so no adjustment is needed.
    """
    try:
        resp = _NavResponse.model_validate(payload)
    except ValidationError as exc:
        raise BadResponseError(f"mfapi {scheme_code}: unexpected payload shape: {exc}") from exc
    if resp.status != "SUCCESS":
        raise BadResponseError(f"mfapi {scheme_code}: status={resp.status!r}")
    if not resp.data or resp.meta.scheme_code == 0:
        raise NotFoundError(f"mfapi {scheme_code}: no such scheme (empty data)")
    if resp.meta.scheme_code != scheme_code:
        raise BadResponseError(
            f"mfapi {scheme_code}: response is for scheme {resp.meta.scheme_code}"
        )

    try:
        dates = pd.to_datetime([r.date for r in resp.data], format="%d-%m-%Y")
    except ValueError as exc:
        raise BadResponseError(f"mfapi {scheme_code}: unparseable date: {exc}") from exc
    raw = pd.Series([r.nav for r in resp.data], index=dates)
    series, warnings = clean_price_series(raw, label=f"mfapi {scheme_code}")
    return resp.meta, series, warnings


def parse_search_response(payload: Any) -> list[SchemeSearchResult]:
    """Validate an /mf/search payload."""
    if not isinstance(payload, list):
        raise BadResponseError(f"mfapi search: expected a list, got {type(payload).__name__}")
    try:
        return [SchemeSearchResult.model_validate(item) for item in payload]
    except ValidationError as exc:
        raise BadResponseError(f"mfapi search: unexpected item shape: {exc}") from exc


# --- HTTP client ---


def _is_retryable(exc: Exception) -> bool:
    """Timeouts, connection errors, 429 and 5xx are transient; other 4xx are not."""
    if isinstance(exc, httpx.TransportError):  # includes TimeoutException
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return code == 429 or code >= 500
    return False


class MfapiProvider:
    """PriceProvider for Indian mutual funds via mfapi.in."""

    source = SOURCE

    def __init__(
        self,
        client: httpx.Client | None = None,
        *,
        timeout: float = 20.0,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        # Tests inject a client backed by httpx.MockTransport; production builds a real one.
        self._owns_client = client is None
        self._client = client or httpx.Client(
            base_url=BASE_URL,
            timeout=timeout,
            headers={"User-Agent": "aiwealth-seed/0.1"},
        )
        self._max_attempts = max_attempts
        self._base_delay = base_delay
        self._sleep = sleep

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def _get_json(self, path: str, params: dict[str, str] | None = None) -> Any:
        def attempt() -> httpx.Response:
            resp = self._client.get(path, params=params)
            resp.raise_for_status()
            return resp

        try:
            resp = call_with_retry(
                attempt,
                is_retryable=_is_retryable,
                max_attempts=self._max_attempts,
                base_delay=self._base_delay,
                sleep=self._sleep,
            )
        except httpx.HTTPStatusError as exc:
            code = exc.response.status_code
            if code == 404:
                raise NotFoundError(f"mfapi {path}: HTTP 404") from exc
            raise ProviderError(f"mfapi {path}: HTTP {code}") from exc
        except httpx.TransportError as exc:
            raise ProviderError(
                f"mfapi {path}: {type(exc).__name__} after {self._max_attempts} attempts"
            ) from exc

        try:
            return resp.json()
        except ValueError as exc:
            raise BadResponseError(f"mfapi {path}: response is not JSON") from exc

    def search(self, query: str) -> list[SchemeSearchResult]:
        """Search schemes by name. Returns all plans/options; callers filter for Direct-Growth."""
        if not query.strip():
            raise ValueError("query must not be empty")
        return parse_search_response(self._get_json("/mf/search", params={"q": query}))

    def fetch_nav_history(
        self, scheme_code: int, start: date | None = None
    ) -> tuple[SchemeMeta, pd.Series, list[str]]:
        """Full NAV history for a scheme (the API has no date filter; `start` is applied here)."""
        meta, series, warnings = parse_nav_response(
            self._get_json(f"/mf/{scheme_code}"), scheme_code
        )
        if start is not None:
            series = series[series.index >= pd.Timestamp(start)]
            if series.empty:
                raise NotFoundError(f"mfapi {scheme_code}: no NAVs on or after {start}")
        return meta, series, warnings

    def fetch_history(self, asset: AssetRef, start: date | None = None) -> PriceHistory:
        if asset.kind != "mutual_fund":
            raise ValueError(f"MfapiProvider only handles mutual funds, got {asset.kind}")
        try:
            code = int(asset.symbol)
        except ValueError as exc:
            raise ValueError(f"scheme code must be numeric, got {asset.symbol!r}") from exc
        meta, series, warnings = self.fetch_nav_history(code, start)
        for w in warnings:
            logger.warning(w)
        return PriceHistory(
            asset=asset,
            prices=series.rename(asset.id),
            source=SOURCE,
            source_name=meta.scheme_name,
            warnings=tuple(warnings),
        )
