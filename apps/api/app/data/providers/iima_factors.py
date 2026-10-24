"""Indian Fama-French-Carhart factors from IIM Ahmedabad (daily, survivorship-bias adjusted).

Source: https://faculty.iima.ac.in/~iffm/Indian-Fama-French-Momentum/ (data from CMIE
Prowess). Cite: Agarwalla, Jacob and Varma (2013), see CITATION. The page states no licence,
so the file is downloaded into the gitignored cache and never committed.

File (verified 2026-10-25 for release 2025-12):
  DATA/<release>_FourFactors_and_Market_Returns_Daily_SurvivorshipBiasAdjusted.csv
  header: Date,SMB,HML,WML,MF,RF   CRLF line endings, missing values are the literal "NA"
  ~8,000 rows, 1993-10-01 onwards, ISO dates, including some weekend sessions

Facts the parser relies on, and how each was checked:
  - Values are **percent** simple (holding-period) returns per row. Daily RF sums to 5.6%
    over 2025 (91-day T-bill level), and the release's monthly file equals prod(1 + daily) - 1
    to ~1e-15 for every column (summing the daily values, i.e. reading them as log returns,
    is off by up to 0.7 percentage points a month).
  - MF is the market risk premium Rm - Rf: "the value-weighted portfolio of all the stocks
    involved in the estimation of SMB, HML, and WML ... The risk-free rate Rf, computed using
    the 91-days T-bill rate, is deducted from the return of the market portfolio" (working
    paper, section 5.3). It is not the Nifty 50. We rename it MKT_RF.
  - The server redirects /~iffm/ to /iffm/ (HTTP 301), so redirects are followed and the
    final URL is recorded.
"""

import io
import logging
import re
import time
from collections.abc import Callable
from types import TracebackType
from typing import Self

import httpx
import numpy as np
import pandas as pd

from app.data.errors import BadResponseError, NotFoundError, ProviderError
from app.data.models import FACTOR_COLUMNS, FactorHistory
from app.data.providers._retry import call_with_retry

logger = logging.getLogger(__name__)

BASE_URL = "https://faculty.iima.ac.in/~iffm/Indian-Fama-French-Momentum/DATA"
DEFAULT_RELEASE = "2025-12"
DATASET = "iima_ff4_daily"
SOURCE = "IIMA Indian Fama-French-Momentum factors (CMIE Prowess), survivorship-bias adjusted"
CITATION = (
    "Agarwalla, S. K., Jacob, J. and Varma, J. R. (2013). Four factor model in Indian "
    "equities market. Working Paper No. 2013-09-05, Indian Institute of Management Ahmedabad."
)
UNITS_SOURCE = "percent simple returns per row (holding period since the previous row)"

SOURCE_HEADER = ("Date", "SMB", "HML", "WML", "MF", "RF")
_RENAME = {"MF": "MKT_RF"}
_RELEASE_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")

# Sanity bounds applied after converting percent -> decimal. They exist to catch a change of
# units between releases, not to clean data: any violation rejects the whole file.
# Real 2025-12 release: max |value| 0.158 (MF), std(MF) 0.0136, RF in [0.00008, 0.0023].
MAX_ABS_RETURN = 0.5
MKT_STD_RANGE = (0.002, 0.05)
MAX_RF = 0.01


def daily_file_name(release: str) -> str:
    """File name of the survivorship-bias-adjusted daily four-factor file for a release."""
    if not _RELEASE_RE.match(release):
        raise ValueError(f"release must look like 'YYYY-MM', got {release!r}")
    return f"{release}_FourFactors_and_Market_Returns_Daily_SurvivorshipBiasAdjusted.csv"


# --- Pure parsing (unit-tested directly with a synthetic fixture in the real format) ---


def parse_factor_csv(content: bytes, label: str = "IIMA factors") -> tuple[pd.DataFrame, list[str]]:
    """Parse the daily factor CSV into decimal returns.

    Returns (frame, warnings): columns FACTOR_COLUMNS (MF renamed MKT_RF), values divided by
    100 once here, DatetimeIndex "date". Rows with "NA" are kept as NaN (see FactorHistory)
    and listed in the warnings.

    Raises BadResponseError for anything unexpected: a different header, unparseable dates or
    numbers, duplicate or unsorted dates, or values whose scale does not fit percent returns.
    """
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise BadResponseError(f"{label}: not UTF-8 text") from exc
    try:
        raw = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise BadResponseError(f"{label}: not a CSV file: {exc}") from exc

    header = tuple(c.strip() for c in raw.columns)
    if header != SOURCE_HEADER:
        raise BadResponseError(
            f"{label}: unexpected header {header[:8]} (expected {SOURCE_HEADER}); "
            f"first bytes: {text[:60]!r}"
        )
    if raw.empty:
        raise BadResponseError(f"{label}: no data rows")
    raw.columns = list(header)

    try:
        dates = pd.to_datetime(raw["Date"].str.strip(), format="%Y-%m-%d")
    except ValueError as exc:
        raise BadResponseError(f"{label}: unparseable date: {exc}") from exc
    index = pd.DatetimeIndex(dates, name="date")
    if not index.is_unique:
        dups = sorted({d.date().isoformat() for d in index[index.duplicated()]})
        raise BadResponseError(f"{label}: duplicate dates {dups[:5]}")
    if not index.is_monotonic_increasing:
        raise BadResponseError(f"{label}: dates are not in increasing order")

    values: dict[str, np.ndarray] = {}
    for col in SOURCE_HEADER[1:]:
        cells = raw[col].str.strip().replace("NA", np.nan)
        try:
            values[_RENAME.get(col, col)] = pd.to_numeric(cells, errors="raise").to_numpy(
                dtype="float64"
            )
        except (ValueError, TypeError) as exc:
            raise BadResponseError(f"{label}: non-numeric value in {col}: {exc}") from exc
    frame = pd.DataFrame(values, index=index)[list(FACTOR_COLUMNS)] / 100.0

    _check_scale(frame, label)

    warnings = []
    na_rows = frame[frame.isna().any(axis=1)]
    for day, row in na_rows.iterrows():
        cols = ", ".join(c for c in FACTOR_COLUMNS if np.isnan(row[c]))
        warnings.append(f"{label}: NA in source on {day.date()} ({cols}); kept as NaN")
    return frame, warnings


def _check_scale(frame: pd.DataFrame, label: str) -> None:
    values = frame.to_numpy()
    finite = values[~np.isnan(values)]
    if finite.size == 0:
        raise BadResponseError(f"{label}: every value is NA")
    if not np.isfinite(finite).all():
        raise BadResponseError(f"{label}: infinite values")
    if np.abs(finite).max() > MAX_ABS_RETURN:
        raise BadResponseError(
            f"{label}: |return| up to {np.abs(finite).max():.4g} after /100; "
            "the file no longer looks like percent returns"
        )
    rf = frame["RF"].dropna()
    if (rf < 0).any() or (rf > MAX_RF).any():
        raise BadResponseError(f"{label}: RF outside [0, {MAX_RF}] after /100")
    mkt = frame["MKT_RF"].dropna()
    if len(mkt) >= 2:
        std = float(mkt.std())
        lo, hi = MKT_STD_RANGE
        if not lo <= std <= hi:
            raise BadResponseError(
                f"{label}: std of MKT_RF is {std:.4g} after /100, outside [{lo}, {hi}]; "
                "check the units of this release"
            )


# --- HTTP client ---


def _is_retryable(exc: Exception) -> bool:
    """Timeouts, connection errors, 429 and 5xx are transient; other 4xx are not."""
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return code == 429 or code >= 500
    return False


class IimaFactorProvider:
    """FactorProvider for the IIMA daily four-factor file of one release."""

    source = SOURCE

    def __init__(
        self,
        client: httpx.Client | None = None,
        *,
        release: str = DEFAULT_RELEASE,
        timeout: float = 60.0,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.release = release
        self.file_name = daily_file_name(release)  # validates the release string
        self.url = f"{BASE_URL}/{self.file_name}"
        # Tests inject a client backed by httpx.MockTransport; production builds a real one.
        self._owns_client = client is None
        self._client = client or httpx.Client(
            timeout=timeout,
            follow_redirects=True,
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

    def _download(self) -> httpx.Response:
        def attempt() -> httpx.Response:
            resp = self._client.get(self.url, follow_redirects=True)
            resp.raise_for_status()
            return resp

        try:
            return call_with_retry(
                attempt,
                is_retryable=_is_retryable,
                max_attempts=self._max_attempts,
                base_delay=self._base_delay,
                sleep=self._sleep,
            )
        except httpx.HTTPStatusError as exc:
            code = exc.response.status_code
            if code == 404:
                raise NotFoundError(
                    f"IIMA {self.file_name}: HTTP 404 (is release {self.release} published?)"
                ) from exc
            raise ProviderError(f"IIMA {self.file_name}: HTTP {code}") from exc
        except httpx.TransportError as exc:
            raise ProviderError(
                f"IIMA {self.file_name}: {type(exc).__name__} after {self._max_attempts} attempts"
            ) from exc

    def fetch(self) -> FactorHistory:
        resp = self._download()
        frame, warnings = parse_factor_csv(resp.content, label=f"IIMA {self.release}")
        for w in warnings:
            logger.warning(w)
        return FactorHistory(
            factors=frame,
            dataset=DATASET,
            release=self.release,
            source=SOURCE,
            source_url=str(resp.url),
            source_file=self.file_name,
            raw_csv=resp.content,
            units_source=UNITS_SOURCE,
            citation=CITATION,
            warnings=tuple(warnings),
        )
