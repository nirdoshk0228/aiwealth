"""NSE equities and the Nifty 50 index from Yahoo Finance via yfinance.

Prices are the adjusted close: `Ticker.history(auto_adjust=True)` rewrites OHLC for
splits and dividends, so `Close` is a total-return-style price suitable for returns.
Yahoo's adjustment quality for NSE names is not guaranteed (see docs/data.md).

Why `Ticker.history` and not `yf.download`: `download` swallows every error and returns
an empty frame, so a typo'd ticker, a rate limit and a network outage look identical.
With yfinance exceptions enabled, `history` raises distinct errors we can classify:
  - YFTickerMissingError (incl. YFPricesMissingError, YFTzMissingError) -> NotFoundError
  - YFRateLimitError, OSError (curl_cffi network errors subclass OSError) -> retried

The network call sits behind an injectable `fetcher` so tests replay saved frames.
"""

import logging
import time
from collections.abc import Callable
from datetime import date

import pandas as pd
import yfinance as yf
from yfinance.exceptions import YFRateLimitError, YFTickerMissingError

from app.data.errors import BadResponseError, NotFoundError, ProviderError
from app.data.models import AssetRef, PriceHistory
from app.data.providers._clean import clean_price_series
from app.data.providers._retry import call_with_retry

logger = logging.getLogger(__name__)

SOURCE = "yfinance"
NSE_SUFFIX = ".NS"

# (yahoo_ticker, start or None for full history) -> raw history DataFrame
HistoryFetcher = Callable[[str, date | None], pd.DataFrame]


def yahoo_ticker(asset: AssetRef) -> str:
    """Map an AssetRef to a Yahoo symbol: RELIANCE -> RELIANCE.NS, ^NSEI stays ^NSEI."""
    if asset.kind == "equity":
        sym = asset.symbol.upper()
        return sym if sym.endswith(NSE_SUFFIX) else sym + NSE_SUFFIX
    if asset.kind == "index":
        if not asset.symbol.startswith("^"):
            raise ValueError(f"Yahoo index symbols start with '^', got {asset.symbol!r}")
        return asset.symbol
    raise ValueError(f"YFinanceProvider does not handle {asset.kind}")


def make_yfinance_fetcher(timeout: float = 30.0) -> HistoryFetcher:
    """Build the real network fetcher. Never used in tests."""

    def fetch(ticker: str, start: date | None) -> pd.DataFrame:
        # Process-wide switch: make yfinance raise instead of logging and returning empty.
        yf.config.debug.hide_exceptions = False
        t = yf.Ticker(ticker)
        if start is None:
            return t.history(period="max", auto_adjust=True, timeout=timeout)
        return t.history(start=start, auto_adjust=True, timeout=timeout)

    return fetch


_OHLCV = ("Open", "High", "Low", "Close", "Volume")


def drop_phantom_bars(df: pd.DataFrame, label: str) -> tuple[pd.DataFrame, list[str]]:
    """Remove placeholder bars Yahoo inserts on some exchange holidays.

    A phantom bar has Volume == 0 and Open == High == Low == Close (in practice all equal to
    the previous close), e.g. every NSE stock on 2026-01-15. Keeping it would add a fake 0%
    return and a fake trading day. Equality is exact on purpose: a genuinely illiquid but
    traded day has some volume or some price range and is kept.

    Only for equities: index volume on Yahoo is unreliable (often 0), so the rule would
    misfire on ^NSEI.
    """
    if not set(_OHLCV) <= set(df.columns):
        return df, [f"{label}: OHLCV columns missing; phantom-bar check skipped"]
    phantom = (
        (df["Volume"] == 0)
        & (df["Open"] == df["High"])
        & (df["High"] == df["Low"])
        & (df["Low"] == df["Close"])
    )
    if not phantom.any():
        return df, []
    dates = [str(d.date()) for d in pd.DatetimeIndex(df.index[phantom])]
    shown = ", ".join(dates[:5]) + (", ..." if len(dates) > 5 else "")
    return df[~phantom], [
        f"{label}: dropped {len(dates)} phantom bar(s) (zero volume, flat OHLC): {shown}"
    ]


def _flatten_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Turn yf.download's (field, ticker) MultiIndex into flat field columns."""
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    if ticker in df.columns.get_level_values(1):
        return df.xs(ticker, axis=1, level=1)
    return df.droplevel(1, axis=1)


def parse_history_frame(
    df: pd.DataFrame | None, ticker: str, *, drop_phantom: bool = False
) -> tuple[pd.Series, list[str]]:
    """Extract the adjusted Close series from a yfinance history frame.

    Accepts both flat columns (Ticker.history) and the (field, ticker) MultiIndex that
    yf.download produces, so a future switch of fetch method does not break parsing.
    `drop_phantom` applies `drop_phantom_bars` first (use for equities only).
    """
    if df is None or df.empty:
        raise NotFoundError(f"yfinance {ticker}: empty history")
    df = _flatten_columns(df, ticker)
    if "Close" not in df.columns:
        raise BadResponseError(f"yfinance {ticker}: no Close column in {list(df.columns)}")
    try:
        pd.DatetimeIndex(df.index)
    except (TypeError, ValueError) as exc:
        raise BadResponseError(f"yfinance {ticker}: index is not dates") from exc

    warnings: list[str] = []
    if drop_phantom:
        df, warnings = drop_phantom_bars(df, label=f"yfinance {ticker}")
        if df.empty:
            raise BadResponseError(f"yfinance {ticker}: only phantom bars")
    series, clean_warnings = clean_price_series(df["Close"], label=f"yfinance {ticker}")
    return series, warnings + clean_warnings


def _http_status(exc: Exception) -> int | None:
    """HTTP status carried by a curl_cffi HTTPError (an OSError subclass), if any."""
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return status if isinstance(status, int) else None


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, YFTickerMissingError):
        return False
    status = _http_status(exc)
    if status is not None:  # HTTP error: only 429 and 5xx are transient
        return status == 429 or status >= 500
    return isinstance(exc, YFRateLimitError | OSError)


def _today_ist() -> date:
    return pd.Timestamp.now(tz="Asia/Kolkata").date()


class YFinanceProvider:
    """PriceProvider for NSE equities (.NS) and Yahoo indices (^NSEI)."""

    source = SOURCE

    def __init__(
        self,
        fetcher: HistoryFetcher | None = None,
        *,
        max_attempts: int = 3,
        base_delay: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
        today: Callable[[], date] = _today_ist,
    ) -> None:
        self._fetch = fetcher or make_yfinance_fetcher()
        self._max_attempts = max_attempts
        self._base_delay = base_delay
        self._sleep = sleep
        self._today = today

    def fetch_history(self, asset: AssetRef, start: date | None = None) -> PriceHistory:
        ticker = yahoo_ticker(asset)
        try:
            df = call_with_retry(
                lambda: self._fetch(ticker, start),
                is_retryable=_is_retryable,
                max_attempts=self._max_attempts,
                base_delay=self._base_delay,
                sleep=self._sleep,
            )
        except YFTickerMissingError as exc:
            raise NotFoundError(f"yfinance {ticker}: {exc}") from exc
        except (YFRateLimitError, OSError) as exc:
            status = _http_status(exc)
            if status == 404:
                raise NotFoundError(f"yfinance {ticker}: HTTP 404") from exc
            if status is not None and not _is_retryable(exc):
                raise ProviderError(f"yfinance {ticker}: HTTP {status}") from exc
            raise ProviderError(
                f"yfinance {ticker}: {type(exc).__name__} after {self._max_attempts} attempts"
            ) from exc

        series, warnings = parse_history_frame(df, ticker, drop_phantom=asset.kind == "equity")

        # During (or before the close of) a trading session Yahoo returns a bar for today
        # holding the live price. It is not a close, so caching it would corrupt the last
        # return. Keep only bars strictly before today's IST date (costs at most one day).
        today = pd.Timestamp(self._today())
        if (series.index >= today).any():
            warnings.append(f"yfinance {ticker}: dropped bar(s) dated on/after {today.date()}")
            series = series[series.index < today]
            if series.empty:
                raise NotFoundError(
                    f"yfinance {ticker}: no completed sessions before {today.date()}"
                )
        for w in warnings:
            logger.warning(w)
        return PriceHistory(
            asset=asset,
            prices=series.rename(asset.id),
            source=SOURCE,
            source_name=ticker,
            warnings=tuple(warnings),
        )
