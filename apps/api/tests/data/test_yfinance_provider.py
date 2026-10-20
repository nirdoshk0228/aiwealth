"""YFinanceProvider tests. The network fetcher is replaced by a stub that replays saved real
Ticker.history() output (tests/data/fixtures/yf_history_*.csv, captured 2026-10-15)."""

from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from yfinance.exceptions import YFRateLimitError, YFTickerMissingError

from app.data.errors import BadResponseError, NotFoundError, ProviderError
from app.data.models import AssetRef
from app.data.providers.yfinance_provider import (
    YFinanceProvider,
    parse_history_frame,
    yahoo_ticker,
)

FIXTURES = Path(__file__).parent / "fixtures"
RELIANCE = AssetRef(kind="equity", symbol="RELIANCE", name="Reliance Industries")
NIFTY = AssetRef(kind="index", symbol="^NSEI", name="Nifty 50")


def load_frame(name: str) -> pd.DataFrame:
    # Saved with tz-aware Asia/Kolkata timestamps, exactly as Ticker.history returns them.
    df = pd.read_csv(FIXTURES / name, index_col="Date")
    df.index = pd.to_datetime(df.index, utc=True).tz_convert("Asia/Kolkata")
    return df


def stub_fetcher(frames: dict[str, pd.DataFrame]):
    calls: list[tuple[str, date | None]] = []

    def fetch(ticker: str, start: date | None) -> pd.DataFrame:
        calls.append((ticker, start))
        return frames[ticker]

    return fetch, calls


def test_yahoo_ticker_mapping() -> None:
    assert yahoo_ticker(RELIANCE) == "RELIANCE.NS"
    assert yahoo_ticker(AssetRef(kind="equity", symbol="tcs.ns", name="TCS")) == "TCS.NS"
    assert yahoo_ticker(NIFTY) == "^NSEI"
    with pytest.raises(ValueError):
        yahoo_ticker(AssetRef(kind="index", symbol="NSEI", name="no caret"))
    with pytest.raises(ValueError):
        yahoo_ticker(AssetRef(kind="mutual_fund", symbol="122639", name="MF"))


def test_fetch_equity_returns_adjusted_close_on_ist_dates() -> None:
    fetch, calls = stub_fetcher({"RELIANCE.NS": load_frame("yf_history_RELIANCE.NS.csv")})
    hist = YFinanceProvider(fetch, sleep=lambda s: None).fetch_history(RELIANCE)

    assert calls == [("RELIANCE.NS", None)]
    s = hist.prices
    assert s.name == "equity:RELIANCE"
    assert hist.source == "yfinance" and hist.source_name == "RELIANCE.NS"
    assert len(s) == 10
    assert s.index.tz is None
    # IST midnight must map to the same calendar date (a naive UTC conversion would give 31 Aug).
    assert s.index[0] == pd.Timestamp("2026-09-01")
    assert s.iloc[0] == pytest.approx(1309.0)
    assert s.index[-1] == pd.Timestamp("2026-09-15")
    assert s.iloc[-1] == pytest.approx(1235.300048828125)
    # 14 Sep 2026 is absent in the source (exchange holiday) and must not be invented.
    assert pd.Timestamp("2026-09-14") not in s.index


def test_fetch_index_and_start_is_passed_through() -> None:
    fetch, calls = stub_fetcher({"^NSEI": load_frame("yf_history_NSEI.csv")})
    hist = YFinanceProvider(fetch).fetch_history(NIFTY, start=date(2026, 9, 1))
    assert calls == [("^NSEI", date(2026, 9, 1))]
    assert hist.prices.iloc[-1] == pytest.approx(23118.599609375)


def test_parse_accepts_download_style_multiindex() -> None:
    flat = load_frame("yf_history_RELIANCE.NS.csv")[["Close", "Open"]]
    multi = flat.copy()
    multi.columns = pd.MultiIndex.from_tuples([("Close", "RELIANCE.NS"), ("Open", "RELIANCE.NS")])
    s_flat, _ = parse_history_frame(flat, "RELIANCE.NS")
    s_multi, _ = parse_history_frame(multi, "RELIANCE.NS")
    pd.testing.assert_series_equal(s_flat, s_multi, check_names=False)


def test_empty_frame_is_not_found() -> None:
    with pytest.raises(NotFoundError):
        parse_history_frame(pd.DataFrame(), "X.NS")


def test_missing_close_column_is_bad_response() -> None:
    df = load_frame("yf_history_RELIANCE.NS.csv").drop(columns=["Close"])
    with pytest.raises(BadResponseError, match="no Close"):
        parse_history_frame(df, "RELIANCE.NS")


def test_nan_and_nonpositive_closes_dropped_with_warning() -> None:
    df = load_frame("yf_history_RELIANCE.NS.csv")
    df.iloc[2, df.columns.get_loc("Close")] = float("nan")
    df.iloc[3, df.columns.get_loc("Close")] = 0.0
    s, warnings = parse_history_frame(df, "RELIANCE.NS")
    assert len(s) == 8
    assert warnings == ["yfinance RELIANCE.NS: dropped 2 non-numeric/non-positive value(s)"]


def test_unknown_ticker_fails_fast_as_not_found() -> None:
    calls = {"n": 0}

    def fetch(ticker: str, start: date | None) -> pd.DataFrame:
        calls["n"] += 1
        raise YFTickerMissingError(ticker, "possibly delisted; no timezone found")

    sleeps: list[float] = []
    with pytest.raises(NotFoundError):
        YFinanceProvider(fetch, sleep=sleeps.append).fetch_history(RELIANCE)
    assert calls["n"] == 1 and sleeps == []


def test_rate_limit_and_network_errors_are_retried() -> None:
    errors: list[Exception] = [YFRateLimitError(), TimeoutError("read timed out")]
    frame = load_frame("yf_history_RELIANCE.NS.csv")

    def fetch(ticker: str, start: date | None) -> pd.DataFrame:
        if errors:
            raise errors.pop(0)
        return frame

    sleeps: list[float] = []
    hist = YFinanceProvider(fetch, base_delay=2.0, sleep=sleeps.append).fetch_history(RELIANCE)
    assert sleeps == [2.0, 4.0]
    assert len(hist.prices) == 10


def test_persistent_network_error_raises_provider_error() -> None:
    def fetch(ticker: str, start: date | None) -> pd.DataFrame:
        raise ConnectionError("unreachable")

    with pytest.raises(ProviderError, match="after 3 attempts"):
        YFinanceProvider(fetch, sleep=lambda s: None).fetch_history(RELIANCE)


def test_programming_errors_are_not_retried() -> None:
    calls = {"n": 0}

    def fetch(ticker: str, start: date | None) -> pd.DataFrame:
        calls["n"] += 1
        raise KeyError("bug")

    with pytest.raises(KeyError):
        YFinanceProvider(fetch, sleep=lambda s: None).fetch_history(RELIANCE)
    assert calls["n"] == 1


def _curl_http_error(status: int) -> Exception:
    """The real curl_cffi HTTPError (an OSError) as raised inside yfinance."""
    from types import SimpleNamespace

    from curl_cffi.requests.exceptions import HTTPError

    return HTTPError(f"HTTP Error {status}: ", 0, SimpleNamespace(status_code=status))


def test_http_404_fails_fast_as_not_found() -> None:
    calls = {"n": 0}

    def fetch(ticker: str, start: date | None) -> pd.DataFrame:
        calls["n"] += 1
        raise _curl_http_error(404)

    sleeps: list[float] = []
    with pytest.raises(NotFoundError, match="HTTP 404"):
        YFinanceProvider(fetch, sleep=sleeps.append).fetch_history(RELIANCE)
    assert calls["n"] == 1 and sleeps == []


def test_http_403_fails_fast_as_provider_error() -> None:
    def fetch(ticker: str, start: date | None) -> pd.DataFrame:
        raise _curl_http_error(403)

    sleeps: list[float] = []
    with pytest.raises(ProviderError, match="HTTP 403"):
        YFinanceProvider(fetch, sleep=sleeps.append).fetch_history(RELIANCE)
    assert sleeps == []


def test_http_503_is_retried() -> None:
    errors = [_curl_http_error(503)]
    frame = load_frame("yf_history_RELIANCE.NS.csv")

    def fetch(ticker: str, start: date | None) -> pd.DataFrame:
        if errors:
            raise errors.pop(0)
        return frame

    sleeps: list[float] = []
    YFinanceProvider(fetch, sleep=sleeps.append).fetch_history(RELIANCE)
    assert sleeps == [2.0]


def test_todays_intraday_bar_is_dropped() -> None:
    # Pretend "today" is 15 Sep 2026: the 15 Sep row is a live price, not a close.
    fetch, _ = stub_fetcher({"RELIANCE.NS": load_frame("yf_history_RELIANCE.NS.csv")})
    hist = YFinanceProvider(fetch, today=lambda: date(2026, 9, 15)).fetch_history(RELIANCE)
    assert hist.prices.index[-1] == pd.Timestamp("2026-09-11")
    assert len(hist.prices) == 9
    assert any("on/after 2026-09-15" in w for w in hist.warnings)


def test_only_todays_bar_raises_not_found() -> None:
    frame = load_frame("yf_history_RELIANCE.NS.csv").iloc[-1:]
    fetch, _ = stub_fetcher({"RELIANCE.NS": frame})
    with pytest.raises(NotFoundError, match="no completed sessions"):
        YFinanceProvider(fetch, today=lambda: date(2026, 9, 15)).fetch_history(RELIANCE)


def _with_phantom_on_14_sep(frame: pd.DataFrame) -> pd.DataFrame:
    """Insert a Yahoo-style phantom bar on 14 Sep 2026: flat OHLC = previous close, volume 0."""
    prev_close = frame.loc[frame.index[-2], "Close"]  # 11 Sep close
    ts = pd.Timestamp("2026-09-14", tz="Asia/Kolkata")
    row = pd.DataFrame(
        {
            "Open": prev_close,
            "High": prev_close,
            "Low": prev_close,
            "Close": prev_close,
            "Volume": 0,
            "Dividends": 0.0,
            "Stock Splits": 0.0,
        },
        index=pd.DatetimeIndex([ts], name=frame.index.name),
    )
    return pd.concat([frame, row]).sort_index()


def test_phantom_bar_dropped_for_equities_with_warning() -> None:
    frame = _with_phantom_on_14_sep(load_frame("yf_history_RELIANCE.NS.csv"))
    fetch, _ = stub_fetcher({"RELIANCE.NS": frame})
    hist = YFinanceProvider(fetch).fetch_history(RELIANCE)
    assert pd.Timestamp("2026-09-14") not in hist.prices.index
    assert len(hist.prices) == 10
    assert any("1 phantom bar(s)" in w and "2026-09-14" in w for w in hist.warnings)


def test_phantom_rule_not_applied_to_index() -> None:
    frame = _with_phantom_on_14_sep(load_frame("yf_history_NSEI.csv"))
    fetch, _ = stub_fetcher({"^NSEI": frame})
    hist = YFinanceProvider(fetch).fetch_history(NIFTY)
    assert pd.Timestamp("2026-09-14") in hist.prices.index
    assert not any("phantom" in w for w in hist.warnings)


def test_zero_volume_bar_with_a_price_range_is_kept() -> None:
    # Zero volume alone is not enough: a bar that moved is not a placeholder.
    frame = _with_phantom_on_14_sep(load_frame("yf_history_RELIANCE.NS.csv"))
    frame.loc[pd.Timestamp("2026-09-14", tz="Asia/Kolkata"), "High"] += 1.0
    s, warnings = parse_history_frame(frame, "RELIANCE.NS", drop_phantom=True)
    assert pd.Timestamp("2026-09-14") in s.index
    assert warnings == []


def test_phantom_check_skipped_without_ohlcv() -> None:
    frame = load_frame("yf_history_RELIANCE.NS.csv")[["Close"]]
    s, warnings = parse_history_frame(frame, "RELIANCE.NS", drop_phantom=True)
    assert len(s) == 10
    assert warnings == ["yfinance RELIANCE.NS: OHLCV columns missing; phantom-bar check skipped"]
