"""align_factors: hand-computed compounding and every kind of calendar mismatch."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from app.engine.factors.align import FACTOR_COLUMNS, align_factors
from app.engine.returns import simple_returns


def dates(*days: str) -> pd.DatetimeIndex:
    return pd.DatetimeIndex([f"2025-10-{d}" for d in days], name="date")


def frame(days: tuple[str, ...], **cols: list[float]) -> pd.DataFrame:
    """Factor frame on 2025-10-<days>; columns not given are filled with distinct small values."""
    n = len(days)
    data = {
        c: cols.get(c, list(0.001 * (k + 1) + 0.0001 * np.arange(n)))
        for k, c in enumerate(FACTOR_COLUMNS)
    }
    return pd.DataFrame(data, index=dates(*days))[list(FACTOR_COLUMNS)]


def test_identical_calendars_pass_values_through() -> None:
    f = frame(("16", "17", "20", "21"))
    out = align_factors(f, dates("16", "17", "20", "21"))

    pd.testing.assert_frame_equal(out.factors, f.iloc[1:], check_freq=False, atol=1e-15)
    r = out.report
    assert (r.rows, r.rows_with_factors) == (3, 3)
    assert r.folded_factor_days == r.calendar_days_missing == r.rows_without_factors == ()
    assert (r.rows_before_coverage, r.rows_after_coverage) == (0, 0)


def test_off_calendar_factor_day_is_compounded_into_the_next_row() -> None:
    # Saturday 2025-10-18 is a factor (special) session but not on our calendar.
    f = frame(
        ("16", "17", "18", "20", "21"),
        MKT_RF=[0.0, 0.0, 0.01, -0.02, 0.0],
        RF=[0.0, 0.0, 0.0001, 0.0003, 0.0],
        SMB=[0.0, 0.0, 0.02, 0.01, 0.0],
        HML=[0.0, 0.0, -0.01, 0.01, 0.0],
        WML=[0.0, 0.0, 0.0, 0.05, 0.0],
    )
    out = align_factors(f, dates("16", "17", "20", "21"))
    mon = out.factors.loc["2025-10-20"]

    # By hand: SMB = 1.02 * 1.01 - 1;  HML = 0.99 * 1.01 - 1;  RF = 1.0001 * 1.0003 - 1
    assert mon["SMB"] == pytest.approx(0.0302, abs=1e-15)
    assert mon["HML"] == pytest.approx(-0.0001, abs=1e-15)
    assert mon["WML"] == pytest.approx(0.05, abs=1e-15)
    assert mon["RF"] == pytest.approx(0.00040003, abs=1e-15)
    # Total market: 1.0101 * 0.9803 = 0.99020103; excess = 0.99020103 - 1.00040003
    assert mon["MKT_RF"] == pytest.approx(-0.010199, abs=1e-15)
    # Compounding the excess return directly (IIMA's monthly convention) differs by ~1e-6.
    assert 1.01 * 0.98 - 1 == pytest.approx(-0.0102, abs=1e-15)

    assert out.report.folded_factor_days == (date(2025, 10, 18),)
    assert out.report.rows_with_factors == 3


def test_calendar_day_missing_from_factors_blanks_both_rows_it_touches() -> None:
    # Factors have no Tuesday 21st, so their Wednesday return covers Mon -> Wed, which spans
    # our rows for the 21st and the 22nd. Neither row may get it.
    f = frame(("16", "17", "20", "22"))
    out = align_factors(f, dates("16", "17", "20", "21", "22"))

    assert out.factors.loc["2025-10-17":"2025-10-20"].notna().all().all()
    assert out.factors.loc["2025-10-21":"2025-10-22"].isna().all().all()
    r = out.report
    assert r.calendar_days_missing == (date(2025, 10, 21),)
    assert r.rows_without_factors == (date(2025, 10, 21), date(2025, 10, 22))
    assert r.rows_with_factors == 2


def test_rows_outside_factor_coverage_are_nan_and_counted() -> None:
    f = frame(("16", "17", "20", "21"))
    out = align_factors(f, dates("14", "15", "16", "17", "20", "21", "22", "23"))

    valid = out.factors.notna().all(axis=1)
    assert list(valid[valid].index.day) == [17, 20, 21]
    r = out.report
    # Rows 15 and 16 start before the first factor date (16 needs a factor close on 15).
    assert r.rows_before_coverage == 2
    assert r.rows_after_coverage == 2
    assert r.first_row_after_coverage == date(2025, 10, 22)
    assert r.rows_without_factors == ()  # nothing missing *inside* the coverage
    assert "rows after factor coverage: 2 (from 2025-10-22)" in r.summary()


def test_na_in_a_folded_day_propagates_instead_of_being_skipped() -> None:
    f = frame(("16", "17", "18", "20", "21"), SMB=[0.0, 0.0, np.nan, 0.01, 0.0])
    out = align_factors(f, dates("16", "17", "20", "21"))

    assert out.factors.loc["2025-10-20"].isna().all()
    assert out.factors.loc[["2025-10-17", "2025-10-21"]].notna().all().all()
    assert out.report.rows_without_factors == (date(2025, 10, 20),)


def test_na_on_a_calendar_day_blanks_that_row() -> None:
    f = frame(("16", "17", "20"), MKT_RF=[0.0, np.nan, 0.01])
    out = align_factors(f, dates("16", "17", "20"))
    assert out.factors.loc["2025-10-17"].isna().all()
    assert out.factors.loc["2025-10-20"].notna().all()


def test_output_rows_match_simple_returns_of_prices_on_the_calendar() -> None:
    cal = dates("16", "17", "20", "21")
    prices = pd.DataFrame({"a": [100.0, 101.0, 99.0, 100.0]}, index=cal)
    out = align_factors(frame(("15", "16", "17", "20", "21", "22")), cal)
    pd.testing.assert_index_equal(out.factors.index, simple_returns(prices).index)
    assert list(out.factors.columns) == list(FACTOR_COLUMNS)


def test_extra_factor_columns_are_ignored() -> None:
    f = frame(("16", "17")).assign(EXTRA=1.0)
    assert list(align_factors(f, dates("16", "17")).factors.columns) == list(FACTOR_COLUMNS)


@pytest.mark.parametrize(
    ("factors", "calendar", "match"),
    [
        (frame(("16", "17")).drop(columns="WML"), dates("16", "17"), "missing columns"),
        (frame(("16", "17")).iloc[:0], dates("16", "17"), "empty"),
        (frame(("16", "17")).replace(0.001, np.inf), dates("16", "17"), "finite"),
        (frame(("16", "17")), dates("16"), "at least 2 dates"),
        (frame(("16", "17")), dates("17", "16"), "strictly increasing"),
        (frame(("16", "17")), dates("16", "17").tz_localize("Asia/Kolkata"), "tz-naive"),
        (frame(("16", "17")), dates("16", "17") + pd.Timedelta(hours=15), "midnight"),
        (frame(("16", "17")).reset_index(drop=True), dates("16", "17"), "DatetimeIndex"),
    ],
)
def test_invalid_inputs_raise(factors: pd.DataFrame, calendar: pd.Index, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        align_factors(factors, calendar)  # type: ignore[arg-type]
