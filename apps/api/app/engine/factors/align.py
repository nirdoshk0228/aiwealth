"""Align daily factor returns to the master trading calendar.

Pure pandas: no I/O, no imports from app.data (enforced by tests/test_architecture.py).

Conventions
-----------
- `factors`: columns FACTOR_COLUMNS, **decimal simple returns**. A factor row dated f covers
  the period from the previous row of the factor file to f (close to close). MKT_RF is the
  market return minus RF.
- `calendar`: the *price* calendar, i.e. `AlignedPrices.prices.index`. The output has one row
  per calendar[1:], the same index as `simple_returns(prices)`; row c_i covers (c_{i-1}, c_i].

Rule
----
Row c_i gets factor data iff both c_{i-1} and c_i are dates in the factor file. Then all factor
rows dated in (c_{i-1}, c_i] are compounded:

    SMB, HML, WML, RF:  prod(1 + x) - 1
    MKT_RF:             prod(1 + MKT_RF + RF) - prod(1 + RF)

MKT_RF is rebuilt from the compounded total market return and the compounded RF, because an
excess return over several days is (total over the period) - (RF over the period). IIMA's own
monthly file compounds MF directly, prod(1 + MF) - 1; the two differ by roughly MKT_RF * RF, about
1e-6 for a two-day fold. SMB, HML and WML are long-short returns, and compounding them is the
usual approximation.

Otherwise the row is NaN, and the report says why:
  - outside coverage: c_{i-1} is before the factor file starts, or c_i is after it ends;
  - a calendar day the factor file lacks: the factor return around it spans two of our rows, so
    both rows lose their factor data (they are not compounded into a wrong period);
  - a factor row being compounded is NA in the source (NaN propagates; it is never skipped).
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

# Must equal app.data.models.FACTOR_COLUMNS (checked by tests/data/test_factor_repository.py).
FACTOR_COLUMNS = ("MKT_RF", "SMB", "HML", "WML", "RF")
_LONG_SHORT = ("SMB", "HML", "WML")


@dataclass(frozen=True)
class FactorAlignmentReport:
    """Everything `align_factors` did, so no factor day changes or disappears silently."""

    factor_first: date
    factor_last: date
    rows: int  # = len(calendar) - 1
    rows_with_factors: int
    rows_before_coverage: int
    rows_after_coverage: int
    first_row_after_coverage: date | None
    # Factor dates in (calendar[0], calendar[-1]] that are not calendar days; each one was
    # compounded into the next calendar row (which may still be NaN for another reason).
    folded_factor_days: tuple[date, ...]
    # Calendar days inside the factor coverage that the factor file does not have.
    calendar_days_missing: tuple[date, ...]
    # Rows inside the coverage left without factor data (missing days, NA in the source).
    rows_without_factors: tuple[date, ...]

    def summary(self) -> str:
        def days(ds: Sequence[date], limit: int = 12) -> str:
            shown = ", ".join(str(d) for d in ds[:limit])
            return f"{len(ds)}" + (f" [{shown}{', ...' if len(ds) > limit else ''}]" if ds else "")

        after = (
            f"{self.rows_after_coverage} (from {self.first_row_after_coverage})"
            if self.rows_after_coverage
            else "0"
        )
        return "\n".join(
            [
                "Factor alignment report",
                f"  factor file covers: {self.factor_first} -> {self.factor_last}",
                f"  return rows: {self.rows}, with factor data: {self.rows_with_factors}",
                f"  rows before factor coverage: {self.rows_before_coverage}",
                f"  rows after factor coverage: {after}",
                f"  factor days not on our calendar (compounded into the next row): "
                f"{days(self.folded_factor_days)}",
                f"  calendar days missing from the factor file: {days(self.calendar_days_missing)}",
                f"  rows inside coverage without factor data: {days(self.rows_without_factors)}",
            ]
        )


@dataclass(frozen=True)
class AlignedFactors:
    factors: pd.DataFrame  # index = calendar[1:], columns FACTOR_COLUMNS, NaN where no data
    report: FactorAlignmentReport


def _check_dates(index: pd.Index, label: str) -> pd.DatetimeIndex:
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError(f"{label}: index must be a DatetimeIndex")
    if index.tz is not None:
        raise ValueError(f"{label}: index must be tz-naive")
    if not index.is_monotonic_increasing or not index.is_unique:
        raise ValueError(f"{label}: dates must be strictly increasing")
    if not (index == index.normalize()).all():
        raise ValueError(f"{label}: dates must be calendar dates (midnight timestamps)")
    return index


def _dates(idx: pd.Index) -> tuple[date, ...]:
    return tuple(d.date() for d in pd.DatetimeIndex(idx))


def align_factors(factors: pd.DataFrame, calendar: pd.DatetimeIndex) -> AlignedFactors:
    """Put factor returns on the rows of `simple_returns(prices)` for a price calendar.

    Args:
        factors: decimal factor returns, columns FACTOR_COLUMNS (extra columns are ignored),
            DatetimeIndex of the factor file's dates. NaN allowed (NA in the source).
        calendar: the price calendar (at least 2 dates).

    Returns:
        AlignedFactors: a frame indexed by calendar[1:] (NaN rows where there is no factor
        data, see the module docstring) and a FactorAlignmentReport.
    """
    missing_cols = [c for c in FACTOR_COLUMNS if c not in factors.columns]
    if missing_cols:
        raise ValueError(f"factors: missing columns {missing_cols}")
    fdates = _check_dates(factors.index, "factors")
    cal = _check_dates(calendar, "calendar")
    if factors.empty:
        raise ValueError("factors: frame is empty")
    if len(cal) < 2:
        raise ValueError("calendar: need at least 2 dates to define one return row")
    fac = factors[list(FACTOR_COLUMNS)].astype("float64")
    if np.isinf(fac.to_numpy()).any():
        raise ValueError("factors: values must be finite or NaN")

    rows = cal[1:]
    # Each factor row goes to the first calendar date on or after it; rows dated on/before
    # calendar[0] or after calendar[-1] belong to no return row.
    pos = cal.searchsorted(fdates, side="left")
    used = (pos >= 1) & (pos < len(cal))
    sel = fac.iloc[np.flatnonzero(used)]
    target = cal[pos[used]]

    growth = pd.DataFrame(
        {
            "MKT": (1.0 + sel["MKT_RF"] + sel["RF"]).to_numpy(),
            "RF": (1.0 + sel["RF"]).to_numpy(),
            **{c: (1.0 + sel[c]).to_numpy() for c in _LONG_SHORT},
        },
        index=target,
    )
    compounded = growth.groupby(level=0).prod().reindex(rows)
    has_nan = (
        pd.Series(sel.isna().any(axis=1).to_numpy(), index=target)
        .groupby(level=0)
        .any()
        .reindex(rows, fill_value=False)
        .to_numpy(dtype=bool)
    )

    out = pd.DataFrame(
        {
            "MKT_RF": compounded["MKT"] - compounded["RF"],
            "RF": compounded["RF"] - 1.0,
            **{c: compounded[c] - 1.0 for c in _LONG_SHORT},
        },
        index=rows,
    )[list(FACTOR_COLUMNS)]

    is_fdate = cal.isin(fdates)
    valid = is_fdate[1:] & is_fdate[:-1] & ~has_nan
    out.loc[~valid, :] = np.nan

    first, last = fdates[0], fdates[-1]
    before = cal[:-1] < first
    after = rows > last
    inside = ~before & ~after
    report = FactorAlignmentReport(
        factor_first=first.date(),
        factor_last=last.date(),
        rows=len(rows),
        rows_with_factors=int(valid.sum()),
        rows_before_coverage=int(before.sum()),
        rows_after_coverage=int(after.sum()),
        first_row_after_coverage=rows[after][0].date() if after.any() else None,
        folded_factor_days=_dates(fdates[used & ~fdates.isin(cal)]),
        calendar_days_missing=_dates(cal[(cal >= first) & (cal <= last) & ~is_fdate]),
        rows_without_factors=_dates(rows[inside & ~valid]),
    )
    return AlignedFactors(factors=out, report=report)
