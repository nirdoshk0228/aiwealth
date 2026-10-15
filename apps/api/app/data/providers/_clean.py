"""Shared post-parse cleaning so every provider returns series with the same invariants."""

import pandas as pd

from app.data.errors import BadResponseError


def clean_price_series(raw: pd.Series, label: str) -> tuple[pd.Series, list[str]]:
    """Normalise a raw price series to the `PriceHistory.prices` invariants.

    Steps:
      1. Index -> tz-naive midnight dates. tz-aware timestamps are first converted to
         Asia/Kolkata so an NSE session date never shifts to the previous UTC day.
      2. Values -> float; non-numeric, non-finite and <= 0 values are dropped
         (a price/NAV of 0 is a data error, not a real observation).
      3. Sort ascending; on duplicate dates keep the first occurrence in source order.

    Rows are dropped rather than raising so that one bad print does not discard years of
    history; every drop is reported in the returned warnings.

    Raises:
        BadResponseError: nothing usable remains after cleaning.
    """
    warnings: list[str] = []

    idx = pd.DatetimeIndex(raw.index)
    if idx.tz is not None:
        idx = idx.tz_convert("Asia/Kolkata").tz_localize(None)
    idx = idx.normalize()

    values = pd.to_numeric(pd.Series(raw.to_numpy(), index=idx), errors="coerce").astype(float)
    valid = values.notna() & values.gt(0) & values.abs().lt(float("inf"))
    n_bad = int((~valid).sum())
    if n_bad:
        warnings.append(f"{label}: dropped {n_bad} non-numeric/non-positive value(s)")
    values = values[valid]

    dup = values.index.duplicated(keep="first")
    if dup.any():
        warnings.append(f"{label}: dropped {int(dup.sum())} duplicate date(s), kept first")
        values = values[~dup]

    values = values.sort_index()
    if values.empty:
        raise BadResponseError(f"{label}: no valid prices after cleaning")
    values.index.name = "date"
    return values, warnings
