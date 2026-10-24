"""Provider interface."""

from datetime import date
from typing import Protocol

from app.data.models import AssetRef, FactorHistory, PriceHistory


class PriceProvider(Protocol):
    """Fetches a full daily price history for one asset from an external source."""

    source: str

    def fetch_history(self, asset: AssetRef, start: date | None = None) -> PriceHistory:
        """Return the asset's history from `start` (inclusive), or all available if None.

        Raises:
            NotFoundError: the source has no data for this asset.
            BadResponseError: the payload was malformed or empty after cleaning.
            ProviderError: transient failures persisted after retries.
        """
        ...


class FactorProvider(Protocol):
    """Fetches one complete daily factor file (all dates) from an external source."""

    source: str

    def fetch(self) -> FactorHistory:
        """Return the parsed file.

        Raises:
            NotFoundError: the requested release/file does not exist.
            BadResponseError: the file's header, dates, units or values are not as expected.
            ProviderError: transient failures persisted after retries.
        """
        ...
