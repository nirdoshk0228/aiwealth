"""Exceptions raised by the data layer.

Callers (the seed script, later the refresh job) catch `ProviderError` to handle any
provider failure, or a subclass when the reaction differs (e.g. skip a delisted asset).
"""


class ProviderError(Exception):
    """A market data provider failed (network exhausted retries, unexpected payload, ...)."""


class NotFoundError(ProviderError):
    """The source has no data for the requested identifier (bad scheme code / ticker)."""


class BadResponseError(ProviderError):
    """The source answered, but the payload was malformed or unusable."""


class NotInCacheError(LookupError):
    """The requested series has not been seeded into the cache.

    Deliberately not a ProviderError: reading the cache never falls back to the network.
    Run scripts/seed_data.py to populate it.
    """
