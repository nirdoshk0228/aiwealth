"""Service-level errors that routers translate into HTTP responses.

Services catch the engine's ValueError for *input* problems and re-raise AnalyticsInputError,
so a router can answer 422 without treating every ValueError (which could be a bug) as the
client's fault.
"""


class AnalyticsInputError(ValueError):
    """The request is well-formed but cannot be computed (unknown asset, too little data)."""


class MarketDataUnavailable(RuntimeError):
    """Market data is not loaded yet, or could not be loaded (not seeded, DB unreachable)."""


class NotFound(LookupError):
    """A requested resource (scenario, portfolio) does not exist or is not the caller's."""
