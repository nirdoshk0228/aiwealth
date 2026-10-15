"""Minimal retry with exponential backoff (no extra dependency)."""

import logging
import time
from collections.abc import Callable
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def call_with_retry(  # noqa: UP047 - TypeVar keeps Python 3.11 support (requires-python)
    fn: Callable[[], T],
    *,
    is_retryable: Callable[[Exception], bool],
    max_attempts: int = 3,
    base_delay: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call `fn`, retrying transient failures.

    Waits base_delay * 2**(k-1) seconds before retry k (1s, 2s, 4s, ... by default).
    Non-retryable exceptions propagate immediately; after `max_attempts` the last
    retryable exception propagates unchanged so the caller can map it.

    `sleep` is injectable so tests run instantly and can assert the backoff schedule.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:
            if attempt == max_attempts or not is_retryable(exc):
                raise
            delay = base_delay * 2 ** (attempt - 1)
            logger.warning(
                "Attempt %d/%d failed (%s: %s); retrying in %.1fs",
                attempt,
                max_attempts,
                type(exc).__name__,
                exc,
                delay,
            )
            sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover
