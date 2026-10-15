import pytest

from app.data.providers._retry import call_with_retry


def test_returns_first_success_without_sleeping() -> None:
    sleeps: list[float] = []
    assert call_with_retry(lambda: 42, is_retryable=lambda e: True, sleep=sleeps.append) == 42
    assert sleeps == []


def test_backoff_doubles_and_last_error_propagates() -> None:
    sleeps: list[float] = []

    def boom() -> None:
        raise TimeoutError("x")

    with pytest.raises(TimeoutError):
        call_with_retry(
            boom, is_retryable=lambda e: True, max_attempts=4, base_delay=0.5, sleep=sleeps.append
        )
    assert sleeps == [0.5, 1.0, 2.0]


def test_non_retryable_propagates_immediately() -> None:
    sleeps: list[float] = []

    def boom() -> None:
        raise ValueError("x")

    with pytest.raises(ValueError):
        call_with_retry(boom, is_retryable=lambda e: False, sleep=sleeps.append)
    assert sleeps == []


def test_invalid_max_attempts() -> None:
    with pytest.raises(ValueError):
        call_with_retry(lambda: 1, is_retryable=lambda e: True, max_attempts=0)
