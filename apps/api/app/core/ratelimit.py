"""In-process rate limiting for the chat endpoint (LLM calls spend a free quota).

Two limits: a sliding window per client key (e.g. IP) and a daily cap across all clients. State
lives in memory, so it is per API instance and resets on restart; for this project (one small
instance) that is enough, and it avoids running Redis.
"""

import threading
import time
from collections import deque
from collections.abc import Callable


class RateLimited(Exception):
    def __init__(self, message: str, retry_after: int) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class RateLimiter:
    def __init__(
        self,
        *,
        limit: int,
        window_seconds: int,
        daily_limit: int,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.limit = limit
        self.window = window_seconds
        self.daily_limit = daily_limit
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}
        self._day: int | None = None
        self._day_count = 0
        self._lock = threading.Lock()

    def check(self, key: str) -> None:
        """Record one request for `key`, or raise RateLimited without recording it."""
        now = self._clock()
        with self._lock:
            day = int(now // 86_400)
            if day != self._day:
                self._day, self._day_count = day, 0
            if self._day_count >= self.daily_limit:
                raise RateLimited(
                    "the assistant's daily limit is reached; please try again tomorrow",
                    retry_after=int(86_400 - now % 86_400),
                )
            q = self._hits.setdefault(key, deque())
            while q and q[0] <= now - self.window:
                q.popleft()
            if len(q) >= self.limit:
                raise RateLimited(
                    f"too many questions; at most {self.limit} per {self.window} seconds",
                    retry_after=int(q[0] + self.window - now) + 1,
                )
            q.append(now)
            self._day_count += 1
