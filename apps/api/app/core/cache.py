"""A small thread-safe in-process LRU cache (Redis was dropped from the stack).

Used for results that are expensive and deterministic for a given input, e.g. a rolling VaR
backtest. Each API process has its own cache; a restart or a market-data reload clears it
(the market-data load time is part of every key).
"""

import threading
from collections import OrderedDict
from collections.abc import Callable, Hashable


class LRUCache[V]:
    def __init__(self, maxsize: int = 128) -> None:
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        self.maxsize = maxsize
        self._data: OrderedDict[Hashable, V] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get_or_compute(self, key: Hashable, compute: Callable[[], V]) -> V:
        """Return the cached value for `key`, computing and storing it on a miss.

        `compute` runs outside the lock, so two concurrent misses may both compute; the
        result is the same (deterministic inputs) and the second write is harmless.
        """
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                self.hits += 1
                return self._data[key]
            self.misses += 1
        value = compute()
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)
        return value

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        return len(self._data)
