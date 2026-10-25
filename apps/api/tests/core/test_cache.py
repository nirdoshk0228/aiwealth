import pytest

from app.core.cache import LRUCache


def test_hit_miss_and_eviction_order() -> None:
    cache: LRUCache[int] = LRUCache(maxsize=2)
    calls: list[str] = []

    def compute(k: str) -> int:
        calls.append(k)
        return len(k)

    assert cache.get_or_compute("a", lambda: compute("a")) == 1
    assert cache.get_or_compute("bb", lambda: compute("bb")) == 2
    assert cache.get_or_compute("a", lambda: compute("a")) == 1  # hit; "a" now most recent
    cache.get_or_compute("ccc", lambda: compute("ccc"))  # evicts "bb", the least recent
    cache.get_or_compute("a", lambda: compute("a"))
    cache.get_or_compute("bb", lambda: compute("bb"))
    assert calls == ["a", "bb", "ccc", "bb"]
    assert (cache.hits, cache.misses, len(cache)) == (2, 4, 2)


def test_errors_are_not_cached() -> None:
    cache: LRUCache[int] = LRUCache(maxsize=2)

    def boom() -> int:
        raise ValueError("bad")

    with pytest.raises(ValueError):
        cache.get_or_compute("k", boom)
    assert cache.get_or_compute("k", lambda: 5) == 5


def test_maxsize_must_be_positive() -> None:
    with pytest.raises(ValueError):
        LRUCache(maxsize=0)
