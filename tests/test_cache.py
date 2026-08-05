"""Pytest coverage for ``LRUCache`` (capacity, updates, missing keys)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lru_cache import LRUCache


def test_missing_key_returns_sentinel() -> None:
    cache = LRUCache(2)
    assert cache.get("missing") == -1


def test_put_get_and_key_update() -> None:
    cache = LRUCache(2)
    cache.put(1, 10)
    assert cache.get(1) == 10
    cache.put(1, 99)
    assert cache.get(1) == 99


def test_capacity_eviction_lru_order() -> None:
    cache = LRUCache(2)
    cache.put(1, 1)
    cache.put(2, 2)
    assert cache.get(1) == 1  # 1 becomes most-recent
    cache.put(3, 3)  # evicts key 2 (least recently used)
    assert cache.get(2) == -1
    assert cache.get(1) == 1
    assert cache.get(3) == 3


def test_zero_capacity_is_noop() -> None:
    cache = LRUCache(0)
    cache.put(1, 1)
    assert cache.get(1) == -1
    assert len(cache) == 0
