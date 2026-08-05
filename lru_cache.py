"""LRU Cache — capacity-bounded least-recently-used map."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Hashable


class LRUCache:
    """Least-recently-used cache with O(1) ``get`` / ``put``."""

    def __init__(self, capacity: int) -> None:
        if int(capacity) < 0:
            raise ValueError("capacity must be >= 0")
        self.capacity = int(capacity)
        self._data: OrderedDict[Hashable, Any] = OrderedDict()

    def get(self, key: Hashable) -> Any:
        """Return value for ``key``, or ``-1`` when missing (LeetCode-style)."""
        if key not in self._data:
            return -1
        self._data.move_to_end(key)
        return self._data[key]

    def put(self, key: Hashable, value: Any) -> None:
        """Insert or update ``key``. Evicts least-recently-used when over capacity."""
        if self.capacity <= 0:
            return
        if key in self._data:
            self._data.move_to_end(key)
            self._data[key] = value
            return
        if len(self._data) >= self.capacity:
            self._data.popitem(last=False)
        self._data[key] = value

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: object) -> bool:
        return key in self._data


__all__ = ("LRUCache",)
