"""Simulate 100 LRU cache reads/writes and print hit/miss summary."""

from __future__ import annotations

import json
import random
from typing import Any

from lru_cache import LRUCache


def run_simulation(
    *,
    capacity: int = 16,
    operations: int = 100,
    key_space: int = 40,
    seed: int = 42,
) -> dict[str, Any]:
    """Run mixed get/put traffic against ``LRUCache`` and return stats."""
    rng = random.Random(seed)
    cache = LRUCache(capacity)
    hits = 0
    misses = 0
    puts = 0
    gets = 0

    for i in range(operations):
        key = rng.randint(0, key_space - 1)
        # Alternate / bias toward reads after warm-up.
        if i < capacity or rng.random() < 0.45:
            cache.put(key, i)
            puts += 1
        else:
            gets += 1
            val = cache.get(key)
            if val == -1:
                misses += 1
            else:
                hits += 1

    total_lookups = hits + misses
    summary = {
        "capacity": capacity,
        "operations": operations,
        "puts": puts,
        "gets": gets,
        "hits": hits,
        "misses": misses,
        "hit_rate": (hits / total_lookups) if total_lookups else 0.0,
        "size": len(cache),
    }
    return summary


def main() -> None:
    summary = run_simulation()
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
