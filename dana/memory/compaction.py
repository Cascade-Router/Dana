"""Exponential weight decay and compaction for episodic memory."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any

from dana.memory.store import EpisodicMemoryStore

# Decay rates (per hour). Preferences never decay.
LAMBDA_PROCEDURAL = 0.05
LAMBDA_PREFERENCE = 0.0
WEIGHT_PRUNE_THRESHOLD = 0.15


class CompactionEngine:
    """Prune stale procedural traces via exponential memory decay."""

    def __init__(
        self,
        *,
        time_fn: Callable[[], float] | None = None,
        weight_threshold: float = WEIGHT_PRUNE_THRESHOLD,
        lambda_procedural: float = LAMBDA_PROCEDURAL,
    ) -> None:
        self._time_fn: Callable[[], float] = time_fn or time.time
        self.weight_threshold = float(weight_threshold)
        self.lambda_procedural = float(lambda_procedural)

    def decay_lambda(self, category: str) -> float:
        cat = str(category or "").strip()
        if cat == "user_preference":
            return LAMBDA_PREFERENCE
        # Procedural execution traces and similar non-preference facts.
        return self.lambda_procedural

    def decay_weight(
        self,
        category: str,
        base_weight: float,
        created_at: float,
        *,
        now: float | None = None,
    ) -> float:
        """weight = base_weight * exp(-lambda * delta_hours)."""
        current = float(self._time_fn() if now is None else now)
        delta_hours = max(0.0, (current - float(created_at)) / 3600.0)
        lam = self.decay_lambda(category)
        return float(base_weight) * math.exp(-lam * delta_hours)

    def compact_memory(self, store: EpisodicMemoryStore) -> dict[str, Any]:
        """Delete non-preference facts whose decayed weight falls below threshold."""
        now = float(self._time_fn())
        pruned: list[dict[str, Any]] = []
        kept = 0
        for fact in store.list_facts(include_expired=True):
            cat = str(fact.get("category") or "")
            if cat == "user_preference":
                kept += 1
                continue
            base = float(fact.get("confidence_score") or 1.0)
            created = float(fact.get("timestamp") or fact.get("created_at") or 0.0)
            weight = self.decay_weight(cat, base, created, now=now)
            if weight < self.weight_threshold:
                fact_id = fact.get("id")
                if fact_id is not None and store.delete_fact(int(fact_id)):
                    pruned.append(
                        {
                            "id": int(fact_id),
                            "category": cat,
                            "key": fact.get("key"),
                            "weight": weight,
                        }
                    )
                continue
            kept += 1
        return {
            "ok": True,
            "pruned": len(pruned),
            "kept": kept,
            "details": pruned,
            "threshold": self.weight_threshold,
        }
