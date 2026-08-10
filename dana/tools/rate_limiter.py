"""Shared rate-limiter implementation for physical actuators.

Each actuator (mouse, keyboard, scroll, window, drag, clipboard) used to
copy-paste an identical lock + timestamp + cooldown check. This module is
the single implementation; every actuator gets its own *named* instance via
``get_limiter(name)`` so different actuator types stay independently
throttled — a click immediately followed by typing into the just-clicked
field (one composite tool call spanning two actuators) is not rejected
merely because a different actuator type fired a moment ago. What's shared
is the code, not a single global counter across actuator types.
"""

from __future__ import annotations

import threading
import time

MIN_ACTUATION_INTERVAL_S = 0.35


class ActuatorRateLimiter:
    """One independent cooldown gate, e.g. one per actuator type."""

    def __init__(self, *, min_interval_s: float = MIN_ACTUATION_INTERVAL_S) -> None:
        self._min_interval_s = min_interval_s
        self._lock = threading.Lock()
        self._last_actuation_ts = 0.0

    def check_and_mark(self) -> tuple[bool, str]:
        """Refuse a second actuation inside the cooldown window.

        Returns ``(True, "")`` and marks the actuation timestamp when the
        cooldown has elapsed; returns ``(False, reason)`` without marking
        otherwise.
        """
        now = time.monotonic()
        with self._lock:
            elapsed = now - self._last_actuation_ts
            if elapsed < self._min_interval_s:
                wait = self._min_interval_s - elapsed
                return False, f"rate_limited: wait {wait:.2f}s between actuations"
            self._last_actuation_ts = now
        return True, ""

    def reset(self) -> None:
        """Test-only: clear this limiter's cooldown."""
        with self._lock:
            self._last_actuation_ts = 0.0


_registry: dict[str, ActuatorRateLimiter] = {}
_registry_lock = threading.Lock()


def get_limiter(name: str) -> ActuatorRateLimiter:
    """Return the process-wide limiter for actuator ``name``, creating it on first use."""
    with _registry_lock:
        limiter = _registry.get(name)
        if limiter is None:
            limiter = ActuatorRateLimiter()
            _registry[name] = limiter
        return limiter


def reset() -> None:
    """Test-only: reset every registered actuator's cooldown."""
    with _registry_lock:
        limiters = list(_registry.values())
    for limiter in limiters:
        limiter.reset()


__all__ = ("MIN_ACTUATION_INTERVAL_S", "ActuatorRateLimiter", "get_limiter", "reset")
