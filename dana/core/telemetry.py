from __future__ import annotations

import threading
from collections import deque
from typing import Any


class AsyncRingBuffer:
    """Thread-safe fixed-size ring buffer for low-latency telemetry."""

    def __init__(self, *, capacity: int = 500) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._lock = threading.Lock()
        self._events: deque[dict[str, Any]] = deque(maxlen=capacity)

    def append(self, event: dict[str, Any]) -> None:
        with self._lock:
            self._events.append(event)

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._events)

    def clear(self) -> None:
        with self._lock:
            self._events.clear()


class NeuralStreamEmitter:
    """O(1), non-blocking structured event emitter for the UI/telemetry stream."""

    def __init__(self, buffer: AsyncRingBuffer) -> None:
        self._buffer = buffer

    def emit(self, event_type: str, payload: dict[str, Any] | None = None) -> None:
        event = {"type": event_type, "payload": dict(payload or {})}
        self._buffer.append(event)
