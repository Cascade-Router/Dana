"""Thread-safe Live Trace event bus (worker threads → Tk main thread).

LangGraph / agentic workers must only call ``emit_trace_event`` (non-blocking).
The CustomTkinter UI drains the queue via ``root.after`` polling.
"""

from __future__ import annotations

import queue
import threading
from typing import Any

from dana.schema import (
    NodeEnterEvent,
    NodeExitEvent,
    StateUpdateEvent,
    ToolExecutionEvent,
    TraceEvent,
    TraceEventType,
)


class TraceEventBus:
    """Process-wide singleton queue for Live Trace events."""

    _instance: TraceEventBus | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._q: queue.Queue[TraceEvent] = queue.Queue(maxsize=512)
        self._enabled = True

    @classmethod
    def instance(cls) -> TraceEventBus:
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = bool(enabled)

    def emit(self, event: TraceEvent) -> None:
        if not self._enabled:
            return
        try:
            self._q.put_nowait(event)
        except queue.Full:
            try:
                _ = self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(event)
            except queue.Full:
                pass
        except Exception:  # noqa: BLE001
            pass

    def drain(self, *, max_items: int = 64) -> list[TraceEvent]:
        """Non-blocking drain for the Tk main thread."""
        out: list[TraceEvent] = []
        for _ in range(max(1, int(max_items))):
            try:
                out.append(self._q.get_nowait())
            except queue.Empty:
                break
        return out


def get_trace_bus() -> TraceEventBus:
    return TraceEventBus.instance()


def emit_trace_event(
    event_type: TraceEventType | str,
    payload: dict[str, Any] | None = None,
    **kwargs: Any,
) -> None:
    """Helper for LangGraph nodes / agentic workers (always thread-safe)."""
    data: dict[str, Any] = {}
    if isinstance(payload, dict):
        data.update(payload)
    elif payload is not None:
        data["message"] = str(payload)
    data.update(kwargs)
    keys = data.get("state_keys") or ()
    if isinstance(keys, str):
        state_keys = (keys,)
    else:
        try:
            state_keys = tuple(str(k) for k in keys)
        except Exception:  # noqa: BLE001
            state_keys = ()
    et = str(event_type or "status").strip().lower()
    if et not in {
        "node_enter",
        "node_exit",
        "tool_execution",
        "state_update",
        "mode",
        "status",
    }:
        et = "status"
    latency = data.get("latency_ms")
    try:
        latency_f = float(latency) if latency is not None else None
    except (TypeError, ValueError):
        latency_f = None
    event = TraceEvent(
        event_type=et,  # type: ignore[arg-type]
        node=str(data.get("node") or data.get("stage") or ""),
        message=str(data.get("message") or ""),
        mode=str(data.get("mode") or ""),
        tool=str(data.get("tool") or ""),
        latency_ms=latency_f,
        payload=str(data.get("payload") or data.get("snippet") or "")[:2000],
        state_keys=state_keys,
    )
    get_trace_bus().emit(event)
