"""Donna CustomTkinter UI packages (Live Trace, etc.).

Heavy modules (``trace_bus`` → ``donna.schema`` / optional ``langgraph``) are
loaded lazily so ``import donna.ui.logo`` stays lightweight.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "TraceEventBus",
    "emit_trace_event",
    "get_trace_bus",
]

_LAZY_EXPORTS = frozenset(__all__)


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        from donna.ui import trace_bus as _trace_bus

        value = getattr(_trace_bus, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _LAZY_EXPORTS)
