"""Donna CustomTkinter UI package.

Keep this module empty of backend imports. Submodules such as ``donna.ui.logo``
must load without pulling ``donna.schema``, ``pydantic``, or ``langgraph``.

Live Trace helpers live in ``donna.ui.trace_bus`` — import them explicitly:
``from donna.ui.trace_bus import emit_trace_event, get_trace_bus``.
"""

from __future__ import annotations
