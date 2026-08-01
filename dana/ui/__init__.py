"""Dana CustomTkinter UI package.

Keep this module empty of backend imports. Submodules such as ``dana.ui.logo``
must load without pulling ``dana.schema``, ``pydantic``, or ``langgraph``.

Live Trace helpers live in ``dana.ui.trace_bus`` — import them explicitly:
``from dana.ui.trace_bus import emit_trace_event, get_trace_bus``.

STATE_CHANGE indicators live in ``dana.ui.status_bus``:
``from dana.ui.status_bus import emit_state_change, drain_state_changes``.
"""

from __future__ import annotations
