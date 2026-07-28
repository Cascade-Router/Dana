"""Approved plugin tools loaded from dana/tools/plugins/ (roadmap deployments)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from dana.tools.schema import ToolCall

# Explicit allowlist — never auto-import arbitrary modules from disk.
_PLUGIN_HANDLERS: dict[str, Callable[[ToolCall], str]] = {}


def _register_builtin_plugins() -> None:
    if _PLUGIN_HANDLERS:
        return
    from dana.tools.plugins.dispatch_swarm import (
        TOOL_ID as SWARM_ID,
        handle_tool_call as swarm_handler,
    )
    from dana.tools.plugins.dispatch_watchdog import (
        TOOL_ID as WATCHDOG_ID,
        handle_tool_call as watchdog_handler,
    )
    from dana.tools.plugins.file_jail_enforcer import (
        TOOL_ID as JAIL_ID,
        handle_tool_call as jail_handler,
    )
    from dana.tools.plugins.kill_watchdog import (
        TOOL_ID as KILL_WATCHDOG_ID,
        handle_tool_call as kill_watchdog_handler,
    )
    _PLUGIN_HANDLERS[JAIL_ID] = jail_handler
    _PLUGIN_HANDLERS[SWARM_ID] = swarm_handler
    _PLUGIN_HANDLERS[WATCHDOG_ID] = watchdog_handler
    _PLUGIN_HANDLERS[KILL_WATCHDOG_ID] = kill_watchdog_handler


def list_plugin_ids() -> list[str]:
    _register_builtin_plugins()
    return sorted(_PLUGIN_HANDLERS)


def resolve_plugin_handler(tool_id: str) -> Callable[[ToolCall], str] | None:
    """Return a handler if tool_id is an approved plugin; else None."""
    _register_builtin_plugins()
    return _PLUGIN_HANDLERS.get(str(tool_id or "").strip())


def dispatch_plugin(call: ToolCall) -> Any:
    handler = resolve_plugin_handler(call.tool_id)
    if handler is None:
        raise KeyError(f"No plugin registered for {call.tool_id}")
    return handler(call)
