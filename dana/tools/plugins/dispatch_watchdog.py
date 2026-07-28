"""Plugin: dispatch_watchdog — fire-and-forget Watchdog LangGraph thread."""

from __future__ import annotations

from typing import Any

TOOL_ID = "dispatch_watchdog"


def execute(task: str) -> str:
    from dana.tools.langchain_tools import dispatch_watchdog_impl

    return dispatch_watchdog_impl(task)


def handle_tool_call(call: Any) -> str:
    args = getattr(call, "arguments", None) or {}
    if not isinstance(args, dict):
        args = {}
    task = args.get("task")
    if task is None or not str(task).strip():
        task = args.get("query")
    if task is None or not str(task).strip():
        return "ERROR: missing task"
    return execute(str(task).strip())
