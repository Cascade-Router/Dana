"""Plugin: kill_watchdog — stop a registered background Watchdog job."""

from __future__ import annotations

from typing import Any

TOOL_ID = "kill_watchdog"


def execute(task_id: str) -> str:
    from dana.tools.langchain_tools import kill_watchdog_impl

    return kill_watchdog_impl(task_id)


def handle_tool_call(call: Any) -> str:
    args = getattr(call, "arguments", None) or {}
    if not isinstance(args, dict):
        args = {}
    task_id = args.get("task_id")
    if task_id is None or not str(task_id).strip():
        task_id = args.get("id")
    if task_id is None or not str(task_id).strip():
        return "ERROR: missing task_id"
    return execute(str(task_id).strip())
