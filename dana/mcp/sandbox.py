"""Sandboxed MCP tool execution.

``dana/mcp/client.py`` only discovers tools and advertises them to the
planning LLM — nothing there actually invokes ``tools/call`` on a live
server. This module is that missing execution path, and the *only* one the
Meta-Broker/worker graph should use: an MCP server is an arbitrary,
third-party process, no more trustworthy than LLM-generated code, so a call
here gets the same blast-radius containment generated code already receives:

  - a system-health gate before starting (``dana.system_health``),
  - a hard wall-clock timeout with process-tree kill on hang
    (mirrors ``dana.tools.system_repl``'s subprocess jail),
  - a before/after file snapshot with automatic rollback on failure
    (reuses ``dana.graph.runtime_harness``'s epic artifact tracker —
    the same mechanism that protects generated code).

``MCPClient._request`` blocks on a plain ``readline()`` with no way to
interrupt it in place, so a wedged server can only be recovered from by
killing its process tree; that also invalidates the cached connection so
the next call reconnects a fresh server instead of reusing a corrupted pipe.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
from typing import Any

from dana.graph.runtime_harness import (
    begin_epic_artifact_tracking,
    commit_epic_artifact_tracking,
    rollback_scratch_workspace,
    run_validation_harness,
)
from dana.mcp.client import MCPClient, _parse_server_specs
from dana.system_health import check_system_health, kill_process_tree

DEFAULT_MCP_TIMEOUT_S = 15.0
MAX_RESULT_CHARS = 2000

_FILE_TOKEN_RE = re.compile(r"([\w./\\-]+\.\w{1,8})\b")

_clients_lock = threading.Lock()
_clients: dict[str, MCPClient] = {}


def _truncate(text: str, limit: int = MAX_RESULT_CHARS) -> str:
    raw = text if isinstance(text, str) else str(text or "")
    if len(raw) <= limit:
        return raw
    return f"...[truncated to last {limit} chars]\n{raw[-limit:]}"


def _guess_watch_paths(arguments: dict[str, Any]) -> list[str]:
    """Heuristic: scan argument values for file-looking tokens to snapshot.

    Same idea as runtime_harness's own epic-goal scan — we don't know an MCP
    tool's semantics, so this is a best-effort net, not a guarantee.
    """
    blob = " ".join(str(v) for v in (arguments or {}).values())
    return sorted({m.group(1).replace("\\", "/") for m in _FILE_TOKEN_RE.finditer(blob)})


def get_mcp_client(server_id: str, *, env_var: str = "DANA_MCP_SERVERS") -> MCPClient:
    """Return a persistent, connected client for ``server_id`` (spawn on first use).

    Reconnects automatically if a previous connection was killed after a
    timeout or crash (see ``_drop_client``).
    """
    with _clients_lock:
        cached = _clients.get(server_id)
        if cached is not None and cached.is_alive():
            return cached
        raw = (os.environ.get(env_var) or "").strip()
        for sid, cmd in _parse_server_specs(raw):
            if sid != server_id:
                continue
            exe = shutil.which(cmd[0]) or cmd[0]
            client = MCPClient([exe, *cmd[1:]], server_id=sid)
            client.connect()
            _clients[server_id] = client
            return client
        raise KeyError(f"No MCP server configured with id={server_id!r} in {env_var}")


def _drop_client(server_id: str) -> None:
    """Forget a cached client after its process was killed — force reconnect."""
    with _clients_lock:
        _clients.pop(server_id, None)


def call_mcp_tool_sandboxed(
    client: MCPClient,
    tool_name: str,
    arguments: dict[str, Any] | None = None,
    *,
    workspace_path: str | None = None,
    run_key: str | None = None,
    validate_command: str | None = None,
    timeout_s: float = DEFAULT_MCP_TIMEOUT_S,
    extra_watch_paths: list[str] | None = None,
) -> dict[str, Any]:
    """Invoke one MCP ``tools/call`` inside the staging + rollback sandbox.

    Mirrors ``run_validation_harness``'s contract: never raises, always
    returns a structured dict with a ``success`` key. On timeout or RPC
    failure, the server subprocess is killed (process tree) and any files
    the call might have touched are rolled back via the same
    ``rollback_scratch_workspace`` machinery that protects generated epics.
    On success, the snapshot is committed (discarded) rather than restored.
    """
    from dana.paths import PROJECT_ROOT

    workspace = str(workspace_path or PROJECT_ROOT)
    key = run_key or f"mcp-{client.server_id}-{tool_name}-{int(time.time() * 1000)}"
    args = dict(arguments or {})
    watch_paths = sorted(set(_guess_watch_paths(args)) | set(extra_watch_paths or []))
    if watch_paths:
        begin_epic_artifact_tracking(workspace, watch_paths, run_key=key)

    try:
        check_system_health()
    except SystemError as exc:
        return {
            "success": False,
            "result": None,
            "error": str(exc),
            "timed_out": False,
            "rolled_back": False,
            "duration_s": 0.0,
        }

    outcome: dict[str, Any] = {}

    def _target() -> None:
        try:
            outcome["result"] = client.call_tool(tool_name, args)
        except Exception as exc:  # noqa: BLE001
            outcome["error"] = exc

    t0 = time.monotonic()
    thread = threading.Thread(target=_target, name=f"mcp-call-{tool_name}", daemon=True)
    thread.start()
    thread.join(timeout=max(0.1, float(timeout_s)))
    duration = time.monotonic() - t0

    if thread.is_alive():
        # Hung server: readline() inside client.call_tool() can't be
        # interrupted from here. Killing the process tree forces that
        # blocked read to hit EOF, which lets the stuck thread unwind on
        # its own shortly after (we don't wait on it further).
        pid = client.pid()
        if pid:
            kill_process_tree(pid)
        _drop_client(client.server_id)
        rollback = rollback_scratch_workspace(workspace, run_key=key)
        return {
            "success": False,
            "result": None,
            "error": (
                f"MCP tool {tool_name!r} timed out after {timeout_s}s on "
                f"server {client.server_id!r}; process tree killed."
            ),
            "timed_out": True,
            "rolled_back": True,
            "rollback": rollback,
            "duration_s": duration,
        }

    if "error" in outcome:
        rollback = rollback_scratch_workspace(workspace, run_key=key)
        return {
            "success": False,
            "result": None,
            "error": f"{type(outcome['error']).__name__}: {outcome['error']}",
            "timed_out": False,
            "rolled_back": True,
            "rollback": rollback,
            "duration_s": duration,
        }

    result = outcome.get("result")

    if validate_command:
        harness_result = run_validation_harness(workspace, validate_command, timeout_s=timeout_s)
        if not harness_result.get("success"):
            rollback = rollback_scratch_workspace(workspace, run_key=key)
            return {
                "success": False,
                "result": result,
                "error": f"post-call validation failed: {harness_result.get('stderr')}",
                "timed_out": bool(harness_result.get("timed_out")),
                "rolled_back": True,
                "rollback": rollback,
                "validation": harness_result,
                "duration_s": duration,
            }

    if watch_paths:
        commit_epic_artifact_tracking(key)
    return {
        "success": True,
        "result": result,
        "error": None,
        "timed_out": False,
        "rolled_back": False,
        "duration_s": duration,
    }


def mcp_tool_call(
    server_id: str,
    tool_name: str,
    arguments: dict[str, Any] | None = None,
    *,
    validate_command: str | None = None,
    timeout_s: float = DEFAULT_MCP_TIMEOUT_S,
) -> str:
    """Sandboxed entrypoint for calling a live MCP tool.

    This is the ONLY sanctioned way generated/agent code should invoke an
    MCP server — it never touches ``MCPClient`` directly. Returns a
    plain-text observation, matching ``shell_execute``/``python_repl``'s
    contract, so it slots into the same tool-output conventions.
    """
    try:
        client = get_mcp_client(server_id)
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: mcp_tool_call could not connect to server {server_id!r}: {exc}"

    outcome = call_mcp_tool_sandboxed(
        client,
        tool_name,
        arguments,
        validate_command=validate_command,
        timeout_s=timeout_s,
    )
    if outcome.get("timed_out"):
        return f"WARNING: {outcome['error']}"
    if not outcome.get("success"):
        return (
            "--- EXECUTION ERROR ---\n"
            f"Tool: {tool_name}\n"
            f"Server: {server_id}\n"
            f"{_truncate(str(outcome.get('error')))}"
        )
    try:
        result_text = json.dumps(outcome.get("result"), default=str)
    except Exception:  # noqa: BLE001
        result_text = str(outcome.get("result"))
    return f"exit_code=0\nresult:\n{_truncate(result_text)}"


__all__ = (
    "DEFAULT_MCP_TIMEOUT_S",
    "call_mcp_tool_sandboxed",
    "get_mcp_client",
    "mcp_tool_call",
)
