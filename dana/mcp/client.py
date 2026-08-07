"""Lightweight MCP (Model Context Protocol) client — stdio / JSON-RPC.

Discovers tools from local or remote MCP servers so the Spec Compiler and
Meta-Broker can advertise live endpoints (ROS2, filesystem, DB schema, …).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MCPTool:
    name: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)
    server: str = ""

    def as_prompt_line(self) -> str:
        desc = (self.description or "").strip().replace("\n", " ")
        if len(desc) > 160:
            desc = desc[:157] + "..."
        schema_keys = ""
        props = (self.input_schema or {}).get("properties") or {}
        if isinstance(props, dict) and props:
            schema_keys = " args=[" + ", ".join(sorted(props.keys())[:8]) + "]"
        server_bit = f" server={self.server!r}" if self.server else ""
        return f"- `{self.name}`{server_bit}{schema_keys}: {desc or '(no description)'}"


class MCPClient:
    """Minimal JSON-RPC 2.0 MCP client over stdio (initialize + tools/list)."""

    def __init__(
        self,
        command: list[str] | None = None,
        *,
        server_id: str = "default",
        timeout_s: float = 8.0,
    ) -> None:
        self.command = list(command or [])
        self.server_id = server_id
        self.timeout_s = float(timeout_s)
        self._proc: subprocess.Popen[str] | None = None
        self._rpc_id = 0
        self._lock = threading.Lock()
        self._tools: list[MCPTool] = []

    @property
    def tools(self) -> list[MCPTool]:
        return list(self._tools)

    def connect(self) -> None:
        if not self.command:
            raise RuntimeError("MCPClient: empty command")
        if self._proc is not None and self._proc.poll() is None:
            return
        self._proc = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "dana-mcp-client", "version": "0.1"},
            },
        )
        # Notify initialized (no response required by many servers).
        self._notify("notifications/initialized", {})
        self.refresh_tools()

    def pid(self) -> int | None:
        """PID of the underlying server process, or None if not connected."""
        return self._proc.pid if self._proc is not None else None

    def is_alive(self) -> bool:
        """True while the server subprocess is running (poll() is None)."""
        return self._proc is not None and self._proc.poll() is None

    def close(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
        except Exception:  # noqa: BLE001
            pass

    def refresh_tools(self) -> list[MCPTool]:
        result = self._request("tools/list", {})
        tools_raw = (result or {}).get("tools") or []
        out: list[MCPTool] = []
        for row in tools_raw:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name") or "").strip()
            if not name:
                continue
            out.append(
                MCPTool(
                    name=name,
                    description=str(row.get("description") or ""),
                    input_schema=dict(row.get("inputSchema") or {}),
                    server=self.server_id,
                )
            )
        self._tools = out
        return out

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        return self._request(
            "tools/call",
            {"name": name, "arguments": dict(arguments or {})},
        )

    def _next_id(self) -> int:
        self._rpc_id += 1
        return self._rpc_id

    def _notify(self, method: str, params: dict[str, Any]) -> None:
        self._write({"jsonrpc": "2.0", "method": method, "params": params})

    def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if self._proc is None or self._proc.stdin is None or self._proc.stdout is None:
            raise RuntimeError("MCPClient: not connected")
        req_id = self._next_id()
        payload = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params,
        }
        with self._lock:
            self._write(payload)
            deadline = time.time() + self.timeout_s
            while time.time() < deadline:
                line = self._proc.stdout.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if msg.get("id") != req_id:
                    continue
                if "error" in msg:
                    raise RuntimeError(f"MCP error: {msg['error']}")
                result = msg.get("result")
                return result if isinstance(result, dict) else {"value": result}
        raise TimeoutError(f"MCP RPC timeout method={method}")

    def _write(self, payload: dict[str, Any]) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.write(json.dumps(payload) + "\n")
        self._proc.stdin.flush()


def _parse_server_specs(raw: str) -> list[tuple[str, list[str]]]:
    """Parse ``DONNA_MCP_SERVERS`` — ``id=cmd arg1 arg2;id2=cmd2``."""
    out: list[tuple[str, list[str]]] = []
    for chunk in (raw or "").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" in chunk:
            sid, cmd = chunk.split("=", 1)
        else:
            sid, cmd = f"server{len(out)+1}", chunk
        parts = [p for p in cmd.strip().split() if p]
        if parts:
            out.append((sid.strip() or f"server{len(out)+1}", parts))
    return out


def discover_mcp_tools(*, env_var: str = "DONNA_MCP_SERVERS") -> list[MCPTool]:
    """Best-effort discovery across configured MCP servers (never raises)."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:  # noqa: BLE001
        pass
    raw = (os.environ.get(env_var) or "").strip()
    if not raw:
        return []
    tools: list[MCPTool] = []
    for sid, cmd in _parse_server_specs(raw):
        # Resolve executable on PATH when possible.
        exe = shutil.which(cmd[0]) or cmd[0]
        client = MCPClient([exe, *cmd[1:]], server_id=sid)
        try:
            client.connect()
            tools.extend(client.tools)
        except Exception as exc:  # noqa: BLE001
            print(f"[MCP] discover failed server={sid}: {exc}", flush=True)
        finally:
            client.close()
    return tools


def format_mcp_tools_block(tools: list[MCPTool] | None = None) -> str:
    """Prompt block injected into Spec Compiler / Meta-Broker / Worker context."""
    rows = tools if tools is not None else discover_mcp_tools()
    if not rows:
        return (
            "### MCP Tools\n"
            "(none discovered — set DONNA_MCP_SERVERS to enable live tool endpoints)\n"
        )
    lines = [
        "### MCP Tools (live endpoints — generate code that can call these)",
        "When useful, prefer invoking these MCP tools instead of inventing stubs. "
        "To call one, generate code that calls "
        "`dana.mcp.sandbox.mcp_tool_call(server_id, tool_name, arguments)`. "
        "NEVER instantiate `dana.mcp.client.MCPClient` directly — that bypasses "
        "the sandboxed timeout / process-tree-kill / snapshot-rollback path.",
    ]
    for t in rows:
        lines.append(t.as_prompt_line())
    return "\n".join(lines)


__all__ = (
    "MCPClient",
    "MCPTool",
    "discover_mcp_tools",
    "format_mcp_tools_block",
)
