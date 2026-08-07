"""Sandboxed MCP tool execution — timeout/kill, snapshot/rollback, commit."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from dana.mcp import sandbox


class _FakeClient:
    """Duck-typed MCPClient double — exposes only what sandbox.py calls."""

    def __init__(self, *, on_call=None, hang_s: float | None = None, pid: int = 4242):
        self.server_id = "fake-server"
        self._on_call = on_call
        self._hang_s = hang_s
        self._pid = pid

    def call_tool(self, name, arguments):
        if self._hang_s is not None:
            time.sleep(self._hang_s)
        if self._on_call is not None:
            return self._on_call(name, arguments)
        return {"ok": True, "name": name, "arguments": arguments}

    def pid(self):
        return self._pid

    def is_alive(self):
        return True


@pytest.fixture(autouse=True)
def _clean_tracker():
    """Runtime harness tracker is module-global — never leak state across tests."""
    yield
    from dana.graph.runtime_harness import _EPIC_FILE_TRACKER

    _EPIC_FILE_TRACKER.clear()


def test_successful_call_commits_no_rollback(tmp_path: Path) -> None:
    client = _FakeClient()
    result = sandbox.call_mcp_tool_sandboxed(
        client, "do_thing", {"x": 1}, workspace_path=str(tmp_path), run_key="t1"
    )
    assert result["success"] is True
    assert result["rolled_back"] is False
    assert result["result"] == {"ok": True, "name": "do_thing", "arguments": {"x": 1}}
    from dana.graph.runtime_harness import _EPIC_FILE_TRACKER

    assert "t1" not in _EPIC_FILE_TRACKER  # committed, not left dangling


def test_failed_call_rolls_back_touched_file(tmp_path: Path) -> None:
    target = tmp_path / "victim.py"
    target.write_text("ORIGINAL\n", encoding="utf-8")

    def _boom(name, arguments):
        # Simulate the MCP server mutating a file, then the RPC erroring out.
        target.write_text("CORRUPTED_BY_ROGUE_MCP_SERVER\n", encoding="utf-8")
        raise RuntimeError("server-side failure mid-write")

    client = _FakeClient(on_call=_boom)
    result = sandbox.call_mcp_tool_sandboxed(
        client,
        "write_file",
        {"path": "victim.py"},
        workspace_path=str(tmp_path),
        run_key="t2",
    )
    assert result["success"] is False
    assert result["rolled_back"] is True
    assert "server-side failure" in result["error"]
    assert target.read_text(encoding="utf-8") == "ORIGINAL\n", (
        "rollback must restore the file the rogue MCP call corrupted"
    )


def test_hung_server_times_out_kills_tree_and_rolls_back(tmp_path: Path) -> None:
    target = tmp_path / "victim.py"
    target.write_text("ORIGINAL\n", encoding="utf-8")

    client = _FakeClient(hang_s=5.0, pid=9999)
    sandbox._clients["fake-server"] = client  # simulate a cached live connection

    with patch.object(sandbox, "kill_process_tree") as mock_kill:
        result = sandbox.call_mcp_tool_sandboxed(
            client,
            "write_file",
            {"path": "victim.py"},
            workspace_path=str(tmp_path),
            run_key="t3",
            timeout_s=0.2,
        )

    assert result["success"] is False
    assert result["timed_out"] is True
    assert result["rolled_back"] is True
    mock_kill.assert_called_once_with(9999)
    # Timeout must drop the cached client so the next call reconnects fresh.
    assert "fake-server" not in sandbox._clients


def test_validate_command_failure_rolls_back(tmp_path: Path) -> None:
    target = tmp_path / "victim.py"
    target.write_text("ORIGINAL\n", encoding="utf-8")

    def _mutate(name, arguments):
        target.write_text("BAD_STATE\n", encoding="utf-8")
        return {"ok": True}

    client = _FakeClient(on_call=_mutate)
    result = sandbox.call_mcp_tool_sandboxed(
        client,
        "write_file",
        {"path": "victim.py"},
        workspace_path=str(tmp_path),
        run_key="t4",
        validate_command="python -c \"import sys; sys.exit(1)\"",
    )
    assert result["success"] is False
    assert result["rolled_back"] is True
    assert target.read_text(encoding="utf-8") == "ORIGINAL\n"


def test_mcp_tool_call_string_formatting_success() -> None:
    with patch.object(sandbox, "get_mcp_client", return_value=_FakeClient()):
        text = sandbox.mcp_tool_call("fake-server", "do_thing", {"x": 1})
    assert text.startswith("exit_code=0")
    assert "do_thing" in text


def test_mcp_tool_call_string_formatting_timeout() -> None:
    client = _FakeClient(hang_s=5.0, pid=1234)
    with patch.object(sandbox, "get_mcp_client", return_value=client), patch.object(
        sandbox, "kill_process_tree"
    ):
        text = sandbox.mcp_tool_call("fake-server", "do_thing", {}, timeout_s=0.2)
    assert text.startswith("WARNING:")
    assert "timed out" in text


def test_mcp_tool_call_string_formatting_error() -> None:
    def _boom(name, arguments):
        raise RuntimeError("nope")

    client = _FakeClient(on_call=_boom)
    with patch.object(sandbox, "get_mcp_client", return_value=client):
        text = sandbox.mcp_tool_call("fake-server", "do_thing", {})
    assert text.startswith("--- EXECUTION ERROR ---")
    assert "nope" in text


def test_get_mcp_client_reuses_alive_connection() -> None:
    client = _FakeClient()
    sandbox._clients["fake-server"] = client
    try:
        assert sandbox.get_mcp_client("fake-server") is client
    finally:
        sandbox._clients.pop("fake-server", None)


def test_prompt_block_points_at_sandboxed_entrypoint() -> None:
    from dana.mcp.client import MCPTool, format_mcp_tools_block

    tools = [MCPTool(name="fs.read", description="read a file", server="fake-server")]
    block = format_mcp_tools_block(tools)
    assert "dana.mcp.sandbox.mcp_tool_call" in block
    assert "server='fake-server'" in block or 'server="fake-server"' in block
