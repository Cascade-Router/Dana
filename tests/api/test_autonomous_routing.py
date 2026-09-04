"""Tests for autonomous semantic routing: the `load_capability` tool lets
the agent unlock a capability domain (here, the real "os_tools" filesystem
+ script-execution domain — see tests/plugins/os/ for that domain's own
dedicated coverage) mid-loop, merging into session["agent_loaded_capabilities"]
alongside the frontend's own "update_context"-driven session["active_plugins"]
— see dana.api.server._effective_capabilities and dana.core.react_dispatch's
_CAPABILITY_TOOL_IDS / _tool_ids_for_plugins / _llm_tools_schema.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

import dana.core.react_dispatch as react_dispatch
from dana.api import server as server_module
from dana.platform.mock import MockControlPlane, MockFreeCADEngine
from dana.tools.schema import ToolCall


class _FakeProvider:
    """Queues successive per-turn responses — kept local rather than
    imported from test_ws_chat.py, matching this suite's convention of each
    test module owning its own small fixture surface."""

    def __init__(self, turns: list[list[ToolCall] | str]) -> None:
        self._turns = list(turns)
        self.calls: list[dict[str, Any]] = []

    def complete_with_tool_calls(self, messages: Any, *, tools: Any, provider: Any = None, **kwargs: Any) -> dict:
        self.calls.append({"tools": tools})
        turn = self._turns.pop(0) if self._turns else "Done."
        if isinstance(turn, str):
            return {"content": turn, "tool_calls": [], "provider": "test"}
        return {"content": "", "tool_calls": turn, "provider": "test"}


def _mock_llm(monkeypatch: pytest.MonkeyPatch, *turns: list[ToolCall] | str) -> _FakeProvider:
    fake = _FakeProvider(list(turns))
    monkeypatch.setattr(react_dispatch, "ModelProvider", lambda **_kwargs: fake)
    return fake


@pytest.fixture(autouse=True)
def _mock_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server_module, "get_cad_engine", lambda: MockFreeCADEngine())
    monkeypatch.setattr(server_module, "get_control_plane", lambda: MockControlPlane())


@pytest.fixture
def client() -> TestClient:
    return TestClient(server_module.app)


def _drain_until(ws: Any, msg_type: str, limit: int = 20) -> dict[str, Any]:
    for _ in range(limit):
        msg = ws.receive_json()
        if msg.get("type") == msg_type:
            return msg
    raise AssertionError(f"never received a {msg_type!r} message")


# --------------------------------------------------------------------------
# Unit-level: the tool handler and capability-domain registry in isolation.
# --------------------------------------------------------------------------


def test_load_capability_unlocks_os_tools() -> None:
    """os_tools is now the real filesystem domain (dana.plugins.os.
    file_system — see tests/plugins/os/test_file_system.py for its own
    dedicated coverage), with 3 real tools, not the original single mock."""
    result = react_dispatch._tool_load_capability({"domain": "os_tools"}, None, None)
    assert result["ok"] is True
    assert result["domain"] == "os_tools"
    assert result["unlocked_tools"] == [
        "analyze_desktop_screen",
        "edit_file",
        "execute_terminal_command",
        "list_background_services",
        "list_directory",
        "read_file",
        "run_python_script",
        "search_files",
        "start_background_service",
        "stop_background_service",
        "write_file",
    ]
    # Message is deliberately concise now (no per-tool-name enumeration —
    # see _tool_load_capability's own comment); unlocked_tools above is
    # already the authoritative, programmatically-checked list.
    assert str(len(result["unlocked_tools"])) in result["message"]


def test_load_capability_unknown_domain_reports_error_not_a_crash() -> None:
    result = react_dispatch._tool_load_capability({"domain": "nonexistent"}, None, None)
    assert result["ok"] is False
    assert "nonexistent" in result["error"]


def test_load_capability_always_in_core_tools() -> None:
    schema = react_dispatch._llm_tools_schema(frozenset())
    names = {t["function"]["name"] for t in schema}
    assert "load_capability" in names
    assert "list_directory" not in names  # os_tools not active yet


def test_list_directory_offered_once_os_tools_capability_active() -> None:
    schema = react_dispatch._llm_tools_schema(frozenset({"os_tools"}))
    names = {t["function"]["name"] for t in schema}
    assert "list_directory" in names


def test_tool_ids_for_plugins_merges_frontend_and_agent_style_sets_identically() -> None:
    """_tool_ids_for_plugins doesn't distinguish WHERE a capability name
    came from — dana.api.server._effective_capabilities is what unions
    active_plugins and agent_loaded_capabilities before calling in; this
    just confirms the merged frozenset behaves exactly like any other."""
    merged = frozenset({"freecad"}) | frozenset({"os_tools"})
    ids = react_dispatch._tool_ids_for_plugins(merged)
    assert "list_directory" in ids
    assert "create_freecad_box" in ids


# --------------------------------------------------------------------------
# End-to-end: dispatching load_capability over a real /ws/chat session
# actually unlocks list_directory for the IMMEDIATE next turn.
# --------------------------------------------------------------------------


def test_load_capability_exposes_list_directory_on_the_next_turn(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _mock_llm(
        monkeypatch,
        [ToolCall(tool_id="load_capability", arguments={"domain": "os_tools"})],
        [ToolCall(tool_id="list_directory", arguments={"path": "."})],
        "Here's what's in that directory.",
    )
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # ready
        ws.send_json({"text": "what's in the current directory?"})

        load_call = _drain_until(ws, "tool_dispatch_start")
        assert load_call["tool_name"] == "load_capability"
        load_result = _drain_until(ws, "tool_dispatch_end")
        assert load_result["status"] == "success"
        assert load_result["output"]["unlocked_tools"] == [
            "analyze_desktop_screen",
            "edit_file",
            "execute_terminal_command",
            "list_background_services",
            "list_directory",
            "read_file",
            "run_python_script",
            "search_files",
            "start_background_service",
            "stop_background_service",
            "write_file",
        ]

        list_call = _drain_until(ws, "tool_dispatch_start")
        assert list_call["tool_name"] == "list_directory"
        list_result = _drain_until(ws, "tool_dispatch_end")
        assert list_result["status"] == "success"

        assistant = _drain_until(ws, "assistant_message")
        assert assistant["content"] == "Here's what's in that directory."

    # Checked only after the whole websocket session has closed (TestClient
    # joins the server-side task on exit), so there's no race between the
    # server thread still appending to fake.calls and this read.
    # fake.calls[0] is the very first LLM turn — proposes load_capability,
    # before anything is unlocked. fake.calls[1] is the IMMEDIATE next turn
    # — the one that actually proposed list_directory — proving the tool
    # was unlocked in time for it, not one turn later.
    assert "list_directory" not in {t["function"]["name"] for t in fake.calls[0]["tools"]}
    assert "list_directory" in {t["function"]["name"] for t in fake.calls[1]["tools"]}


def test_deactivating_frontend_plugin_does_not_clear_agent_loaded_capability(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Constraint: "update_context" (the frontend's plugin-tab state) must
    only ever touch session["active_plugins"] — a capability the agent
    loaded on its own initiative must survive the frontend deactivating
    (or never having activated) any plugin."""
    fake = _mock_llm(
        monkeypatch,
        [ToolCall(tool_id="load_capability", arguments={"domain": "os_tools"})],
        "Loaded it.",
        [ToolCall(tool_id="list_directory", arguments={"path": "."})],
        "Still works.",
    )
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # ready
        ws.send_json({"type": "update_context", "active_plugins": ["freecad"]})
        ws.send_json({"text": "load os tools"})
        _drain_until(ws, "tool_dispatch_end")  # load_capability's own dispatch result
        _drain_until(ws, "assistant_message")

        # Frontend closes the CAD tab entirely.
        ws.send_json({"type": "update_context", "active_plugins": []})

        ws.send_json({"text": "now list the directory"})
        list_call = _drain_until(ws, "tool_dispatch_start")
        assert list_call["tool_name"] == "list_directory"
        _drain_until(ws, "assistant_message")

    last_tool_names = {t["function"]["name"] for t in fake.calls[-1]["tools"]}
    assert "list_directory" in last_tool_names  # agent_loaded_capabilities survived the deactivation
    assert "create_freecad_box" not in last_tool_names  # active_plugins really was cleared, though
