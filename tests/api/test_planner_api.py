"""Tests for the Task Planner / Executive Function feature: state
transitions in dana.plugins.planning.task_board, the create_plan/
mark_task_completed ReAct tools' wiring through dana.core.react_dispatch
(TOOL_HANDLERS/_CORE_TOOL_IDS/never-mutating), the read-only
dana.api.planner REST API, and the "## Current Active Plan" system-prompt
injection that anchors the LLM across turns.

Every test relies on tests/conftest.py's autouse `_reset_task_board_plan`
fixture to reset the module-global `_ACTIVE_PLAN` on teardown — none of
these leak state into other tests in this file or the wider suite.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from dana.api import server as server_module
from dana.platform.mock import MockControlPlane, MockFreeCADEngine
from dana.plugins.planning import task_board
from dana.tools.schema import ToolCall


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
# task_board state transitions — create_plan
# --------------------------------------------------------------------------


def test_create_plan_sets_objective_and_tasks_with_first_active() -> None:
    result = task_board.create_plan("Build a to-do app", ["Scaffold project", "Write backend", "Write frontend"])
    assert result["ok"] is True
    plan = result["plan"]
    assert plan["objective"] == "Build a to-do app"
    assert [t["status"] for t in plan["tasks"]] == ["active", "pending", "pending"]
    assert [t["id"] for t in plan["tasks"]] == [1, 2, 3]
    assert plan["current_task_id"] == 1


def test_create_plan_rejects_empty_objective() -> None:
    result = task_board.create_plan("", ["a"])
    assert result["ok"] is False
    assert "objective" in result["error"]


def test_create_plan_rejects_empty_tasks_list() -> None:
    result = task_board.create_plan("Do something", [])
    assert result["ok"] is False
    assert "tasks" in result["error"]


def test_create_plan_filters_blank_task_strings() -> None:
    result = task_board.create_plan("Objective", ["real task", "   ", ""])
    assert result["ok"] is True
    assert [t["description"] for t in result["plan"]["tasks"]] == ["real task"]


def test_create_plan_replaces_a_previous_plan_entirely() -> None:
    task_board.create_plan("First objective", ["a", "b"])
    result = task_board.create_plan("Second objective", ["x", "y", "z"])
    assert result["ok"] is True
    plan = result["plan"]
    assert plan["objective"] == "Second objective"
    assert len(plan["tasks"]) == 3
    assert plan["current_task_id"] == 1


# --------------------------------------------------------------------------
# task_board state transitions — mark_task_completed
# --------------------------------------------------------------------------


def test_mark_task_completed_without_next_clears_current_task_id() -> None:
    task_board.create_plan("Objective", ["a", "b"])
    result = task_board.mark_task_completed(1)
    assert result["ok"] is True
    plan = result["plan"]
    assert plan["tasks"][0]["status"] == "completed"
    assert plan["tasks"][1]["status"] == "pending"  # NOT auto-promoted
    assert plan["current_task_id"] is None


def test_mark_task_completed_with_next_promotes_it_to_active() -> None:
    task_board.create_plan("Objective", ["a", "b", "c"])
    result = task_board.mark_task_completed(1, next_task_id=2)
    assert result["ok"] is True
    plan = result["plan"]
    assert plan["tasks"][0]["status"] == "completed"
    assert plan["tasks"][1]["status"] == "active"
    assert plan["tasks"][2]["status"] == "pending"
    assert plan["current_task_id"] == 2


def test_mark_task_completed_unknown_task_id_is_rejected_without_mutating() -> None:
    task_board.create_plan("Objective", ["a", "b"])
    result = task_board.mark_task_completed(99)
    assert result["ok"] is False
    assert "no task with id 99" in result["error"]
    assert task_board.get_active_plan()["tasks"][0]["status"] == "active"  # unchanged


def test_mark_task_completed_unknown_next_task_id_is_rejected_atomically() -> None:
    """Validates BOTH ids before mutating anything — an invalid
    next_task_id must not leave task_id marked completed anyway."""
    task_board.create_plan("Objective", ["a", "b"])
    result = task_board.mark_task_completed(1, next_task_id=99)
    assert result["ok"] is False
    assert "no task with id 99" in result["error"]
    assert task_board.get_active_plan()["tasks"][0]["status"] == "active"  # task 1 untouched


def test_mark_task_completed_next_task_id_same_as_task_id_is_rejected() -> None:
    task_board.create_plan("Objective", ["a", "b"])
    result = task_board.mark_task_completed(1, next_task_id=1)
    assert result["ok"] is False
    assert "must be different" in result["error"]


def test_mark_task_completed_with_no_active_plan_reports_error() -> None:
    result = task_board.mark_task_completed(1)
    assert result["ok"] is False
    assert "no active plan" in result["error"]


def test_get_active_plan_returns_an_independent_copy_not_a_live_reference() -> None:
    task_board.create_plan("Objective", ["a"])
    snapshot = task_board.get_active_plan()
    snapshot["tasks"][0]["status"] = "completed"  # mutate the returned copy only
    assert task_board.get_active_plan()["tasks"][0]["status"] == "active"  # module state unaffected


# --------------------------------------------------------------------------
# GET /api/planner
# --------------------------------------------------------------------------


def test_get_planner_empty_state(client: TestClient) -> None:
    resp = client.get("/api/planner")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "plan": {"objective": "", "tasks": [], "current_task_id": None}}


def test_get_planner_reflects_a_real_created_plan(client: TestClient) -> None:
    task_board.create_plan("Ship the feature", ["Design", "Implement", "Test"])
    resp = client.get("/api/planner")
    body = resp.json()
    assert body["ok"] is True
    assert body["plan"]["objective"] == "Ship the feature"
    assert len(body["plan"]["tasks"]) == 3
    assert body["plan"]["current_task_id"] == 1


def test_get_planner_reflects_progress_after_mark_task_completed(client: TestClient) -> None:
    task_board.create_plan("Ship the feature", ["Design", "Implement"])
    task_board.mark_task_completed(1, next_task_id=2)
    resp = client.get("/api/planner")
    body = resp.json()
    assert body["plan"]["tasks"][0]["status"] == "completed"
    assert body["plan"]["tasks"][1]["status"] == "active"
    assert body["plan"]["current_task_id"] == 2


# --------------------------------------------------------------------------
# Tool execution — dispatch_tool_call / registry wiring
# --------------------------------------------------------------------------


def test_dispatch_tool_call_create_plan_end_to_end() -> None:
    import dana.core.react_dispatch as rd

    call = ToolCall(tool_id="create_plan", arguments={"objective": "Ship it", "tasks": ["a", "b"]})
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None)
    assert result.ok is True
    assert result.payload["plan"]["objective"] == "Ship it"


def test_dispatch_tool_call_create_plan_rejects_non_list_tasks() -> None:
    import dana.core.react_dispatch as rd

    call = ToolCall(tool_id="create_plan", arguments={"objective": "Ship it", "tasks": "not-a-list"})
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None)
    assert result.ok is False
    assert "tasks" in result.payload.get("raw_error", "")


def test_dispatch_tool_call_mark_task_completed_end_to_end() -> None:
    import dana.core.react_dispatch as rd

    task_board.create_plan("Ship it", ["a", "b"])
    call = ToolCall(tool_id="mark_task_completed", arguments={"task_id": 1, "next_task_id": 2})
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None)
    assert result.ok is True
    assert result.payload["plan"]["tasks"][1]["status"] == "active"


def test_dispatch_tool_call_mark_task_completed_without_next_task_id() -> None:
    import dana.core.react_dispatch as rd

    task_board.create_plan("Ship it", ["a"])
    call = ToolCall(tool_id="mark_task_completed", arguments={"task_id": 1})
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None)
    assert result.ok is True
    assert result.payload["plan"]["current_task_id"] is None


def test_dispatch_tool_call_mark_task_completed_rejects_non_integer_task_id() -> None:
    import dana.core.react_dispatch as rd

    task_board.create_plan("Ship it", ["a"])
    call = ToolCall(tool_id="mark_task_completed", arguments={"task_id": "not-a-number"})
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None)
    assert result.ok is False
    assert "task_id must be an integer" in result.payload.get("raw_error", "")


def test_create_plan_and_mark_task_completed_are_registered_core_tools() -> None:
    import dana.core.react_dispatch as rd

    for tool_id in ("create_plan", "mark_task_completed"):
        assert tool_id in rd.TOOL_HANDLERS, tool_id
        assert tool_id in rd._CORE_TOOL_IDS, tool_id


def test_create_plan_and_mark_task_completed_are_never_mutating() -> None:
    import dana.core.react_dispatch as rd

    assert rd.is_mutating_tool("create_plan") is False
    assert rd.is_mutating_tool("mark_task_completed") is False


# --------------------------------------------------------------------------
# Prompt injection — build_system_prompt's "## Current Active Plan" block
# --------------------------------------------------------------------------


def test_build_system_prompt_omits_plan_section_when_none_active() -> None:
    import dana.core.react_dispatch as rd

    prompt = rd.build_system_prompt(None, active_plugins=frozenset())
    assert "Current Active Plan" not in prompt


def test_build_system_prompt_injects_the_active_plan() -> None:
    import dana.core.react_dispatch as rd

    task_board.create_plan("Build a to-do app", ["Scaffold project", "Write backend", "Write frontend"])
    prompt = rd.build_system_prompt(None, active_plugins=frozenset())
    assert "## Current Active Plan" in prompt
    assert "Objective: Build a to-do app" in prompt
    assert "[>] 1. Scaffold project" in prompt
    assert "YOU ARE HERE" in prompt
    assert "[ ] 2. Write backend" in prompt
    assert "[ ] 3. Write frontend" in prompt


def test_build_system_prompt_reflects_progress_after_mark_task_completed() -> None:
    import dana.core.react_dispatch as rd

    task_board.create_plan("Objective", ["a", "b", "c"])
    task_board.mark_task_completed(1, next_task_id=2)
    prompt = rd.build_system_prompt(None, active_plugins=frozenset())
    assert "[x] 1. a" in prompt
    assert "[>] 2. b" in prompt
    assert "[ ] 3. c" in prompt


def test_build_system_prompt_plan_section_is_the_final_block() -> None:
    """The plan is explicitly the LAST thing in the prompt (see
    build_system_prompt's own docstring) — the freshest anchor, not
    buried above the (much larger) engineering rulebook."""
    import dana.core.react_dispatch as rd

    task_board.create_plan("Objective", ["a"])
    prompt = rd.build_system_prompt(None, active_plugins=frozenset())
    assert prompt.rstrip().endswith("once its objective changes.")


def test_ws_chat_system_prompt_actually_includes_the_active_plan(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: the plan reaches the REAL messages array a live ReAct
    turn hands to the LLM, not just build_system_prompt in isolation."""
    import dana.core.react_dispatch as rd

    task_board.create_plan("Ship the feature", ["Design", "Implement"])

    captured: dict[str, Any] = {}

    class _CapturingProvider:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def complete_with_tool_calls(self, messages: Any, *, tools: Any, provider: Any = None, **_kwargs: Any) -> dict:
            captured["messages"] = messages
            return {"content": "done", "tool_calls": [], "provider": "test"}

    monkeypatch.setattr(rd, "ModelProvider", _CapturingProvider)

    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # ready
        ws.send_json({"text": "what's next?"})
        _drain_until(ws, "assistant_message")

    system_message = next(m for m in captured["messages"] if m["role"] == "system")
    assert "## Current Active Plan" in system_message["content"]
    assert "Ship the feature" in system_message["content"]
