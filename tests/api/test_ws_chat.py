"""Integration tests for the ``/ws/chat`` protocol: multi-step ReAct loop
DAG event streaming, canvas-selection context injection, camera automation,
and the HITL approval gate — all layered on dana.core.react_dispatch's
dispatch core.

The LLM is driven now, so every test mocks the one LLM call site
(``dana.core.react_dispatch.ModelProvider``) with a queue of canned
per-iteration responses — these are protocol/wiring tests, not LLM-quality
tests, and must not require a real Ollama daemon to run. ``_FakeProvider``
returns one queued response per call and falls back to a plain "Done."
final turn once the queue is exhausted, so a single queued tool call still
terminates the loop cleanly on the next iteration instead of looping until
the safety-counter cap.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from dana.api import server as server_module
from dana.platform.mock import MockControlPlane, MockFreeCADEngine
from dana.tools.schema import ToolCall


class _FakeProvider:
    def __init__(self, turns: list[list[ToolCall] | str]) -> None:
        self._turns = list(turns)

    def complete_with_tool_calls(self, messages: Any, *, tools: Any, provider: Any = None, **kwargs: Any) -> dict:
        turn = self._turns.pop(0) if self._turns else "Done."
        if isinstance(turn, str):
            return {"content": turn, "tool_calls": [], "provider": "test"}
        return {"content": "", "tool_calls": turn, "provider": "test"}


def _mock_llm(monkeypatch: pytest.MonkeyPatch, *turns: list[ToolCall] | str) -> None:
    """Queue ``turns`` as successive LLM responses for the ReAct loop.

    Each turn is either a ``list[ToolCall]`` (that iteration proposes those
    tool calls) or a plain ``str`` (that iteration's final assistant text,
    no tool calls — the loop stops there). Once the queue is exhausted,
    further calls return a plain "Done." final turn.
    """
    import dana.core.react_dispatch as react_dispatch

    fake = _FakeProvider(list(turns))
    monkeypatch.setattr(react_dispatch, "ModelProvider", lambda: fake)


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


def test_ready_message_on_connect(client: TestClient) -> None:
    with client.websocket_connect("/ws/chat") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "ready"
        assert "driver_state" in msg


def test_safe_tool_streams_dag_events_then_loops_to_final_text(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-mutating tool executes immediately, then the loop asks the LLM
    again with the tool's result appended — this second iteration is what
    actually distinguishes the multi-step loop from the old single-shot
    dispatch, so the test supplies a distinct final-text second turn rather
    than relying on the fallback "Done." to prove the loop-back happened."""
    _mock_llm(
        monkeypatch,
        [ToolCall(tool_id="system_state", arguments={})],
        "The system is healthy and ready.",
    )
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # ready
        ws.send_json({"text": "system status"})

        parse_start = _drain_until(ws, "dag_node_start")
        assert parse_start["node_id"] == "parse-0"
        parse_complete = _drain_until(ws, "dag_node_complete")
        assert parse_complete["node_id"] == "parse-0" and parse_complete["status"] == "success"

        dispatch_start = _drain_until(ws, "dag_node_start")
        assert dispatch_start["node_id"] == "dispatch-0"

        tool_call = _drain_until(ws, "tool_call")
        assert tool_call["tool_id"] == "system_state"

        dispatch_complete = _drain_until(ws, "dag_node_complete")
        assert dispatch_complete["node_id"] == "dispatch-0" and dispatch_complete["status"] == "success"

        tool_result = _drain_until(ws, "tool_result")
        assert tool_result["ok"] is True

        # Second loop iteration: the LLM sees the tool result and produces
        # its own final text — no synthesized "control_plane=..." summary.
        second_parse_start = _drain_until(ws, "dag_node_start")
        assert second_parse_start["node_id"] == "parse-1"

        assistant = _drain_until(ws, "assistant_message")
        assert assistant["content"] == "The system is healthy and ready."


def test_no_tool_call_yields_plain_fallback_message(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_llm(monkeypatch, [])
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # ready
        ws.send_json({"text": "thanks!"})

        parse_complete = _drain_until(ws, "dag_node_complete")
        # A tool-less final turn is a normal (successful) loop termination
        # now, not a parse failure — the old single-shot design had no
        # other reason to return final text besides "couldn't parse".
        assert parse_complete["status"] == "success"

        assistant = _drain_until(ws, "assistant_message")
        assert "tool call" in assistant["content"] or "action" in assistant["content"]


def test_mutating_tool_requires_hitl_approval_then_proceeds(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_llm(monkeypatch, [ToolCall(tool_id="create_freecad_box", arguments={"length": 60, "width": 40, "height": 20})])
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # ready
        ws.send_json({"text": "build a box 60x40x20"})

        approval = _drain_until(ws, "hitl_approval_required")
        request_id = approval["payload"]["request_id"]
        assert approval["payload"]["action_name"] == "create_freecad_box"
        assert "60" in approval["payload"]["description"]

        ws.send_json({"type": "hitl_response", "payload": {"request_id": request_id, "approved": True}})

        tool_call = _drain_until(ws, "tool_call")
        assert tool_call["tool_id"] == "create_freecad_box"

        tool_result = _drain_until(ws, "tool_result")
        assert tool_result["ok"] is True
        assert tool_result["mesh_url"] is not None

        # Loop continues after approval+execution — falls back to "Done."
        # since only one turn was queued, and terminates cleanly.
        assistant = _drain_until(ws, "assistant_message")
        assert assistant["content"] == "Done."


def test_mutating_tool_cancelled_when_not_approved(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_llm(monkeypatch, [ToolCall(tool_id="create_freecad_box", arguments={"length": 60, "width": 40, "height": 20})])
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # ready
        ws.send_json({"text": "build a box 60x40x20"})

        approval = _drain_until(ws, "hitl_approval_required")
        request_id = approval["payload"]["request_id"]

        ws.send_json({"type": "hitl_response", "payload": {"request_id": request_id, "approved": False}})

        assistant = _drain_until(ws, "assistant_message")
        assert "Cancelled" in assistant["content"]


def test_hitl_modify_overrides_parameters_before_dispatch(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_llm(monkeypatch, [ToolCall(tool_id="create_freecad_box", arguments={"length": 60, "width": 40, "height": 20})])
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # ready
        ws.send_json({"text": "build a box 60x40x20"})

        approval = _drain_until(ws, "hitl_approval_required")
        request_id = approval["payload"]["request_id"]

        ws.send_json(
            {
                "type": "hitl_response",
                "payload": {"request_id": request_id, "approved": True, "parameters": {"length": "99"}},
            }
        )

        tool_call = _drain_until(ws, "tool_call")
        assert tool_call["arguments"]["length"] == "99"


def test_canvas_selection_feeds_camera_target(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # ready
        ws.send_json(
            {
                "type": "canvas_selection",
                "payload": {"mesh_id": "current_mesh", "centroid": [5.0, 6.0, 7.0], "normal": [0, 1, 0]},
            }
        )
        _mock_llm(monkeypatch, [ToolCall(tool_id="manipulate_camera", arguments={"preset": "top"})])
        ws.send_json({"text": "look at it from the top"})

        tool_call = _drain_until(ws, "tool_call")
        assert tool_call["tool_id"] == "manipulate_camera"
        assert tool_call["arguments"]["target"] == [5.0, 6.0, 7.0]

        camera_animate = _drain_until(ws, "camera_animate")
        assert camera_animate["target"] == [5.0, 6.0, 7.0]


def test_canvas_selection_injects_target_position_on_mutating_tool(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # ready
        ws.send_json(
            {
                "type": "canvas_selection",
                "payload": {"mesh_id": "current_mesh", "centroid": [1.0, 2.0, 3.0], "normal": [0, 0, 1]},
            }
        )
        # The LLM proposes the box with no anchor of its own — the
        # deterministic fallback in react_dispatch._finalize_call_arguments
        # must inject it since the user said "here".
        _mock_llm(monkeypatch, [ToolCall(tool_id="create_freecad_box", arguments={})])
        ws.send_json({"text": "add a box here"})

        approval = _drain_until(ws, "hitl_approval_required")
        assert approval["payload"]["parameters"]["target_position"] == [1.0, 2.0, 3.0]

        ws.send_json(
            {
                "type": "hitl_response",
                "payload": {"request_id": approval["payload"]["request_id"], "approved": True},
            }
        )
        tool_result = _drain_until(ws, "tool_result")
        assert tool_result["ok"] is True


# --------------------------------------------------------------------------
# Multi-step ReAct loop: the actual new behavior this directive adds.
# --------------------------------------------------------------------------


def test_multi_step_loop_chains_two_tool_calls_before_final_text(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The core new capability: after a non-mutating tool executes, the
    loop asks the LLM again — and the LLM can decide to call ANOTHER tool
    (not just stop), all within one user turn."""
    _mock_llm(
        monkeypatch,
        [ToolCall(tool_id="system_state", arguments={})],
        [ToolCall(tool_id="check_plugin_registry", arguments={})],
        "Checked both system state and the plugin registry.",
    )
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # ready
        ws.send_json({"text": "check everything"})

        first_call = _drain_until(ws, "tool_call")
        assert first_call["tool_id"] == "system_state"
        _drain_until(ws, "tool_result")

        second_parse_start = _drain_until(ws, "dag_node_start")
        assert second_parse_start["node_id"] == "parse-1"

        second_call = _drain_until(ws, "tool_call")
        assert second_call["tool_id"] == "check_plugin_registry"
        _drain_until(ws, "tool_result")

        assistant = _drain_until(ws, "assistant_message")
        assert assistant["content"] == "Checked both system state and the plugin registry."


def test_multi_step_loop_suspends_for_hitl_mid_chain_and_resumes(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-mutating query (get_freecad_bounding_box-style) followed by a
    mutating create call — the loop must pause for approval on the SECOND
    tool, not just the first, and resume with the right messages/loop_count."""
    _mock_llm(
        monkeypatch,
        [ToolCall(tool_id="system_state", arguments={})],
        [ToolCall(tool_id="create_freecad_box", arguments={"length": 10, "width": 10, "height": 10})],
        "Built the box after checking system state.",
    )
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # ready
        ws.send_json({"text": "check status then build a box"})

        _drain_until(ws, "tool_result")  # system_state, no HITL

        approval = _drain_until(ws, "hitl_approval_required")
        assert approval["payload"]["action_name"] == "create_freecad_box"

        ws.send_json(
            {
                "type": "hitl_response",
                "payload": {"request_id": approval["payload"]["request_id"], "approved": True},
            }
        )

        tool_call = _drain_until(ws, "tool_call")
        assert tool_call["tool_id"] == "create_freecad_box"
        tool_result = _drain_until(ws, "tool_result")
        assert tool_result["ok"] is True

        assistant = _drain_until(ws, "assistant_message")
        assert assistant["content"] == "Built the box after checking system state."


def test_react_loop_stops_at_max_iterations(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """A model stuck proposing the same non-mutating tool call forever must
    be forcefully stopped, not left to run indefinitely."""
    endless_tool_call = [ToolCall(tool_id="system_state", arguments={})]
    _mock_llm(monkeypatch, *([endless_tool_call] * 10))  # far more turns than the cap allows
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # ready
        ws.send_json({"text": "loop forever"})

        assistant = _drain_until(ws, "assistant_message", limit=100)
        assert "maximum number of reasoning steps" in assistant["content"]


def test_second_message_while_hitl_pending_is_bounced(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_llm(monkeypatch, [ToolCall(tool_id="create_freecad_box", arguments={"length": 10, "width": 10, "height": 10})])
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # ready
        ws.send_json({"text": "build a box"})
        _drain_until(ws, "hitl_approval_required")

        ws.send_json({"text": "build another box"})
        assistant = _drain_until(ws, "assistant_message")
        assert "pending approval" in assistant["content"]
