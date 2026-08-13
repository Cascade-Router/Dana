"""Integration tests for the ``/ws/chat`` protocol: DAG event streaming,
canvas-selection context injection, camera automation, and the HITL
approval gate — all layered on dana.core.react_dispatch's dispatch core."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from dana.api import server as server_module
from dana.platform.mock import MockControlPlane, MockFreeCADEngine


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


def test_safe_tool_streams_dag_events_without_hitl(client: TestClient) -> None:
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # ready
        ws.send_json({"text": "system status"})

        parse_start = _drain_until(ws, "dag_node_start")
        assert parse_start["node_id"] == "parse"
        parse_complete = _drain_until(ws, "dag_node_complete")
        assert parse_complete["node_id"] == "parse" and parse_complete["status"] == "success"

        dispatch_start = _drain_until(ws, "dag_node_start")
        assert dispatch_start["node_id"] == "dispatch"

        tool_call = _drain_until(ws, "tool_call")
        assert tool_call["tool_id"] == "system_state"

        dispatch_complete = _drain_until(ws, "dag_node_complete")
        assert dispatch_complete["node_id"] == "dispatch" and dispatch_complete["status"] == "success"

        tool_result = _drain_until(ws, "tool_result")
        assert tool_result["ok"] is True

        assistant = _drain_until(ws, "assistant_message")
        assert "control_plane=" in assistant["content"]


def test_mutating_tool_requires_hitl_approval_then_proceeds(client: TestClient) -> None:
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


def test_mutating_tool_cancelled_when_not_approved(client: TestClient) -> None:
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # ready
        ws.send_json({"text": "build a box 60x40x20"})

        approval = _drain_until(ws, "hitl_approval_required")
        request_id = approval["payload"]["request_id"]

        ws.send_json({"type": "hitl_response", "payload": {"request_id": request_id, "approved": False}})

        assistant = _drain_until(ws, "assistant_message")
        assert "Cancelled" in assistant["content"]


def test_hitl_modify_overrides_parameters_before_dispatch(client: TestClient) -> None:
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


def test_canvas_selection_feeds_camera_target(client: TestClient) -> None:
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # ready
        ws.send_json(
            {
                "type": "canvas_selection",
                "payload": {"mesh_id": "current_mesh", "centroid": [5.0, 6.0, 7.0], "normal": [0, 1, 0]},
            }
        )
        ws.send_json({"text": "look at it from the top"})

        tool_call = _drain_until(ws, "tool_call")
        assert tool_call["tool_id"] == "manipulate_camera"
        assert tool_call["arguments"]["target"] == [5.0, 6.0, 7.0]

        camera_animate = _drain_until(ws, "camera_animate")
        assert camera_animate["target"] == [5.0, 6.0, 7.0]


def test_canvas_selection_injects_target_position_on_mutating_tool(client: TestClient) -> None:
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # ready
        ws.send_json(
            {
                "type": "canvas_selection",
                "payload": {"mesh_id": "current_mesh", "centroid": [1.0, 2.0, 3.0], "normal": [0, 0, 1]},
            }
        )
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
