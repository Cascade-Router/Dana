"""Targeted tests for Multi-Modal Verification (Frontier 2): the pure
``build_visual_inspection_result`` helper in ``dana.core.react_dispatch``,
its defensive direct-dispatch behavior for ``take_canvas_screenshot``, and
the full ``visual_capture_request``/``visual_capture_response`` WebSocket
round-trip in ``dana.api.server`` — mirrors tests/api/test_ws_chat.py's
conventions for the structurally-identical HITL round-trip. Not a full
test suite by design; no live VLM/Ollama or FreeCADCmd required.
"""

from __future__ import annotations

import base64
from typing import Any

import pytest
from fastapi.testclient import TestClient

import dana.core.react_dispatch as react_dispatch
from dana.api import server as server_module
from dana.platform.mock import MockControlPlane, MockFreeCADEngine
from dana.tools.schema import ToolCall

# A real (tiny but valid) PNG, long enough that cad_vision's own
# _resolve_to_base64 heuristic (len > 256) treats it as inline image data
# rather than a file path.
_FAKE_PNG_B64 = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 300).decode("ascii")


def _fake_analyze_cad_blueprint_ok(_image_b64: str, **_kwargs: Any) -> str:
    return '{"ok": true, "summary": "A box sits on a plate.", "entities": []}'


def _fake_analyze_cad_blueprint_unavailable(_image_b64: str, **_kwargs: Any) -> str:
    return '{"ok": false, "error": "all VLM providers failed"}'


# --------------------------------------------------------------------------
# build_visual_inspection_result — pure core, VLM call mocked for speed
# --------------------------------------------------------------------------


def test_visual_inspection_result_with_image_and_working_vlm(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr(react_dispatch, "analyze_cad_blueprint", _fake_analyze_cad_blueprint_ok)
    monkeypatch.setattr(react_dispatch, "CAPTURES_DIR", tmp_path)
    monkeypatch.setattr(react_dispatch, "_LAST_CANVAS_SCREENSHOT_PATH", tmp_path / "last_canvas_screenshot.png")

    result = react_dispatch.build_visual_inspection_result(_FAKE_PNG_B64)
    assert result["ok"] is True
    assert result["summary"] == "A box sits on a plate."
    assert result["image_url"] == "/api/vision/last_canvas_screenshot.png"
    assert (tmp_path / "last_canvas_screenshot.png").is_file()


def test_visual_inspection_result_vlm_unavailable_is_still_ok(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """A captured screenshot with no working VLM is a partial success, not
    a tool failure — the LLM still gets confirmation + an image URL."""
    monkeypatch.setattr(react_dispatch, "analyze_cad_blueprint", _fake_analyze_cad_blueprint_unavailable)
    monkeypatch.setattr(react_dispatch, "CAPTURES_DIR", tmp_path)
    monkeypatch.setattr(react_dispatch, "_LAST_CANVAS_SCREENSHOT_PATH", tmp_path / "last_canvas_screenshot.png")

    result = react_dispatch.build_visual_inspection_result(_FAKE_PNG_B64)
    assert result["ok"] is True
    assert result["summary"] is None
    assert "vision analysis unavailable" in result["note"]


def test_visual_inspection_result_no_image_reports_frontend_error() -> None:
    result = react_dispatch.build_visual_inspection_result(None, error="user cancelled capture")
    assert result == {"ok": False, "error": "user cancelled capture"}


def test_visual_inspection_result_invalid_base64_is_rejected() -> None:
    result = react_dispatch.build_visual_inspection_result("not-valid-base64!!!")
    assert result["ok"] is False
    assert "invalid base64" in result["error"]


# --------------------------------------------------------------------------
# take_canvas_screenshot's defensive direct-dispatch behavior
# --------------------------------------------------------------------------


def test_take_canvas_screenshot_is_registered_but_not_directly_dispatchable() -> None:
    """Proves the tool_id resolves through dispatch_tool_call/TOOL_HANDLERS
    (schema-valid, callable at all) while documenting that its real
    behavior only happens via dana.api.server's websocket interception."""
    result = react_dispatch.dispatch_tool_call(
        ToolCall(tool_id="take_canvas_screenshot", arguments={}), engine=None, control_plane=None
    )
    assert result.ok is False
    assert "WebSocket round-trip" in result.message
    assert react_dispatch.is_visual_inspection_tool("take_canvas_screenshot") is True
    assert react_dispatch.is_mutating_tool("take_canvas_screenshot") is False


# --------------------------------------------------------------------------
# Full WS round-trip — mirrors test_ws_chat.py's HITL round-trip pattern
# --------------------------------------------------------------------------


class _FakeProvider:
    def __init__(self, turns: list[list[ToolCall] | str]) -> None:
        self._turns = list(turns)

    def complete_with_tool_calls(self, messages: Any, *, tools: Any, provider: Any = None, **kwargs: Any) -> dict:
        turn = self._turns.pop(0) if self._turns else "Done."
        if isinstance(turn, str):
            return {"content": turn, "tool_calls": [], "provider": "test"}
        return {"content": "", "tool_calls": turn, "provider": "test"}


def _mock_llm(monkeypatch: pytest.MonkeyPatch, *turns: list[ToolCall] | str) -> None:
    fake = _FakeProvider(list(turns))
    monkeypatch.setattr(react_dispatch, "ModelProvider", lambda **_kwargs: fake)


@pytest.fixture(autouse=True)
def _mock_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server_module, "get_cad_engine", lambda: MockFreeCADEngine())
    monkeypatch.setattr(server_module, "get_control_plane", lambda: MockControlPlane())


@pytest.fixture(autouse=True)
def _mock_vision(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr(react_dispatch, "analyze_cad_blueprint", _fake_analyze_cad_blueprint_ok)
    monkeypatch.setattr(react_dispatch, "CAPTURES_DIR", tmp_path)
    monkeypatch.setattr(react_dispatch, "_LAST_CANVAS_SCREENSHOT_PATH", tmp_path / "last_canvas_screenshot.png")


@pytest.fixture
def client() -> TestClient:
    return TestClient(server_module.app)


def _drain_until(ws: Any, msg_type: str, limit: int = 20) -> dict[str, Any]:
    for _ in range(limit):
        msg = ws.receive_json()
        if msg.get("type") == msg_type:
            return msg
    raise AssertionError(f"never received a {msg_type!r} message")


def test_take_canvas_screenshot_suspends_for_visual_capture_then_resumes(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_llm(
        monkeypatch,
        [ToolCall(tool_id="take_canvas_screenshot", arguments={})],
        "The assembly looks correct.",
    )
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # ready
        ws.send_json({"text": "does this look right?"})

        capture_request = _drain_until(ws, "visual_capture_request")
        request_id = capture_request["payload"]["request_id"]

        ws.send_json(
            {
                "type": "visual_capture_response",
                "payload": {"request_id": request_id, "image_b64": _FAKE_PNG_B64},
            }
        )

        tool_result = _drain_until(ws, "tool_result")
        assert tool_result["tool_id"] == "take_canvas_screenshot"
        assert tool_result["ok"] is True
        assert "image_b64" not in tool_result["payload"]  # never re-sent over the wire

        assistant = _drain_until(ws, "assistant_message")
        assert assistant["content"] == "The assembly looks correct."


def test_take_canvas_screenshot_never_hits_hitl_approval(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """A read-only inspection tool must use its own visual_state suspend
    path, never the mutating-tool HITL gate."""
    _mock_llm(monkeypatch, [ToolCall(tool_id="take_canvas_screenshot", arguments={})])
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # ready
        ws.send_json({"text": "check the viewport"})

        msg = _drain_until(ws, "visual_capture_request")
        assert msg["type"] == "visual_capture_request"


def test_second_message_while_visual_capture_pending_is_bounced(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_llm(monkeypatch, [ToolCall(tool_id="take_canvas_screenshot", arguments={})])
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # ready
        ws.send_json({"text": "check the viewport"})
        _drain_until(ws, "visual_capture_request")

        ws.send_json({"text": "another message"})
        assistant = _drain_until(ws, "assistant_message")
        assert "pending action" in assistant["content"]
