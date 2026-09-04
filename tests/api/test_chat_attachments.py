"""Tests for user image-attachment uploads over ``/ws/chat``.

Covers the wire path added for the "3-view technical drawing" / general
visual-Q&A goal: a client attaches one or more images (already resized +
base64-encoded client-side, see ``ChatPanel.tsx``) alongside a chat turn,
and dana.core.react_dispatch.build_user_message must turn that into the
standard OpenAI multimodal ``content`` array before the turn ever reaches
the LLM. Mocks the one LLM call site (``dana.core.react_dispatch.
ModelProvider``) exactly like tests/api/test_ws_chat.py — these are
protocol/wiring tests, not LLM-quality tests.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from dana.api import server as server_module
from dana.core import react_dispatch as rd
from dana.platform.mock import MockControlPlane, MockFreeCADEngine
from dana.tools.schema import ToolCall

_PNG_DATA_URI = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="


@pytest.fixture(autouse=True)
def _mock_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server_module, "get_cad_engine", lambda: MockFreeCADEngine())
    monkeypatch.setattr(server_module, "get_control_plane", lambda: MockControlPlane())


@pytest.fixture(autouse=True)
def _disable_permanent_hitl_whitelist(monkeypatch: pytest.MonkeyPatch) -> None:
    """create_freecad_box is used below as the representative "a mutating
    tool" fixture — written before dana.api.server._HITL_ALWAYS_APPROVED_TOOLS
    permanently exempted FreeCAD's geometry-CRUD tools (create_freecad_box
    included) from HITL approval. Cleared here so those tests keep
    exercising the HITL suspension path unaffected by that later, unrelated
    feature — same fix already applied in tests/api/test_ws_chat.py's and
    tests/api/test_sessions_api.py's fixtures of the same name."""
    monkeypatch.setattr(server_module, "_HITL_ALWAYS_APPROVED_TOOLS", frozenset())


@pytest.fixture
def client() -> TestClient:
    return TestClient(server_module.app)


def _capture_messages(monkeypatch: pytest.MonkeyPatch, *, tool_calls: list[ToolCall] | None = None) -> dict[str, Any]:
    """Mocks ModelProvider so the exact ``messages`` array next_react_turn
    hands to the LLM is inspectable — the only thing these tests need to
    assert on, since dispatch/DAG/HITL plumbing is already covered by
    test_ws_chat.py."""
    captured: dict[str, Any] = {}

    class _CapturingProvider:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def complete_with_tool_calls(self, messages: Any, *, tools: Any, provider: Any = None, **_kwargs: Any) -> dict:
            captured["messages"] = messages
            if tool_calls:
                return {"content": "", "tool_calls": tool_calls, "provider": "test"}
            return {"content": "done", "tool_calls": [], "provider": "test"}

    monkeypatch.setattr(rd, "ModelProvider", _CapturingProvider)
    return captured


def test_attachment_becomes_multimodal_content_array(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_messages(monkeypatch)
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # ready
        ws.send_json({"text": "what shape is this?", "attachments": [_PNG_DATA_URI]})
        _drain_until(ws, "assistant_message")

    user_message = next(m for m in captured["messages"] if m["role"] == "user")
    assert isinstance(user_message["content"], list)
    assert user_message["content"][0] == {"type": "text", "text": "what shape is this?"}
    assert user_message["content"][1] == {"type": "image_url", "image_url": {"url": _PNG_DATA_URI}}


def test_multiple_attachments_all_become_image_url_parts(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    second = _PNG_DATA_URI.replace("iVBOR", "iVBOS")
    captured = _capture_messages(monkeypatch)
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # ready
        ws.send_json({"text": "reconstruct this in 3D", "attachments": [_PNG_DATA_URI, second]})
        _drain_until(ws, "assistant_message")

    user_message = next(m for m in captured["messages"] if m["role"] == "user")
    image_urls = [p["image_url"]["url"] for p in user_message["content"] if p["type"] == "image_url"]
    assert image_urls == [_PNG_DATA_URI, second]


def test_no_attachments_keeps_plain_string_content(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Backward compatibility: a turn with no attachments must still send a
    plain string ``content``, not a single-element content array — every
    existing test/behavior that reads a user message's content as a string
    (e.g. test_react_dispatch.py) must keep working unchanged."""
    captured = _capture_messages(monkeypatch)
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # ready
        ws.send_json({"text": "hello"})
        _drain_until(ws, "assistant_message")

    user_message = next(m for m in captured["messages"] if m["role"] == "user")
    assert user_message["content"] == "hello"


def test_malformed_attachments_are_dropped_not_fatal(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-string entry or a non-image data URI must never crash the turn
    — it silently degrades to a plain-text (or filtered) message instead."""
    captured = _capture_messages(monkeypatch)
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # ready
        ws.send_json(
            {
                "text": "hello",
                "attachments": [123, None, "not-a-data-uri", "data:text/plain;base64,aGk="],
            }
        )
        _drain_until(ws, "assistant_message")

    user_message = next(m for m in captured["messages"] if m["role"] == "user")
    assert user_message["content"] == "hello"


def test_non_list_attachments_field_is_ignored_not_fatal(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_messages(monkeypatch)
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # ready
        ws.send_json({"text": "hello", "attachments": "not-a-list"})
        assistant = _drain_until(ws, "assistant_message")
        assert assistant["content"] == "done"

    user_message = next(m for m in captured["messages"] if m["role"] == "user")
    assert user_message["content"] == "hello"


def test_selection_reference_fallback_still_works_with_attachment_present(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_finalize_call_arguments``'s "this"/"here" selection fallback reads
    ``call.raw_text`` (the ORIGINAL user text) via
    ``dana.api.server._run_react_loop``'s ``extract_user_text`` call — must
    keep matching correctly even though this turn's user message content is
    now a multimodal array, not a plain string."""
    _capture_messages(monkeypatch, tool_calls=[ToolCall(tool_id="create_freecad_box", arguments={})])
    with client.websocket_connect("/ws/chat") as ws:
        ready = ws.receive_json()  # ready
        # Plan-and-Execute FSM: create_freecad_box is now schema-gated on an
        # active plan one level EARLIER than dispatch_tool_call's own
        # Gatekeeper (next_react_turn's own hard_restrict_to during the
        # PLANNING phase — see build_system_prompt/_llm_tools_schema) — a
        # session with no plan yet is never even OFFERED it, so the mocked
        # LLM's hardcoded create_freecad_box tool_call would otherwise be
        # rejected as "not offered" and this turn would end in plain text
        # instead of ever reaching HITL. Pre-open the gate for this test's
        # own session (same lightweight bypass test_plan_gatekeeper_* tests
        # use), since this test is about the selection-reference fallback,
        # not plan-gating itself.
        rd._set_has_plan(True, "test-harness plan", session_id=ready["session_id"])
        ws.send_json({"type": "update_context", "active_plugins": ["freecad"]})
        ws.send_json(
            {
                "type": "canvas_selection",
                "payload": {"mesh_id": "current_mesh", "centroid": [1.0, 2.0, 3.0], "normal": [0, 0, 1]},
            }
        )
        ws.send_json({"text": "add a box here", "attachments": [_PNG_DATA_URI]})

        approval = _drain_until(ws, "hitl_approval_required")
        assert approval["payload"]["parameters"]["target_position"] == [1.0, 2.0, 3.0]


def _drain_until(ws: Any, msg_type: str, limit: int = 20) -> dict[str, Any]:
    for _ in range(limit):
        msg = ws.receive_json()
        if msg.get("type") == msg_type:
            return msg
    raise AssertionError(f"never received a {msg_type!r} message")
