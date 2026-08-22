"""Tests for Local Chat Session Persistence: the /api/sessions REST CRUD
endpoints (dana.api.sessions), title auto-generation, and /ws/chat's
hydrate-on-connect + auto-save-after-every-turn wiring (dana.api.server).

Every test redirects dana.api.sessions.SESSIONS_DIR to a throwaway temp
directory (see the autouse `_sessions_dir` fixture) — none of these ever
touch the real on-disk agent_workspace. The LLM is mocked the same way
tests/api/test_ws_chat.py already does (one call site,
dana.core.react_dispatch.ModelProvider) — these are wiring/persistence
tests, not LLM-quality tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import dana.api.sessions as sessions_module
from dana.api import server as server_module
from dana.platform.mock import MockControlPlane, MockFreeCADEngine
from dana.tools.schema import ToolCall


@pytest.fixture(autouse=True)
def _sessions_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "sessions"
    monkeypatch.setattr(sessions_module, "SESSIONS_DIR", root)
    return root


@pytest.fixture(autouse=True)
def _mock_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server_module, "get_cad_engine", lambda: MockFreeCADEngine())
    monkeypatch.setattr(server_module, "get_control_plane", lambda: MockControlPlane())


@pytest.fixture
def client() -> TestClient:
    return TestClient(server_module.app)


class _FakeProvider:
    def __init__(self, turns: list[list[ToolCall] | str]) -> None:
        self._turns = list(turns)

    def complete_with_tool_calls(self, messages: Any, *, tools: Any, provider: Any = None, **kwargs: Any) -> dict:
        turn = self._turns.pop(0) if self._turns else "Done."
        if isinstance(turn, str):
            return {"content": turn, "tool_calls": [], "provider": "test"}
        return {"content": "", "tool_calls": turn, "provider": "test"}


def _mock_llm(monkeypatch: pytest.MonkeyPatch, *turns: list[ToolCall] | str) -> None:
    import dana.core.react_dispatch as react_dispatch

    fake = _FakeProvider(list(turns))
    monkeypatch.setattr(react_dispatch, "ModelProvider", lambda **_kwargs: fake)


def _drain_until(ws: Any, msg_type: str, limit: int = 20) -> dict[str, Any]:
    for _ in range(limit):
        msg = ws.receive_json()
        if msg.get("type") == msg_type:
            return msg
    raise AssertionError(f"never received a {msg_type!r} message")


# --------------------------------------------------------------------------
# dana.api.sessions — pure storage helpers (derive_title, save/load/delete)
# --------------------------------------------------------------------------


def test_derive_title_uses_first_line_collapsed_and_truncated() -> None:
    assert sessions_module.derive_title("Build a box 60x40x20") == "Build a box 60x40x20"
    assert sessions_module.derive_title("  hello   world  \nsecond line") == "hello world"
    assert sessions_module.derive_title("") == "New chat"
    assert sessions_module.derive_title("   ") == "New chat"
    long_text = "x" * 100
    title = sessions_module.derive_title(long_text)
    assert title.endswith("…")
    assert len(title) == 60


def test_save_and_load_session_round_trips(_sessions_dir: Path) -> None:
    record = sessions_module.save_session(
        "abc123",
        title="Build a box",
        created_at=None,
        messages=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
    )
    assert record["title"] == "Build a box"
    assert record["created_at"]

    loaded = sessions_module.load_session("abc123")
    assert loaded is not None
    assert loaded["title"] == "Build a box"
    assert loaded["messages"] == [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]


def test_load_session_returns_none_when_missing() -> None:
    assert sessions_module.load_session("does-not-exist") is None


def test_load_session_rejects_path_traversal_ids() -> None:
    assert sessions_module.load_session("../../etc/passwd") is None
    assert sessions_module.is_valid_session_id("../../etc/passwd") is False


def test_delete_session_is_idempotent(_sessions_dir: Path) -> None:
    sessions_module.save_session("xyz", title="T", created_at=None, messages=[])
    assert sessions_module.delete_session("xyz") is True
    assert sessions_module.delete_session("xyz") is False
    assert sessions_module.load_session("xyz") is None


def test_list_sessions_sorted_most_recent_first(_sessions_dir: Path) -> None:
    sessions_module.save_session("older", title="Older", created_at=None, messages=[])
    sessions_module.save_session("newer", title="Newer", created_at=None, messages=[])
    # Force a distinguishable ordering regardless of clock resolution.
    older_path = _sessions_dir / "older.json"
    import json as _json

    older_record = _json.loads(older_path.read_text(encoding="utf-8"))
    older_record["updated_at"] = "2020-01-01T00:00:00+00:00"
    older_path.write_text(_json.dumps(older_record), encoding="utf-8")

    listed = sessions_module.list_sessions()
    ids = [s["id"] for s in listed]
    assert ids == ["newer", "older"]
    assert set(listed[0].keys()) == {"id", "title", "updated_at"}


# --------------------------------------------------------------------------
# /api/sessions REST endpoints
# --------------------------------------------------------------------------


def test_get_sessions_empty_when_nothing_saved(client: TestClient) -> None:
    resp = client.get("/api/sessions")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "sessions": []}


def test_get_session_by_id_returns_full_record(client: TestClient, _sessions_dir: Path) -> None:
    sessions_module.save_session(
        "sess-1", title="Hello chat", created_at=None, messages=[{"role": "user", "content": "hi"}]
    )
    resp = client.get("/api/sessions/sess-1")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["session"]["title"] == "Hello chat"
    assert body["session"]["messages"] == [{"role": "user", "content": "hi"}]


def test_get_session_missing_is_404(client: TestClient) -> None:
    resp = client.get("/api/sessions/does-not-exist")
    assert resp.status_code == 404


def test_get_session_invalid_id_is_400(client: TestClient) -> None:
    resp = client.get("/api/sessions/bad$id!")
    assert resp.status_code == 400


def test_delete_session_endpoint(client: TestClient, _sessions_dir: Path) -> None:
    sessions_module.save_session("to-delete", title="T", created_at=None, messages=[])
    resp = client.delete("/api/sessions/to-delete")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "deleted": True}
    assert sessions_module.load_session("to-delete") is None

    # Idempotent — deleting again is still a clean 200, not an error.
    resp2 = client.delete("/api/sessions/to-delete")
    assert resp2.json() == {"ok": True, "deleted": False}


# --------------------------------------------------------------------------
# /ws/chat — hydrate-on-connect + auto-save-after-every-turn
# --------------------------------------------------------------------------


def test_new_connection_with_no_session_id_gets_a_fresh_uuid(client: TestClient) -> None:
    with client.websocket_connect("/ws/chat") as ws:
        ready = ws.receive_json()
        assert ready["type"] == "ready"
        assert isinstance(ready["session_id"], str) and len(ready["session_id"]) > 0


def test_completed_turn_auto_saves_with_derived_title(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_llm(monkeypatch, "Sure, here's the answer.")
    with client.websocket_connect("/ws/chat") as ws:
        ready = ws.receive_json()
        session_id = ready["session_id"]
        ws.send_json({"text": "What is a fillet?"})
        assistant = _drain_until(ws, "assistant_message")
        assert assistant["content"] == "Sure, here's the answer."

    stored = sessions_module.load_session(session_id)
    assert stored is not None
    assert stored["title"] == "What is a fillet?"
    assert stored["messages"] == [
        {"role": "user", "content": "What is a fillet?"},
        {"role": "assistant", "content": "Sure, here's the answer."},
    ]


def test_reconnecting_with_session_id_hydrates_prior_history_and_appends(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_llm(monkeypatch, "First reply.")
    with client.websocket_connect("/ws/chat") as ws:
        ready = ws.receive_json()
        session_id = ready["session_id"]
        ws.send_json({"text": "First message"})
        _drain_until(ws, "assistant_message")

    _mock_llm(monkeypatch, "Second reply.")
    with client.websocket_connect(f"/ws/chat?session_id={session_id}") as ws:
        ready2 = ws.receive_json()
        assert ready2["session_id"] == session_id
        ws.send_json({"text": "Second message"})
        _drain_until(ws, "assistant_message")

    stored = sessions_module.load_session(session_id)
    assert stored is not None
    # Both exchanges present — the second connection's save appended onto
    # the hydrated history rather than starting over from an empty list.
    assert stored["messages"] == [
        {"role": "user", "content": "First message"},
        {"role": "assistant", "content": "First reply."},
        {"role": "user", "content": "Second message"},
        {"role": "assistant", "content": "Second reply."},
    ]
    # Title stays whatever the FIRST turn derived — a later turn in the
    # same session must never overwrite it.
    assert stored["title"] == "First message"


def test_invalid_session_id_query_param_is_ignored_and_gets_a_fresh_id(client: TestClient) -> None:
    with client.websocket_connect("/ws/chat?session_id=../../etc/passwd") as ws:
        ready = ws.receive_json()
        assert ready["session_id"] != "../../etc/passwd"
        assert sessions_module.is_valid_session_id(ready["session_id"])


def test_hitl_cancelled_turn_is_persisted(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_llm(monkeypatch, [ToolCall(tool_id="create_freecad_box", arguments={"length": 10, "width": 10, "height": 10})])
    with client.websocket_connect("/ws/chat") as ws:
        ready = ws.receive_json()
        session_id = ready["session_id"]
        ws.send_json({"type": "update_context", "active_plugins": ["freecad"]})
        ws.send_json({"text": "build a box"})

        approval = _drain_until(ws, "hitl_approval_required")
        ws.send_json(
            {"type": "hitl_response", "payload": {"request_id": approval["payload"]["request_id"], "approved": False}}
        )
        assistant = _drain_until(ws, "assistant_message")
        assert assistant["content"] == "Cancelled — no changes were made."

    stored = sessions_module.load_session(session_id)
    assert stored is not None
    assert stored["messages"] == [
        {"role": "user", "content": "build a box"},
        {"role": "assistant", "content": "Cancelled — no changes were made."},
    ]


def test_aborted_turn_is_persisted(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_llm(monkeypatch, [ToolCall(tool_id="create_freecad_box", arguments={"length": 10, "width": 10, "height": 10})])
    with client.websocket_connect("/ws/chat") as ws:
        ready = ws.receive_json()
        session_id = ready["session_id"]
        ws.send_json({"type": "update_context", "active_plugins": ["freecad"]})
        ws.send_json({"text": "build a box then abort"})

        _drain_until(ws, "hitl_approval_required")
        ws.send_json({"type": "abort_turn"})
        assistant = _drain_until(ws, "assistant_message")
        assert assistant["content"] == "Generation aborted by user."

    stored = sessions_module.load_session(session_id)
    assert stored is not None
    assert stored["messages"][-1] == {"role": "assistant", "content": "Generation aborted by user."}


def test_bounced_pending_action_message_is_not_persisted(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """The "please wait for the pending action" bounce (sent when a second
    message arrives while a HITL approval is still pending) must NOT be
    treated as a completed turn — no user turn was actually accepted."""
    _mock_llm(monkeypatch, [ToolCall(tool_id="create_freecad_box", arguments={"length": 10, "width": 10, "height": 10})])
    with client.websocket_connect("/ws/chat") as ws:
        ready = ws.receive_json()
        session_id = ready["session_id"]
        ws.send_json({"type": "update_context", "active_plugins": ["freecad"]})
        ws.send_json({"text": "build a box"})
        _drain_until(ws, "hitl_approval_required")

        ws.send_json({"text": "build another box"})
        bounce = _drain_until(ws, "assistant_message")
        assert "pending action" in bounce["content"]

    # Nothing was ever saved — the turn is still pending, never completed.
    assert sessions_module.load_session(session_id) is None
