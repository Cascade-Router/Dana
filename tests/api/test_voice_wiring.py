"""Tests that dana.api.server wires VoiceService's on_state callback to a
`voice_state` broadcast and, on a finalized transcript, replays it through
the same dispatch path a client's own chat messages use.

A fake VoiceService stands in for the real one so these tests never touch
real microphone hardware or load Whisper — only the wiring is under test.
"""

from __future__ import annotations

from typing import Any, Callable

import pytest
from fastapi.testclient import TestClient

from dana.api import server as server_module
from dana.platform.mock import MockControlPlane, MockFreeCADEngine


class FakeVoiceService:
    instances: list["FakeVoiceService"] = []

    def __init__(self, on_state: Callable[[str, str], None] | None = None) -> None:
        self.on_state = on_state or (lambda *_a: None)
        self.started = False
        self.stopped = False
        FakeVoiceService.instances.append(self)

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def emit(self, state: str, transcript: str = "") -> None:
        self.on_state(state, transcript)


@pytest.fixture(autouse=True)
def _mock_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server_module, "get_cad_engine", lambda: MockFreeCADEngine())
    monkeypatch.setattr(server_module, "get_control_plane", lambda: MockControlPlane())


@pytest.fixture
def fake_voice(monkeypatch: pytest.MonkeyPatch) -> Any:
    FakeVoiceService.instances = []
    monkeypatch.setattr(server_module, "VoiceService", FakeVoiceService)
    yield FakeVoiceService


def _drain_until(ws: Any, msg_type: str, limit: int = 20) -> dict[str, Any]:
    for _ in range(limit):
        msg = ws.receive_json()
        if msg.get("type") == msg_type:
            return msg
    raise AssertionError(f"never received a {msg_type!r} message")


def test_lifespan_starts_and_stops_the_voice_service(fake_voice: Any) -> None:
    with TestClient(server_module.app):
        assert len(fake_voice.instances) == 1
        assert fake_voice.instances[0].started is True
    assert fake_voice.instances[0].stopped is True


def test_voice_state_event_broadcasts_to_connected_client(fake_voice: Any) -> None:
    with TestClient(server_module.app) as client:
        service = fake_voice.instances[0]
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # ready
            service.emit("listening", "")
            msg = _drain_until(ws, "voice_state")
            assert msg == {"type": "voice_state", "state": "listening", "transcript": ""}


def test_finalized_transcript_dispatches_through_connected_session(fake_voice: Any) -> None:
    with TestClient(server_module.app) as client:
        service = fake_voice.instances[0]
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # ready
            service.emit("speaking", "system status")

            _drain_until(ws, "voice_state")
            tool_call = _drain_until(ws, "tool_call")
            assert tool_call["tool_id"] == "system_state"
            tool_result = _drain_until(ws, "tool_result")
            assert tool_result["ok"] is True
