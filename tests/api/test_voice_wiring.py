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
from dana.tools.schema import ToolCall


class _FakeProvider:
    def __init__(self, tool_calls: list[ToolCall]) -> None:
        self._tool_calls = tool_calls

    def complete_with_tool_calls(self, messages: Any, *, tools: Any, provider: Any = None, **kwargs: Any) -> dict:
        return {"content": "", "tool_calls": self._tool_calls, "provider": "test"}


class FakeVoiceService:
    instances: list["FakeVoiceService"] = []

    def __init__(self, on_state: Callable[[str, str], None] | None = None) -> None:
        self.on_state = on_state or (lambda *_a: None)
        self.started = False
        self.stopped = False
        self.listen_requests = 0
        self.cancels = 0
        self.finish_turns = 0
        FakeVoiceService.instances.append(self)

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def emit(self, state: str, transcript: str = "") -> None:
        self.on_state(state, transcript)

    # -- push-to-talk controls exercised via "voice_control"/"audio_playback_complete" --

    def request_listen(self) -> bool:
        self.listen_requests += 1
        return True

    def cancel(self) -> None:
        self.cancels += 1

    def finish_turn(self) -> None:
        self.finish_turns += 1


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


def test_finalized_transcript_dispatches_through_connected_session(
    fake_voice: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """VoiceService hands off a transcript on the "processing" transition
    (state stays "processing" but now carries the transcript) — NOT
    "speaking". "speaking" is reserved for the assistant's own TTS
    playback (see test_final_turn_triggers_tts_and_speaking_state); the two
    used to be conflated, which made the orb flash "speaking" for the
    user's own words instead of the assistant's reply.
    """
    import dana.core.react_dispatch as react_dispatch

    monkeypatch.setattr(
        react_dispatch,
        "ModelProvider",
        lambda **_kwargs: _FakeProvider([ToolCall(tool_id="system_state", arguments={})]),
    )
    with TestClient(server_module.app) as client:
        service = fake_voice.instances[0]
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # ready
            service.emit("processing", "system status")

            _drain_until(ws, "voice_state")
            tool_call = _drain_until(ws, "tool_call")
            assert tool_call["tool_id"] == "system_state"
            tool_result = _drain_until(ws, "tool_result")
            assert tool_result["ok"] is True


def test_final_turn_triggers_tts_and_speaking_state(
    fake_voice: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """A plain-text ("final") turn should synthesize a reply via the
    existing dana.audio.multi_voice_tts pipeline, broadcast the assistant
    as "speaking", and hand the client a fetchable /api/audio/ URL."""
    import dana.core.react_dispatch as react_dispatch

    monkeypatch.setattr(react_dispatch, "ModelProvider", lambda **_kwargs: _FakeProvider([]))

    fake_wav = tmp_path / "reply.wav"
    fake_wav.write_bytes(b"RIFF0000WAVEfmt ")
    # _speak_reply imports synthesize_speech lazily (see its docstring), so
    # the patch target is the source module, not dana.api.server.
    monkeypatch.setattr("dana.audio.multi_voice_tts.synthesize_speech", lambda text, **kwargs: fake_wav)

    with TestClient(server_module.app) as client:
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # ready
            ws.send_json({"text": "hello"})

            assistant_msg = _drain_until(ws, "assistant_message")
            assert assistant_msg["content"]

            speaking = _drain_until(ws, "voice_state")
            assert speaking == {"type": "voice_state", "state": "speaking", "transcript": assistant_msg["content"]}

            audio_msg = _drain_until(ws, "assistant_audio")
            assert audio_msg["audio_url"].startswith("/api/audio/")

        resp = client.get(audio_msg["audio_url"])
        assert resp.status_code == 200
        assert resp.content == fake_wav.read_bytes()


def test_voice_control_listen_and_cancel_reach_the_service(
    fake_voice: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    import dana.core.react_dispatch as react_dispatch

    monkeypatch.setattr(react_dispatch, "ModelProvider", lambda **_kwargs: _FakeProvider([]))
    fake_wav = tmp_path / "reply.wav"
    fake_wav.write_bytes(b"RIFF0000WAVEfmt ")
    monkeypatch.setattr("dana.audio.multi_voice_tts.synthesize_speech", lambda text, **kwargs: fake_wav)

    with TestClient(server_module.app) as client:
        service = fake_voice.instances[0]
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # ready
            ws.send_json({"type": "voice_control", "action": "listen"})
            ws.send_json({"type": "voice_control", "action": "cancel"})
            # A real chat message, just so the test can synchronize on a
            # reply before asserting — proof the two voice_control frames
            # above were actually processed (not just still in flight).
            ws.send_json({"text": "ping so the test can synchronize on a reply"})
            _drain_until(ws, "assistant_message")

    assert service.listen_requests == 1
    assert service.cancels == 1


def test_audio_playback_complete_finishes_turn_and_broadcasts_idle(fake_voice: Any) -> None:
    with TestClient(server_module.app) as client:
        service = fake_voice.instances[0]
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # ready
            ws.send_json({"type": "audio_playback_complete"})
            idle = _drain_until(ws, "voice_state")
            assert idle == {"type": "voice_state", "state": "idle", "transcript": ""}

    assert service.finish_turns == 1
