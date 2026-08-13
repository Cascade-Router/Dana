"""Unit tests for the headless VoiceService state machine.

Every hardware/model dependency is monkeypatched — these tests exercise the
idle/listening/processing/speaking transitions and the graceful no-hardware
fallback, not real microphone or Whisper behavior.
"""

from __future__ import annotations

import time

import pytest

from dana.services.voice_service import VoiceService


def _wait_for(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_no_hardware_stays_idle_without_crashing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(VoiceService, "_probe_hardware", staticmethod(lambda: False))
    states: list[tuple[str, str]] = []
    service = VoiceService(on_state=lambda s, t: states.append((s, t)))

    assert service.hardware_available is False
    service.start()
    time.sleep(0.1)
    assert service.state == "idle"
    service.stop()
    assert all(s == "idle" for s, _ in states)


def test_listening_cycle_with_no_speech_returns_to_idle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(VoiceService, "_probe_hardware", staticmethod(lambda: True))
    monkeypatch.setattr(VoiceService, "_start_whisper_background_load", staticmethod(lambda: None))
    monkeypatch.setattr(VoiceService, "_capture_utterance", lambda self: None)

    states: list[str] = []
    service = VoiceService(on_state=lambda s, _t: states.append(s))
    service.start()

    assert _wait_for(lambda: "listening" in states)
    service.stop()

    assert states[0] == "listening"
    assert "processing" not in states
    assert "speaking" not in states


def test_finalized_transcript_reaches_speaking_state_then_idles(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(VoiceService, "_probe_hardware", staticmethod(lambda: True))
    monkeypatch.setattr(VoiceService, "_start_whisper_background_load", staticmethod(lambda: None))

    call_count = {"n": 0}

    def fake_capture(self: VoiceService):
        call_count["n"] += 1
        return object() if call_count["n"] == 1 else None

    monkeypatch.setattr(VoiceService, "_capture_utterance", fake_capture)
    monkeypatch.setattr(VoiceService, "_transcribe", staticmethod(lambda audio: "build a box 60x40x20"))

    events: list[tuple[str, str]] = []
    service = VoiceService(on_state=lambda s, t: events.append((s, t)))
    service.start()

    assert _wait_for(lambda: any(s == "speaking" for s, _ in events))
    service.stop()

    speaking = [t for s, t in events if s == "speaking"]
    assert speaking == ["build a box 60x40x20"]
    assert events[-1][0] == "idle"


def test_broken_listener_does_not_kill_worker_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(VoiceService, "_probe_hardware", staticmethod(lambda: False))

    def bad_listener(_s: str, _t: str) -> None:
        raise RuntimeError("boom")

    service = VoiceService(on_state=bad_listener)
    service.start()
    time.sleep(0.1)
    assert service.state == "idle"
    service.stop()


def test_stop_is_idempotent_and_resets_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(VoiceService, "_probe_hardware", staticmethod(lambda: False))
    service = VoiceService()
    service.start()
    service.stop()
    service.stop()
    assert service.state == "idle"
