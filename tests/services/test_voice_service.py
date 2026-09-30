"""Unit tests for the headless VoiceService state machine.

Every hardware/model dependency is monkeypatched — these tests exercise the
idle/listening/processing/speaking transitions and the graceful no-hardware
fallback, not real microphone or Whisper behavior.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
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


def test_listening_only_starts_after_request_listen(monkeypatch: pytest.MonkeyPatch) -> None:
    """Push-to-talk: the worker parks (no "listening") until request_listen()
    is called — it no longer free-runs a hot mic the moment start() fires."""
    monkeypatch.setattr(VoiceService, "_probe_hardware", staticmethod(lambda: True))
    monkeypatch.setattr(VoiceService, "_start_whisper_background_load", staticmethod(lambda: None))
    monkeypatch.setattr(VoiceService, "_capture_utterance", lambda self: None)

    states: list[str] = []
    service = VoiceService(on_state=lambda s, _t: states.append(s))
    service.start()
    time.sleep(0.15)
    assert "listening" not in states  # parked, not auto-listening

    assert service.request_listen() is True
    assert _wait_for(lambda: "listening" in states)
    service.stop()

    assert "processing" not in states
    assert "speaking" not in states


def test_finalized_transcript_hands_off_on_processing_and_waits_for_finish_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A finalized transcript is handed off on the "processing" state (not
    "speaking" — that's reserved for the assistant's own TTS playback, see
    dana.api.server._speak_reply) and the service stays there, re-armable
    only once finish_turn() is called — it does not auto-idle on its own."""
    monkeypatch.setattr(VoiceService, "_probe_hardware", staticmethod(lambda: True))
    monkeypatch.setattr(VoiceService, "_start_whisper_background_load", staticmethod(lambda: None))
    monkeypatch.setattr(VoiceService, "_capture_utterance", lambda self: object())
    monkeypatch.setattr(VoiceService, "_transcribe", staticmethod(lambda audio: "build a box 60x40x20"))

    events: list[tuple[str, str]] = []
    service = VoiceService(on_state=lambda s, t: events.append((s, t)))
    service.start()
    service.request_listen()

    assert _wait_for(lambda: any(s == "processing" and t for s, t in events))
    time.sleep(0.1)
    assert service.state == "processing"  # not auto-idled
    assert "speaking" not in [s for s, _t in events]

    # Busy — a second request_listen() while still handling the first is a no-op.
    assert service.request_listen() is False

    service.finish_turn()
    assert _wait_for(lambda: service.state == "idle")
    service.stop()

    handoff = [(s, t) for s, t in events if s == "processing" and t]
    assert handoff == [("processing", "build a box 60x40x20")]


def test_cancel_aborts_an_in_flight_listen(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(VoiceService, "_probe_hardware", staticmethod(lambda: True))
    monkeypatch.setattr(VoiceService, "_start_whisper_background_load", staticmethod(lambda: None))

    def fake_capture(self: VoiceService):
        # Simulate a slow capture that notices cancellation, the way the
        # real sd.InputStream read-loop checks _cancel_event each frame.
        _wait_for(lambda: self._cancel_event.is_set(), timeout=1.0)
        return None

    monkeypatch.setattr(VoiceService, "_capture_utterance", fake_capture)

    states: list[str] = []
    service = VoiceService(on_state=lambda s, _t: states.append(s))
    service.start()
    service.request_listen()

    assert _wait_for(lambda: service.state == "listening")
    service.cancel()
    assert _wait_for(lambda: service.state == "idle")
    service.stop()

    assert "processing" not in states


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


def test_transcribe_logs_failures_instead_of_swallowing_them(monkeypatch: pytest.MonkeyPatch) -> None:
    import dana.audio.stt as stt
    import dana.services.voice_service as voice_service

    logged: list[str] = []
    monkeypatch.setattr(voice_service, "log_exception", lambda _t, msg, *, exc=None: logged.append(f"{msg}: {exc}"))
    monkeypatch.setattr(voice_service, "log", lambda _t, msg, **_kw: logged.append(msg))

    def not_loaded(timeout: float = 0.0):
        raise RuntimeError("Whisper failed to load")

    monkeypatch.setattr(stt, "ensure_whisper_bundle", not_loaded)
    assert VoiceService._transcribe(np.zeros(1600, dtype=np.int16)) == ""
    assert logged == ["transcription failed: Whisper failed to load"]

    def still_loading(timeout: float = 0.0):
        raise TimeoutError("Whisper background load timed out")

    logged.clear()
    monkeypatch.setattr(stt, "ensure_whisper_bundle", still_loading)
    assert VoiceService._transcribe(np.zeros(1600, dtype=np.int16)) == ""
    assert logged and "still loading" in logged[0]


def test_audio_modules_import_without_pytest_path_injection() -> None:
    """Regression: dana.audio.stt / multi_voice_tts imported the bare module
    ``spatial_context`` (via shared_state), which only resolved because
    tests/conftest.py puts scripts/diagnostics on sys.path. Import them the
    way the launcher does — only the repo root on sys.path — and check the
    legacy shared_state stack (vault daemon client, Popen patch) stays out."""
    repo_root = Path(__file__).resolve().parents[2]
    code = (
        "import sys\n"
        "import dana.audio.stt, dana.audio.multi_voice_tts, dana.services.voice_service\n"
        "assert 'spatial_context' not in sys.modules\n"
        "assert 'dana.core.shared_state' not in sys.modules\n"
        "assert 'dana.vault_service' not in sys.modules\n"
    )
    env = {**os.environ, "PYTHONPATH": str(repo_root)}
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=tempfile.gettempdir(), env=env, capture_output=True, text=True, timeout=240
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
