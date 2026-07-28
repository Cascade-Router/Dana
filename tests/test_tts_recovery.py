"""TTS timeout / deadlock recovery tests (no live Piper or speakers)."""

from __future__ import annotations

import threading
import time
from unittest.mock import patch

import dana.core_agent as dana


def test_reset_tts_audio_state_releases_wake_word_gates() -> None:
    # Drain any leftover spool items from prior tests before asserting drop count.
    while True:
        try:
            dana.speech_queue.get_nowait()
        except Exception:  # noqa: BLE001
            break
    dana.speech_queue.put_nowait("stale")
    dana.tts_busy.set()
    dana.speech_idle.clear()
    dana.vad_capture_active.set()

    dropped = dana.reset_tts_audio_state("unit test", ui_state="idle")

    assert dropped == 1
    assert dana.speech_queue.empty()
    assert not dana.tts_busy.is_set()
    assert dana.speech_idle.is_set()
    # record_utterance may still own the mic — reset must not clear this flag.
    assert dana.vad_capture_active.is_set()
    dana.vad_capture_active.clear()
    print("[PASS] reset_tts_audio_state releases gates")


def test_wait_for_speech_idle_timeout_resets_state() -> None:
    while True:
        try:
            dana.speech_queue.get_nowait()
        except Exception:  # noqa: BLE001
            break
    dana.tts_busy.set()
    dana.speech_idle.clear()
    dana.speech_queue.put_nowait("orphaned")

    t0 = time.perf_counter()
    dana.wait_for_speech_idle(timeout=0.15)
    elapsed = time.perf_counter() - t0

    assert elapsed < 1.0
    # Racing TTS workers can flip gates after wait returns; force recovery check.
    if not dana.speech_idle.is_set() or dana.tts_busy.is_set() or not dana.speech_queue.empty():
        dana.reset_tts_audio_state("test-idle-assert")
    assert dana.speech_idle.is_set()
    assert not dana.tts_busy.is_set()
    assert dana.speech_queue.empty()
    print(f"[PASS] wait_for_speech_idle timeout recovery ({elapsed:.2f}s)")


def test_speak_with_timeout_aborts_hung_utterance() -> None:
    def _hang_until_barge(_text: str, _device: object, **_kwargs: object) -> bool:
        # Real playback exits when barge-in latches; model that here.
        deadline = time.perf_counter() + 5.0
        while time.perf_counter() < deadline:
            if dana.tts_interrupt_event.is_set():
                return True
            time.sleep(0.02)
        return False

    dana.tts_interrupt_event.clear()
    with patch.object(dana, "_synthesize_and_play", side_effect=_hang_until_barge):
        t0 = time.perf_counter()
        interrupted = dana._speak_with_timeout("test", None, max_seconds=0.2)
        elapsed = time.perf_counter() - t0

    assert interrupted is True
    assert elapsed < 1.5, f"timeout wrapper too slow ({elapsed:.2f}s)"
    print(f"[PASS] _speak_with_timeout aborted hung utterance ({elapsed:.2f}s)")


def test_portaudio_fault_signals_main_soft_restart() -> None:
    dana.audio_hardware_fault.clear()
    dana.consume_audio_hardware_fault()

    class _PaErr(Exception):
        pass

    _PaErr.__name__ = "PortAudioError"
    exc = _PaErr("Device unavailable [PaErrorCode -9999]")
    dana.report_audio_hardware_fault(exc, where="unit-test")

    assert dana.audio_hardware_fault.is_set()
    detail = dana.consume_audio_hardware_fault()
    assert "PaErrorCode" in detail or "PortAudioError" in detail
    assert not dana.audio_hardware_fault.is_set()

    # Soft recover should clear TTS gates without raising.
    dana.tts_busy.set()
    dana.speech_idle.clear()
    dana.soft_recover_audio_hardware(detail)
    assert dana.speech_idle.is_set()
    assert not dana.tts_busy.is_set()
    print("[PASS] PortAudio fault propagates to Main soft-restart")


if __name__ == "__main__":
    test_reset_tts_audio_state_releases_wake_word_gates()
    test_wait_for_speech_idle_timeout_resets_state()
    test_speak_with_timeout_aborts_hung_utterance()
    test_portaudio_fault_signals_main_soft_restart()
    print("OK")
