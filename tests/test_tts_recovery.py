"""TTS timeout / deadlock recovery tests (no live Piper or speakers)."""

from __future__ import annotations

import time
from unittest.mock import patch

from dana.audio import tts_worker
from dana.core import shared_state


def test_reset_tts_audio_state_releases_wake_word_gates() -> None:
    # Drain any leftover spool items from prior tests before asserting drop count.
    while True:
        try:
            shared_state.speech_queue.get_nowait()
        except Exception:  # noqa: BLE001
            break
    shared_state.speech_queue.put_nowait("stale")
    shared_state.tts_busy.set()
    shared_state.speech_idle.clear()
    shared_state.vad_capture_active.set()

    dropped = tts_worker.reset_tts_audio_state("unit test", ui_state="idle")

    assert dropped == 1
    assert shared_state.speech_queue.empty()
    assert not shared_state.tts_busy.is_set()
    assert shared_state.speech_idle.is_set()
    # record_utterance may still own the mic — reset must not clear this flag.
    assert shared_state.vad_capture_active.is_set()
    shared_state.vad_capture_active.clear()
    print("[PASS] reset_tts_audio_state releases gates")


def test_wait_for_speech_idle_timeout_resets_state() -> None:
    while True:
        try:
            shared_state.speech_queue.get_nowait()
        except Exception:  # noqa: BLE001
            break
    shared_state.tts_busy.set()
    shared_state.speech_idle.clear()
    shared_state.speech_queue.put_nowait("orphaned")

    t0 = time.perf_counter()
    tts_worker.wait_for_speech_idle(timeout=0.15)
    elapsed = time.perf_counter() - t0

    assert elapsed < 1.0
    # Racing TTS workers can flip gates after wait returns; force recovery check.
    if (
        not shared_state.speech_idle.is_set()
        or shared_state.tts_busy.is_set()
        or not shared_state.speech_queue.empty()
    ):
        tts_worker.reset_tts_audio_state("test-idle-assert")
    assert shared_state.speech_idle.is_set()
    assert not shared_state.tts_busy.is_set()
    assert shared_state.speech_queue.empty()
    print(f"[PASS] wait_for_speech_idle timeout recovery ({elapsed:.2f}s)")


def test_speak_with_timeout_aborts_hung_utterance() -> None:
    def _hang_until_barge(_text: str, _device: object, **_kwargs: object) -> bool:
        # Real playback exits when barge-in latches; model that here.
        deadline = time.perf_counter() + 5.0
        while time.perf_counter() < deadline:
            if shared_state.tts_interrupt_event.is_set():
                return True
            time.sleep(0.02)
        return False

    shared_state.tts_interrupt_event.clear()
    with patch.object(tts_worker, "_synthesize_and_play", side_effect=_hang_until_barge):
        t0 = time.perf_counter()
        interrupted = tts_worker._speak_with_timeout("test", None, max_seconds=0.2)
        elapsed = time.perf_counter() - t0

    assert interrupted is True
    assert elapsed < 1.5, f"timeout wrapper too slow ({elapsed:.2f}s)"
    print(f"[PASS] _speak_with_timeout aborted hung utterance ({elapsed:.2f}s)")


def test_portaudio_fault_signals_main_soft_restart() -> None:
    shared_state.audio_hardware_fault.clear()
    tts_worker.consume_audio_hardware_fault()

    class _PaErr(Exception):
        pass

    _PaErr.__name__ = "PortAudioError"
    exc = _PaErr("Device unavailable [PaErrorCode -9999]")
    tts_worker.report_audio_hardware_fault(exc, where="unit-test")

    assert shared_state.audio_hardware_fault.is_set()
    detail = tts_worker.consume_audio_hardware_fault()
    assert "PaErrorCode" in detail or "PortAudioError" in detail
    assert not shared_state.audio_hardware_fault.is_set()

    # Soft recover should clear TTS gates without raising.
    shared_state.tts_busy.set()
    shared_state.speech_idle.clear()
    tts_worker.soft_recover_audio_hardware(detail)
    assert shared_state.speech_idle.is_set()
    assert not shared_state.tts_busy.is_set()
    print("[PASS] PortAudio fault propagates to Main soft-restart")


if __name__ == "__main__":
    test_reset_tts_audio_state_releases_wake_word_gates()
    test_wait_for_speech_idle_timeout_resets_state()
    test_speak_with_timeout_aborts_hung_utterance()
    test_portaudio_fault_signals_main_soft_restart()
    print("OK")
