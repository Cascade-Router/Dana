"""Unit tests for VAD-triggered TTS barge-in (no live mic/speakers)."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import numpy as np

from dana.audio import tts_manager, tts_worker
from dana.core import shared_state
from dana.core.constants import BARGE_IN_PLAYBACK_GRACE_MS, VAD_FRAME_SAMPLES
from dana.audio.tts_worker import TtsWorker, get_tts_worker
from dana.audio.vad_consumer import trigger_tts_barge_in


def test_interrupt_flushes_queue_and_sets_event() -> None:
    tts_worker._bind_tts_barge_controller()
    shared_state.tts_interrupt_event.clear()
    shared_state._tts_barge.end_playback()
    shared_state._tts_barge.begin_playback(interruptible=True)
    shared_state.speech_queue.put_nowait(("chunk-a", True))
    shared_state.speech_queue.put_nowait(("chunk-b", True))
    shared_state.tts_busy.set()

    dropped = trigger_tts_barge_in(reason="unit-test")

    assert shared_state.tts_interrupt_event.is_set()
    assert dropped >= 2
    assert shared_state.speech_queue.empty()
    shared_state._tts_barge.end_playback()
    print("[PASS] interrupt flushes spool + latches barge-in event")


def test_uninterruptible_ux_ack_ignores_barge_in() -> None:
    tts_worker._bind_tts_barge_controller()
    shared_state.tts_interrupt_event.clear()
    shared_state.speech_queue.put_nowait(("Yes?", False))
    shared_state._tts_barge.begin_playback(interruptible=False)
    shared_state.tts_busy.set()

    dropped = trigger_tts_barge_in(reason="self-bleed")

    assert dropped == 0
    assert not shared_state.tts_interrupt_event.is_set()
    assert not shared_state.speech_queue.empty()
    shared_state._tts_barge.end_playback()
    tts_manager.flush_tts_queue()
    shared_state.tts_busy.clear()
    print("[PASS] uninterruptible UX ack ignores barge-in")


def test_tts_worker_skips_chunk_when_barge_latched() -> None:
    worker = TtsWorker()
    worker.interrupt(reason="prelatch")
    assert worker.consume_if_set() is True
    assert worker.is_set() is False
    assert worker.consume_if_set() is False
    print("[PASS] barge latch consume/clear for next spool item")


def test_playback_grace_suppresses_onset_window() -> None:
    """First 400ms after begin_playback must report in_playback_grace."""
    worker = TtsWorker()
    worker.begin_playback(interruptible=True)
    assert worker.in_playback_grace(grace_s=0.4) is True
    time.sleep(0.45)
    assert worker.in_playback_grace(grace_s=0.4) is False
    worker.end_playback()
    assert worker.in_playback_grace(grace_s=0.4) is False
    assert BARGE_IN_PLAYBACK_GRACE_MS == 400.0
    print("[PASS] playback grace window")


def test_half_duplex_mic_drop_does_not_interrupt() -> None:
    """While TTS is busy, half-duplex drop must flush mic and never latch barge-in."""
    tts_worker._bind_tts_barge_controller()
    shared_state.tts_interrupt_event.clear()
    shared_state.stop_event.clear()
    shared_state.tts_busy.set()
    shared_state.vad_capture_active.clear()
    shared_state._tts_barge.begin_playback(interruptible=True)
    # Poison the mic queue with "speech-like" frames.
    for _ in range(8):
        shared_state.audio_buffer_queue.put_nowait(np.ones(VAD_FRAME_SAMPLES, dtype=np.float32) * 0.2)
    stop_flag = threading.Event()

    def _stop_soon() -> None:
        time.sleep(0.15)
        shared_state.tts_busy.clear()
        stop_flag.set()

    threading.Thread(target=_stop_soon, daemon=True).start()
    tts_worker.half_duplex_mic_drop(stop_flag)
    assert not shared_state.tts_interrupt_event.is_set()
    shared_state._tts_barge.end_playback()
    print("[PASS] half-duplex mic drop does not interrupt TTS")


def test_active_stream_abort_on_interrupt() -> None:
    stream = MagicMock()
    worker = get_tts_worker(barge_in_event=shared_state.tts_interrupt_event)
    tts_worker._bind_tts_barge_controller()
    shared_state.tts_interrupt_event.clear()
    worker.register_output_stream(stream)

    worker.interrupt(reason="abort-stream")

    stream.abort.assert_called()
    worker.unregister_output_stream(stream)
    print("[PASS] interrupt aborts registered OutputStream")


def test_play_pcm_respects_barge_in_quickly() -> None:
    """Simulate long PCM; interrupt from another thread mid-playback."""
    tts_worker._bind_tts_barge_controller()
    shared_state.tts_interrupt_event.clear()
    shared_state.stop_event.clear()

    # ~2s of silence @ 16 kHz — interrupt should cut well before full duration.
    audio = np.zeros(32000, dtype=np.float32)
    t0 = time.perf_counter()

    def _barge() -> None:
        time.sleep(0.05)
        trigger_tts_barge_in(reason="sim-barge")

    threading.Thread(target=_barge, daemon=True).start()
    # Use fallback path if OutputStream unavailable in CI — still checks event.
    interrupted = tts_worker._play_pcm_interruptible(audio, 16000, None)
    elapsed = time.perf_counter() - t0

    assert interrupted is True
    assert elapsed < 1.5, f"barge-in too slow ({elapsed:.2f}s)"
    print(f"[PASS] playback aborted on barge-in ({elapsed:.2f}s)")
