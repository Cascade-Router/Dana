"""Barge-in interrupt plumbing tests (no live mic/speaker required)."""

from __future__ import annotations

import threading
import time

import numpy as np

from dana.audio import tts_manager, tts_worker
from dana.core import shared_state


def test_flush_speech_queue() -> None:
    tts_manager.flush_speech_queue()
    shared_state.speech_queue.put_nowait("one")
    shared_state.speech_queue.put_nowait("two")
    assert tts_manager.flush_speech_queue() == 2
    assert shared_state.speech_queue.empty()
    print("[PASS] flush_speech_queue")


def test_play_pcm_respects_interrupt_event() -> None:
    shared_state.tts_interrupt_event.clear()
    # ~1s of silence @ 16 kHz — interrupt after 80ms.
    audio = np.zeros(16000, dtype=np.float32)

    def _trip() -> None:
        time.sleep(0.08)
        shared_state.tts_interrupt_event.set()

    threading.Thread(target=_trip, daemon=True).start()
    t0 = time.perf_counter()
    interrupted = tts_worker._play_pcm_interruptible(audio, 16000, None)
    elapsed = time.perf_counter() - t0
    shared_state.tts_interrupt_event.clear()
    assert interrupted is True
    assert elapsed < 0.6, f"playback did not abort quickly ({elapsed:.2f}s)"
    print(f"[PASS] interruptible playback aborted in {elapsed:.2f}s")


def test_tts_interrupt_event_exists() -> None:
    assert isinstance(shared_state.tts_interrupt_event, threading.Event)
    print("[PASS] tts_interrupt_event is a threading.Event")


if __name__ == "__main__":
    test_tts_interrupt_event_exists()
    test_flush_speech_queue()
    test_play_pcm_respects_interrupt_event()
    print("OK")
