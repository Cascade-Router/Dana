"""Barge-in interrupt plumbing tests (no live mic/speaker required)."""

from __future__ import annotations

import threading
import time

import numpy as np

import dana.core_agent as dana


def test_flush_speech_queue() -> None:
    dana.flush_speech_queue()
    dana.speech_queue.put_nowait("one")
    dana.speech_queue.put_nowait("two")
    assert dana.flush_speech_queue() == 2
    assert dana.speech_queue.empty()
    print("[PASS] flush_speech_queue")


def test_play_pcm_respects_interrupt_event() -> None:
    dana.tts_interrupt_event.clear()
    # ~1s of silence @ 16 kHz — interrupt after 80ms.
    audio = np.zeros(16000, dtype=np.float32)

    def _trip() -> None:
        time.sleep(0.08)
        dana.tts_interrupt_event.set()

    threading.Thread(target=_trip, daemon=True).start()
    t0 = time.perf_counter()
    interrupted = dana._play_pcm_interruptible(audio, 16000, None)
    elapsed = time.perf_counter() - t0
    dana.tts_interrupt_event.clear()
    assert interrupted is True
    assert elapsed < 0.6, f"playback did not abort quickly ({elapsed:.2f}s)"
    print(f"[PASS] interruptible playback aborted in {elapsed:.2f}s")


def test_tts_interrupt_event_exists() -> None:
    assert isinstance(dana.tts_interrupt_event, threading.Event)
    print("[PASS] tts_interrupt_event is a threading.Event")


if __name__ == "__main__":
    test_tts_interrupt_event_exists()
    test_flush_speech_queue()
    test_play_pcm_respects_interrupt_event()
    print("OK")
