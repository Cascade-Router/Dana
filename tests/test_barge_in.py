"""Barge-in interrupt plumbing tests (no live mic/speaker required)."""

from __future__ import annotations

import threading
import time



from dana.audio import tts_manager
from dana.core import shared_state


def test_flush_speech_queue() -> None:
    tts_manager.flush_speech_queue()
    shared_state.speech_queue.put_nowait("one")
    shared_state.speech_queue.put_nowait("two")
    assert tts_manager.flush_speech_queue() == 2
    assert shared_state.speech_queue.empty()
    print("[PASS] flush_speech_queue")


def test_tts_interrupt_event_exists() -> None:
    assert isinstance(shared_state.tts_interrupt_event, threading.Event)
    print("[PASS] tts_interrupt_event is a threading.Event")


if __name__ == "__main__":
    test_tts_interrupt_event_exists()
    test_flush_speech_queue()
    test_play_pcm_respects_interrupt_event()
    print("OK")
