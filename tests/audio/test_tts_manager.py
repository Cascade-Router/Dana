"""Unit tests for TTSManager speech queue + vision debug gate."""

from __future__ import annotations

import queue
import threading
import time

import pytest


def test_tts_manager_sequential_queue() -> None:
    from dana.audio.tts_manager import TTSManager

    mgr = TTSManager(maxsize=8)
    seen: list[str] = []
    done = threading.Event()

    def _worker() -> None:
        while not done.is_set():
            try:
                item = mgr.speech_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                break
            text = item[0] if isinstance(item, tuple) else str(item)
            seen.append(str(text))
            time.sleep(0.02)

    mgr.bind(worker=_worker)
    mgr.start()
    mgr.enqueue("alpha", interruptible=False)
    mgr.enqueue("beta", interruptible=False)
    mgr.enqueue("gamma", interruptible=False)
    deadline = time.time() + 2.0
    while len(seen) < 3 and time.time() < deadline:
        time.sleep(0.05)
    done.set()
    try:
        mgr.speech_queue.put_nowait(None)
    except queue.Full:
        pass
    assert seen == ["alpha", "beta", "gamma"]


def test_vision_overlay_gated_without_debug(monkeypatch) -> None:
    monkeypatch.delenv("DONNA_DEBUG_VISION", raising=False)
    from dana.vision.overlay import RoiOverlay, update_roi, vision_debug_enabled

    assert vision_debug_enabled() is False
    ov = RoiOverlay()
    ov.start()
    # Gate: ready is set immediately, no Tk thread.
    assert ov._ready.is_set()
    assert ov._thread is None or not ov._thread.is_alive()
    update_roi((10, 10, 50, 50), "should not open")
    assert ov._thread is None or not ov._thread.is_alive()


def test_status_bus_message_updates_not_dropped() -> None:
    from dana.ui.status_bus import StatusEventBus, drain_state_changes, emit_state_change

    bus = StatusEventBus()
    StatusEventBus._instance = bus
    emit_state_change("executing", tool="meta_broker", message="Starting Epic 1")
    emit_state_change("executing", tool="meta_broker", message="Epic 1 validated OK")
    events = drain_state_changes(max_items=16)
    assert len(events) == 2
    assert events[-1]["message"] == "Epic 1 validated OK"
