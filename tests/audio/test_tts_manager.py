"""Unit tests for TTSManager speech queue + vision debug gate."""

from __future__ import annotations

import queue
import threading
import time



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
