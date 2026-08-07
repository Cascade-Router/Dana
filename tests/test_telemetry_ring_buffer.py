import threading
import time

from dana.core.telemetry import AsyncRingBuffer, NeuralStreamEmitter


def test_ring_buffer_eviction_is_thread_safe_and_uses_bounded_capacity() -> None:
    buf = AsyncRingBuffer(capacity=8)
    emitter = NeuralStreamEmitter(buf)

    def worker() -> None:
        for i in range(100):
            emitter.emit("event", {"i": i})

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    items = buf.snapshot()
    assert len(items) <= 8
    assert len(items) > 0


def test_emitter_is_non_blocking_and_reports_recent_events() -> None:
    buf = AsyncRingBuffer(capacity=4)
    emitter = NeuralStreamEmitter(buf)
    start = time.perf_counter()
    for i in range(20):
        emitter.emit("tick", {"i": i})
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    assert elapsed_ms < 1.0
    assert len(buf.snapshot()) <= 4
