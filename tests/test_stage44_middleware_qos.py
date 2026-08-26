"""Stage 4.4 — middleware QoS (toast async).

The priority/torch-thread tests that used to live here (daemon-module nice()
side effect, vision_poller's torch thread cap) tested
dana/middleware/actuator_executor.py and dana/middleware/vision_poller.py,
both removed as dead legacy code (zero live callers — see the 2026-08-26
connectivity audit). Only the still-live toast_notify coverage remains.
"""

from __future__ import annotations

import time


def test_toast_async_returns_immediately(monkeypatch) -> None:  # noqa: ANN001
    from dana.middleware import toast_notify as tn

    started = time.perf_counter()
    blocked = {"n": 0}

    def _slow_toast(*_a, **_k):  # noqa: ANN001
        blocked["n"] += 1
        time.sleep(0.35)
        return True

    monkeypatch.setattr(tn, "show_silent_toast", _slow_toast)
    tn.show_silent_toast_async("Dana Task", "Dana Task: draft_cursor_prompt completed.")
    elapsed = time.perf_counter() - started
    assert elapsed < 0.2, f"async toast blocked caller for {elapsed:.3f}s"
    # Allow background thread to run.
    deadline = time.time() + 2.0
    while blocked["n"] == 0 and time.time() < deadline:
        time.sleep(0.02)
    assert blocked["n"] == 1
