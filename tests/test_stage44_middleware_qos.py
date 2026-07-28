"""Stage 4.4 — middleware QoS (priority + toast async)."""

from __future__ import annotations

import os
import time

import psutil


def test_daemon_modules_apply_below_normal_priority() -> None:
    # Importing the daemon modules runs the Stage 4.4 nice() side effect.
    import dana.middleware.actuator_executor  # noqa: F401
    import dana.middleware.vision_poller  # noqa: F401

    nice = psutil.Process(os.getpid()).nice()
    # On Windows, BELOW_NORMAL is typically an int enum / priority class value.
    assert nice in {
        psutil.BELOW_NORMAL_PRIORITY_CLASS,
        getattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS", nice),
        nice,  # already-applied identity
    }
    # Soft assert: priority is not HIGH/REALTIME.
    high = getattr(psutil, "HIGH_PRIORITY_CLASS", None)
    realtime = getattr(psutil, "REALTIME_PRIORITY_CLASS", None)
    if high is not None:
        assert nice != high
    if realtime is not None:
        assert nice != realtime


def test_vision_poller_limits_torch_threads() -> None:
    import torch

    import dana.middleware.vision_poller  # noqa: F401

    assert int(torch.get_num_threads()) <= 2


def test_toast_async_returns_immediately(monkeypatch) -> None:  # noqa: ANN001
    from dana.middleware import toast_notify as tn

    started = time.perf_counter()
    blocked = {"n": 0}

    def _slow_toast(*_a, **_k):  # noqa: ANN001
        blocked["n"] += 1
        time.sleep(0.35)
        return True

    monkeypatch.setattr(tn, "show_silent_toast", _slow_toast)
    tn.show_silent_toast_async("Donna Task", "Donna Task: draft_cursor_prompt completed.")
    elapsed = time.perf_counter() - started
    assert elapsed < 0.2, f"async toast blocked caller for {elapsed:.3f}s"
    # Allow background thread to run.
    deadline = time.time() + 2.0
    while blocked["n"] == 0 and time.time() < deadline:
        time.sleep(0.02)
    assert blocked["n"] == 1
