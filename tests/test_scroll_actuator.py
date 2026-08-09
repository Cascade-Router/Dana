"""Pytest coverage for the foundational scroll actuator (dry-run, no real hardware input).

Clears ``DANA_OS_DRY_RUN`` from the ambient environment before each test:
at least one other test module in this suite (``tests/test_e2e_lifecycle.py``)
sets it at import time via a raw ``os.environ[...] = ...`` (not
``monkeypatch``), which leaks into every test that runs afterward in the
same pytest process. Without this, tests here that expect non-dry-run
behavior by default silently get dry-run instead when the full suite runs
in an order where that file is collected first.
"""
from __future__ import annotations

import dana.tools.scroll_actuator as scroll_actuator
from dana.tools.os_control import WHEEL_DELTA
from dana.tools.scroll_actuator import ScrollActuator, scroll

import pytest


@pytest.fixture(autouse=True)
def _reset_rate_limiter(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("DANA_OS_DRY_RUN", raising=False)
    scroll_actuator._last_actuation_ts = 0.0
    yield
    scroll_actuator._last_actuation_ts = 0.0


def _stub_actuator():
    calls: list[tuple[int, int]] = []

    def scroll_fn(dx, dy):
        calls.append((dx, dy))

    return ScrollActuator(scroll_fn=scroll_fn), calls


def test_scroll_down_sends_one_negative_vertical_tick_per_count() -> None:
    actuator, calls = _stub_actuator()
    result = actuator.scroll("down", ticks=3)
    assert result == {"ok": True, "direction": "down", "ticks": 3, "dry_run": False}
    assert calls == [(0, -WHEEL_DELTA)] * 3


def test_scroll_up_sends_positive_vertical_ticks() -> None:
    actuator, calls = _stub_actuator()
    actuator.scroll("up", ticks=2)
    assert calls == [(0, WHEEL_DELTA)] * 2


def test_scroll_left_sends_negative_horizontal_ticks() -> None:
    actuator, calls = _stub_actuator()
    actuator.scroll("left", ticks=1)
    assert calls == [(-WHEEL_DELTA, 0)]


def test_scroll_right_sends_positive_horizontal_ticks() -> None:
    actuator, calls = _stub_actuator()
    actuator.scroll("right", ticks=4)
    assert calls == [(WHEEL_DELTA, 0)] * 4


def test_scroll_is_case_and_whitespace_insensitive() -> None:
    actuator, calls = _stub_actuator()
    result = actuator.scroll("  DOWN  ", ticks=1)
    assert result["ok"] is True
    assert result["direction"] == "down"


def test_scroll_rejects_unknown_direction_without_calling_backend() -> None:
    actuator, calls = _stub_actuator()
    result = actuator.scroll("sideways", ticks=1)
    assert result["ok"] is False
    assert "unknown direction" in result["error"]
    assert calls == []


def test_scroll_rejects_non_positive_ticks_without_calling_backend() -> None:
    actuator, calls = _stub_actuator()
    result = actuator.scroll("down", ticks=0)
    assert result["ok"] is False
    assert "positive" in result["error"]
    assert calls == []


def test_scroll_rejects_excessive_ticks_without_calling_backend() -> None:
    actuator, calls = _stub_actuator()
    result = actuator.scroll("down", ticks=999)
    assert result["ok"] is False
    assert "too large" in result["error"]
    assert calls == []


def test_scroll_rate_limits_rapid_successive_calls() -> None:
    actuator, calls = _stub_actuator()
    first = actuator.scroll("down", ticks=1)
    second = actuator.scroll("down", ticks=1)
    assert first["ok"] is True
    assert second["ok"] is False
    assert "rate_limited" in second["error"]
    assert calls == [(0, -WHEEL_DELTA)]


def test_scroll_surfaces_backend_failure_after_partial_ticks() -> None:
    calls: list[tuple[int, int]] = []

    def flaky_scroll_fn(dx, dy):
        calls.append((dx, dy))
        if len(calls) == 2:
            raise OSError("SendInput mouse wheel failed (sent=0)")

    actuator = ScrollActuator(scroll_fn=flaky_scroll_fn)
    result = actuator.scroll("down", ticks=5)
    assert result["ok"] is False
    assert result["ticks_completed"] == 1
    assert "SendInput mouse wheel failed" in result["error"]
    assert len(calls) == 2


def test_scroll_dry_run_validates_but_never_calls_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DANA_OS_DRY_RUN", "1")
    actuator, calls = _stub_actuator()
    result = actuator.scroll("down", ticks=5)
    assert result == {"ok": True, "direction": "down", "ticks": 5, "dry_run": True}
    assert calls == []


def test_scroll_dry_run_still_validates_direction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DANA_OS_DRY_RUN", "1")
    actuator, calls = _stub_actuator()
    result = actuator.scroll("sideways", ticks=1)
    assert result["ok"] is False


def test_scroll_respects_kill_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    import dana.middleware.kill_switch as kill_switch

    monkeypatch.setattr(kill_switch, "halt_if_requested", lambda: True)
    actuator, calls = _stub_actuator()
    result = actuator.scroll("down", ticks=3)
    assert result["ok"] is False
    assert result["halted"] is True
    assert calls == []


def test_module_level_scroll_uses_real_os_control_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Dry-run end-to-end: no dependency injection, no real hardware call.
    monkeypatch.setenv("DANA_OS_DRY_RUN", "1")
    result = scroll("up", ticks=2)
    assert result == {"ok": True, "direction": "up", "ticks": 2, "dry_run": True}
