"""Pytest coverage for the foundational keyboard actuator (dry-run, no real hardware input).

Clears ``DONNA_OS_DRY_RUN`` from the ambient environment before each test:
at least one other test module in this suite (``tests/test_e2e_lifecycle.py``)
sets it at import time via a raw ``os.environ[...] = ...`` (not
``monkeypatch``), which leaks into every test that runs afterward in the
same pytest process. Without this, tests here that expect non-dry-run
behavior by default silently get dry-run instead when the full suite runs
in an order where that file is collected first.
"""
from __future__ import annotations

import dana.tools.keyboard_actuator as keyboard_actuator
from dana.tools.keyboard_actuator import KeyboardActuator, type_text

import pytest


@pytest.fixture(autouse=True)
def _reset_rate_limiter(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("DONNA_OS_DRY_RUN", raising=False)
    keyboard_actuator._last_actuation_ts = 0.0
    yield
    keyboard_actuator._last_actuation_ts = 0.0


def _stub_actuator():
    calls: list[str] = []

    def type_fn(text):
        calls.append(text)
        return {"ok": True, "chars_typed": len(text)}

    return KeyboardActuator(type_fn=type_fn), calls


def test_type_text_calls_backend_with_exact_text() -> None:
    actuator, calls = _stub_actuator()
    result = actuator.type_text("hello world")
    assert result == {"ok": True, "chars_typed": 11, "dry_run": False}
    assert calls == ["hello world"]


def test_type_text_rejects_empty_text_without_calling_backend() -> None:
    actuator, calls = _stub_actuator()
    result = actuator.type_text("   ")
    assert result == {"ok": False, "error": "empty text"}
    assert calls == []


def test_type_text_rejects_oversized_text_without_calling_backend() -> None:
    actuator, calls = _stub_actuator()
    result = actuator.type_text("x" * 2001)
    assert result["ok"] is False
    assert "too long" in result["error"]
    assert calls == []


def test_type_text_rate_limits_rapid_successive_calls() -> None:
    actuator, calls = _stub_actuator()
    first = actuator.type_text("hello")
    second = actuator.type_text("world")
    assert first["ok"] is True
    assert second["ok"] is False
    assert "rate_limited" in second["error"]
    assert calls == ["hello"]


def test_type_text_surfaces_backend_failure() -> None:
    def failing_type_fn(text):
        return {"ok": False, "error": "SendInput failed (sent=0)"}

    actuator = KeyboardActuator(type_fn=failing_type_fn)
    result = actuator.type_text("hello")
    assert result["ok"] is False
    assert result["error"] == "SendInput failed (sent=0)"


def test_type_text_dry_run_validates_but_never_calls_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DONNA_OS_DRY_RUN", "1")
    actuator, calls = _stub_actuator()
    result = actuator.type_text("hello world")
    assert result == {"ok": True, "chars_typed": 11, "dry_run": True}
    assert calls == []


def test_type_text_dry_run_still_enforces_empty_text_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DONNA_OS_DRY_RUN", "1")
    actuator, calls = _stub_actuator()
    result = actuator.type_text("")
    assert result["ok"] is False


def test_type_text_respects_kill_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    import dana.middleware.kill_switch as kill_switch

    monkeypatch.setattr(kill_switch, "halt_if_requested", lambda: True)
    actuator, calls = _stub_actuator()
    result = actuator.type_text("hello world")
    assert result["ok"] is False
    assert result["halted"] is True
    assert calls == []


def test_module_level_type_text_uses_real_os_control_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Dry-run end-to-end: no dependency injection, no real hardware call.
    monkeypatch.setenv("DONNA_OS_DRY_RUN", "1")
    result = type_text("hello world")
    assert result == {"ok": True, "chars_typed": 11, "dry_run": True}
