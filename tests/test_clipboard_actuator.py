"""Pytest coverage for the foundational clipboard actuator (dry-run, no real
Win32 clipboard access).

Clears ``DANA_OS_DRY_RUN`` from the ambient environment before each test:
at least one other test module in this suite (``tests/test_e2e_lifecycle.py``)
sets it at import time via a raw ``os.environ[...] = ...`` (not
``monkeypatch``), which leaks into every test that runs afterward in the
same pytest process — matching the convention in
``tests/test_keyboard_actuator.py``/``tests/test_mouse_actuator.py``.

Reads are never gated by dry-run (they have no OS side effects), so every
read test injects ``read_fn`` directly rather than relying on
``DANA_OS_DRY_RUN``. Writes go through the full actuator pipeline, so
write tests follow the same DI-stub + rate-limiter-reset pattern as every
other actuator test file in this suite.
"""
from __future__ import annotations

import dana.tools.clipboard_actuator as clipboard_actuator
from dana.tools.clipboard_actuator import (
    ClipboardActuator,
    read_clipboard_text,
    write_clipboard_text,
)

import pytest


@pytest.fixture(autouse=True)
def _reset_rate_limiter(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("DANA_OS_DRY_RUN", raising=False)
    clipboard_actuator._last_actuation_ts = 0.0
    yield
    clipboard_actuator._last_actuation_ts = 0.0


def _stub_actuator(clipboard_text: str | None = "hello clipboard"):
    read_calls: list[bool] = []
    write_calls: list[str] = []

    def read_fn():
        read_calls.append(True)
        return clipboard_text

    def write_fn(text):
        write_calls.append(text)

    actuator = ClipboardActuator(read_fn=read_fn, write_fn=write_fn)
    return actuator, read_calls, write_calls


# --------------------------------------------------------------------------
# read_text
# --------------------------------------------------------------------------


def test_read_text_returns_backend_text() -> None:
    actuator, read_calls, _ = _stub_actuator("hello clipboard")
    result = actuator.read_text()
    assert result == {
        "ok": True,
        "text": "hello clipboard",
        "empty": False,
        "truncated": False,
        "chars": len("hello clipboard"),
    }
    assert read_calls == [True]


def test_read_text_treats_none_backend_result_as_empty() -> None:
    actuator, _, _ = _stub_actuator(None)
    result = actuator.read_text()
    assert result == {"ok": True, "text": "", "empty": True, "truncated": False, "chars": 0}


def test_read_text_treats_whitespace_only_text_as_empty() -> None:
    actuator, _, _ = _stub_actuator("   ")
    result = actuator.read_text()
    assert result["ok"] is True
    assert result["empty"] is True


def test_read_text_truncates_when_over_max_chars() -> None:
    actuator, _, _ = _stub_actuator("x" * 100)
    result = actuator.read_text(max_chars=10)
    assert result["ok"] is True
    assert result["truncated"] is True
    assert result["text"] == "x" * 10
    assert result["chars"] == 10


def test_read_text_surfaces_backend_failure() -> None:
    def failing_read_fn():
        raise OSError("OpenClipboard failed: 5")

    actuator = ClipboardActuator(read_fn=failing_read_fn)
    result = actuator.read_text()
    assert result["ok"] is False
    assert "OpenClipboard failed" in result["error"]


def test_read_text_is_never_gated_by_dry_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DANA_OS_DRY_RUN", "1")
    actuator, read_calls, _ = _stub_actuator("hello clipboard")
    result = actuator.read_text()
    assert result["ok"] is True
    assert result["text"] == "hello clipboard"
    assert "dry_run" not in result
    assert read_calls == [True]


# --------------------------------------------------------------------------
# write_text
# --------------------------------------------------------------------------


def test_write_text_calls_backend_with_exact_text() -> None:
    actuator, _, write_calls = _stub_actuator()
    result = actuator.write_text("hello world")
    assert result == {"ok": True, "chars": 11, "dry_run": False}
    assert write_calls == ["hello world"]


def test_write_text_rejects_oversized_text_without_calling_backend() -> None:
    actuator, _, write_calls = _stub_actuator()
    result = actuator.write_text("x" * 100, max_chars=10)
    assert result["ok"] is False
    assert "too large" in result["error"]
    assert write_calls == []


def test_write_text_rate_limits_rapid_successive_calls() -> None:
    actuator, _, write_calls = _stub_actuator()
    first = actuator.write_text("hello")
    second = actuator.write_text("world")
    assert first["ok"] is True
    assert second["ok"] is False
    assert "rate_limited" in second["error"]
    assert write_calls == ["hello"]


def test_write_text_surfaces_backend_failure() -> None:
    def failing_write_fn(text):
        raise OSError("SetClipboardData failed: 5")

    actuator = ClipboardActuator(write_fn=failing_write_fn)
    result = actuator.write_text("hello")
    assert result["ok"] is False
    assert "SetClipboardData failed" in result["error"]


def test_write_text_dry_run_validates_but_never_calls_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DANA_OS_DRY_RUN", "1")
    actuator, _, write_calls = _stub_actuator()
    result = actuator.write_text("hello world")
    assert result == {"ok": True, "chars": 11, "dry_run": True}
    assert write_calls == []


def test_write_text_dry_run_still_enforces_size_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DANA_OS_DRY_RUN", "1")
    actuator, _, write_calls = _stub_actuator()
    result = actuator.write_text("x" * 100, max_chars=10)
    assert result["ok"] is False
    assert "too large" in result["error"]
    assert write_calls == []


def test_write_text_respects_kill_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    import dana.middleware.kill_switch as kill_switch

    monkeypatch.setattr(kill_switch, "halt_if_requested", lambda: True)
    actuator, _, write_calls = _stub_actuator()
    result = actuator.write_text("hello world")
    assert result["ok"] is False
    assert result["halted"] is True
    assert write_calls == []


def test_module_level_read_clipboard_text_uses_real_os_control_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dana.tools.os_control as os_control

    monkeypatch.setattr(os_control, "read_clipboard_text", lambda: "from os_control")
    result = read_clipboard_text()
    assert result == {
        "ok": True,
        "text": "from os_control",
        "empty": False,
        "truncated": False,
        "chars": len("from os_control"),
    }


def test_module_level_write_clipboard_text_uses_real_os_control_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Dry-run end-to-end: no dependency injection, no real hardware call.
    monkeypatch.setenv("DANA_OS_DRY_RUN", "1")
    result = write_clipboard_text("hello world")
    assert result == {"ok": True, "chars": 11, "dry_run": True}
