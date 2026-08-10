"""Pytest coverage for the foundational window actuator (dry-run, no real
Win32 window manipulation).

Every test injects ``list_windows_fn``/``focus_fn`` stubs so the
matching/safety pipeline runs with no real desktop windows touched, and
resets the module-wide rate limiter between tests so cases don't interfere
with each other. Also clears ``DANA_OS_DRY_RUN`` from the ambient
environment before each test, matching the convention in
``tests/test_mouse_actuator.py``/``tests/test_scroll_actuator.py`` (some
other test module in this suite sets it at import time via a raw
``os.environ[...] = ...``, which would otherwise leak into these tests).
"""
from __future__ import annotations

import dana.tools.window_actuator as window_actuator
from dana.tools import rate_limiter
from dana.tools.window_actuator import WindowActuator, focus_by_title

import pytest

_WINDOWS = [
    {"hwnd": 1001, "title": "Cursor - project_name", "pid": 100},
    {"hwnd": 1002, "title": "Mozilla Firefox", "pid": 200},
    {"hwnd": 1003, "title": "Notepad", "pid": 300},
]


@pytest.fixture(autouse=True)
def _reset_rate_limiter(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("DANA_OS_DRY_RUN", raising=False)
    rate_limiter.reset()
    yield
    rate_limiter.reset()


def _stub_actuator(windows=None, focus_result: bool = True):
    calls: dict[str, list] = {"focus": []}
    win_list = windows if windows is not None else list(_WINDOWS)

    def list_windows_fn():
        return win_list

    def focus_fn(hwnd):
        calls["focus"].append(hwnd)
        return focus_result

    actuator = WindowActuator(list_windows_fn=list_windows_fn, focus_fn=focus_fn)
    return actuator, calls


def test_focus_by_title_matches_exact_title_and_focuses() -> None:
    actuator, calls = _stub_actuator()
    result = actuator.focus_by_title("Notepad")
    assert result["ok"] is True
    assert result["dry_run"] is False
    assert result["window"]["hwnd"] == 1003
    assert calls["focus"] == [1003]


def test_focus_by_title_partial_match_when_no_exact_title() -> None:
    actuator, calls = _stub_actuator()
    result = actuator.focus_by_title("Cursor")
    assert result["ok"] is True
    assert result["window"]["title"] == "Cursor - project_name"
    assert calls["focus"] == [1001]


def test_focus_by_title_prefers_exact_match_over_partial_match() -> None:
    windows = [
        {"hwnd": 1, "title": "Terminal", "pid": 10},
        {"hwnd": 2, "title": "Term", "pid": 20},
    ]
    actuator, calls = _stub_actuator(windows=windows)
    result = actuator.focus_by_title("Term")
    assert result["ok"] is True
    # "Term" fullmatches hwnd=2's title exactly; hwnd=1 is only a substring
    # match, so the exact match wins even though it's later in the list.
    assert result["window"]["hwnd"] == 2
    assert calls["focus"] == [2]


def test_focus_by_title_is_case_insensitive() -> None:
    actuator, calls = _stub_actuator()
    result = actuator.focus_by_title("notepad")
    assert result["ok"] is True
    assert result["window"]["title"] == "Notepad"


def test_focus_by_title_no_match_without_calling_backend() -> None:
    actuator, calls = _stub_actuator()
    result = actuator.focus_by_title("Zoom")
    assert result["ok"] is False
    assert "no window title matched" in result["error"]
    assert calls["focus"] == []


def test_focus_by_title_rejects_invalid_regex_without_listing_windows() -> None:
    listed = {"count": 0}

    def list_windows_fn():
        listed["count"] += 1
        return list(_WINDOWS)

    actuator = WindowActuator(list_windows_fn=list_windows_fn, focus_fn=lambda hwnd: True)
    result = actuator.focus_by_title("[invalid(regex")
    assert result["ok"] is False
    assert "invalid title_regex" in result["error"]
    assert listed["count"] == 0


def test_focus_by_title_rejects_empty_pattern() -> None:
    actuator, calls = _stub_actuator()
    result = actuator.focus_by_title("   ")
    assert result["ok"] is False
    assert "non-empty" in result["error"]
    assert calls["focus"] == []


def test_focus_by_title_surfaces_list_windows_failure() -> None:
    def list_windows_fn():
        raise OSError("EnumWindows failed: 5")

    actuator = WindowActuator(list_windows_fn=list_windows_fn, focus_fn=lambda hwnd: True)
    result = actuator.focus_by_title("Notepad")
    assert result["ok"] is False
    assert "could not list windows" in result["error"]


def test_focus_by_title_surfaces_setforegroundwindow_denial() -> None:
    actuator, calls = _stub_actuator(focus_result=False)
    result = actuator.focus_by_title("Notepad")
    assert result["ok"] is False
    assert "SetForegroundWindow reported failure" in result["error"]
    assert calls["focus"] == [1003]


def test_focus_by_title_rate_limits_rapid_successive_calls() -> None:
    actuator, calls = _stub_actuator()
    first = actuator.focus_by_title("Notepad")
    second = actuator.focus_by_title("Notepad")
    assert first["ok"] is True
    assert second["ok"] is False
    assert "rate_limited" in second["error"]
    assert calls["focus"] == [1003]


def test_focus_by_title_dry_run_validates_but_never_focuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DANA_OS_DRY_RUN", "1")
    actuator, calls = _stub_actuator()
    result = actuator.focus_by_title("Notepad")
    assert result["ok"] is True
    assert result["dry_run"] is True
    assert result["window"]["hwnd"] == 1003
    assert calls["focus"] == []


def test_focus_by_title_dry_run_still_enforces_no_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DANA_OS_DRY_RUN", "1")
    actuator, calls = _stub_actuator()
    result = actuator.focus_by_title("Zoom")
    assert result["ok"] is False
    assert calls["focus"] == []


def test_focus_by_title_respects_kill_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    import dana.middleware.kill_switch as kill_switch

    monkeypatch.setattr(kill_switch, "halt_if_requested", lambda: True)
    actuator, calls = _stub_actuator()
    result = actuator.focus_by_title("Notepad")
    assert result["ok"] is False
    assert result["halted"] is True
    assert calls["focus"] == []


def test_module_level_focus_by_title_uses_real_os_control_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Dry-run end-to-end: no dependency injection, so the real (mocked)
    # os_control.get_active_windows backend is exercised for the listing
    # step even though the focus step itself is skipped by dry-run.
    monkeypatch.setenv("DANA_OS_DRY_RUN", "1")
    import dana.tools.os_control as os_control

    monkeypatch.setattr(
        os_control, "get_active_windows", lambda: [{"hwnd": 42, "title": "Notepad", "pid": 5}]
    )
    result = focus_by_title("Notepad")
    assert result["ok"] is True
    assert result["dry_run"] is True
    assert result["window"]["hwnd"] == 42
