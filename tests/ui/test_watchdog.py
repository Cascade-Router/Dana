"""Unit tests for the ambient Shell Watchdog (no real Windows toasts)."""

from __future__ import annotations

from pathlib import Path

import pytest

from donna.ui import notifications, watchdog
from donna.ui.watchdog import ShellWatchdog, extract_trace_window, summarize_error


PYTEST_FAIL_STDERR = """\
============================= test session starts ==============================
collected 1 item

tests/ui/test_example.py::test_boom FAILED                               [100%]

=================================== FAILURES ===================================
__________________________________ test_boom ___________________________________

    def test_boom():
>       raise ImportError("cannot import name 'foo' from 'startup_tray'")
E       ImportError: cannot import name 'foo' from 'startup_tray'

tests/ui/test_example.py:4: ImportError
=========================== short test summary info ============================
FAILED tests/ui/test_example.py::test_boom - ImportError: cannot import name 'foo'
============================== 1 failed in 0.05s ===============================
Exit status 1
"""

CLEAN_SUCCESS_LOG = """\
============================= test session starts ==============================
collected 2 items

tests/ui/test_example.py::test_ok PASSED                                 [ 50%]
tests/ui/test_example.py::test_ok2 PASSED                                [100%]

============================== 2 passed in 0.02s ===============================
"""


@pytest.fixture(autouse=True)
def _reset_shared(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    watchdog.reset_shared_watchdog_for_tests()
    monkeypatch.setattr(watchdog, "_pref_path", lambda: tmp_path / "shell_watchdog.json")
    yield
    watchdog.reset_shared_watchdog_for_tests()


def test_error_stream_triggers_notification_payload() -> None:
    payloads: list[dict] = []

    def _capture_notify(title: str, message: str) -> bool:
        payloads.append({"title": title, "message": message})
        return True

    plans: list[tuple[str, str]] = []

    def _fake_planner(trace: str, summary: str) -> dict:
        plans.append((trace, summary))
        return {"status": "planned", "intended_goal": summary}

    events: list[tuple[str, str]] = []

    def _on_error(trace: str, summary: str) -> None:
        events.append((trace, summary))
        notifications.notify_shell_error(
            trace,
            summary,
            notify=_capture_notify,
            submit=_fake_planner,
        )

    wd = ShellWatchdog(enabled=True, on_error=_on_error)
    emitted = wd.process_buffer(PYTEST_FAIL_STDERR)

    assert emitted, "failing pytest stderr must emit at least one error event"
    assert events, "on_error callback must fire"
    trace, summary = events[0]
    assert "ImportError" in trace or "ImportError" in summary
    assert len(trace.splitlines()) <= 15
    assert payloads, "notification payload must be produced"
    assert payloads[0]["title"] == notifications.WATCHDOG_TOAST_TITLE
    assert "ImportError" in payloads[0]["message"] or "detected" in payloads[0]["message"]
    assert plans, "planner handoff hook must be invoked"
    assert "ImportError" in plans[0][0] or plans[0][1]


def test_clean_output_remains_silent() -> None:
    events: list[tuple[str, str]] = []
    wd = ShellWatchdog(enabled=True, on_error=lambda t, s: events.append((t, s)))
    emitted = wd.process_buffer(CLEAN_SUCCESS_LOG)
    assert emitted == []
    assert events == []


def test_disabled_watchdog_is_silent() -> None:
    events: list[tuple[str, str]] = []
    wd = ShellWatchdog(enabled=False, on_error=lambda t, s: events.append((t, s)))
    emitted = wd.process_buffer(PYTEST_FAIL_STDERR)
    assert emitted == []
    assert events == []


def test_traceback_window_and_summary() -> None:
    lines = [
        "running startup_tray.py",
        "Traceback (most recent call last):",
        '  File "startup_tray.py", line 10, in <module>',
        "    import missing_mod",
        "ModuleNotFoundError: No module named 'missing_mod'",
        "Exit status 1",
    ]
    # Match on ModuleNotFoundError line (index 4).
    window = extract_trace_window(lines, 4, window=15)
    assert "ModuleNotFoundError" in window
    assert len(window.splitlines()) <= 15
    summary = summarize_error(window, lines[4])
    assert "ModuleNotFoundError" in summary
    assert "startup_tray.py" in summary


def test_feed_line_streaming() -> None:
    events: list[tuple[str, str]] = []
    wd = ShellWatchdog(enabled=True, on_error=lambda t, s: events.append((t, s)))
    wd.feed_line("ok so far")
    assert events == []
    wd.feed_line("ImportError: cannot import name 'x' from 'startup_tray.py'")
    assert events, "ImportError line must trigger"
    assert "ImportError" in events[0][1]


def test_pref_toggle_default_off(tmp_path: Path) -> None:
    assert watchdog.is_shell_watchdog_enabled() is False
    watchdog.set_shell_watchdog_enabled(True)
    assert watchdog.is_shell_watchdog_enabled() is True
    assert (tmp_path / "shell_watchdog.json").is_file()
    watchdog.set_shell_watchdog_enabled(False)
    assert watchdog.is_shell_watchdog_enabled() is False


def test_tray_toggle_updates_shared(monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import MagicMock

    wd = watchdog.get_shared_watchdog()
    assert wd.enabled is False
    icon = MagicMock()
    watchdog.toggle_shell_watchdog(icon, None)
    assert watchdog.is_shell_watchdog_enabled() is True
    assert wd.enabled is True
    icon.update_menu.assert_called_once()
    watchdog.toggle_shell_watchdog(icon, None)
    assert watchdog.is_shell_watchdog_enabled() is False


def test_notify_shell_error_injectable() -> None:
    seen: list[str] = []
    out = notifications.notify_shell_error(
        "Traceback\nImportError: boom",
        "ImportError detected in startup_tray.py",
        notify=lambda t, m: seen.append(m) or True,
        submit=lambda tr, s: {"ok": True, "summary": s},
    )
    assert out["title"] == "Dānā Shell Watchdog"
    assert out["notified"] is True
    assert out["plan"] == {
        "ok": True,
        "summary": "ImportError detected in startup_tray.py",
    }
    assert seen and "ImportError" in seen[0]


def test_submit_to_planner_resilient(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_a, **_k):  # noqa: ANN001
        raise RuntimeError("no planner")

    monkeypatch.setattr(
        "donna.agentic_planning.build_structured_plan",
        _boom,
        raising=False,
    )
    # Import path may fail before attribute set — still must not raise.
    result = notifications.submit_to_planner("trace", "summary")
    # Either None (caught) or a plan if real planner was used before patch.
    assert result is None or isinstance(result, dict)
