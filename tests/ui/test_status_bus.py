"""Headless smoke tests for STATE_CHANGE status bus + labels."""

from __future__ import annotations

import pytest


def test_emit_drain_and_format_labels() -> None:
    from dana.ui.status_bus import (
        StatusEventBus,
        drain_state_changes,
        emit_state_change,
        format_system_status_line,
        friendly_tool_label,
    )

    # Isolate from other tests / leftover process state.
    bus = StatusEventBus()
    StatusEventBus._instance = bus

    assert friendly_tool_label("execute_powershell") == "PowerShell"
    assert format_system_status_line("routing") == "Supervisor Routing..."
    assert (
        format_system_status_line("executing", tool="execute_powershell")
        == "Executing PowerShell..."
    )
    assert format_system_status_line("listening") == "Listening"
    assert format_system_status_line("processing") == "Processing"
    assert format_system_status_line("idle") == "Idle"

    emit_state_change("listening")
    emit_state_change("routing", message="Supervisor Routing...")
    emit_state_change("executing", tool="execute_powershell")
    events = drain_state_changes(max_items=16)
    assert events
    assert events[0]["event"] == "STATE_CHANGE"
    assert events[0]["status"] == "listening"
    assert events[-1]["status"] == "executing"
    assert events[-1]["tool"] == "execute_powershell"

    # Coalesce identical consecutive snapshots.
    emit_state_change("executing", tool="execute_powershell")
    assert drain_state_changes() == []

    emit_state_change("idle")
    assert drain_state_changes()[-1]["status"] == "idle"


def test_donna_gui_has_status_indicators() -> None:
    try:
        from dana.core_agent import DonnaGUI
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"DonnaGUI unavailable: {exc}")

    try:
        app = DonnaGUI()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Tk unavailable: {exc}")
    try:
        assert getattr(app, "_vad_mic_lbl", None) is not None
        assert getattr(app, "_system_status_lbl", None) is not None
        from dana.ui.status_bus import emit_state_change

        emit_state_change("executing", tool="execute_powershell")
        app._poll_state_changes()
        # Drain already consumed; apply explicitly for assertion.
        app._apply_state_change(
            {
                "event": "STATE_CHANGE",
                "status": "executing",
                "tool": "execute_powershell",
                "message": "",
            }
        )
        text = str(app._system_status_lbl.cget("text") or "")
        assert "PowerShell" in text
    finally:
        try:
            app.destroy()
        except Exception:  # noqa: BLE001
            pass
