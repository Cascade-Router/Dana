"""GUI tabs — Assistant / Perception / Memory & Settings."""

from __future__ import annotations

import pytest


def _make_gui():
    try:
        from dana.core_agent import DonnaGUI

        return DonnaGUI()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"DonnaGUI/Tk unavailable: {exc}")


def test_tab_order_and_dashboard_widgets() -> None:
    app = _make_gui()
    try:
        tabs = app._tabs
        names = list(getattr(tabs, "_tab_dict", {}).keys())
        assert names == [
            "Assistant & Tasks",
            "Perception",
            "Memory & Settings",
        ], names
        assert "Dashboard" not in names
        assert "Live Trace" not in names
        assert "Dictation" not in names
        assert "Behavior" not in names
        assert "Audio" not in names
        assert "Stats" not in names

        assert tabs.get() == "Assistant & Tasks"
        assert app.transcript_box is not None
        assert app.status_value is not None
        assert app.mic_menu is not None
        assert app.speaker_menu is not None
        assert hasattr(app, "dictation_btn")
        assert hasattr(app, "_behavior_sliders")
        assert getattr(app, "task_tracker_view", None) is not None
        assert "Dana" in str(app.title())
        assert "Donna" not in str(app.title())
        assert "STOP DANA" in str(app.stop_donna_btn.cget("text"))

        tk_text = app._transcript_tk()
        assert tk_text is not None
        configured = set(tk_text.tag_names())
        for tag in ("jason", "llama", "deepseek", "vision", "typist", "default"):
            assert tag in configured

        app._select_tab("Memory & Settings")
        assert tabs.get() == "Memory & Settings"
        app._select_tab("Perception")
        assert tabs.get() == "Perception"
        app._select_tab("Assistant & Tasks")
        assert tabs.get() == "Assistant & Tasks"
    finally:
        try:
            app.destroy()
        except Exception:  # noqa: BLE001
            pass


def test_gui_source_declares_three_tabs() -> None:
    """Headless: assert tab consolidation without requiring a live Tk display."""
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "dana" / "core_agent.py").read_text(
        encoding="utf-8"
    )
    assert 'tabs.add("Assistant & Tasks")' in src
    assert 'tabs.add("Perception")' in src
    assert 'tabs.add("Memory & Settings")' in src
    assert 'tabs.add("Dashboard")' not in src
    assert 'tabs.add("Live Trace")' not in src
    assert 'self.title("Dana — Control Dashboard")' in src
    assert 'text="STOP DANA"' in src
    assert "TaskTrackerView" in src

    app = _make_gui()
    try:
        app._select_tab("Memory & Settings")
        app._set_dictation_ui(True)
        app._set_behavior_controls_locked(True)
        assert app._behavior_locked is True
        hint = app._behavior_lock_hint
        assert hint is not None
        assert "LOCKED" in str(hint.cget("text")).upper()
        app._set_dictation_ui(False)
        app._set_behavior_controls_locked(False)
        assert "unlocked" in str(hint.cget("text")).lower()
    finally:
        try:
            app.destroy()
        except Exception:  # noqa: BLE001
            pass
