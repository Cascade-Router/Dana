"""Stage 8.9.4 — Dashboard tab order, Settings merge, transcript tags."""

from __future__ import annotations


def test_tab_order_and_dashboard_widgets() -> None:
    from dana.core_agent import DonnaGUI

    app = DonnaGUI()
    try:
        tabs = app._tabs
        names = list(getattr(tabs, "_tab_dict", {}).keys())
        assert names == [
            "Dashboard",
            "Dictation",
            "Behavior",
            "Live Trace",
            "Settings",
        ], names
        assert "Transcript" not in names
        assert "Audio" not in names
        assert "Stats" not in names

        assert tabs.get() == "Dashboard"
        assert app.transcript_box is not None
        assert app.status_value is not None
        assert app.mic_menu is not None
        assert app.speaker_menu is not None
        assert hasattr(app, "dictation_btn")
        assert hasattr(app, "_behavior_sliders")

        tk_text = app._transcript_tk()
        assert tk_text is not None
        configured = set(tk_text.tag_names())
        for tag in ("jason", "llama", "deepseek", "vision", "typist", "default"):
            assert tag in configured

        # Name-based tab switch (not fragile index bindings).
        app._select_tab("Behavior")
        assert tabs.get() == "Behavior"
        app._select_tab("Dashboard")
        assert tabs.get() == "Dashboard"
    finally:
        try:
            app.destroy()
        except Exception:  # noqa: BLE001
            pass


def test_behavior_lock_survives_tab_reorder() -> None:
    from dana.core_agent import DonnaGUI

    app = DonnaGUI()
    try:
        app._select_tab("Dictation")
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
