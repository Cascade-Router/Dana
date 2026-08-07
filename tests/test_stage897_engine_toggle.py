"""Stage 8.9.7 — Engine ENGAGE / STANDBY toggle + Behavior lock."""

from __future__ import annotations


def test_engine_engage_standby_locks_behavior() -> None:
    from dana.core_agent import (
        DonnaGUI,
        is_engine_engaged,
        set_engine_engaged,
    )

    set_engine_engaged(False)
    app = DonnaGUI()
    try:
        # DonnaGUI now auto-engages the engine during construction
        # (engage_engine() runs unconditionally from _build_unified_canvas),
        # so a fresh instance is never left in STANDBY — establish that
        # starting state explicitly before testing the toggle transitions.
        app.standby_engine()
        assert app.engine_active is False
        assert is_engine_engaged() is False
        # app._engage_btn / _standby_btn: standby was merged into the single
        # engage/standby toggle button (see core_agent.py: "legacy; merged
        # into toggle"); _engine_status_lbl is likewise now None — its
        # ACTIVE/STANDBY/STOPPED text moved to the compact header HUD label
        # (_header_status_lbl), which drops the "Local Engine" suffix.
        assert app._engage_btn is not None
        assert "STANDBY" in str(app._header_status_lbl.cget("text"))

        # Sliders editable in STANDBY.
        app._set_behavior_controls_locked(False)
        assert app._behavior_locked is False

        app.engage_engine()
        assert app.engine_active is True
        assert is_engine_engaged() is True
        assert app._behavior_locked is True
        assert "ACTIVE" in str(app._header_status_lbl.cget("text"))

        app.standby_engine()
        assert app.engine_active is False
        assert is_engine_engaged() is False
        assert app._behavior_locked is False
        assert "STANDBY" in str(app._header_status_lbl.cget("text"))
    finally:
        set_engine_engaged(False)
        try:
            app.destroy()
        except Exception:  # noqa: BLE001
            pass


def test_standby_blocks_chat_and_dictation() -> None:
    from dana.core_agent import DonnaGUI, set_engine_engaged

    set_engine_engaged(False)
    app = DonnaGUI()
    try:
        # DonnaGUI auto-engages on construction — force STANDBY first so
        # _require_engine() below has something to actually refuse.
        app.standby_engine()
        # _engine_warn_lbl is intentionally None now (see core_agent.py) and
        # _flash_engine_warning() no-ops without it — no toast widget
        # replaced it. The real, current guarantee is that a STANDBY task
        # attempt is refused and routed back to the Assistant tab.
        assert app._require_engine() is False
        assert app._tabs.get() == "Assistant & Tasks"

        # Dictation ON blocked in STANDBY.
        app._dictation_active = False
        app._toggle_dictation_mode()
        assert app._dictation_active is False

        app.engage_engine()
        assert app._require_engine() is True
        app._dashboard_start_chat()  # should not warn when engaged
    finally:
        set_engine_engaged(False)
        try:
            app.destroy()
        except Exception:  # noqa: BLE001
            pass


def test_stop_donna_still_independent() -> None:
    """Kill switch remains a separate hard-exit path."""
    from dana.core_agent import DonnaGUI, set_engine_engaged

    set_engine_engaged(False)
    app = DonnaGUI()
    try:
        assert hasattr(app, "kill_donna_processes")
        assert hasattr(app, "_on_stop_donna_clicked")
        assert app.stop_donna_btn is not None
        assert "STOP" in str(app.stop_donna_btn.cget("text")).upper()
        # Soft STANDBY must not clear the kill button.
        app.engage_engine()
        app.standby_engine()
        assert app.stop_donna_btn.winfo_exists()
    finally:
        set_engine_engaged(False)
        try:
            app.destroy()
        except Exception:  # noqa: BLE001
            pass


def test_stop_callback_halts_engine(monkeypatch) -> None:
    """STOP DANA must deactivate the engine even when kill-script is stubbed."""
    from dana.core_agent import (
        DonnaGUI,
        is_engine_engaged,
        set_engine_engaged,
        stop_event,
    )

    set_engine_engaged(False)
    stop_event.clear()
    app = DonnaGUI()
    try:
        app.engage_engine()
        assert app.engine_active is True
        assert is_engine_engaged() is True

        monkeypatch.setattr(
            app,
            "kill_donna_processes",
            lambda: {"ok": True, "pid": 1, "path": "stub"},
        )
        # Avoid deferred Popen; halt must be synchronous in the click handler.
        monkeypatch.setattr(app, "after", lambda *_a, **_k: None)

        app._on_stop_donna_clicked()

        assert app.engine_active is False
        assert is_engine_engaged() is False
        assert bool(getattr(app, "_engine_stopped", False)) is True
        pill = str(app._header_status_lbl.cget("text"))
        assert "STOPPED" in pill
        assert stop_event.is_set()
    finally:
        set_engine_engaged(False)
        stop_event.clear()
        try:
            from dana.middleware.kill_switch import clear_global_halt

            clear_global_halt()
        except Exception:  # noqa: BLE001
            pass
        try:
            app.destroy()
        except Exception:  # noqa: BLE001
            pass
