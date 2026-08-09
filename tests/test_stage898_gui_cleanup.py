"""Stage 8.9.8 — header dedupe, accent unification, no footer kill switch."""

from __future__ import annotations


def test_header_pill_only_and_accent() -> None:
    from dana.core_agent import (
        DanaGUI,
        _UI_ACCENT,
        _TRACE_STATUS_ICONS,
    )

    app = DanaGUI()
    try:
        # No redundant brand / plain Mode label — mode_badge itself was later
        # removed as a further dedupe (see core_agent.py: "removed redundant
        # CHAT badge"); status now lives solely in _header_status_lbl.
        assert app.mode_dot is None
        assert app.mode_label is None
        assert app.mode_badge is None

        # The standalone "Save & Apply" button (and app.save_btn) was removed
        # entirely in a later dedupe — accent unification is still verified
        # below via the behavior sliders (and separately via engage/send/stop
        # buttons in tests/ui/test_theme_chat.py).
        assert app.save_btn is None
        # Behavior sliders use the unified accent.
        slider = next(iter(app._behavior_sliders.values()), None)
        assert slider is not None
        assert _UI_ACCENT.lower() in str(slider.cget("progress_color")).lower()

        # Trace icons are ASCII (no emoji tofu).
        assert _TRACE_STATUS_ICONS["active"] == "[~]"
        assert "✅" not in _TRACE_STATUS_ICONS.values()

        # DanaGUI auto-engages the engine on construction (see
        # engage_engine() called from _build_unified_canvas), which locks
        # the Behavior Mixer immediately — so the hint shows the
        # locked/amber state, not the old standby/gray default.
        assert app._behavior_lock_hint is not None
        assert "#f59e0b" in str(app._behavior_lock_hint.cget("text_color")).lower()
    finally:
        try:
            app.destroy()
        except Exception:  # noqa: BLE001
            pass
