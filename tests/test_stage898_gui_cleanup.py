"""Stage 8.9.8 — header dedupe, accent unification, no footer kill switch."""

from __future__ import annotations


def test_header_pill_only_and_accent() -> None:
    from donna.core_agent import (
        DonnaGUI,
        _UI_ACCENT,
        _TRACE_STATUS_ICONS,
    )

    app = DonnaGUI()
    try:
        # No redundant brand / plain Mode label.
        assert app.mode_dot is None
        assert app.mode_label is None
        assert app.mode_badge is not None
        assert "CHAT" in str(app.mode_badge.cget("text")).upper()

        # Save accent is teal, not default CTk blue.
        assert _UI_ACCENT.lower() in str(app.save_btn.cget("fg_color")).lower()
        assert "#0288d1" not in str(app.save_btn.cget("fg_color")).lower()
        # Behavior sliders use the unified accent.
        slider = next(iter(app._behavior_sliders.values()), None)
        assert slider is not None
        assert _UI_ACCENT.lower() in str(slider.cget("progress_color")).lower()

        # Trace icons are ASCII (no emoji tofu).
        assert _TRACE_STATUS_ICONS["active"] == "[~]"
        assert "✅" not in _TRACE_STATUS_ICONS.values()

        # Behavior standby hint is neutral gray.
        assert app._behavior_lock_hint is not None
        assert "#888888" in str(app._behavior_lock_hint.cget("text_color")).lower()
    finally:
        try:
            app.destroy()
        except Exception:  # noqa: BLE001
            pass
