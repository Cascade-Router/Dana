"""Stage 8.9.5/8.9.9 — Orb logo pulse (PNG LANCZOS or smooth polygon)."""

from __future__ import annotations


def test_orb_icon_glyph_and_font_pulse() -> None:
    """Legacy name kept; asserts Stage 8.9.9 logo pipeline (no Unicode glyph)."""
    import customtkinter as ctk

    from dana.audio.multi_voice_tts import PERSONA_COLORS, set_active_tts_agent
    from dana.ui.assistive_orb import (
        AssistiveTouchOrb,
        _ICON_SIZE_MAX,
        _ICON_SIZE_MIN,
    )

    root = ctk.CTk()
    root.withdraw()
    set_active_tts_agent("jason")
    orb = AssistiveTouchOrb(
        root,
        agent_getter=lambda: "jason",
        dictation_getter=lambda: False,
        mode_getter=lambda: "chat",
    )
    root.update_idletasks()
    root.update()

    icons = orb._canvas.find_withtag("icon")
    assert len(icons) >= 1
    assert orb._canvas.type(icons[0]) in {"image", "polygon"}

    orb.pulse_animation()
    root.update_idletasks()
    assert orb._logo_mode in {"png", "polygon"}
    assert orb._accent().lower() == PERSONA_COLORS["jason"].lower()

    cx, cy = orb._icon_center()
    # Image items expose center coords; polygons expose bbox midpoints.
    if orb._canvas.type(icons[0]) == "image":
        coords = orb._canvas.coords(icons[0])
        assert abs(coords[0] - cx) < 0.5
        assert abs(coords[1] - cy) < 0.5
    assert _ICON_SIZE_MIN <= _ICON_SIZE_MAX

    orb.destroy()
    root.destroy()
