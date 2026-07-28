"""Stage 8.9.9 — high-fidelity LANCZOS logo pipeline."""

from __future__ import annotations


def test_load_premium_logo_lanczos_ctkimage() -> None:
    from dana.ui.logo import (
        load_premium_logo,
        load_premium_logo_pil,
        resolve_logo_path,
        ui_assets_dir,
    )

    assert ui_assets_dir().is_dir()
    path = resolve_logo_path()
    assert path is not None
    assert path.name in {
        "dana_logo_highres.png",
        "donna_logo_highres.png",
        "donna_logo.png",
        "orb_logo.png",
    }

    pil = load_premium_logo_pil((48, 48))
    assert pil is not None
    assert pil.size == (48, 48)

    ctk_img = load_premium_logo((36, 36))
    assert ctk_img is not None
    # CTkImage stores the requested size.
    assert tuple(ctk_img._size) == (36, 36)  # noqa: SLF001


def test_gui_header_and_dashboard_use_ctkimage() -> None:
    from dana.core_agent import DonnaGUI

    app = DonnaGUI()
    try:
        assert app._header_logo_img is not None
        assert app._header_logo_lbl is not None
        assert str(app._header_logo_lbl.cget("text")) == ""
        assert app._dash_logo_img is not None
    finally:
        try:
            app.destroy()
        except Exception:  # noqa: BLE001
            pass


def test_orb_uses_png_or_smooth_polygon_not_glyph() -> None:
    import customtkinter as ctk

    from dana.audio.multi_voice_tts import PERSONA_COLORS, set_active_tts_agent
    from dana.ui import assistive_orb as orb_mod
    from dana.ui.assistive_orb import AssistiveTouchOrb, _ICON_SIZE_MAX, _ICON_SIZE_MIN

    # Glyph remnants must be gone.
    assert not hasattr(orb_mod, "_ICON_GLYPH")
    assert not hasattr(orb_mod, "_ICON_FONT_FAMILY")

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
    assert icons
    # Must not be a text glyph item.
    for item in icons:
        assert orb._canvas.type(item) in {"image", "polygon"}

    phase0 = orb._pulse_phase
    orb.pulse_animation()
    assert orb._pulse_phase != phase0
    assert orb._logo_mode in {"png", "polygon"}
    assert orb._accent().lower() == PERSONA_COLORS["jason"].lower()
    assert _ICON_SIZE_MIN <= _ICON_SIZE_MAX

    orb.destroy()
    root.destroy()
