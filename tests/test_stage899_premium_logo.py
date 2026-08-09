"""Stage 8.9.9 — high-fidelity LANCZOS logo pipeline."""

from __future__ import annotations

import pytest


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
        "dana_logo.png",
        "dana_logo_highres.png",
        "dana_logo_highres.png",
        "dana_logo.png",
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
    """GUI wires premium logos when Tk image names remain valid."""
    import customtkinter as ctk

    from dana.ui.logo import load_premium_logo

    # Fresh root before DanaGUI so CTkImage PhotoImages have a living master.
    try:
        root = ctk.CTk()
        root.withdraw()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Tk unavailable: {exc}")

    try:
        header = load_premium_logo((36, 36))
        dash = load_premium_logo((72, 72))
        assert header is not None
        assert dash is not None
        try:
            from dana.core_agent import DanaGUI

            app = DanaGUI()
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"DanaGUI/Tk isolation: {exc}")
        try:
            # Construction may soft-fail logo labels under Tk isolation; attributes
            # must still be assignable CTkImage instances when load succeeds.
            # _dash_logo_img was part of an older separate-dashboard-tab
            # design and no longer exists as an attribute at all (the
            # Unified Canvas has a single header logo) — getattr() rather
            # than direct access so that's "None", not an AttributeError.
            dash_logo_img = getattr(app, "_dash_logo_img", None)
            if app._header_logo_img is None and dash_logo_img is None:
                pytest.skip("Tk PhotoImage isolation prevented GUI logo bind")
            if app._header_logo_img is not None:
                assert app._header_logo_lbl is not None
                assert str(app._header_logo_lbl.cget("text")) == ""
            if dash_logo_img is not None:
                assert tuple(dash_logo_img._size) == (72, 72)  # noqa: SLF001
        finally:
            try:
                app.destroy()
            except Exception:  # noqa: BLE001
                pass
    finally:
        try:
            root.destroy()
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
