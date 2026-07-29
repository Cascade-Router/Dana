"""Transparent logo alpha-masking + cached RGBA asset generators."""

from __future__ import annotations

from PIL import Image, ImageDraw


def test_make_transparent_logo_keys_solid_black_background() -> None:
    from dana.ui.logo import make_transparent_logo

    # Solid black canvas with an opaque white disc — classic keyed logo fixture.
    img = Image.new("RGB", (64, 64), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse((16, 16, 48, 48), fill=(240, 240, 240))

    out = make_transparent_logo(img, dark_threshold=15)
    assert out is not None
    assert out.mode == "RGBA"
    assert out.size == (64, 64)

    corner_a = out.getpixel((0, 0))[3]
    assert corner_a == 0, "solid black background must become fully transparent"

    # Interior of the white mark must remain visible (non-zero alpha).
    center_a = out.getpixel((32, 32))[3]
    assert center_a > 200


def test_make_transparent_logo_preserves_existing_alpha() -> None:
    from dana.ui.logo import make_transparent_logo

    img = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse((6, 6, 26, 26), fill=(30, 144, 255, 255))

    out = make_transparent_logo(img, dark_threshold=15)
    assert out.mode == "RGBA"
    assert out.getpixel((0, 0))[3] == 0
    assert out.getpixel((16, 16))[3] > 200


def test_cached_generators_return_rgba_pil() -> None:
    from dana.ui.logo import get_toast_logo, get_tray_icon

    tray = get_tray_icon((32, 32))
    assert tray is not None
    assert tray.mode == "RGBA"
    assert tray.size == (32, 32)

    toast = get_toast_logo((64, 64))
    assert toast is not None
    assert toast.mode == "RGBA"
    assert toast.size == (64, 64)


def test_get_overlay_logo_rgba_path() -> None:
    """CTkImage when available; otherwise keyed PIL RGBA (headless-safe)."""
    from dana.ui.logo import get_overlay_logo

    # Bust cache kind so this test controls the first materialization.
    from dana.ui import logo as logo_mod

    logo_mod._ASSET_CACHE.pop(("overlay", 48, 48), None)  # noqa: SLF001

    overlay = get_overlay_logo((48, 48))
    assert overlay is not None

    try:
        import customtkinter as ctk

        if isinstance(overlay, ctk.CTkImage):
            assert tuple(overlay._size) == (48, 48)  # noqa: SLF001
            # Underlying light image must stay RGBA.
            light = getattr(overlay, "_light_image", None) or getattr(
                overlay, "light_image", None
            )
            if light is not None:
                assert light.mode == "RGBA"
            return
    except Exception:  # noqa: BLE001
        pass

    # Graceful fallback — always assert the PIL RGBA path.
    assert getattr(overlay, "mode", None) == "RGBA"
    assert overlay.size == (48, 48)
