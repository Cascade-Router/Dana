"""System-tray startup toggle helpers (pystray checked-menu binding)."""

from __future__ import annotations

from typing import Any


def tray_icon_image(size: tuple[int, int] = (32, 32)) -> Any | None:
    """Clean RGBA tray glyph via ``get_tray_icon`` (transparent backdrop)."""
    try:
        from dana.ui.logo import get_tray_icon

        return get_tray_icon(size=size)
    except Exception:  # noqa: BLE001
        return None


def build_tray_image(mode: str = "idle", size: int = 64) -> Any | None:
    """Branded pystray image with optional listening pip; RGBA transparent bg."""
    try:
        from PIL import Image, ImageDraw

        from dana.ui.logo import get_tray_icon
    except Exception:  # noqa: BLE001
        return None

    logo = get_tray_icon(size=(size, size))
    if logo is None:
        return None
    try:
        img = logo.convert("RGBA")
    except Exception:  # noqa: BLE001
        img = logo
    if mode == "listening":
        try:
            draw = ImageDraw.Draw(img)
            # Scale pip with icon size (defaults match legacy 64px tray art).
            s = max(16, int(size))
            x0 = int(round(s * 42 / 64))
            y0 = int(round(s * 8 / 64))
            x1 = int(round(s * 56 / 64))
            y1 = int(round(s * 22 / 64))
            ix0 = int(round(s * 45 / 64))
            iy0 = int(round(s * 11 / 64))
            ix1 = int(round(s * 53 / 64))
            iy1 = int(round(s * 19 / 64))
            draw.ellipse((x0, y0, x1, y1), fill=(250, 250, 250, 255))
            draw.ellipse((ix0, iy0, ix1, iy1), fill=(34, 197, 94, 255))
        except Exception:  # noqa: BLE001
            pass
    return img


def check_startup_registry_status(_item: Any = None) -> bool:
    """Return True when Dānā is registered for login/startup (quiet)."""
    try:
        from dana.tools.setup_startup import is_startup_enabled

        return bool(is_startup_enabled())
    except Exception:  # noqa: BLE001
        return False


def toggle_run_on_startup(icon: Any = None, _item: Any = None) -> None:
    """Toggle OS login/startup registration and refresh the tray checkmark."""
    try:
        from dana.tools.setup_startup import (
            disable_startup,
            enable_startup,
            is_startup_enabled,
        )
    except Exception:  # noqa: BLE001
        return

    try:
        if is_startup_enabled():
            disable_startup()
        else:
            enable_startup()
    except Exception:  # noqa: BLE001
        return

    # Force pystray to re-evaluate ``checked=`` on next menu open.
    if icon is not None:
        try:
            update = getattr(icon, "update_menu", None)
            if callable(update):
                update()
        except Exception:  # noqa: BLE001
            pass
