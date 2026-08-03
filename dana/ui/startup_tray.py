"""System-tray startup toggle helpers (pystray checked-menu binding)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Packaged icon / logo candidates (resolved via ``get_resource_path``).
_TRAY_ASSET_RELS = (
    "dana/assets/dana_icon.ico",
    "dana/ui/assets/dana_icon.png",
    "dana/assets/donna.ico",
    "dana/ui/assets/dana_logo_highres.png",
    "dana/ui/assets/orb_logo.png",
    "dana/assets/orb_logo.png",
    "dana/ui/assets/donna_logo_highres.png",
)


def resolve_tray_asset_path() -> Path | None:
    """Return the first existing tray .ico/.png via MEIPASS-aware lookup."""
    try:
        from dana.resources import get_resource_path
    except Exception:  # noqa: BLE001
        return None
    for rel in _TRAY_ASSET_RELS:
        try:
            path = get_resource_path(rel)
        except Exception:  # noqa: BLE001
            continue
        if path.is_file():
            return path
    return None


def tray_icon_image(size: tuple[int, int] = (32, 32)) -> Any | None:
    """Clean RGBA tray glyph via ``get_tray_icon`` (transparent backdrop)."""
    try:
        from dana.ui.logo import get_tray_icon

        img = get_tray_icon(size=size)
        if img is not None:
            return img
    except Exception:  # noqa: BLE001
        pass
    return _load_tray_asset_pil(size)


def _load_tray_asset_pil(size: tuple[int, int]) -> Any | None:
    """Fallback: load .ico/.png through ``get_resource_path`` + alpha mask."""
    path = resolve_tray_asset_path()
    if path is None:
        return None
    try:
        from PIL import Image

        from dana.ui.logo import make_transparent_logo

        img = Image.open(path).convert("RGBA")
        w, h = int(size[0]), int(size[1])
        if w > 0 and h > 0 and img.size != (w, h):
            img = img.resize((w, h), Image.Resampling.LANCZOS)
        return make_transparent_logo(img)
    except Exception:  # noqa: BLE001
        return None


def build_tray_image(mode: str = "idle", size: int = 64) -> Any | None:
    """Branded pystray image with optional listening pip; RGBA transparent bg."""
    try:
        from PIL import Image, ImageDraw

        from dana.ui.logo import get_tray_icon
    except Exception:  # noqa: BLE001
        return None

    logo = None
    try:
        logo = get_tray_icon(size=(size, size))
    except Exception:  # noqa: BLE001
        logo = None
    if logo is None:
        logo = _load_tray_asset_pil((size, size))
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
