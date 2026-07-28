"""Stage 8.9.9 — High-fidelity logo loader (PIL LANCZOS → CTkImage / PhotoImage)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Preferred filenames under dana/ui/assets/ (first hit wins).
_LOGO_CANDIDATES = (
    "dana_logo_highres.png",
    "donna_logo_highres.png",
    "donna_logo.png",
    "orb_logo.png",
)


def ui_assets_dir() -> Path:
    """``dana/ui/assets`` — drop artist-rendered PNG/SVG exports here."""
    return Path(__file__).resolve().parent / "assets"


def resolve_logo_path() -> Path | None:
    """Return the first existing premium logo path, or None."""
    root = ui_assets_dir()
    for name in _LOGO_CANDIDATES:
        path = root / name
        if path.is_file():
            return path
    return None


def app_icon_path() -> Path:
    """Canonical Windows ``.ico`` path (``dana/assets/donna.ico``).

    Resolves via ``dana.paths.PROJECT_ROOT`` so packaged + repo launches agree.
    """
    try:
        from dana.paths import PROJECT_ROOT

        return Path(PROJECT_ROOT) / "dana" / "assets" / "donna.ico"
    except Exception:  # noqa: BLE001
        return Path(__file__).resolve().parents[1] / "assets" / "donna.ico"


def resolve_app_icon_path() -> Path | None:
    """Return ``dana/assets/donna.ico`` when present."""
    path = app_icon_path()
    return path if path.is_file() else None


def apply_window_icon(root: Any) -> bool:
    """Bind ``donna.ico`` to a Tk / CustomTkinter root for taskbar + title bar.

    Returns True when an icon was applied. Safe no-op on failure / non-Windows.
    """
    ico = resolve_app_icon_path()
    if ico is None:
        return False
    ico_s = str(ico)
    applied = False
    try:
        root.iconbitmap(ico_s)
        applied = True
    except Exception:  # noqa: BLE001
        pass
    try:
        root.wm_iconbitmap(ico_s)
        applied = True
    except Exception:  # noqa: BLE001
        pass
    # iconphoto helps some CTk / multi-monitor taskbar hosts prefer our bitmap.
    try:
        from PIL import Image, ImageTk

        img = Image.open(ico).convert("RGBA")
        photo = ImageTk.PhotoImage(img.resize((64, 64), Image.Resampling.LANCZOS))
        root.iconphoto(True, photo)
        # Keep a reference so Tk GC does not drop the PhotoImage.
        root._donna_iconphoto = photo  # type: ignore[attr-defined]
        applied = True
    except Exception:  # noqa: BLE001
        pass
    return applied


def load_app_icon_pil(size: tuple[int, int] = (64, 64)) -> Any | None:
    """Load the multi-resolution app ``.ico`` (or PNG fallback) as RGBA PIL image."""
    try:
        from PIL import Image
    except Exception:  # noqa: BLE001
        return None
    ico = resolve_app_icon_path()
    try:
        if ico is not None:
            img = Image.open(ico)
            # Prefer an exact frame when the ICO embeds multiple sizes.
            target = (int(size[0]), int(size[1]))
            best = None
            try:
                n = int(getattr(img, "n_frames", 1) or 1)
            except Exception:  # noqa: BLE001
                n = 1
            for i in range(max(1, n)):
                try:
                    img.seek(i)
                except EOFError:
                    break
                frame = img.convert("RGBA")
                if frame.size == target:
                    return frame
                best = frame
            if best is not None:
                return best.resize(target, Image.Resampling.LANCZOS)
            return img.convert("RGBA").resize(target, Image.Resampling.LANCZOS)
        return load_premium_logo_pil(size)
    except Exception:  # noqa: BLE001
        return None


def _load_pil(path: Path):
    from PIL import Image

    return Image.open(path).convert("RGBA")


def load_premium_logo_pil(
    size: tuple[int, int],
    *,
    tint: str | None = None,
) -> Any | None:
    """Load + LANCZOS-resize the premium logo; optional hex tint for orb pulses."""
    path = resolve_logo_path()
    if path is None:
        return None
    try:
        from PIL import Image, ImageEnhance

        img = _load_pil(path)
        w, h = int(size[0]), int(size[1])
        if w < 1 or h < 1:
            return None
        img = img.resize((w, h), Image.Resampling.LANCZOS)
        if tint:
            img = _tint_rgba(img, tint)
        # Mild contrast keep edges crisp on dark chrome.
        try:
            img = ImageEnhance.Sharpness(img).enhance(1.05)
        except Exception:  # noqa: BLE001
            pass
        return img
    except Exception:  # noqa: BLE001
        return None


def _tint_rgba(img: Any, hex_color: str) -> Any:
    """Recolor opaque pixels to ``hex_color`` while preserving alpha edges."""
    from PIL import Image

    color = (hex_color or "").strip().lstrip("#")
    if len(color) != 6:
        return img
    try:
        rgb = (int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16))
    except ValueError:
        return img
    base = img.convert("RGBA")
    alpha = base.getchannel("A")
    solid = Image.new("RGBA", base.size, (*rgb, 255))
    solid.putalpha(alpha)
    return solid


def load_premium_logo(size: tuple[int, int]) -> Any | None:
    """Return a ``CTkImage`` sized with LANCZOS, or None if asset missing."""
    img = load_premium_logo_pil(size)
    if img is None:
        return None
    try:
        import customtkinter as ctk

        return ctk.CTkImage(light_image=img, dark_image=img, size=size)
    except Exception:  # noqa: BLE001
        return None


def load_premium_logo_photoimage(
    master: Any,
    size: tuple[int, int],
    *,
    tint: str | None = None,
) -> Any | None:
    """Tk ``PhotoImage`` for Canvas (Assistive Orb) — LANCZOS scaled + tinted."""
    img = load_premium_logo_pil(size, tint=tint)
    if img is None:
        return None
    try:
        from PIL import ImageTk

        return ImageTk.PhotoImage(img, master=master)
    except Exception:  # noqa: BLE001
        return None
