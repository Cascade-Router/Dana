"""Stage 8.9.9 — High-fidelity logo loader (PIL LANCZOS → CTkImage / PhotoImage).

RGBA alpha-masking: solid dark / missing-alpha backgrounds are keyed to
transparent with soft edge anti-aliasing for tray, HUD, and toast surfaces.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Preferred filenames under dana/ui/assets/ (first hit wins).
_LOGO_CANDIDATES = (
    "dana_logo_highres.png",
    "orb_logo.png",
    "donna_logo_highres.png",  # legacy fallback
    "donna_logo.png",  # legacy fallback
)

# Cached generators — key: (kind, width, height)
_ASSET_CACHE: dict[tuple[str, int, int], Any] = {}


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
        photo = ImageTk.PhotoImage(
            img.resize((64, 64), Image.Resampling.LANCZOS),
            master=root,
        )
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
                frame = make_transparent_logo(img.convert("RGBA"))
                if frame.size == target:
                    return frame
                best = frame
            if best is not None:
                return make_transparent_logo(
                    best.resize(target, Image.Resampling.LANCZOS)
                )
            return make_transparent_logo(
                img.convert("RGBA").resize(target, Image.Resampling.LANCZOS)
            )
        return load_premium_logo_pil(size)
    except Exception:  # noqa: BLE001
        return None


def _load_pil(path: Path):
    from PIL import Image

    return Image.open(path).convert("RGBA")


def make_transparent_logo(pil_image: Any, dark_threshold: int = 15) -> Any:
    """Key solid dark backgrounds to full transparency with soft edge AA.

    When the source is missing meaningful alpha (fully opaque) or samples as a
    solid black backdrop, pixels with RGB channels below ``dark_threshold``
    become ``RGBA(0, 0, 0, 0)``. A short luminance ramp plus a light Gaussian
    blur on the alpha channel softens the cutout edge.
    """
    from PIL import Image, ImageFilter

    if pil_image is None:
        return None
    base = pil_image.convert("RGBA")
    alpha = base.getchannel("A")
    try:
        a_min, _a_max = alpha.getextrema()
    except Exception:  # noqa: BLE001
        a_min = 255
    missing_alpha = int(a_min) >= 250

    px = base.load()
    w, h = base.size
    corners = (
        px[0, 0],
        px[w - 1, 0],
        px[0, h - 1],
        px[w - 1, h - 1],
    )
    thr = max(0, int(dark_threshold))
    # Opaque near-black corners only — already-transparent corners must not re-key.
    solid_black_bg = all(
        int(c[0]) < thr
        and int(c[1]) < thr
        and int(c[2]) < thr
        and int(c[3]) >= 250
        for c in corners
    )
    if not missing_alpha and not solid_black_bg:
        return base

    fade = max(8, thr)
    out_pixels: list[tuple[int, int, int, int]] = []
    for r, g, b, a in base.getdata():
        mx = max(int(r), int(g), int(b))
        if mx <= thr:
            out_pixels.append((0, 0, 0, 0))
            continue
        if mx < thr + fade:
            t = (mx - thr) / float(fade)
            out_pixels.append((int(r), int(g), int(b), int(round(int(a) * t))))
            continue
        out_pixels.append((int(r), int(g), int(b), int(a)))

    out = Image.new("RGBA", base.size)
    out.putdata(out_pixels)
    # Soften keyed edges without destroying already-transparent interiors.
    soft_a = out.getchannel("A").filter(ImageFilter.GaussianBlur(radius=0.65))
    out.putalpha(soft_a)
    return out


def _rgba_source(size: tuple[int, int]) -> Any | None:
    """Load app icon or premium PNG as sized RGBA with dark-bg keying."""
    img = load_app_icon_pil(size)
    if img is None:
        img = load_premium_logo_pil(size)
    if img is None:
        return None
    return make_transparent_logo(img.convert("RGBA"))


def get_tray_icon(size: tuple[int, int] = (32, 32)) -> Any | None:
    """Cached clean RGBA PIL image for pystray (transparent backdrop)."""
    w, h = int(size[0]), int(size[1])
    key = ("tray", w, h)
    cached = _ASSET_CACHE.get(key)
    if cached is not None:
        try:
            return cached.copy()
        except Exception:  # noqa: BLE001
            return cached
    img = _rgba_source((w, h))
    if img is None:
        return None
    _ASSET_CACHE[key] = img
    try:
        return img.copy()
    except Exception:  # noqa: BLE001
        return img


def get_overlay_logo(size: tuple[int, int] = (48, 48)) -> Any | None:
    """Cached RGBA ``CTkImage`` for the floating HUD (PIL fallback if CTk missing)."""
    w, h = int(size[0]), int(size[1])
    key = ("overlay", w, h)
    cached = _ASSET_CACHE.get(key)
    if cached is not None:
        return cached
    img = _rgba_source((w, h))
    if img is None:
        return None
    try:
        import tkinter as tk

        import customtkinter as ctk

        if getattr(tk, "_default_root", None) is None:
            sentinel = ctk.CTk()
            sentinel.withdraw()
            tk._donna_logo_sentinel = sentinel  # type: ignore[attr-defined]
        ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(w, h))
        _ASSET_CACHE[key] = ctk_img
        return ctk_img
    except Exception:  # noqa: BLE001
        # Headless / missing customtkinter — return keyed PIL RGBA.
        _ASSET_CACHE[key] = img
        return img


def get_toast_logo(size: tuple[int, int] = (64, 64)) -> Any | None:
    """Cached anti-aliased RGBA PIL logo for Windows notification icons."""
    w, h = int(size[0]), int(size[1])
    key = ("toast", w, h)
    cached = _ASSET_CACHE.get(key)
    if cached is not None:
        try:
            return cached.copy()
        except Exception:  # noqa: BLE001
            return cached
    img = _rgba_source((w, h))
    if img is None:
        return None
    # Extra light AA pass for toast downscales.
    try:
        from PIL import ImageFilter

        soft = img.copy()
        soft.putalpha(soft.getchannel("A").filter(ImageFilter.GaussianBlur(radius=0.35)))
        img = soft
    except Exception:  # noqa: BLE001
        pass
    _ASSET_CACHE[key] = img
    try:
        return img.copy()
    except Exception:  # noqa: BLE001
        return img


def toast_logo_path(size: tuple[int, int] = (64, 64)) -> Path | None:
    """Persist ``get_toast_logo`` to a temp PNG for APIs that need a file path."""
    img = get_toast_logo(size)
    if img is None:
        return None
    try:
        import tempfile

        cache_key = ("toast_path", int(size[0]), int(size[1]))
        cached = _ASSET_CACHE.get(cache_key)
        if isinstance(cached, Path) and cached.is_file():
            return cached
        fd, name = tempfile.mkstemp(prefix="dana_toast_", suffix=".png")
        import os

        os.close(fd)
        path = Path(name)
        img.save(path, format="PNG")
        _ASSET_CACHE[cache_key] = path
        return path
    except Exception:  # noqa: BLE001
        return None


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

        img = make_transparent_logo(_load_pil(path))
        w, h = int(size[0]), int(size[1])
        if w < 1 or h < 1:
            return None
        img = img.resize((w, h), Image.Resampling.LANCZOS)
        # Re-key after downscale so LANCZOS fringe blacks stay transparent.
        img = make_transparent_logo(img)
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
        import tkinter as tk

        import customtkinter as ctk

        # CTkImage builds PhotoImage against the default root. Keep a withdrawn
        # sentinel root alive so images survive across DonnaGUI destroy cycles.
        if getattr(tk, "_default_root", None) is None:
            sentinel = ctk.CTk()
            sentinel.withdraw()
            tk._donna_logo_sentinel = sentinel  # type: ignore[attr-defined]
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
