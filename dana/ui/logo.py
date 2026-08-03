"""Stage 8.9.9 — High-fidelity logo loader (PIL LANCZOS → CTkImage / PhotoImage).

RGBA alpha-masking: solid dark / missing-alpha backgrounds are keyed to
transparent with soft edge anti-aliasing for tray, HUD, and toast surfaces.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Preferred filenames under dana/ui/assets/ (first hit wins).
_LOGO_CANDIDATES = (
    "dana_logo_highres.png",
    "orb_logo.png",
    "donna_logo_highres.png",  # legacy fallback
    "donna_logo.png",  # legacy fallback
)

# Windows HUD / orb chroma-key (must not appear in opaque logo pixels).
_LWA_COLORKEY_RGB = (0, 0, 1)  # #000001
_LWA_COLORKEY_LIFT = (1, 1, 2)  # nearby non-key RGB

# Cached generators — key: (kind, width, height)
_ASSET_CACHE: dict[tuple[str, int, int], Any] = {}


def _frozen_root() -> Path | None:
    """PyInstaller extract root (``sys._MEIPASS``) or onedir exe directory."""
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        try:
            return Path(meipass).resolve()
        except Exception:  # noqa: BLE001
            return Path(str(meipass))
    if bool(getattr(sys, "frozen", False)):
        try:
            return Path(sys.executable).resolve().parent
        except Exception:  # noqa: BLE001
            return None
    return None


def _asset_search_roots() -> list[Path]:
    """Ordered roots for logo / icon discovery (dev + frozen)."""
    roots: list[Path] = []
    try:
        from dana.resources import get_resource_path

        roots.append(get_resource_path("dana/ui/assets"))
        roots.append(get_resource_path("dana/assets"))
    except Exception:  # noqa: BLE001
        pass
    roots.extend(
        [
            Path(__file__).resolve().parent / "assets",  # dana/ui/assets
            Path(__file__).resolve().parents[1] / "assets",  # dana/assets
        ]
    )
    frozen = _frozen_root()
    if frozen is not None:
        roots.extend(
            [
                frozen / "dana" / "ui" / "assets",
                frozen / "dana" / "assets",
                frozen / "ui" / "assets",
                frozen / "assets",
            ]
        )
    out: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        try:
            key = str(root.resolve())
        except Exception:  # noqa: BLE001
            key = str(root)
        if key in seen:
            continue
        seen.add(key)
        out.append(root)
    return out


def ui_assets_dir() -> Path:
    """``dana/ui/assets`` — drop artist-rendered PNG/SVG exports here."""
    try:
        from dana.resources import get_resource_path

        via = get_resource_path("dana/ui/assets")
        if via.is_dir():
            return via
    except Exception:  # noqa: BLE001
        pass
    for root in _asset_search_roots():
        # Prefer the UI assets tree when present (dev or ``--add-data``).
        if root.name == "assets" and root.parent.name == "ui" and root.is_dir():
            return root
    here = Path(__file__).resolve().parent / "assets"
    frozen = _frozen_root()
    if frozen is not None:
        for candidate in (
            frozen / "dana" / "ui" / "assets",
            frozen / "ui" / "assets",
            frozen / "assets",
        ):
            if candidate.is_dir():
                return candidate
    return here


def resolve_logo_path() -> Path | None:
    """Return the first existing premium logo path, or None."""
    try:
        from dana.resources import get_resource_path

        for name in _LOGO_CANDIDATES:
            for rel in (f"dana/ui/assets/{name}", f"dana/assets/{name}"):
                path = get_resource_path(rel)
                if path.is_file():
                    return path
    except Exception:  # noqa: BLE001
        pass
    for root in _asset_search_roots():
        if not root.is_dir():
            continue
        for name in _LOGO_CANDIDATES:
            path = root / name
            if path.is_file():
                return path
    return None


# Preferred Windows app icon filenames (first hit wins).
_APP_ICON_RELS = (
    "dana/assets/dana_icon.ico",
    "dana/assets/donna.ico",  # legacy fallback
)


def app_icon_path() -> Path:
    """Canonical Windows ``.ico`` path (``dana/assets/dana_icon.ico``).

    Resolves via ``dana.resources.get_resource_path`` (``sys._MEIPASS``) so
    packaged + repo launches agree when assets are bundled with ``--add-data``.
    Falls back to legacy ``donna.ico`` when the hyper-mint squircle is absent.
    """
    try:
        from dana.resources import get_resource_path

        for rel in _APP_ICON_RELS:
            via = get_resource_path(rel)
            if via.is_file():
                return via
    except Exception:  # noqa: BLE001
        pass
    asset_dir = Path(__file__).resolve().parents[1] / "assets"
    candidates: list[Path] = [
        asset_dir / "dana_icon.ico",
        asset_dir / "donna.ico",
    ]
    frozen = _frozen_root()
    if frozen is not None:
        candidates.extend(
            [
                frozen / "dana" / "assets" / "dana_icon.ico",
                frozen / "dana" / "assets" / "donna.ico",
                frozen / "assets" / "dana_icon.ico",
                frozen / "assets" / "donna.ico",
            ]
        )
    for path in candidates:
        if path.is_file():
            return path
    try:
        from dana.resources import get_resource_path

        return get_resource_path("dana/assets/dana_icon.ico")
    except Exception:  # noqa: BLE001
        return candidates[0]


def resolve_app_icon_path() -> Path | None:
    """Return ``dana/assets/dana_icon.ico`` (or legacy ``donna.ico``) when present."""
    path = app_icon_path()
    return path if path.is_file() else None


def apply_window_icon(root: Any) -> bool:
    """Bind app ``.ico`` to a Tk / CustomTkinter root for taskbar + title bar.

    Uses ``iconbitmap`` / ``wm_iconbitmap`` / ``iconphoto`` (wm_iconphoto path).
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
    # iconphoto / wm_iconphoto helps some CTk / multi-monitor taskbar hosts.
    try:
        from PIL import Image, ImageTk

        img = Image.open(ico).convert("RGBA")
        photo = ImageTk.PhotoImage(
            img.resize((64, 64), Image.Resampling.LANCZOS),
            master=root,
        )
        try:
            root.wm_iconphoto(True, photo)
        except Exception:  # noqa: BLE001
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


def _scrub_lwa_colorkey(pil_image: Any) -> Any:
    """Lift opaque pixels matching ``#000001`` so LWA_COLORKEY cannot punch holes."""
    from PIL import Image

    if pil_image is None:
        return None
    base = pil_image.convert("RGBA")
    kr, kg, kb = _LWA_COLORKEY_RGB
    lr, lg, lb = _LWA_COLORKEY_LIFT
    out_pixels: list[tuple[int, int, int, int]] = []
    touched = False
    for r, g, b, a in base.getdata():
        if int(a) > 0 and int(r) == kr and int(g) == kg and int(b) == kb:
            out_pixels.append((lr, lg, lb, int(a)))
            touched = True
            continue
        out_pixels.append((int(r), int(g), int(b), int(a)))
    if not touched:
        return base
    out = Image.new("RGBA", base.size)
    out.putdata(out_pixels)
    return out


def make_transparent_logo(pil_image: Any, dark_threshold: int = 12) -> Any:
    """Key solid dark backgrounds to full transparency with soft edge AA.

    When the source is missing meaningful alpha (fully opaque) or samples as a
    solid black backdrop, pixels with RGB channels below ``dark_threshold``
    become ``RGBA(0, 0, 0, 0)``. A short luminance ramp plus a light Gaussian
    blur on the alpha channel softens the cutout edge.

    True-alpha PNGs (preferred for HUD / orb) skip dark keying so near-black
    logo art is preserved; opaque ``#000001`` pixels are scrubbed so Windows
    ``LWA_COLORKEY`` cannot punch holes in the mark.
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
    # Keep threshold tight — aggressive values punch holes in dark logo strokes.
    thr = max(0, min(int(dark_threshold), 15))
    # Opaque near-black corners only — already-transparent corners must not re-key.
    solid_black_bg = all(
        int(c[0]) < thr
        and int(c[1]) < thr
        and int(c[2]) < thr
        and int(c[3]) >= 250
        for c in corners
    )
    if not missing_alpha and not solid_black_bg:
        return _scrub_lwa_colorkey(base)

    fade = max(6, thr)
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
    return _scrub_lwa_colorkey(out)


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
