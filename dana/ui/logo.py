"""Stage 8.9.9 — High-fidelity logo loader (PIL LANCZOS → CTkImage / PhotoImage).

RGBA alpha-masking: solid dark / missing-alpha backgrounds are keyed to
transparent with soft edge anti-aliasing for tray, HUD, and toast surfaces.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Preferred filenames (first hit wins). Root ``assets/`` is the SSoT.
_LOGO_CANDIDATES = (
    "dana_logo.png",
    "dana_logo_highres.png",  # legacy
    "orb_logo.png",  # legacy
    "dana_logo_highres.png",  # legacy
    "dana_logo.png",  # legacy
)

# Windows HUD / orb chroma-key (must not appear in opaque logo pixels).
_LWA_COLORKEY_RGB = (0, 0, 1)  # #000001
_LWA_COLORKEY_LIFT = (1, 1, 2)  # nearby non-key RGB

# Cached generators — key: (kind, width, height)
_ASSET_CACHE: dict[tuple[Any, ...], Any] = {}

# Disk cache-bust token for Windows shell / explorer icon caches.
_RUNTIME_ICON_TAG = "v2"


def runtime_icon_cache_dir() -> Path:
    """Per-user temp dir for uniquely named runtime icons (not the source assets)."""
    import tempfile

    path = Path(tempfile.gettempdir()) / "dana_icon_cache"
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception:  # noqa: BLE001
        pass
    return path


def purge_runtime_icon_files() -> None:
    """Delete prior runtime/temp logo exports so Windows cannot fall back to them."""
    import tempfile

    patterns = (
        "dana_logo_runtime_*.png",
        "dana_logo_runtime_*.ico",
        "dana_tray_runtime_*.png",
        "dana_tray_runtime_*.ico",
        "dana_toast_*.png",
    )
    roots = [runtime_icon_cache_dir(), Path(tempfile.gettempdir())]
    for root in roots:
        if not root.is_dir():
            continue
        for pattern in patterns:
            try:
                for path in root.glob(pattern):
                    try:
                        path.unlink(missing_ok=True)
                    except Exception:  # noqa: BLE001
                        pass
            except Exception:  # noqa: BLE001
                pass


def invalidate_logo_cache() -> None:
    """Drop in-memory PIL/CTk/PhotoImage logo caches (force full-bleed reload)."""
    _ASSET_CACHE.clear()
    purge_runtime_icon_files()


def export_runtime_icon(
    size: tuple[int, int] = (64, 64),
    *,
    kind: str = "logo",
    fmt: str = "png",
    img: Any | None = None,
) -> Path | None:
    """Write a full-bleed runtime icon to a uniquely named temp file.

    Never reuses a fixed filename — token includes content hash + timestamp so
    Windows Explorer / taskbar caches cannot keep serving a stale glyph.
    """
    import hashlib
    import time

    try:
        from PIL import Image
    except Exception:  # noqa: BLE001
        return None

    w, h = max(1, int(size[0])), max(1, int(size[1]))
    work = img
    if work is None:
        work = load_premium_logo_pil((w, h))
    if work is None:
        work = load_app_icon_pil((w, h))
    if work is None:
        return None
    try:
        work = make_transparent_logo(work.convert("RGBA"))
        if work.size != (w, h):
            work = (
                full_bleed_rgba(work, (w, h), fill_ratio=0.78)
                or work.resize((w, h), Image.Resampling.LANCZOS)
            )
        digest = hashlib.sha1(work.tobytes()).hexdigest()[:12]
        token = f"{_RUNTIME_ICON_TAG}_{digest}_{int(time.time() * 1000)}"
        prefix = "dana_tray_runtime_" if kind == "tray" else "dana_logo_runtime_"
        ext = "ico" if str(fmt).lower() == "ico" else "png"
        out = runtime_icon_cache_dir() / f"{prefix}{token}.{ext}"
        if ext == "ico":
            # Multi-size ICO helps taskbar + alt-tab hosts.
            sizes = [(16, 16), (32, 32), (48, 48), (w, h)]
            frames = []
            for sw, sh in sizes:
                frame = (
                    full_bleed_rgba(work, (sw, sh), fill_ratio=0.78)
                    or work.resize((sw, sh), Image.Resampling.LANCZOS)
                )
                frames.append(frame)
            frames[0].save(
                out,
                format="ICO",
                sizes=[(f.width, f.height) for f in frames],
                append_images=frames[1:],
            )
        else:
            work.save(out, format="PNG")
        return out if out.is_file() else None
    except Exception:  # noqa: BLE001
        return None


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

        roots.append(get_resource_path("assets"))
        roots.append(get_resource_path("dana/ui/assets"))
        roots.append(get_resource_path("dana/assets"))
    except Exception:  # noqa: BLE001
        pass
    roots.extend(
        [
            Path(__file__).resolve().parents[2] / "assets",  # project-root assets/
            Path(__file__).resolve().parent / "assets",  # dana/ui/assets
            Path(__file__).resolve().parents[1] / "assets",  # dana/assets
        ]
    )
    frozen = _frozen_root()
    if frozen is not None:
        roots.extend(
            [
                frozen / "assets",
                frozen / "dana" / "ui" / "assets",
                frozen / "dana" / "assets",
                frozen / "ui" / "assets",
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
            for rel in (f"assets/{name}", f"dana/ui/assets/{name}", f"dana/assets/{name}"):
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


# Preferred Windows app icon filenames (first hit wins). Root assets/ is SSoT.
_APP_ICON_RELS = (
    "assets/dana_logo.ico",
    "dana/assets/dana_icon.ico",  # legacy
    "dana/assets/dana.ico",  # legacy
)


def app_icon_path() -> Path:
    """Canonical Windows ``.ico`` path (``assets/dana_logo.ico``).

    Resolves via ``dana.resources.get_resource_path`` (``sys._MEIPASS``) so
    packaged + repo launches agree when assets are bundled with ``--add-data``.
    """
    try:
        from dana.resources import get_resource_path

        for rel in _APP_ICON_RELS:
            via = get_resource_path(rel)
            if via.is_file():
                return via
    except Exception:  # noqa: BLE001
        pass
    root_assets = Path(__file__).resolve().parents[2] / "assets"
    legacy_dir = Path(__file__).resolve().parents[1] / "assets"
    candidates: list[Path] = [
        root_assets / "dana_logo.ico",
        legacy_dir / "dana_icon.ico",
        legacy_dir / "dana.ico",
    ]
    frozen = _frozen_root()
    if frozen is not None:
        candidates.extend(
            [
                frozen / "assets" / "dana_logo.ico",
                frozen / "dana" / "assets" / "dana_icon.ico",
                frozen / "dana" / "assets" / "dana.ico",
            ]
        )
    for path in candidates:
        if path.is_file():
            return path
    try:
        from dana.resources import get_resource_path

        return get_resource_path("assets/dana_logo.ico")
    except Exception:  # noqa: BLE001
        return candidates[0]


def resolve_app_icon_path() -> Path | None:
    """Return ``assets/dana_logo.ico`` (or legacy ``.ico``) when present."""
    path = app_icon_path()
    return path if path.is_file() else None


def apply_window_icon(root: Any) -> bool:
    """Bind a cache-busted runtime icon to Tk / CTk for taskbar + title bar.

    Exports a uniquely named full-bleed PNG/ICO, then applies ``iconbitmap`` /
    ``wm_iconphoto``. Keeps ``PhotoImage`` alive on the root so Tk GC cannot
    drop it (which would revert to a stale/default glyph).
    """
    # Drop prior runtime files + in-memory logo cache so nothing falls back
    # to a previously exported fixed-looking path.
    purge_runtime_icon_files()
    for key in list(_ASSET_CACHE.keys()):
        try:
            kind = key[0] if key else ""
        except Exception:  # noqa: BLE001
            kind = ""
        if str(kind).startswith(("tray_", "toast", "overlay", "logo")):
            _ASSET_CACHE.pop(key, None)

    png_path = export_runtime_icon((64, 64), kind="logo", fmt="png")
    ico_path = export_runtime_icon((64, 64), kind="logo", fmt="ico")
    applied = False

    # Prefer unique runtime ICO; fall back to packaged asset only if export fails.
    ico_s = str(ico_path) if ico_path is not None else ""
    if not ico_s:
        legacy = resolve_app_icon_path()
        ico_s = str(legacy) if legacy is not None else ""
    if ico_s:
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
        try:
            _win32_set_icon_supplement(root, ico_s)
        except Exception:  # noqa: BLE001
            pass

    # iconphoto / wm_iconphoto — load from the unique runtime PNG when possible.
    try:
        from PIL import Image, ImageTk

        if png_path is not None and png_path.is_file():
            img = Image.open(png_path).convert("RGBA")
        else:
            img = load_premium_logo_pil((64, 64)) or load_app_icon_pil((64, 64))
            if img is not None:
                img = full_bleed_rgba(img, (64, 64), fill_ratio=0.78) or img
        if img is None:
            return applied
        photo = ImageTk.PhotoImage(img, master=root)
        try:
            root.wm_iconphoto(True, photo)
        except Exception:  # noqa: BLE001
            root.iconphoto(True, photo)
        # Explicit keepalives — prevent GC reverting to a cached/default icon.
        root._icon_keepalive = photo  # type: ignore[attr-defined]
        root._dana_iconphoto = photo  # type: ignore[attr-defined]
        root._dana_runtime_icon_png = png_path  # type: ignore[attr-defined]
        root._dana_runtime_icon_ico = ico_path  # type: ignore[attr-defined]
        applied = True
    except Exception:  # noqa: BLE001
        pass
    return applied


def app_icon_abspath() -> str:
    """Absolute packaged ``assets/dana_logo.ico`` path (dev + MEIPASS).

    Runtime cache-busted icons are produced by ``export_runtime_icon`` /
    ``apply_window_icon`` — this helper remains the SSoT source asset path.
    """
    import os

    try:
        from dana.resources import get_resource_path

        return os.path.abspath(str(get_resource_path("assets/dana_logo.ico")))
    except Exception:  # noqa: BLE001
        return os.path.abspath(str(app_icon_path()))


def _win32_set_icon_supplement(root_window: Any, ico_s: str) -> None:
    """Optional WM_SETICON after iconbitmap — never the primary path."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        WM_SETICON = 0x0080
        ICON_SMALL = 0
        ICON_BIG = 1
        IMAGE_ICON = 1
        LR_LOADFROMFILE = 0x00000010
        LR_DEFAULTSIZE = 0x00000020

        hwnd_direct = 0
        try:
            hwnd_direct = int(root_window.winfo_id())
        except Exception:  # noqa: BLE001
            hwnd_direct = 0
        hwnd_parent = 0
        if hwnd_direct:
            try:
                hwnd_parent = int(user32.GetParent(hwnd_direct))
            except Exception:  # noqa: BLE001
                hwnd_parent = 0
        targets: list[int] = []
        for hwnd in (hwnd_direct, hwnd_parent):
            if hwnd and hwnd not in targets:
                targets.append(hwnd)
        if not targets:
            return

        LoadImageW = user32.LoadImageW
        LoadImageW.argtypes = [
            wintypes.HINSTANCE,
            wintypes.LPCWSTR,
            wintypes.UINT,
            ctypes.c_int,
            ctypes.c_int,
            wintypes.UINT,
        ]
        LoadImageW.restype = wintypes.HANDLE
        hicon = LoadImageW(
            None,
            ico_s,
            IMAGE_ICON,
            0,
            0,
            LR_LOADFROMFILE | LR_DEFAULTSIZE,
        )
        if not hicon:
            return
        root_window._dana_hicon = hicon  # type: ignore[attr-defined]
        for hwnd in targets:
            user32.SendMessageW(hwnd, WM_SETICON, ICON_SMALL, hicon)
            user32.SendMessageW(hwnd, WM_SETICON, ICON_BIG, hicon)
    except Exception:  # noqa: BLE001
        pass


def force_apply_window_icon(root_window: Any, ico_path: str | Path | None = None) -> bool:
    """Apply titlebar/taskbar icon via cache-busted runtime export + keepalive.

    Prefer ``schedule_window_icon`` after window init so CTk chrome cannot
    overwrite the binding. When ``ico_path`` is omitted, exports a unique
    runtime ICO/PNG. Win32 WM_SETICON is a light supplement only.
    """
    if ico_path is None:
        return apply_window_icon(root_window)

    ico_s = str(ico_path)
    if not ico_s:
        return apply_window_icon(root_window)
    applied = False
    try:
        root_window.iconbitmap(ico_s)
        applied = True
    except Exception:  # noqa: BLE001
        pass
    try:
        root_window.wm_iconbitmap(ico_s)
        applied = True
    except Exception:  # noqa: BLE001
        pass
    _win32_set_icon_supplement(root_window, ico_s)
    # Still bind PhotoImage from the provided file for hosts that ignore .ico.
    try:
        from PIL import Image, ImageTk

        img = Image.open(ico_s).convert("RGBA")
        img = full_bleed_rgba(img, (64, 64), fill_ratio=0.78) or img.resize(
            (64, 64), Image.Resampling.LANCZOS
        )
        photo = ImageTk.PhotoImage(img, master=root_window)
        try:
            root_window.wm_iconphoto(True, photo)
        except Exception:  # noqa: BLE001
            root_window.iconphoto(True, photo)
        root_window._icon_keepalive = photo  # type: ignore[attr-defined]
        root_window._dana_iconphoto = photo  # type: ignore[attr-defined]
        applied = True
    except Exception:  # noqa: BLE001
        pass
    return applied


def schedule_window_icon(root: Any, delay_ms: int = 100) -> None:
    """Post-init icon bind using a uniquely named runtime export."""

    def _apply() -> None:
        apply_window_icon(root)

    try:
        root.after(int(delay_ms), _apply)
        # Second pass — some CTk builds reset the icon after first layout.
        root.after(int(delay_ms) + 400, _apply)
    except Exception:  # noqa: BLE001
        _apply()


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


def full_bleed_rgba(
    img: Any,
    size: tuple[int, int],
    *,
    fill_ratio: float = 0.78,
) -> Any | None:
    """Crop to opaque content and scale the mark to ``fill_ratio`` of the box.

    Fixes tiny green glyphs sitting inside oversized dark padding for taskbar,
    tray, and titlebar icons.
    """
    try:
        from PIL import Image
    except Exception:  # noqa: BLE001
        return None
    if img is None:
        return None
    try:
        base = make_transparent_logo(img.convert("RGBA"))
        w, h = max(1, int(size[0])), max(1, int(size[1]))
        alpha = base.getchannel("A")
        # Ignore soft AA fringe / near-clear pixels so dark padding is cropped.
        try:
            mask = alpha.point(lambda p: 255 if p > 24 else 0)
            bbox = mask.getbbox()
        except Exception:  # noqa: BLE001
            bbox = alpha.getbbox()
        if bbox is None:
            canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            return canvas
        cropped = base.crop(bbox)
        cw, ch = cropped.size
        ratio = max(0.05, min(0.95, float(fill_ratio)))
        target = max(1.0, ratio * float(min(w, h)))
        scale = target / float(max(cw, ch, 1))
        nw = max(1, int(round(cw * scale)))
        nh = max(1, int(round(ch * scale)))
        mark = cropped.resize((nw, nh), Image.Resampling.LANCZOS)
        # Clamp so a bad crop never exceeds the destination tile.
        if nw > w or nh > h:
            fit = min(w / float(nw), h / float(nh), 1.0)
            nw = max(1, int(round(nw * fit)))
            nh = max(1, int(round(nh * fit)))
            mark = mark.resize((nw, nh), Image.Resampling.LANCZOS)
        canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ox = (w - nw) // 2
        oy = (h - nh) // 2
        canvas.paste(mark, (ox, oy), mark.split()[-1])
        return canvas
    except Exception:  # noqa: BLE001
        return None


def _bolster_tray_glyph(img: Any) -> Any:
    """Full-bleed crop + contrast/stroke so the mark reads on dark taskbars."""
    try:
        from PIL import Image, ImageEnhance, ImageFilter
    except Exception:  # noqa: BLE001
        return img
    try:
        base = img.convert("RGBA")
        w, h = base.size
        # Upscale work buffer for cleaner LANCZOS edges.
        work_size = (max(w * 2, 64), max(h * 2, 64))
        hi = full_bleed_rgba(base, work_size, fill_ratio=0.78) or base.resize(
            work_size, Image.Resampling.LANCZOS
        )
        hi = ImageEnhance.Contrast(hi).enhance(1.4)
        hi = ImageEnhance.Color(hi).enhance(1.18)
        hi = ImageEnhance.Sharpness(hi).enhance(1.25)
        alpha = hi.getchannel("A")
        stroke_a = alpha.filter(ImageFilter.MaxFilter(3))
        stroke = Image.new("RGBA", hi.size, (6, 95, 70, 255))
        stroke.putalpha(stroke_a)
        out = Image.alpha_composite(stroke, hi)
        if out.size != (w, h):
            out = out.resize((w, h), Image.Resampling.LANCZOS)
        return make_transparent_logo(out)
    except Exception:  # noqa: BLE001
        return img


def get_tray_icon(size: tuple[int, int] = (32, 32)) -> Any | None:
    """Cached clean RGBA PIL image for pystray (transparent backdrop).

    Rebuilds from a uniquely named runtime PNG so the OS / pystray cannot keep
    serving a stale shell-cached glyph from a fixed path.
    """
    w, h = int(size[0]), int(size[1])
    key = ("tray_fullbleed_runtime", w, h)
    cached = _ASSET_CACHE.get(key)
    if cached is not None:
        try:
            return cached.copy()
        except Exception:  # noqa: BLE001
            return cached
    img = _rgba_source((max(w * 2, 64), max(h * 2, 64)))
    if img is None:
        img = _rgba_source((w, h))
    if img is None:
        return None
    img = _bolster_tray_glyph(
        full_bleed_rgba(img, (w, h), fill_ratio=0.78) or img
    )
    try:
        from PIL import Image

        if img.size != (w, h):
            img = img.resize((w, h), Image.Resampling.LANCZOS)
        # Persist + reload via unique path (cache-bust for file-based consumers).
        path = export_runtime_icon((w, h), kind="tray", fmt="png", img=img)
        if path is not None and path.is_file():
            img = Image.open(path).convert("RGBA")
            _ASSET_CACHE[("tray_runtime_path", w, h)] = path
    except Exception:  # noqa: BLE001
        pass
    _ASSET_CACHE[key] = img
    try:
        return img.copy()
    except Exception:  # noqa: BLE001
        return img


def get_overlay_logo(size: tuple[int, int] = (48, 48)) -> Any | None:
    """Cached RGBA ``CTkImage`` for the floating HUD (PIL fallback if CTk missing)."""
    w, h = int(size[0]), int(size[1])
    key = ("overlay_fullbleed", w, h)
    cached = _ASSET_CACHE.get(key)
    if cached is not None:
        return cached
    img = _rgba_source((w, h))
    if img is None:
        return None
    img = full_bleed_rgba(img, (w, h), fill_ratio=0.78) or img
    try:
        import tkinter as tk

        import customtkinter as ctk

        if getattr(tk, "_default_root", None) is None:
            try:
                from dana.ui.theme import apply_dana_ctk_theme

                apply_dana_ctk_theme()
            except Exception:  # noqa: BLE001
                pass
            sentinel = ctk.CTk()
            sentinel.withdraw()
            tk._dana_logo_sentinel = sentinel  # type: ignore[attr-defined]
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
    key = ("toast_fullbleed", w, h)
    cached = _ASSET_CACHE.get(key)
    if cached is not None:
        try:
            return cached.copy()
        except Exception:  # noqa: BLE001
            return cached
    img = _rgba_source((w, h))
    if img is None:
        return None
    img = full_bleed_rgba(img, (w, h), fill_ratio=0.78) or img
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
    """Load + full-bleed LANCZOS resize; optional hex tint for orb pulses."""
    path = resolve_logo_path()
    if path is None:
        return None
    try:
        from PIL import Image, ImageEnhance

        img = make_transparent_logo(_load_pil(path))
        w, h = int(size[0]), int(size[1])
        if w < 1 or h < 1:
            return None
        img = full_bleed_rgba(img, (w, h), fill_ratio=0.78) or img.resize(
            (w, h), Image.Resampling.LANCZOS
        )
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
        # sentinel root alive so images survive across DanaGUI destroy cycles.
        if getattr(tk, "_default_root", None) is None:
            try:
                from dana.ui.theme import apply_dana_ctk_theme

                apply_dana_ctk_theme()
            except Exception:  # noqa: BLE001
                pass
            sentinel = ctk.CTk()
            sentinel.withdraw()
            tk._dana_logo_sentinel = sentinel  # type: ignore[attr-defined]
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
