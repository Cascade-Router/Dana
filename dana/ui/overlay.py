"""Apple-style floating HUD overlay (Dynamic Island pill).

Frameless, topmost window with Windows color-key transparency so only the
rounded slate pill is visible. Safe to import on non-Windows / headless CI —
Tk / CustomTkinter failures become no-ops.

On Windows, ``-transparentcolor`` alone often still hit-tests the keyed pixels;
``apply_colorkey_hit_test`` installs ``LWA_COLORKEY`` so transparent chrome does
not intercept desktop clicks while opaque HUD content stays interactive.
"""

from __future__ import annotations

import sys
from typing import Any

# Chroma-key must not appear in drawn HUD pixels (near-black, not pure black).
TRANSPARENT_KEY = "#000001"
_PILL_FG = "#0a0e17"
_PILL_BORDER = "#1e293b"
_PILL_RADIUS = 20
_LABEL_FG = "#9CA3AF"  # STANDBY muted gray
_LABEL_ACTIVE = "#10b981"  # ACTIVE emerald


def _parse_hex_rgb(key: str) -> tuple[int, int, int] | None:
    raw = (key or "").strip().lstrip("#")
    if len(raw) != 6:
        return None
    try:
        return int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16)
    except ValueError:
        return None


def apply_colorkey_hit_test(root: Any, *, key: str = TRANSPARENT_KEY) -> bool:
    """Make color-keyed pixels click-through via Win32 ``LWA_COLORKEY``.

    Opaque widgets keep hit-testing. No-op off Windows / on failure.
    """
    if root is None or sys.platform != "win32":
        return False
    rgb = _parse_hex_rgb(key)
    if rgb is None:
        return False
    try:
        import ctypes

        user32 = ctypes.windll.user32
        GWL_EXSTYLE = -20
        WS_EX_LAYERED = 0x00080000
        LWA_COLORKEY = 0x00000001
        hwnd = int(root.winfo_id())
        if hwnd <= 0:
            return False
        style = int(user32.GetWindowLongW(hwnd, GWL_EXSTYLE))
        user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style | WS_EX_LAYERED)
        # COLORREF is 0x00BBGGRR
        colorref = int(rgb[0]) | (int(rgb[1]) << 8) | (int(rgb[2]) << 16)
        return bool(user32.SetLayeredWindowAttributes(hwnd, colorref, 0, LWA_COLORKEY))
    except Exception:  # noqa: BLE001
        return False


def apply_windows_transparency(root: Any, *, key: str = TRANSPARENT_KEY) -> bool:
    """Enable Windows ``-transparentcolor`` keying + color-key hit-test.

    No-op off Windows / on failure. Geometry should still be clamped to content
    so the layered window does not span unused desktop area.
    """
    if root is None:
        return False
    try:
        root.configure(bg=key)
    except Exception:  # noqa: BLE001
        try:
            root.configure(fg_color=key)
        except Exception:  # noqa: BLE001
            pass
    if sys.platform != "win32":
        return False
    keyed = False
    try:
        root.wm_attributes("-transparentcolor", key)
        keyed = True
    except Exception:  # noqa: BLE001
        keyed = False
    try:
        root.update_idletasks()
    except Exception:  # noqa: BLE001
        pass
    hit = apply_colorkey_hit_test(root, key=key)
    return bool(keyed or hit)


def clamp_toplevel_to_content(
    root: Any,
    *,
    content: Any | None = None,
    pad_x: int = 0,
    pad_y: int = 0,
    min_w: int = 1,
    min_h: int = 1,
    max_w: int | None = None,
    max_h: int | None = None,
    x: int | None = None,
    y: int | None = None,
) -> tuple[int, int]:
    """Resize a toplevel to its content req size (plus pad). Returns (w, h)."""
    if root is None:
        return (0, 0)
    try:
        root.update_idletasks()
    except Exception:  # noqa: BLE001
        pass
    target = content if content is not None else root
    try:
        req_w = int(target.winfo_reqwidth() or 0)
        req_h = int(target.winfo_reqheight() or 0)
    except Exception:  # noqa: BLE001
        req_w, req_h = 0, 0
    if req_w <= 1 or req_h <= 1:
        try:
            req_w = max(req_w, int(root.winfo_reqwidth() or 0))
            req_h = max(req_h, int(root.winfo_reqheight() or 0))
        except Exception:  # noqa: BLE001
            pass
    w = max(min_w, req_w + max(0, int(pad_x)))
    h = max(min_h, req_h + max(0, int(pad_y)))
    if max_w is not None:
        w = min(w, int(max_w))
    if max_h is not None:
        h = min(h, int(max_h))
    try:
        cur_x = int(root.winfo_x()) if x is None else int(x)
        cur_y = int(root.winfo_y()) if y is None else int(y)
    except Exception:  # noqa: BLE001
        cur_x, cur_y = 0, 0
        if x is not None:
            cur_x = int(x)
        if y is not None:
            cur_y = int(y)
    try:
        root.geometry(f"{w}x{h}+{cur_x}+{cur_y}")
        root.minsize(w, h)
        root.maxsize(w, h)
    except Exception:  # noqa: BLE001
        try:
            root.geometry(f"{w}x{h}+{cur_x}+{cur_y}")
        except Exception:  # noqa: BLE001
            pass
    return (w, h)


def build_hud_pill(parent: Any, **kwargs: Any) -> Any | None:
    """Rounded slate ``CTkFrame`` pill (corner_radius=20). Returns None if CTk missing."""
    try:
        import customtkinter as ctk
    except Exception:  # noqa: BLE001
        return None
    opts = {
        "corner_radius": _PILL_RADIUS,
        "fg_color": _PILL_FG,
        "border_width": 1,
        "border_color": _PILL_BORDER,
    }
    opts.update(kwargs)
    try:
        return ctk.CTkFrame(parent, **opts)
    except Exception:  # noqa: BLE001
        return None


class FloatingStatusHud:
    """Minimal always-on-top status pill with a transparent RGBA logo on the left."""

    def __init__(
        self,
        master: Any | None = None,
        *,
        text: str = "STANDBY",
        logo_size: tuple[int, int] = (22, 22),
    ) -> None:
        self.root: Any | None = None
        self.pill: Any | None = None
        self._logo_img: Any | None = None
        self._logo_lbl: Any | None = None
        self._label: Any | None = None
        try:
            import customtkinter as ctk
        except Exception:  # noqa: BLE001
            return

        try:
            if master is None:
                self.root = ctk.CTk()
            else:
                self.root = ctk.CTkToplevel(master)
            self.root.title("Dānā HUD")
            self.root.overrideredirect(True)
            try:
                self.root.attributes("-topmost", True)
            except Exception:  # noqa: BLE001
                pass
            apply_windows_transparency(self.root, key=TRANSPARENT_KEY)
            try:
                self.root.configure(fg_color=TRANSPARENT_KEY)
            except Exception:  # noqa: BLE001
                pass

            self.pill = build_hud_pill(self.root)
            if self.pill is None:
                self.destroy()
                return
            self.pill.pack(padx=6, pady=6)

            row = ctk.CTkFrame(self.pill, fg_color="transparent")
            row.pack(padx=14, pady=10)

            try:
                from dana.ui.logo import get_overlay_logo, load_premium_logo

                self._logo_img = get_overlay_logo(logo_size)
                if self._logo_img is None:
                    # Direct CTkImage path if cache/generator returned None.
                    self._logo_img = load_premium_logo(logo_size)
            except Exception:  # noqa: BLE001
                self._logo_img = None

            if self._logo_img is not None:
                try:
                    # CTkImage path — no black box; keyed RGBA embeds cleanly.
                    # Never substitute a text glyph / status dot for the mark.
                    logo_lbl = ctk.CTkLabel(
                        row,
                        text="",
                        image=self._logo_img,
                        fg_color="transparent",
                    )
                    logo_lbl.pack(side="left", padx=(0, 10))
                    self._logo_lbl = logo_lbl
                except Exception:  # noqa: BLE001
                    # PIL fallback (headless tests) — skip widget bind.
                    try:
                        from PIL import Image as _PilImage

                        if isinstance(self._logo_img, _PilImage.Image):
                            self._logo_img = ctk.CTkImage(
                                light_image=self._logo_img,
                                dark_image=self._logo_img,
                                size=logo_size,
                            )
                            logo_lbl = ctk.CTkLabel(
                                row,
                                text="",
                                image=self._logo_img,
                                fg_color="transparent",
                            )
                            logo_lbl.pack(side="left", padx=(0, 10))
                            self._logo_lbl = logo_lbl
                    except Exception:  # noqa: BLE001
                        pass

            self._label = ctk.CTkLabel(
                row,
                text=text,
                text_color=_LABEL_FG,
                fg_color="transparent",
                font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            )
            self._label.pack(side="left")

            try:
                self._fit_to_content(x=None, y=16, center_x=True)
            except Exception:  # noqa: BLE001
                pass
        except Exception:  # noqa: BLE001
            self.destroy()

    def _fit_to_content(
        self,
        *,
        x: int | None = None,
        y: int | None = None,
        center_x: bool = False,
    ) -> None:
        """Clamp the frameless root to the pill bounds (no full-screen chrome)."""
        if self.root is None:
            return
        try:
            self.root.update_idletasks()
        except Exception:  # noqa: BLE001
            pass
        place_x = x
        place_y = 16 if y is None else y
        if center_x:
            try:
                sw = int(self.root.winfo_screenwidth() or 1280)
                # Pre-measure so centering uses the clamped width.
                probe = self.pill if self.pill is not None else self.root
                pw = max(160, int(probe.winfo_reqwidth() or 200) + 12)
                place_x = max(8, (sw - pw) // 2)
            except Exception:  # noqa: BLE001
                place_x = 8
        clamp_toplevel_to_content(
            self.root,
            content=self.pill,
            pad_x=12,
            pad_y=12,
            min_w=160,
            min_h=56,
            max_w=480,
            max_h=120,
            x=place_x,
            y=place_y,
        )
        apply_colorkey_hit_test(self.root, key=TRANSPARENT_KEY)

    def set_text(self, text: str) -> None:
        if self._label is not None:
            try:
                label = str(text or "")
                color = _LABEL_FG
                upper = label.strip().upper()
                if upper in {"ACTIVE", "ENGAGE", "ENGAGED"} or "ACTIVE" in upper:
                    color = _LABEL_ACTIVE
                elif upper in {"STANDBY", "IDLE"} or "STANDBY" in upper:
                    color = _LABEL_FG
                self._label.configure(text=label, text_color=color)
            except Exception:  # noqa: BLE001
                pass
            try:
                self._fit_to_content()
            except Exception:  # noqa: BLE001
                pass

    def destroy(self) -> None:
        root = self.root
        self.root = None
        self.pill = None
        self._logo_img = None
        self._logo_lbl = None
        self._label = None
        if root is not None:
            try:
                root.destroy()
            except Exception:  # noqa: BLE001
                pass
