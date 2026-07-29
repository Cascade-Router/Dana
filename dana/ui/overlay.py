"""Apple-style floating HUD overlay (Dynamic Island pill).

Frameless, topmost window with Windows color-key transparency so only the
rounded slate pill is visible. Safe to import on non-Windows / headless CI —
Tk / CustomTkinter failures become no-ops.
"""

from __future__ import annotations

import sys
from typing import Any

# Chroma-key must not appear in drawn HUD pixels (near-black, not pure black).
TRANSPARENT_KEY = "#000001"
_PILL_FG = "#0F172A"
_PILL_BORDER = "#1E293B"
_PILL_RADIUS = 20
_LABEL_FG = "#E2E8F0"


def apply_windows_transparency(root: Any, *, key: str = TRANSPARENT_KEY) -> bool:
    """Enable Windows ``-transparentcolor`` keying. No-op off Windows / on failure."""
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
    try:
        root.wm_attributes("-transparentcolor", key)
        return True
    except Exception:  # noqa: BLE001
        return False


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
        text: str = "Dānā",
        logo_size: tuple[int, int] = (48, 48),
    ) -> None:
        self.root: Any | None = None
        self.pill: Any | None = None
        self._logo_img: Any | None = None
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
                from dana.ui.logo import get_overlay_logo

                self._logo_img = get_overlay_logo(logo_size)
            except Exception:  # noqa: BLE001
                self._logo_img = None

            if self._logo_img is not None:
                try:
                    # CTkImage path — no black box; keyed RGBA embeds cleanly.
                    logo_lbl = ctk.CTkLabel(
                        row,
                        text="",
                        image=self._logo_img,
                        fg_color="transparent",
                    )
                    logo_lbl.pack(side="left", padx=(0, 10))
                except Exception:  # noqa: BLE001
                    # PIL fallback (headless tests) — skip widget bind.
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
                self.root.update_idletasks()
                sw = int(self.root.winfo_screenwidth() or 1280)
                w = max(160, int(self.root.winfo_reqwidth() or 200))
                h = max(56, int(self.root.winfo_reqheight() or 64))
                x = max(8, (sw - w) // 2)
                y = 16
                self.root.geometry(f"{w}x{h}+{x}+{y}")
            except Exception:  # noqa: BLE001
                pass
        except Exception:  # noqa: BLE001
            self.destroy()

    def set_text(self, text: str) -> None:
        if self._label is not None:
            try:
                self._label.configure(text=text)
            except Exception:  # noqa: BLE001
                pass

    def destroy(self) -> None:
        root = self.root
        self.root = None
        self.pill = None
        self._logo_img = None
        self._label = None
        if root is not None:
            try:
                root.destroy()
            except Exception:  # noqa: BLE001
                pass
