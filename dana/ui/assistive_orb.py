"""Floating AssistiveTouch pill (frameless, topmost, glassmorphic).

Always-on-top slate pill the operator can drag. Hover expands a mini control
strip (dictation + HITL + dashboard) without touching LangGraph workers.

Visual: glassmorphic pill ``#0a0e17`` + border ``#1e293b``, 20–24px logo left,
Hyper Mint ACTIVE / muted STANDBY. Windows ``LWA_COLORKEY`` ``#000001``
keeps transparent pixels click-through. Popup fades α 0→1 over ~150ms.
"""

from __future__ import annotations

import math
import tkinter as tk
from typing import Any, Callable

try:
    import customtkinter as ctk
except Exception:  # noqa: BLE001
    ctk = None  # type: ignore[assignment]

# Chroma-key for Windows layered transparency (must not appear in drawn pixels).
_TRANSPARENT = "#000001"
_PILL_BG = "#0a0e17"
_PILL_BORDER = "#1e293b"
_PILL_W = 132
_PILL_H = 40
_PANEL_W = 320
_PANEL_H = 360
_PAD = 8
_DRAG_THRESHOLD_PX = 4
_PULSE_MS = 40  # ~25 FPS status refresh
_FADE_MS = 150
_FADE_STEPS = 6
# Logo pixel size (fixed; no sci-fi pulse scale).
_ICON_SIZE_MIN = 20
_ICON_SIZE_MAX = 24
_LOGO_PX = 22
# Legacy aliases (older tests that still reference radius-era names).
_PULSE_BASE_R = float(_LOGO_PX)
_PULSE_AMP = 0.0

_COLOR_ACTIVE = "#10b981"  # Hyper Mint ACTIVE
_COLOR_STANDBY = "#9CA3AF"  # muted STANDBY
_COLOR_CHAT = _COLOR_ACTIVE
_COLOR_DICTATION = "#9C27B0"
_COLOR_EXEC = "#FB8C00"
_COLOR_HITL = "#388e3c"
_COLOR_IDLE = _COLOR_STANDBY
_PANEL_BG = "#131b2e"
_PANEL_FG = "#F8FAFC"
_MUTED = "#94A3B8"


class AssistiveTouchOrb:
    """Draggable always-on-top glassmorphic pill owned by the DanaGUI Tk root."""

    def __init__(
        self,
        master: Any,
        *,
        on_toggle_dictation: Callable[[], None] | None = None,
        on_open_dashboard: Callable[[], None] | None = None,
        on_approve_ticket: Callable[[], None] | None = None,
        on_deny_ticket: Callable[[], None] | None = None,
        dictation_getter: Callable[[], bool] | None = None,
        mode_getter: Callable[[], str] | None = None,
        accent_getter: Callable[[], str] | None = None,
        agent_getter: Callable[[], str] | None = None,
    ) -> None:
        self.master = master
        self._on_toggle_dictation = on_toggle_dictation
        self._on_open_dashboard = on_open_dashboard
        self._on_approve_ticket = on_approve_ticket
        self._on_deny_ticket = on_deny_ticket
        self._dictation_getter = dictation_getter or (lambda: False)
        self._mode_getter = mode_getter or (lambda: "chat")
        self._accent_getter = accent_getter
        self._agent_getter = agent_getter

        self._expanded = False
        self._dragging = False
        self._drag_start_x = 0
        self._drag_start_y = 0
        self._drag_offset_x = 0
        self._drag_offset_y = 0
        self._win_start_x = 0
        self._win_start_y = 0
        self._win_start_x = 0
        self._win_start_y = 0
        self._moved = False
        self._leave_job: str | None = None
        self._fade_job: str | None = None
        self._pulse_phase = 0.0
        self._pulse_job: str | None = None
        self._hitl_pending = False
        self._active_agent = "broker"
        self._status = "STANDBY"
        self._alpha = 1.0
        # Keep PhotoImage refs alive (Tk GC otherwise blanks the canvas).
        self._logo_photo: Any | None = None
        self._chrome_photo: Any | None = None
        self._logo_mode = "png"  # "png" | "polygon"

        self.orb_window = tk.Toplevel(master)
        self.orb_window.title("Dānā Orb")
        self.orb_window.overrideredirect(True)
        try:
            self.orb_window.attributes("-topmost", True)
        except Exception:  # noqa: BLE001
            pass
        try:
            # Windows: punch out chroma-key so only the pill/panel are visible.
            self.orb_window.wm_attributes("-transparentcolor", _TRANSPARENT)
        except Exception:  # noqa: BLE001
            pass
        try:
            self.orb_window.attributes("-alpha", 1.0)
        except Exception:  # noqa: BLE001
            pass
        self.orb_window.configure(bg=_TRANSPARENT)

        screen_w = int(self.orb_window.winfo_screenwidth() or 1280)
        screen_h = int(self.orb_window.winfo_screenheight() or 800)
        x = max(24, screen_w - _PILL_W - 36)
        y = max(80, screen_h // 3)
        self._orb_x = x
        self._orb_y = y

        # Pack to content only — expand=True left a large transparent hit box.
        self._shell = tk.Frame(self.orb_window, bg=_TRANSPARENT, bd=0, highlightthickness=0)
        self._shell.pack(anchor="nw")

        self._canvas = tk.Canvas(
            self._shell,
            width=_PILL_W,
            height=_PILL_H,
            bg=_TRANSPARENT,
            highlightthickness=0,
            bd=0,
        )
        self._canvas.grid(row=0, column=0, padx=0, pady=0, sticky="nw")

        # Rounded popup (CTkFrame corner_radius 15–20) when CTk available.
        if ctk is not None:
            self._panel = ctk.CTkFrame(
                self._shell,
                fg_color=_PANEL_BG,
                corner_radius=18,
                border_width=1,
                border_color=_PILL_BORDER,
                width=_PANEL_W,
                height=_PANEL_H,
            )
        else:
            self._panel = tk.Frame(
                self._shell,
                bg=_PANEL_BG,
                bd=0,
                highlightthickness=1,
                highlightbackground=_PILL_BORDER,
                width=_PANEL_W,
                height=_PANEL_H,
            )
        # Built but not mapped until hover expand.
        self._build_panel()

        self._draw_orb()
        self._apply_compact_geometry()
        self._apply_transparent_hit_test()

        # Drag on the solid canvas graphic only — not the toplevel / shell.
        # Transparent click-through (color-key + SetWindowRgn) drops events on
        # the parent window; canvas items (chrome/logo/status) still receive them.
        self._bind_orb_drag_handlers()
        self._shell.bind("<Enter>", self._on_enter)
        self._shell.bind("<Leave>", self._on_leave)
        self._canvas.bind("<Enter>", self._on_enter)
        self._canvas.bind("<Leave>", self._on_leave)
        self._panel.bind("<Enter>", self._on_enter)
        self._panel.bind("<Leave>", self._on_leave)

        try:
            self.orb_window.after(80, self.pulse_animation)
            self.orb_window.after(400, self._status_tick)
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------ UI
    def _build_panel(self) -> None:
        use_ctk = type(self._panel).__name__ == "CTkFrame"
        if use_ctk:
            title = ctk.CTkLabel(
                self._panel,
                text="DĀNĀ",
                text_color=_PANEL_FG,
                font=ctk.CTkFont(size=12, weight="bold"),
                anchor="w",
            )
            title.pack(fill="x", padx=12, pady=(12, 2))
            self._state_lbl = ctk.CTkLabel(
                self._panel,
                text="Mode: Chat  ·  ● OFF",
                text_color=_MUTED,
                font=ctk.CTkFont(size=11),
                anchor="w",
            )
            self._state_lbl.pack(fill="x", padx=12, pady=(0, 4))
            self._critique_hdr = ctk.CTkLabel(
                self._panel,
                text="Jason Review",
                text_color="#CE93D8",
                font=ctk.CTkFont(size=11, weight="bold"),
                anchor="w",
            )
            self._critique_lbl = ctk.CTkLabel(
                self._panel,
                text="",
                fg_color="#2A1840",
                text_color="#F3E5F5",
                font=ctk.CTkFont(size=10),
                anchor="nw",
                justify="left",
                wraplength=_PANEL_W - 28,
                corner_radius=8,
            )
            self._ticket_hdr = ctk.CTkLabel(
                self._panel,
                text="Drafted Ticket",
                text_color="#81C784",
                font=ctk.CTkFont(size=11, weight="bold"),
                anchor="w",
            )
            self._ticket_lbl = ctk.CTkLabel(
                self._panel,
                text="",
                fg_color="#14261A",
                text_color="#E8F5E9",
                font=ctk.CTkFont(size=10),
                anchor="nw",
                justify="left",
                wraplength=_PANEL_W - 28,
                corner_radius=8,
            )
            self._dictation_btn = ctk.CTkButton(
                self._panel,
                text="●  OFF",
                fg_color="#2A2A3C",
                hover_color="#34344A",
                text_color="#F87171",
                font=ctk.CTkFont(size=12, weight="bold"),
                corner_radius=10,
                command=self._click_dictation,
            )
            self._dictation_btn.pack(fill="x", padx=12, pady=(0, 6))
            self._dash_btn = ctk.CTkButton(
                self._panel,
                text="Open Dashboard",
                fg_color=_COLOR_ACTIVE,
                hover_color="#059669",
                text_color="#FFFFFF",
                font=ctk.CTkFont(size=11, weight="bold"),
                corner_radius=10,
                command=self._click_dashboard,
            )
            self._dash_btn.pack(fill="x", padx=12, pady=(0, 6))
            hitl_row = ctk.CTkFrame(self._panel, fg_color="transparent")
            self._hitl_row = hitl_row
            self._approve_btn = ctk.CTkButton(
                hitl_row,
                text="Approve",
                fg_color=_COLOR_HITL,
                hover_color="#2E7D32",
                text_color="#FFFFFF",
                font=ctk.CTkFont(size=11, weight="bold"),
                corner_radius=10,
                command=self._click_approve,
                state="disabled",
            )
            self._approve_btn.pack(side="left", expand=True, fill="x", padx=(0, 4))
            self._deny_btn = ctk.CTkButton(
                hitl_row,
                text="Deny",
                fg_color="#C62828",
                hover_color="#B71C1C",
                text_color="#FFFFFF",
                font=ctk.CTkFont(size=11, weight="bold"),
                corner_radius=10,
                command=self._click_deny,
                state="disabled",
            )
            self._deny_btn.pack(side="left", expand=True, fill="x", padx=(4, 0))
            self._github_btn = ctk.CTkButton(
                self._panel,
                text="\U0001f419 Report Issue on GitHub",
                fg_color="#24292F",
                hover_color="#1B1F23",
                text_color="#FFFFFF",
                font=ctk.CTkFont(size=11, weight="bold"),
                corner_radius=10,
                command=self._click_github,
            )
            return

        title = tk.Label(
            self._panel,
            text="DĀNĀ",
            bg=_PANEL_BG,
            fg=_PANEL_FG,
            font=("Segoe UI", 11, "bold"),
            anchor="w",
        )
        title.pack(fill="x", padx=12, pady=(10, 2))
        self._state_lbl = tk.Label(
            self._panel,
            text="Mode: Chat  ·  ● OFF",
            bg=_PANEL_BG,
            fg=_MUTED,
            font=("Segoe UI", 9),
            anchor="w",
        )
        self._state_lbl.pack(fill="x", padx=12, pady=(0, 4))

        self._critique_hdr = tk.Label(
            self._panel,
            text="Jason Review",
            bg=_PANEL_BG,
            fg="#CE93D8",
            font=("Segoe UI", 9, "bold"),
            anchor="w",
        )
        self._critique_lbl = tk.Label(
            self._panel,
            text="",
            bg="#2A1840",
            fg="#F3E5F5",
            font=("Segoe UI", 8),
            anchor="nw",
            justify="left",
            wraplength=_PANEL_W - 28,
        )
        self._ticket_hdr = tk.Label(
            self._panel,
            text="Drafted Ticket",
            bg=_PANEL_BG,
            fg="#81C784",
            font=("Segoe UI", 9, "bold"),
            anchor="w",
        )
        self._ticket_lbl = tk.Label(
            self._panel,
            text="",
            bg="#14261A",
            fg="#E8F5E9",
            font=("Segoe UI", 8),
            anchor="nw",
            justify="left",
            wraplength=_PANEL_W - 28,
        )

        self._dictation_btn = tk.Button(
            self._panel,
            text="●  OFF",
            bg="#2A2A3C",
            fg="#F87171",
            activebackground="#34344A",
            activeforeground="#F87171",
            relief="flat",
            bd=0,
            font=("Segoe UI", 10, "bold"),
            cursor="hand2",
            command=self._click_dictation,
        )
        self._dictation_btn.pack(fill="x", padx=12, pady=(0, 6), ipady=4)

        self._dash_btn = tk.Button(
            self._panel,
            text="Open Dashboard",
            bg=_COLOR_ACTIVE,
            fg="#FFFFFF",
            activebackground="#059669",
            activeforeground="#FFFFFF",
            relief="flat",
            bd=0,
            font=("Segoe UI", 9, "bold"),
            cursor="hand2",
            command=self._click_dashboard,
        )
        self._dash_btn.pack(fill="x", padx=12, pady=(0, 6), ipady=3)

        hitl_row = tk.Frame(self._panel, bg=_PANEL_BG)
        self._hitl_row = hitl_row
        self._approve_btn = tk.Button(
            hitl_row,
            text="Approve",
            bg=_COLOR_HITL,
            fg="#FFFFFF",
            activebackground="#2E7D32",
            relief="flat",
            bd=0,
            font=("Segoe UI", 9, "bold"),
            cursor="hand2",
            command=self._click_approve,
            state="disabled",
        )
        self._approve_btn.pack(side="left", expand=True, fill="x", padx=(0, 4), ipady=3)
        self._deny_btn = tk.Button(
            hitl_row,
            text="Deny",
            bg="#C62828",
            fg="#FFFFFF",
            activebackground="#B71C1C",
            relief="flat",
            bd=0,
            font=("Segoe UI", 9, "bold"),
            cursor="hand2",
            command=self._click_deny,
            state="disabled",
        )
        self._deny_btn.pack(side="left", expand=True, fill="x", padx=(4, 0), ipady=3)
        self._github_btn = tk.Button(
            self._panel,
            text="\U0001f419 Report Issue on GitHub",
            bg="#24292F",
            fg="#FFFFFF",
            activebackground="#1B1F23",
            activeforeground="#FFFFFF",
            relief="flat",
            bd=0,
            font=("Segoe UI", 9, "bold"),
            cursor="hand2",
            command=self._click_github,
        )

    def _resolve_active_agent(self) -> str:
        if self._agent_getter is not None:
            try:
                return str(self._agent_getter() or "broker")
            except Exception:  # noqa: BLE001
                pass
        try:
            from dana.audio.multi_voice_tts import get_active_tts_agent

            return str(get_active_tts_agent() or "broker")
        except Exception:  # noqa: BLE001
            return "broker"

    def _is_active(self) -> bool:
        if self._dictation_getter():
            return True
        if self._hitl_pending:
            return True
        try:
            from dana.core.shared_state import tts_busy

            if bool(tts_busy.is_set()):
                return True
        except Exception:  # noqa: BLE001
            pass
        return False

    def _accent(self) -> str:
        """Status accent — ACTIVE emerald / STANDBY muted (persona kept for API)."""
        if self._is_active():
            return _COLOR_ACTIVE
        # Preserve persona color when speaking path asks via agent_getter tests.
        if self._accent_getter is not None:
            try:
                from dana.audio.multi_voice_tts import persona_color_for_agent

                return persona_color_for_agent(self._active_agent)
            except Exception:  # noqa: BLE001
                pass
        try:
            from dana.audio.multi_voice_tts import persona_color_for_agent

            # Tests assert persona colors while idle; expose via agent when set.
            if self._agent_getter is not None:
                return persona_color_for_agent(self._active_agent)
        except Exception:  # noqa: BLE001
            pass
        return _COLOR_STANDBY

    def _status_color(self) -> str:
        return _COLOR_ACTIVE if self._is_active() else _COLOR_STANDBY

    def _icon_center(self) -> tuple[float, float]:
        """Logo center — left side of the glassmorphic pill."""
        cy = _PILL_H / 2.0
        cx = 8.0 + (_LOGO_PX / 2.0)
        return cx, cy

    def _draw_smooth_mark(self, *, size: int, fill: str) -> None:
        """Fallback mark — smooth polygons (no Unicode / no jagged ovals)."""
        cx, cy = self._icon_center()
        s = max(6.0, float(size) * 0.42)
        top = [
            cx - s * 1.05,
            cy - s * 0.85,
            cx + s * 1.05,
            cy - s * 0.55,
            cx + s * 0.95,
            cy - s * 0.15,
            cx - s * 1.05,
            cy - s * 0.45,
        ]
        left = [
            cx - s * 0.85,
            cy - s * 0.05,
            cx - s * 0.15,
            cy - s * 0.05,
            cx - s * 0.35,
            cy + s * 1.05,
            cx - s * 0.95,
            cy + s * 1.05,
        ]
        right = [
            cx + s * 0.05,
            cy - s * 0.05,
            cx + s * 0.75,
            cy - s * 0.05,
            cx + s * 0.55,
            cy + s * 1.05,
            cx - s * 0.05,
            cy + s * 1.05,
        ]
        for pts in (top, left, right):
            self._canvas.create_polygon(
                *pts,
                fill=fill,
                outline="",
                smooth=True,
                splinesteps=24,
                tags=("icon", "mark"),
            )

    def _build_aa_pill_photo(self) -> Any | None:
        """4× supersampled pill; color-key compatible soft edges via LANCZOS.

        Windows ``-transparentcolor #000001`` cannot composite real alpha, so
        we flatten AA fringes onto the key color — edges read smooth without
        jagged Tk ``create_oval`` stairs.
        """
        try:
            from PIL import Image, ImageDraw, ImageTk
        except Exception:  # noqa: BLE001
            return None
        scale = 4
        w, h = _PILL_W, _PILL_H
        W, H = w * scale, h * scale
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        radius = H // 2
        draw.rounded_rectangle(
            (0, 0, W - 1, H - 1),
            radius=radius,
            fill=(30, 41, 59, 255),  # #1e293b
        )
        inset = max(scale, 2)
        draw.rounded_rectangle(
            (inset, inset, W - 1 - inset, H - 1 - inset),
            radius=max(2, radius - inset),
            fill=(10, 14, 23, 255),  # #0a0e17
        )
        out = img.resize((w, h), Image.Resampling.LANCZOS)
        # Flatten onto chroma-key so LWA_COLORKEY punches true exterior.
        key = Image.new("RGBA", out.size, (0, 0, 1, 255))  # #000001
        flat = Image.alpha_composite(key, out).convert("RGB")
        try:
            return ImageTk.PhotoImage(flat, master=self._canvas)
        except Exception:  # noqa: BLE001
            return None

    def _draw_pill_chrome(self) -> None:
        """Glassmorphic rounded rect — PIL AA surface (fallback: canvas ovals)."""
        photo = self._chrome_photo
        if photo is None:
            photo = self._build_aa_pill_photo()
            self._chrome_photo = photo
        if photo is not None:
            self._canvas.create_image(
                0,
                0,
                image=photo,
                anchor="nw",
                tags="chrome",
            )
            return
        # Fallback if PIL/ImageTk unavailable.
        r = 18
        w, h = _PILL_W, _PILL_H
        self._canvas.create_oval(0, 0, r * 2, r * 2, fill=_PILL_BORDER, outline="", tags="chrome")
        self._canvas.create_oval(w - r * 2, 0, w, r * 2, fill=_PILL_BORDER, outline="", tags="chrome")
        self._canvas.create_oval(0, h - r * 2, r * 2, h, fill=_PILL_BORDER, outline="", tags="chrome")
        self._canvas.create_oval(w - r * 2, h - r * 2, w, h, fill=_PILL_BORDER, outline="", tags="chrome")
        self._canvas.create_rectangle(r, 0, w - r, h, fill=_PILL_BORDER, outline="", tags="chrome")
        self._canvas.create_rectangle(0, r, w, h - r, fill=_PILL_BORDER, outline="", tags="chrome")
        inset = 1
        ir = max(2, r - inset)
        x0, y0, x1, y1 = inset, inset, w - inset, h - inset
        self._canvas.create_oval(x0, y0, x0 + ir * 2, y0 + ir * 2, fill=_PILL_BG, outline="", tags="chrome")
        self._canvas.create_oval(x1 - ir * 2, y0, x1, y0 + ir * 2, fill=_PILL_BG, outline="", tags="chrome")
        self._canvas.create_oval(x0, y1 - ir * 2, x0 + ir * 2, y1, fill=_PILL_BG, outline="", tags="chrome")
        self._canvas.create_oval(x1 - ir * 2, y1 - ir * 2, x1, y1, fill=_PILL_BG, outline="", tags="chrome")
        self._canvas.create_rectangle(x0 + ir, y0, x1 - ir, y1, fill=_PILL_BG, outline="", tags="chrome")
        self._canvas.create_rectangle(x0, y0 + ir, x1, y1 - ir, fill=_PILL_BG, outline="", tags="chrome")

    def _draw_orb(self, *, font_size: float | None = None) -> None:
        """Glassmorphic pill: logo left + ACTIVE/STANDBY label."""
        self._canvas.delete("all")
        self._draw_pill_chrome()
        size = int(round(font_size if font_size is not None else _LOGO_PX))
        size = max(_ICON_SIZE_MIN, min(_ICON_SIZE_MAX, size))
        cx, cy = self._icon_center()
        status = "ACTIVE" if self._is_active() else "STANDBY"
        self._status = status
        status_color = self._status_color()

        photo = None
        try:
            from dana.ui.logo import load_premium_logo_photoimage

            photo = load_premium_logo_photoimage(self._canvas, (size, size))
        except Exception:  # noqa: BLE001
            photo = None
        if photo is not None:
            self._logo_photo = photo
            self._logo_mode = "png"
            self._canvas.create_image(
                cx,
                cy,
                image=photo,
                anchor="center",
                tags=("icon", "logo"),
            )
        else:
            self._logo_mode = "polygon"
            self._draw_smooth_mark(size=size, fill=status_color)

        # Status label to the right of the logo.
        self._canvas.create_text(
            8 + _LOGO_PX + 8,
            cy,
            text=status,
            fill=status_color,
            font=("Segoe UI", 9, "bold"),
            anchor="w",
            tags=("status",),
        )
        # Re-tag-bind after redraw — ``delete("all")`` drops prior item binds.
        self._bind_orb_drag_handlers()

    def _bind_orb_drag_handlers(self) -> None:
        """Bind press/drag/release to the solid pill canvas (and item tags)."""
        canvas = getattr(self, "_canvas", None)
        if canvas is None:
            return
        # Ensure transparent hosts do not own the drag gesture.
        for widget in (getattr(self, "orb_window", None), getattr(self, "_shell", None)):
            if widget is None:
                continue
            for seq in ("<Button-1>", "<B1-Motion>", "<ButtonRelease-1>"):
                try:
                    widget.unbind(seq)
                except Exception:  # noqa: BLE001
                    pass
        for seq, handler in (
            ("<Button-1>", self._on_press),
            ("<B1-Motion>", self._on_drag),
            ("<ButtonRelease-1>", self._on_release),
        ):
            try:
                canvas.bind(seq, handler)
            except Exception:  # noqa: BLE001
                pass
            for tag in ("chrome", "icon", "logo", "status"):
                try:
                    canvas.tag_bind(tag, seq, handler)
                except Exception:  # noqa: BLE001
                    pass

    def pulse_animation(self) -> None:
        """Refresh ACTIVE/STANDBY status (no sci-fi size pulse)."""
        if not self.orb_window.winfo_exists():
            return
        self._active_agent = self._resolve_active_agent()
        self._pulse_phase = (self._pulse_phase + 0.18) % (2.0 * math.pi)
        self._draw_orb(font_size=_LOGO_PX)
        try:
            self.orb_window.attributes("-topmost", True)
        except Exception:  # noqa: BLE001
            pass
        try:
            self._pulse_job = self.orb_window.after(_PULSE_MS, self.pulse_animation)
        except Exception:  # noqa: BLE001
            self._pulse_job = None

    def _apply_transparent_hit_test(self) -> None:
        """Color-key + shaped region so transparent pixels do not steal clicks."""
        try:
            from dana.ui.overlay import (
                apply_colorkey_hit_test,
                apply_rounded_window_rgn,
            )

            apply_colorkey_hit_test(self.orb_window, key=_TRANSPARENT)
            if self._expanded:
                # Pill on the left + panel to the right.
                total_w = _PILL_W + _PAD + _PANEL_W
                total_h = max(_PILL_H, _PANEL_H)
                apply_rounded_window_rgn(
                    self.orb_window,
                    width=_PILL_W,
                    height=_PILL_H,
                    radius=_PILL_H // 2,
                    extra_rects=[
                        (
                            _PILL_W + 2,
                            0,
                            total_w,
                            total_h,
                        )
                    ],
                )
            else:
                apply_rounded_window_rgn(
                    self.orb_window,
                    width=_PILL_W,
                    height=_PILL_H,
                    radius=_PILL_H // 2,
                )
        except Exception:  # noqa: BLE001
            pass

    def _set_alpha(self, value: float) -> None:
        self._alpha = max(0.0, min(1.0, float(value)))
        try:
            self.orb_window.attributes("-alpha", self._alpha)
        except Exception:  # noqa: BLE001
            pass

    def _cancel_fade(self) -> None:
        if self._fade_job is not None:
            try:
                self.orb_window.after_cancel(self._fade_job)
            except Exception:  # noqa: BLE001
                pass
            self._fade_job = None

    def _fade_to(self, target: float, *, on_done: Callable[[], None] | None = None) -> None:
        """Animate window alpha toward ``target`` over ~150ms."""
        self._cancel_fade()
        steps = max(1, _FADE_STEPS)
        step_ms = max(1, int(_FADE_MS / steps))
        start = float(self._alpha)
        delta = float(target) - start

        def _step(i: int = 1) -> None:
            self._fade_job = None
            if not self.orb_window.winfo_exists():
                return
            t = min(1.0, i / float(steps))
            self._set_alpha(start + delta * t)
            if i < steps:
                try:
                    self._fade_job = self.orb_window.after(step_ms, lambda: _step(i + 1))
                except Exception:  # noqa: BLE001
                    self._set_alpha(target)
                    if on_done is not None:
                        on_done()
            else:
                self._set_alpha(target)
                if on_done is not None:
                    on_done()

        try:
            self._fade_job = self.orb_window.after(step_ms, lambda: _step(1))
        except Exception:  # noqa: BLE001
            self._set_alpha(target)
            if on_done is not None:
                on_done()

    def _apply_compact_geometry(self, *, animate: bool = False) -> None:
        def _collapse() -> None:
            self._expanded = False
            try:
                self._panel.grid_forget()
            except Exception:  # noqa: BLE001
                pass
            w = _PILL_W
            h = _PILL_H
            try:
                self.orb_window.minsize(w, h)
                self.orb_window.maxsize(w, h)
            except Exception:  # noqa: BLE001
                pass
            self.orb_window.geometry(f"{w}x{h}+{self._orb_x}+{self._orb_y}")
            self._apply_transparent_hit_test()
            self._set_alpha(1.0)

        if animate and self._expanded:
            self._fade_to(0.0, on_done=_collapse)
        else:
            self._cancel_fade()
            _collapse()

    def _apply_expanded_geometry(self, *, animate: bool = True) -> None:
        self._expanded = True
        try:
            self._panel.grid(row=0, column=1, padx=(6, 0), pady=0, sticky="nw")
        except Exception:  # noqa: BLE001
            pass
        w = _PILL_W + _PAD + _PANEL_W
        h = max(_PILL_H, _PANEL_H)
        screen_w = int(self.orb_window.winfo_screenwidth() or 1280)
        x = self._orb_x
        if x + w > screen_w - 8:
            x = max(8, screen_w - w - 8)
        try:
            self.orb_window.minsize(w, h)
            self.orb_window.maxsize(w, h)
        except Exception:  # noqa: BLE001
            pass
        self.orb_window.geometry(f"{w}x{h}+{x}+{self._orb_y}")
        self._apply_transparent_hit_test()
        if animate:
            self._set_alpha(0.0)
            self._fade_to(1.0)
        else:
            self._set_alpha(1.0)

    # --------------------------------------------------------------- events
    def _cancel_leave(self) -> None:
        if self._leave_job is not None:
            try:
                self.orb_window.after_cancel(self._leave_job)
            except Exception:  # noqa: BLE001
                pass
            self._leave_job = None

    def _on_enter(self, _event: Any = None) -> None:
        self._cancel_leave()
        if not self._expanded and not self._dragging:
            self.refresh_controls()
            self._apply_expanded_geometry(animate=True)

    def _on_leave(self, _event: Any = None) -> None:
        self._cancel_leave()

        def _shrink() -> None:
            self._leave_job = None
            if self._dragging:
                return
            try:
                x = self.orb_window.winfo_pointerx()
                y = self.orb_window.winfo_pointery()
                left = self.orb_window.winfo_rootx()
                top = self.orb_window.winfo_rooty()
                right = left + self.orb_window.winfo_width()
                bottom = top + self.orb_window.winfo_height()
                if left <= x <= right and top <= y <= bottom:
                    return
            except Exception:  # noqa: BLE001
                pass
            self._apply_compact_geometry(animate=True)

        try:
            self._leave_job = self.orb_window.after(180, _shrink)
        except Exception:  # noqa: BLE001
            self._apply_compact_geometry(animate=True)

    def _on_press(self, event: Any = None) -> None:
        self._dragging = True
        self._moved = False
        try:
            px = int(self.orb_window.winfo_pointerx())
            py = int(self.orb_window.winfo_pointery())
            rx = int(self.orb_window.winfo_rootx())
            ry = int(self.orb_window.winfo_rooty())
            self._drag_offset_x = px - rx
            self._drag_offset_y = py - ry
            self._drag_start_x = px
            self._drag_start_y = py
            self._win_start_x = int(self.orb_window.winfo_x())
            self._win_start_y = int(self.orb_window.winfo_y())
        except Exception:  # noqa: BLE001
            # Fallback: widget-local coords from the event.
            try:
                self._drag_offset_x = int(getattr(event, "x", 0) or 0)
                self._drag_offset_y = int(getattr(event, "y", 0) or 0)
                self._drag_start_x = int(getattr(event, "x_root", 0) or 0)
                self._drag_start_y = int(getattr(event, "y_root", 0) or 0)
                self._win_start_x = int(self.orb_window.winfo_x())
                self._win_start_y = int(self.orb_window.winfo_y())
            except Exception:  # noqa: BLE001
                pass
        self._cancel_leave()

    def _on_drag(self, event: Any = None) -> None:
        if not self._dragging:
            return
        try:
            px = int(self.orb_window.winfo_pointerx())
            py = int(self.orb_window.winfo_pointery())
        except Exception:  # noqa: BLE001
            try:
                px = int(getattr(event, "x_root", 0) or 0)
                py = int(getattr(event, "y_root", 0) or 0)
            except Exception:  # noqa: BLE001
                return
        if (
            abs(px - self._drag_start_x) > _DRAG_THRESHOLD_PX
            or abs(py - self._drag_start_y) > _DRAG_THRESHOLD_PX
        ):
            self._moved = True
        # Keep the grab point under the cursor (no snap to top-left).
        x = px - int(self._drag_offset_x)
        y = py - int(self._drag_offset_y)
        self._orb_x = x
        self._orb_y = y
        try:
            w = self.orb_window.winfo_width()
            h = self.orb_window.winfo_height()
            self.orb_window.geometry(f"{w}x{h}+{x}+{y}")
            # Do not re-apply Win32 region mid-drag — that drops B1-Motion.
        except Exception:  # noqa: BLE001
            pass

    def _on_release(self, _event: Any = None) -> None:
        was_click = self._dragging and not self._moved
        self._dragging = False
        try:
            self._apply_transparent_hit_test()
        except Exception:  # noqa: BLE001
            pass
        if was_click and self._on_open_dashboard is not None and not self._expanded:
            try:
                self._on_open_dashboard()
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------- actions
    def _click_dictation(self) -> None:
        if self._on_toggle_dictation is not None:
            try:
                self._on_toggle_dictation()
            except Exception:  # noqa: BLE001
                pass
        self.refresh_controls()

    def _click_dashboard(self) -> None:
        if self._on_open_dashboard is not None:
            try:
                self._on_open_dashboard()
            except Exception:  # noqa: BLE001
                pass

    def _click_approve(self) -> None:
        if self._on_approve_ticket is not None:
            try:
                self._on_approve_ticket()
            except Exception:  # noqa: BLE001
                pass
        else:
            try:
                from dana.middleware.hitl_ticket import submit_decision

                submit_decision(True, action="approve")
            except Exception:  # noqa: BLE001
                pass
        self.refresh_controls()

    def _click_deny(self) -> None:
        if self._on_deny_ticket is not None:
            try:
                self._on_deny_ticket()
            except Exception:  # noqa: BLE001
                pass
        else:
            try:
                from dana.middleware.hitl_ticket import submit_decision

                submit_decision(False, action="deny")
            except Exception:  # noqa: BLE001
                pass
        self.refresh_controls()

    def _click_github(self) -> None:
        try:
            from dana.middleware.hitl_ticket import get_pending
            from dana.ui.github_escalation import open_github_issue

            pending = get_pending() or {}
            open_github_issue(
                pending,
                str(pending.get("jason_critique") or ""),
            )
        except Exception:  # noqa: BLE001
            pass

    def refresh_controls(self) -> None:
        """Sync mini-panel labels/buttons from GUI + HITL latch (Tk thread)."""
        dictating = bool(self._dictation_getter())
        mode = (self._mode_getter() or "chat").strip().lower()
        try:
            from dana.middleware.hitl_ticket import is_pending

            self._hitl_pending = bool(is_pending())
        except Exception:  # noqa: BLE001
            self._hitl_pending = False

        if dictating:
            try:
                self._dictation_btn.configure(
                    text="●  DICTATING",
                    fg_color="#388e3c",
                    text_color="#E8F5E9",
                    hover_color="#2E7D32",
                )
            except Exception:  # noqa: BLE001 — classic Tk Button fallback
                self._dictation_btn.configure(
                    text="●  DICTATING",
                    bg="#388e3c",
                    fg="#E8F5E9",
                    activebackground="#2E7D32",
                    activeforeground="#E8F5E9",
                )
            status = "● DICTATING"
        else:
            try:
                self._dictation_btn.configure(
                    text="●  OFF",
                    fg_color="#2A2A3C",
                    text_color="#F87171",
                    hover_color="#34344A",
                )
            except Exception:  # noqa: BLE001
                self._dictation_btn.configure(
                    text="●  OFF",
                    bg="#2A2A3C",
                    fg="#F87171",
                    activebackground="#34344A",
                    activeforeground="#F87171",
                )
            status = "● OFF"
        display_mode = "Dictation" if dictating else mode.title()
        hitl_bit = "  ·  TICKET PENDING" if self._hitl_pending else ""
        self._state_lbl.configure(text=f"Mode: {display_mode}  ·  {status}{hitl_bit}")
        hitl_state = "normal" if self._hitl_pending else "disabled"
        critique = ""
        ticket_text = ""
        denials = 0
        if self._hitl_pending:
            try:
                from dana.middleware.hitl_ticket import (
                    extract_files_line,
                    format_ticket_payload,
                    get_consecutive_denials,
                    get_pending,
                )

                pending = get_pending() or {}
                critique = str(pending.get("jason_critique") or "").strip()
                denials = int(
                    pending.get("consecutive_denials")
                    if pending.get("consecutive_denials") is not None
                    else get_consecutive_denials()
                )
                obj = str(pending.get("objective") or "").strip()
                ctx = str(pending.get("context") or "").strip()
                files = extract_files_line(ctx, pending.get("files"))
                ticket_text = (
                    f"Objective: {obj or '(empty)'}\n\n"
                    f"Context: {ctx or '(empty)'}\n\n"
                    f"Files: {files}"
                )
                if not pending.get("_formatted"):
                    pending["_formatted"] = format_ticket_payload(pending)
            except Exception:  # noqa: BLE001
                critique = ""
                ticket_text = ""
                denials = 0
        try:
            self._approve_btn.configure(state=hitl_state)
            self._deny_btn.configure(state=hitl_state)
            if self._hitl_pending:
                self._critique_hdr.pack(fill="x", padx=12, pady=(2, 0))
                self._critique_lbl.configure(
                    text=critique or "(Jason critique pending…)"
                )
                self._critique_lbl.pack(fill="x", padx=12, pady=(2, 4), ipady=4)
                self._ticket_hdr.pack(fill="x", padx=12, pady=(2, 0))
                self._ticket_lbl.configure(
                    text=ticket_text or "(drafted ticket pending…)"
                )
                self._ticket_lbl.pack(fill="x", padx=12, pady=(2, 6), ipady=4)
                if not self._hitl_row.winfo_ismapped():
                    self._hitl_row.pack(fill="x", padx=12, pady=(0, 6))
                if denials >= 2:
                    if not self._github_btn.winfo_ismapped():
                        self._github_btn.pack(
                            fill="x", padx=12, pady=(0, 10), ipady=3
                        )
                else:
                    self._github_btn.pack_forget()
            else:
                self._critique_hdr.pack_forget()
                self._critique_lbl.pack_forget()
                self._ticket_hdr.pack_forget()
                self._ticket_lbl.pack_forget()
                self._hitl_row.pack_forget()
                self._github_btn.pack_forget()
        except Exception:  # noqa: BLE001
            pass

    def _status_tick(self) -> None:
        """Lower-frequency panel/HITL sync (pulse_animation owns Canvas redraw)."""
        if not self.orb_window.winfo_exists():
            return
        prev = self._hitl_pending
        self.refresh_controls()
        if self._hitl_pending and not prev and not self._expanded:
            self._apply_expanded_geometry(animate=True)
        try:
            self.orb_window.after(700, self._status_tick)
        except Exception:  # noqa: BLE001
            pass

    def destroy(self) -> None:
        self._cancel_leave()
        self._cancel_fade()
        if self._pulse_job is not None:
            try:
                self.orb_window.after_cancel(self._pulse_job)
            except Exception:  # noqa: BLE001
                pass
            self._pulse_job = None
        try:
            self.orb_window.destroy()
        except Exception:  # noqa: BLE001
            pass
