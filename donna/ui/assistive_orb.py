"""Stage 8.7 / 8.8 / 8.9.9 — Floating AssistiveTouch orb (frameless, topmost).

A always-on-top orb that the operator can drag anywhere on the desktop.
Hover expands a mini control strip (dictation + HITL + dashboard) without
touching the LangGraph worker threads.

Stage 8.8 — Canvas ``pulse_animation`` color tracks active TTS ``agent_id``.
Stage 8.9.9 — High-fidelity LANCZOS PNG logo (PhotoImage); Unicode glyph removed.
Fallback: mathematically smooth Canvas polygons (``smooth=True``).
"""

from __future__ import annotations

import math
import tkinter as tk
from typing import Any, Callable

# Chroma-key for Windows layered transparency (must not appear in drawn pixels).
_TRANSPARENT = "#010101"
_ORB_DIAMETER = 64
_PANEL_W = 320
_PANEL_H = 360
_PAD = 8
_DRAG_THRESHOLD_PX = 4
_PULSE_MS = 40  # ~25 FPS Canvas redraw
# Stage 8.9.9 — logo pixel size range for LANCZOS pulse (no font glyphs).
_ICON_SIZE_MIN = 30
_ICON_SIZE_MAX = 45
# Legacy aliases (older tests that still reference radius-era names).
_PULSE_BASE_R = float((_ICON_SIZE_MIN + _ICON_SIZE_MAX) // 2)
_PULSE_AMP = float((_ICON_SIZE_MAX - _ICON_SIZE_MIN) // 2)

_COLOR_CHAT = "#00E676"
_COLOR_DICTATION = "#9C27B0"
_COLOR_EXEC = "#FB8C00"
_COLOR_HITL = "#388e3c"
_COLOR_IDLE = "#9CA3AF"
_PANEL_BG = "#1E1E2E"
_PANEL_FG = "#F3F4F6"
_MUTED = "#9AA0A6"


class AssistiveTouchOrb:
    """Draggable always-on-top orb owned by the DonnaGUI Tk root."""

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
        self._win_start_x = 0
        self._win_start_y = 0
        self._moved = False
        self._leave_job: str | None = None
        self._pulse_phase = 0.0
        self._pulse_job: str | None = None
        self._hitl_pending = False
        self._active_agent = "broker"
        # Keep PhotoImage refs alive (Tk GC otherwise blanks the canvas).
        self._logo_photo: Any | None = None
        self._logo_mode = "png"  # "png" | "polygon"

        self.orb_window = tk.Toplevel(master)
        self.orb_window.title("Donna Orb")
        self.orb_window.overrideredirect(True)
        try:
            self.orb_window.attributes("-topmost", True)
        except Exception:  # noqa: BLE001
            pass
        try:
            # Windows: punch out chroma-key so only the orb/panel are visible.
            self.orb_window.wm_attributes("-transparentcolor", _TRANSPARENT)
        except Exception:  # noqa: BLE001
            pass
        self.orb_window.configure(bg=_TRANSPARENT)

        screen_w = int(self.orb_window.winfo_screenwidth() or 1280)
        screen_h = int(self.orb_window.winfo_screenheight() or 800)
        x = max(24, screen_w - _ORB_DIAMETER - 36)
        y = max(80, screen_h // 3)
        self._orb_x = x
        self._orb_y = y

        self._shell = tk.Frame(self.orb_window, bg=_TRANSPARENT, bd=0, highlightthickness=0)
        self._shell.pack(fill="both", expand=True)

        self._canvas = tk.Canvas(
            self._shell,
            width=_ORB_DIAMETER,
            height=_ORB_DIAMETER,
            bg=_TRANSPARENT,
            highlightthickness=0,
            bd=0,
        )
        self._canvas.grid(row=0, column=0, padx=0, pady=0, sticky="nw")

        self._panel = tk.Frame(
            self._shell,
            bg=_PANEL_BG,
            bd=0,
            highlightthickness=1,
            highlightbackground="#2A2A3C",
            width=_PANEL_W,
            height=_PANEL_H,
        )
        # Built but not mapped until hover expand.
        self._build_panel()

        self._draw_orb()
        self._apply_compact_geometry()

        # Drag + hover (Tk main thread only).
        for widget in (self._canvas, self.orb_window, self._shell):
            widget.bind("<Button-1>", self._on_press)
            widget.bind("<B1-Motion>", self._on_drag)
            widget.bind("<ButtonRelease-1>", self._on_release)
        self._shell.bind("<Enter>", self._on_enter)
        self._shell.bind("<Leave>", self._on_leave)
        self._canvas.bind("<Enter>", self._on_enter)
        self._canvas.bind("<Leave>", self._on_leave)
        self._panel.bind("<Enter>", self._on_enter)
        self._panel.bind("<Leave>", self._on_leave)

        # Stage 8.8 — smooth Canvas pulse + status / HITL latch (Tk main thread).
        try:
            self.orb_window.after(80, self.pulse_animation)
            self.orb_window.after(400, self._status_tick)
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------ UI
    def _build_panel(self) -> None:
        title = tk.Label(
            self._panel,
            text="DONNA",
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

        # Stage 8.9 — Jason critique shown above Approve / Deny.
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
        # Stage 8.9.6 — full drafted ticket above Approve / Deny.
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
            bg="#00ADB5",
            fg="#FFFFFF",
            activebackground="#008E95",
            activeforeground="#FFFFFF",
            relief="flat",
            bd=0,
            font=("Segoe UI", 9, "bold"),
            cursor="hand2",
            command=self._click_dashboard,
        )
        self._dash_btn.pack(fill="x", padx=12, pady=(0, 6), ipady=3)

        hitl_row = tk.Frame(self._panel, bg=_PANEL_BG)
        # Packed dynamically when HITL is pending (critique sits above buttons).
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
        # Stage 8.9.3 — packed only when consecutive_denials >= 2.
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
            from donna.audio.multi_voice_tts import get_active_tts_agent

            return str(get_active_tts_agent() or "broker")
        except Exception:  # noqa: BLE001
            return "broker"

    def _accent(self) -> str:
        """Pulse fill color — persona colors win over mode accents when speaking."""
        if self._dictation_getter():
            return _COLOR_DICTATION
        if self._hitl_pending:
            return _COLOR_HITL
        try:
            from donna.audio.multi_voice_tts import persona_color_for_agent

            return persona_color_for_agent(self._active_agent)
        except Exception:  # noqa: BLE001
            pass
        if self._accent_getter is not None:
            try:
                return str(self._accent_getter() or _COLOR_CHAT)
            except Exception:  # noqa: BLE001
                pass
        mode = (self._mode_getter() or "chat").strip().lower()
        if mode in {"developer", "execution", "agentic"}:
            return _COLOR_EXEC
        return _COLOR_CHAT

    def _icon_center(self) -> tuple[float, float]:
        """Canvas midpoint — logo stays centered as size changes."""
        cx = cy = _ORB_DIAMETER / 2.0
        return cx, cy

    def _draw_smooth_mark(self, *, size: int, fill: str) -> None:
        """Fallback mark — smooth polygons (no Unicode / no jagged ovals)."""
        cx, cy = self._icon_center()
        s = max(8.0, float(size) * 0.42)
        # Stylized "Da" wedges as mathematically smooth polygons.
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

    def _draw_orb(self, *, font_size: float | None = None) -> None:
        """Stage 8.9.9 — LANCZOS PNG logo (tinted) or smooth polygon fallback."""
        self._canvas.delete("all")
        accent = self._accent()
        cx, cy = self._icon_center()
        base = (_ICON_SIZE_MIN + _ICON_SIZE_MAX) / 2.0
        size = int(round(font_size if font_size is not None else base))
        size = max(_ICON_SIZE_MIN, min(_ICON_SIZE_MAX, size))
        photo = None
        try:
            from donna.ui.logo import load_premium_logo_photoimage

            photo = load_premium_logo_photoimage(
                self._canvas, (size, size), tint=accent
            )
        except Exception:  # noqa: BLE001
            photo = None
        if photo is not None:
            self._logo_photo = photo  # prevent GC
            self._logo_mode = "png"
            self._canvas.create_image(
                cx,
                cy,
                image=photo,
                anchor="center",
                tags=("icon", "logo"),
            )
            return
        self._logo_mode = "polygon"
        self._draw_smooth_mark(size=size, fill=accent)

    def pulse_animation(self) -> None:
        """Stage 8.9.9 — sine-wave logo scale + persona tint (LANCZOS redraw)."""
        if not self.orb_window.winfo_exists():
            return
        self._active_agent = self._resolve_active_agent()
        speaking = False
        try:
            from donna.core_agent import tts_busy

            speaking = bool(tts_busy.is_set())
        except Exception:  # noqa: BLE001
            speaking = False
        # Full 30–45 swing while speaking / HITL; gentler breath when idle.
        self._pulse_phase = (self._pulse_phase + 0.18) % (2.0 * math.pi)
        wave = math.sin(self._pulse_phase)
        mid = (_ICON_SIZE_MIN + _ICON_SIZE_MAX) / 2.0
        amp = (_ICON_SIZE_MAX - _ICON_SIZE_MIN) / 2.0
        if not (speaking or self._hitl_pending):
            amp *= 0.45
        current_size = int(round(mid + amp * wave))
        current_size = max(_ICON_SIZE_MIN, min(_ICON_SIZE_MAX, current_size))
        # Redraw each frame so LANCZOS resize + tint stay crisp (no font glyphs).
        self._draw_orb(font_size=current_size)
        try:
            self.orb_window.attributes("-topmost", True)
        except Exception:  # noqa: BLE001
            pass
        try:
            self._pulse_job = self.orb_window.after(_PULSE_MS, self.pulse_animation)
        except Exception:  # noqa: BLE001
            self._pulse_job = None

    def _apply_compact_geometry(self) -> None:
        self._expanded = False
        try:
            self._panel.grid_forget()
        except Exception:  # noqa: BLE001
            pass
        w = _ORB_DIAMETER
        h = _ORB_DIAMETER
        self.orb_window.geometry(f"{w}x{h}+{self._orb_x}+{self._orb_y}")

    def _apply_expanded_geometry(self) -> None:
        self._expanded = True
        try:
            self._panel.grid(row=0, column=1, padx=(6, 0), pady=0, sticky="nw")
        except Exception:  # noqa: BLE001
            pass
        w = _ORB_DIAMETER + _PAD + _PANEL_W
        h = max(_ORB_DIAMETER, _PANEL_H)
        # Keep panel on-screen: flip left if near right edge.
        screen_w = int(self.orb_window.winfo_screenwidth() or 1280)
        x = self._orb_x
        if x + w > screen_w - 8:
            x = max(8, screen_w - w - 8)
        self.orb_window.geometry(f"{w}x{h}+{x}+{self._orb_y}")

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
            self._apply_expanded_geometry()

    def _on_leave(self, _event: Any = None) -> None:
        # Debounce: moving between canvas/panel children fires Leave on parents.
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
            self._apply_compact_geometry()

        try:
            self._leave_job = self.orb_window.after(180, _shrink)
        except Exception:  # noqa: BLE001
            self._apply_compact_geometry()

    def _on_press(self, event: Any) -> None:
        self._dragging = True
        self._moved = False
        self._drag_start_x = int(event.x_root)
        self._drag_start_y = int(event.y_root)
        self._win_start_x = int(self.orb_window.winfo_x())
        self._win_start_y = int(self.orb_window.winfo_y())
        self._cancel_leave()

    def _on_drag(self, event: Any) -> None:
        if not self._dragging:
            return
        dx = int(event.x_root) - self._drag_start_x
        dy = int(event.y_root) - self._drag_start_y
        if abs(dx) > _DRAG_THRESHOLD_PX or abs(dy) > _DRAG_THRESHOLD_PX:
            self._moved = True
        x = self._win_start_x + dx
        y = self._win_start_y + dy
        self._orb_x = x
        self._orb_y = y
        try:
            w = self.orb_window.winfo_width()
            h = self.orb_window.winfo_height()
            self.orb_window.geometry(f"{w}x{h}+{x}+{y}")
        except Exception:  # noqa: BLE001
            pass

    def _on_release(self, _event: Any = None) -> None:
        was_click = self._dragging and not self._moved
        self._dragging = False
        if was_click and self._on_open_dashboard is not None and not self._expanded:
            # Compact click → open full dashboard.
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
                from donna.middleware.hitl_ticket import submit_decision

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
                from donna.middleware.hitl_ticket import submit_decision

                submit_decision(False, action="deny")
            except Exception:  # noqa: BLE001
                pass
        self.refresh_controls()

    def _click_github(self) -> None:
        try:
            from donna.middleware.hitl_ticket import get_pending
            from donna.ui.github_escalation import open_github_issue

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
            from donna.middleware.hitl_ticket import is_pending

            self._hitl_pending = bool(is_pending())
        except Exception:  # noqa: BLE001
            self._hitl_pending = False

        if dictating:
            self._dictation_btn.configure(
                text="●  DICTATING",
                bg="#388e3c",
                fg="#E8F5E9",
                activebackground="#2E7D32",
                activeforeground="#E8F5E9",
            )
            status = "● DICTATING"
        else:
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
                from donna.middleware.hitl_ticket import (
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
                # Compact Orb card — full body also lands in Payload Viewer.
                obj = str(pending.get("objective") or "").strip()
                ctx = str(pending.get("context") or "").strip()
                files = extract_files_line(ctx, pending.get("files"))
                ticket_text = (
                    f"Objective: {obj or '(empty)'}\n\n"
                    f"Context: {ctx or '(empty)'}\n\n"
                    f"Files: {files}"
                )
                # Keep formatted string on pending for Live Trace Payload Viewer.
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
        # Pulse loop owns redraw; only refresh labels here.

    def _status_tick(self) -> None:
        """Lower-frequency panel/HITL sync (pulse_animation owns Canvas redraw)."""
        if not self.orb_window.winfo_exists():
            return
        prev = self._hitl_pending
        self.refresh_controls()
        # Auto-expand when a ticket needs approval so the operator notices.
        if self._hitl_pending and not prev and not self._expanded:
            self._apply_expanded_geometry()
        try:
            self.orb_window.after(700, self._status_tick)
        except Exception:  # noqa: BLE001
            pass

    def destroy(self) -> None:
        self._cancel_leave()
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
