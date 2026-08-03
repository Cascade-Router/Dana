"""Chat bubble transcript for the Assistant & Tasks dashboard."""

from __future__ import annotations

import re
from typing import Any

import customtkinter as ctk

from dana.ui import theme as T

_SYSTEM_TAG_RE = re.compile(r"^\[([^\]]+)\]\s*(.*)$", re.DOTALL)
_MD_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_MD_CODE_RE = re.compile(r"`([^`]+)`")


def _strip_simple_markdown(text: str) -> str:
    """Best-effort markdown → plain text for CTkLabel (no HTML renderer)."""
    out = _MD_BOLD_RE.sub(r"\1", text)
    out = _MD_CODE_RE.sub(r"\1", out)
    return out


def _classify_role(speaker: str, agent_id: str | None = None) -> str:
    sp = (speaker or "").strip().lower()
    aid = (agent_id or "").strip().lower()
    if sp.startswith("user") or aid in {"user", "user_text"}:
        return "user"
    system_hints = (
        "vision",
        "system",
        "tool",
        "overlay",
        "uia",
        "grounding",
        "trace",
        "router",
    )
    if any(h in sp for h in system_hints) or any(h in aid for h in system_hints):
        return "system"
    if sp.startswith(("[vision", "vision")):
        return "system"
    return "dana"


class ChatBubbleView(ctk.CTkFrame):
    """Scrollable chat bubbles + hidden CTkTextbox mirror for API compatibility.

    ``self.transcript_box`` is a disabled CTkTextbox that still supports
    ``get`` / ``insert`` / persona tags used by existing tests and callers.
    """

    def __init__(self, master: Any, *, wraplength: int = 420) -> None:
        super().__init__(master, fg_color=T.BG, corner_radius=12)
        self._wraplength = max(200, int(wraplength))
        self._bubbles: list[ctk.CTkFrame] = []

        self._scroll = ctk.CTkScrollableFrame(
            self,
            fg_color="transparent",
            corner_radius=0,
        )
        self._scroll.pack(fill="both", expand=True, padx=4, pady=2)

        # Hidden mirror — keeps get/insert/tag API for tests & legacy callers.
        # Placed off-screen so raw [Speaker] lines never duplicate bubble UI
        # and never add a second scrollbar beside the bubble scroller.
        self.transcript_box = ctk.CTkTextbox(
            self,
            wrap="word",
            font=ctk.CTkFont(family="Segoe UI", size=14),
            fg_color=T.BG,
            text_color=T.TEXT,
            height=1,
            border_width=0,
            width=1,
        )
        # Off-screen so mirror text never paints under the bubble scroller.
        self.transcript_box.place(x=-10_000, y=-10_000)
        try:
            self.bind("<Configure>", self._on_resize, add="+")
        except Exception:  # noqa: BLE001
            pass

    def set_wraplength(self, wraplength: int) -> None:
        """Update bubble text wrap for responsive Conversation width."""
        self._wraplength = max(200, int(wraplength))

    def _on_resize(self, event: Any) -> None:
        try:
            if event is None or getattr(event, "widget", None) is not self:
                return
            width = int(getattr(event, "width", 0) or 0)
        except Exception:  # noqa: BLE001
            return
        if width < 240:
            return
        new_wrap = max(200, min(720, width - 48))
        if abs(new_wrap - self._wraplength) >= 16:
            self._wraplength = new_wrap

    def clear_bubbles(self) -> None:
        for row in self._bubbles:
            try:
                row.destroy()
            except Exception:  # noqa: BLE001
                pass
        self._bubbles.clear()

    def append_bubble(
        self,
        speaker: str,
        text: str,
        *,
        agent_id: str | None = None,
        role: str | None = None,
    ) -> None:
        role_key = (role or _classify_role(speaker, agent_id)).strip().lower()
        body = str(text or "").strip()
        if not body and not speaker:
            return

        # System tags like [Vision Output] → muted badge + remainder.
        badge = ""
        display = body
        m = _SYSTEM_TAG_RE.match(f"[{speaker}] {body}" if speaker else body)
        if role_key == "system" or (
            speaker and speaker.strip().startswith("[") is False
            and any(
                x in speaker.lower()
                for x in ("vision", "system", "tool", "overlay")
            )
        ):
            badge = speaker.strip() if speaker else "System"
            display = body
        elif speaker and not role_key == "user":
            # Keep speaker as small label above bubble for dana/agents.
            pass

        row = ctk.CTkFrame(self._scroll, fg_color="transparent")
        row.pack(fill="x", padx=6, pady=(1, 1))
        self._bubbles.append(row)

        if role_key == "user":
            self._pack_user_bubble(row, display, speaker=speaker)
        elif role_key == "system":
            self._pack_system_badge(row, badge or speaker or "System", display)
        else:
            self._pack_dana_bubble(row, display, speaker=speaker)

        self._scroll_to_latest()

    def _pack_user_bubble(
        self, row: ctk.CTkFrame, text: str, *, speaker: str
    ) -> None:
        # Right-aligned sky bubble.
        spacer = ctk.CTkFrame(row, fg_color="transparent", width=80)
        spacer.pack(side="left", fill="x", expand=True)
        bubble = ctk.CTkFrame(
            row,
            fg_color=T.BUBBLE_USER,
            corner_radius=14,
            border_width=0,
        )
        bubble.pack(side="right", padx=(8, 4), pady=1)
        label = speaker if speaker.lower().startswith("user") else "You"
        ctk.CTkLabel(
            bubble,
            text=label,
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#E0F2FE",
            anchor="e",
        ).pack(fill="x", padx=10, pady=(5, 0))
        ctk.CTkLabel(
            bubble,
            text=_strip_simple_markdown(text),
            font=ctk.CTkFont(size=13),
            text_color=T.BUBBLE_USER_TEXT,
            wraplength=self._wraplength,
            justify="left",
            anchor="w",
        ).pack(fill="x", padx=10, pady=(1, 6))

    def _pack_dana_bubble(
        self, row: ctk.CTkFrame, text: str, *, speaker: str
    ) -> None:
        bubble = ctk.CTkFrame(
            row,
            fg_color=T.BUBBLE_DANA,
            corner_radius=14,
            border_width=1,
            border_color=T.BUBBLE_DANA_BORDER,
        )
        bubble.pack(side="left", padx=(4, 8), pady=1)
        title = speaker.strip() or "Dana"
        ctk.CTkLabel(
            bubble,
            text=title,
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=T.ACCENT,
            anchor="w",
        ).pack(fill="x", padx=10, pady=(5, 0))
        ctk.CTkLabel(
            bubble,
            text=_strip_simple_markdown(text),
            font=ctk.CTkFont(size=13),
            text_color=T.TEXT,
            wraplength=self._wraplength,
            justify="left",
            anchor="w",
        ).pack(fill="x", padx=10, pady=(1, 6))
        spacer = ctk.CTkFrame(row, fg_color="transparent", width=80)
        spacer.pack(side="right", fill="x", expand=True)

    def _scroll_to_latest(self) -> None:
        """Stick the bubble scroller to the newest message (stream-safe)."""
        try:
            self.update_idletasks()
        except Exception:  # noqa: BLE001
            pass
        try:
            canvas = getattr(self._scroll, "_parent_canvas", None)
            if canvas is not None:
                canvas.yview_moveto(1.0)
                try:
                    canvas.see("end")
                except Exception:  # noqa: BLE001
                    pass
        except Exception:  # noqa: BLE001
            pass
        try:
            # CTkScrollableFrame also exposes .yview / children for fallbacks.
            self._scroll._parent_canvas.yview_moveto(1.0)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass

    def _pack_system_badge(
        self, row: ctk.CTkFrame, badge: str, text: str
    ) -> None:
        wrap = ctk.CTkFrame(row, fg_color="transparent")
        wrap.pack(fill="x", padx=4, pady=1)
        tag = badge.strip()
        if tag and not tag.startswith("["):
            tag = f"[{tag}]"
        ctk.CTkLabel(
            wrap,
            text=f"  {tag}  ",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=T.MUTED,
            fg_color=T.BUBBLE_SYSTEM,
            corner_radius=999,
        ).pack(side="left", padx=(0, 8))
        if text:
            ctk.CTkLabel(
                wrap,
                text=_strip_simple_markdown(text),
                font=ctk.CTkFont(size=11),
                text_color=T.MUTED,
                wraplength=self._wraplength + 40,
                justify="left",
                anchor="w",
            ).pack(side="left", fill="x", expand=True)


__all__ = ("ChatBubbleView", "_classify_role", "_strip_simple_markdown")
