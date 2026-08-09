"""Lightweight hover tooltips for CustomTkinter controls.

Installed CustomTkinter builds may lack ``CTkToolTip``; this module provides a
compatible helper used by the Dana control dashboard.
"""

from __future__ import annotations

import tkinter as tk
from typing import Any


class CTkToolTip:
    """Delayed hover tooltip bound to a widget."""

    def __init__(
        self,
        widget: Any,
        text: str,
        *,
        delay_ms: int = 450,
        wraplength: int = 280,
    ) -> None:
        self.widget = widget
        self.text = str(text or "").strip()
        self.delay_ms = max(50, int(delay_ms))
        self.wraplength = max(120, int(wraplength))
        self._job: str | None = None
        self._tip: tk.Toplevel | None = None
        if not self.text or widget is None:
            return
        try:
            widget.bind("<Enter>", self._on_enter, add="+")
            widget.bind("<Leave>", self._on_leave, add="+")
            widget.bind("<ButtonPress>", self._on_leave, add="+")
        except Exception:  # noqa: BLE001
            pass

    def _on_enter(self, _event: Any = None) -> None:
        self._cancel()
        try:
            self._job = self.widget.after(self.delay_ms, self._show)
        except Exception:  # noqa: BLE001
            self._job = None

    def _on_leave(self, _event: Any = None) -> None:
        self._cancel()
        self._hide()

    def _cancel(self) -> None:
        if self._job is None:
            return
        try:
            self.widget.after_cancel(self._job)
        except Exception:  # noqa: BLE001
            pass
        self._job = None

    def _hide(self) -> None:
        tip = self._tip
        self._tip = None
        if tip is None:
            return
        try:
            tip.destroy()
        except Exception:  # noqa: BLE001
            pass

    def _show(self) -> None:
        self._job = None
        self._hide()
        if not self.text:
            return
        try:
            if not bool(self.widget.winfo_exists()):
                return
        except Exception:  # noqa: BLE001
            return
        try:
            tip = tk.Toplevel(self.widget)
            tip.wm_overrideredirect(True)
            tip.attributes("-topmost", True)
            tip.configure(bg="#0f172a")
            lbl = tk.Label(
                tip,
                text=self.text,
                justify="left",
                background="#0f172a",
                foreground="#F8FAFC",
                relief="solid",
                borderwidth=1,
                font=("Segoe UI", 9),
                padx=8,
                pady=6,
                wraplength=self.wraplength,
            )
            lbl.pack()
            x = int(self.widget.winfo_rootx()) + 12
            y = int(self.widget.winfo_rooty()) + int(self.widget.winfo_height()) + 6
            tip.geometry(f"+{x}+{y}")
            self._tip = tip
        except Exception:  # noqa: BLE001
            self._tip = None


def attach_tooltip(widget: Any, text: str, **kwargs: Any) -> CTkToolTip | None:
    """Attach a tooltip; returns the helper or None on failure."""
    try:
        return CTkToolTip(widget, text, **kwargs)
    except Exception:  # noqa: BLE001
        return None


__all__ = ("CTkToolTip", "attach_tooltip")
