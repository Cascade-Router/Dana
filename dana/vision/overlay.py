"""Live transparent click-through ROI overlay for CAMGRASPER vision attention.

Runs on a dedicated Tkinter thread so audio / agent work is never blocked.
Windows: layered + transparent + topmost + toolwindow (click-through).
"""

from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from typing import Any, Optional

_OVERLAY_LOCK = threading.Lock()
_OVERLAY: Optional[RoiOverlay] = None


def _enable_click_through(hwnd: int) -> None:
    """Mark a Win32 window as layered + click-through (messages pass through)."""
    try:
        import ctypes

        user32 = ctypes.windll.user32
        GWL_EXSTYLE = -20
        WS_EX_LAYERED = 0x00080000
        WS_EX_TRANSPARENT = 0x00000020
        WS_EX_TOPMOST = 0x00000008
        WS_EX_TOOLWINDOW = 0x00000080
        style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        style |= WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST | WS_EX_TOOLWINDOW
        user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style)
        # Keep always-on-top without activating.
        HWND_TOPMOST = -1
        SWP_NOMOVE = 0x0002
        SWP_NOSIZE = 0x0001
        SWP_NOACTIVATE = 0x0010
        SWP_SHOWWINDOW = 0x0040
        user32.SetWindowPos(
            hwnd,
            HWND_TOPMOST,
            0,
            0,
            0,
            0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW,
        )
    except Exception:  # noqa: BLE001
        pass


class RoiOverlay:
    """Frameless transparent always-on-top ROI box (own Tk mainloop thread)."""

    def __init__(self) -> None:
        self._cmd_q: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._root: tk.Tk | None = None
        self._canvas: tk.Canvas | None = None
        self._box_id: int | None = None
        self._label_id: int | None = None
        self._current: tuple[int, int, int, int] | None = None
        self._label = ""
        self._visible = False

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._ready.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="RoiOverlay",
            daemon=True,
        )
        self._thread.start()
        self._ready.wait(timeout=3.0)

    def stop(self) -> None:
        self._stop.set()
        try:
            self._cmd_q.put_nowait(("stop", None))
        except Exception:  # noqa: BLE001
            pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._ready.clear()

    def update_roi(self, box_coords: Any, label: str = "") -> None:
        """Draw/move the ROI box. ``box_coords`` = (x1, y1, x2, y2) screen pixels."""
        try:
            x1, y1, x2, y2 = (int(round(float(v))) for v in list(box_coords)[:4])
        except Exception:  # noqa: BLE001
            return
        if x2 <= x1 or y2 <= y1:
            return
        self.start()
        try:
            self._cmd_q.put_nowait(("roi", ((x1, y1, x2, y2), str(label or ""))))
        except Exception:  # noqa: BLE001
            pass

    def clear_roi(self) -> None:
        try:
            self._cmd_q.put_nowait(("clear", None))
        except Exception:  # noqa: BLE001
            pass

    def _run(self) -> None:
        try:
            root = tk.Tk()
        except Exception:  # noqa: BLE001
            self._ready.set()
            return
        self._root = root
        root.title("Donna ROI")
        root.overrideredirect(True)
        try:
            root.attributes("-topmost", True)
        except Exception:  # noqa: BLE001
            pass
        # Full virtual desktop covering primary-ish area; geometry updated per ROI.
        root.geometry("1x1+0+0")
        try:
            root.attributes("-alpha", 0.85)
        except Exception:  # noqa: BLE001
            pass
        try:
            root.configure(bg="black")
            root.wm_attributes("-transparentcolor", "black")
        except Exception:  # noqa: BLE001
            pass
        canvas = tk.Canvas(
            root,
            highlightthickness=0,
            bd=0,
            bg="black",
        )
        canvas.pack(fill="both", expand=True)
        self._canvas = canvas
        root.update_idletasks()
        try:
            hwnd = int(root.winfo_id())
            _enable_click_through(hwnd)
        except Exception:  # noqa: BLE001
            pass
        self._ready.set()
        self._poll()
        try:
            root.mainloop()
        except Exception:  # noqa: BLE001
            pass
        finally:
            self._root = None
            self._canvas = None

    def _poll(self) -> None:
        root = self._root
        if root is None or self._stop.is_set():
            if root is not None:
                try:
                    root.quit()
                except Exception:  # noqa: BLE001
                    pass
            return
        try:
            while True:
                cmd, payload = self._cmd_q.get_nowait()
                if cmd == "stop":
                    self._stop.set()
                    break
                if cmd == "clear":
                    self._apply_clear()
                elif cmd == "roi":
                    box, label = payload
                    self._apply_roi(box, label)
        except queue.Empty:
            pass
        if self._stop.is_set():
            try:
                root.quit()
            except Exception:  # noqa: BLE001
                pass
            return
        try:
            root.after(50, self._poll)
        except Exception:  # noqa: BLE001
            pass

    def _apply_clear(self) -> None:
        self._visible = False
        self._current = None
        self._label = ""
        canvas = self._canvas
        root = self._root
        if canvas is not None:
            canvas.delete("all")
            self._box_id = None
            self._label_id = None
        if root is not None:
            try:
                root.withdraw()
            except Exception:  # noqa: BLE001
                pass

    def _apply_roi(self, box: tuple[int, int, int, int], label: str) -> None:
        x1, y1, x2, y2 = box
        pad = 4
        w = max(1, x2 - x1 + pad * 2)
        h = max(1, y2 - y1 + pad * 2)
        root = self._root
        canvas = self._canvas
        if root is None or canvas is None:
            return
        try:
            root.deiconify()
            root.geometry(f"{w}x{h}+{max(0, x1 - pad)}+{max(0, y1 - pad)}")
            canvas.config(width=w, height=h)
            canvas.delete("all")
            # Styled bounding box — cyan attention ring.
            self._box_id = canvas.create_rectangle(
                pad,
                pad,
                w - pad,
                h - pad,
                outline="#22d3ee",
                width=3,
            )
            text = (label or "").strip()
            if text:
                self._label_id = canvas.create_text(
                    pad + 6,
                    pad + 6,
                    anchor="nw",
                    text=text,
                    fill="#ecfeff",
                    font=("Segoe UI", 11, "bold"),
                )
            self._current = box
            self._label = text
            self._visible = True
            root.update_idletasks()
            try:
                hwnd = int(root.winfo_id())
                _enable_click_through(hwnd)
            except Exception:  # noqa: BLE001
                pass
        except Exception:  # noqa: BLE001
            pass


def get_overlay() -> RoiOverlay:
    """Process-wide ROI overlay singleton (lazy)."""
    global _OVERLAY
    with _OVERLAY_LOCK:
        if _OVERLAY is None:
            _OVERLAY = RoiOverlay()
        return _OVERLAY


def update_roi(box_coords: Any, label: str = "") -> None:
    get_overlay().update_roi(box_coords, label)


def clear_roi() -> None:
    overlay = _OVERLAY
    if overlay is not None:
        overlay.clear_roi()


def ensure_overlay_started() -> RoiOverlay:
    ov = get_overlay()
    ov.start()
    return ov
