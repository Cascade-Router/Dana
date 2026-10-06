"""OS Computer Use — screen capture + vision + stealth SendInput keystrokes.

Tools:
  capture_and_analyze_screen — mss screenshot → Cascade MoA vision summary
  execute_os_keystrokes      — hardware scan-code SendInput via ctypes (no pyautogui/pynput)

Closed-loop upgrade (Stage 6.1):
  ``dana.operators.ghost_typist.GhostTypistOperator`` / ``type_stealth_text``
  wraps this SendInput backend with chunked Sense-Evaluate-Act visual guards.

Safety:
  - DANA_OS_DRY_RUN=1 skips real input.
  - Keystroke bursts are rate-limited (chars/sec + cooldown).
  - Chord macros are allowlisted only.
  - Typing uses randomized 40–110 ms human cadence between press/release.
"""

from __future__ import annotations

import ctypes
import io
import os
import re
import threading
import time
from ctypes import wintypes
from typing import Any


_rate_lock = threading.Lock()


_ALLOWED_HOTKEYS: frozenset[tuple[str, ...]] = frozenset(
    {
        ("ctrl", "c"),
        ("ctrl", "v"),
        ("ctrl", "a"),
        ("ctrl", "s"),
        ("ctrl", "z"),
        ("enter",),
        ("tab",),
        ("esc",),
    }
)

# Zero-focus workspace: show/reposition a window WITHOUT activating it.
SW_SHOWNOACTIVATE = 4
SWP_NOACTIVATE = 0x0010
SWP_NOZORDER = 0x0004


_EXTENDED_VKS = frozenset(
    {
        0x21,
        0x22,
        0x23,
        0x24,  # PgUp/PgDn/End/Home
        0x25,
        0x26,
        0x27,
        0x28,  # arrows
        0x2D,
        0x2E,  # Ins/Del
        0x5B,
        0x5C,  # Win
    }
)


def _user32():
    return ctypes.windll.user32


def get_screen_size() -> tuple[int, int]:
    user32 = _user32()
    return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))


def get_active_windows() -> list[dict[str, Any]]:
    """Enumerate visible top-level desktop windows via ``EnumWindows``.

    Filters to windows that are visible (``IsWindowVisible``) and have a
    non-empty title bar — this drops invisible helper windows and
    system-level background processes that never surface a title, without
    needing a hardcoded process-name blocklist. Order matches Win32 Z-order
    (topmost window first).

    Returns a list of ``{"hwnd": int, "title": str, "pid": int}`` dicts.
    """
    if os.name != "nt":
        raise OSError("EnumWindows window listing is Windows-only")
    user32 = _user32()
    windows: list[dict[str, Any]] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def _enum_proc(hwnd: int, _lparam: int) -> bool:
        if not user32.IsWindowVisible(hwnd):
            return True
        length = int(user32.GetWindowTextLengthW(hwnd))
        if length <= 0:
            return True
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        title = (buf.value or "").strip()
        if not title:
            return True
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        windows.append({"hwnd": int(hwnd), "title": title, "pid": int(pid.value)})
        return True

    if not user32.EnumWindows(_enum_proc, 0):
        raise OSError(f"EnumWindows failed: {ctypes.GetLastError()}")
    return windows


class _RECT(ctypes.Structure):
    _fields_ = (
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    )


def get_window_rect(hwnd: int) -> tuple[int, int, int, int]:
    """Return ``(left, top, right, bottom)`` screen coordinates of ``hwnd``.

    Works regardless of focus/foreground state or which monitor the window
    is on — ``GetWindowRect`` reports a window's on-screen position whether
    or not it's active, which is what makes window-targeted screenshotting
    (``capture_window_png_bytes``) possible without ever focusing anything.
    """
    if os.name != "nt":
        raise OSError("GetWindowRect is Windows-only")
    rect = _RECT()
    if not _user32().GetWindowRect(int(hwnd), ctypes.byref(rect)):
        raise OSError(f"GetWindowRect failed: {ctypes.GetLastError()}")
    return int(rect.left), int(rect.top), int(rect.right), int(rect.bottom)


def move_window_no_activate(hwnd: int, x: int, y: int, width: int, height: int) -> bool:
    """Reposition/resize ``hwnd`` without activating it or changing its z-order.

    Zero-focus workspace primitive: ``SWP_NOACTIVATE`` is the whole point —
    the window visibly moves (e.g. onto a second monitor) but never steals
    the foreground lock. ``ShowWindow(SW_SHOWNOACTIVATE)`` afterward covers
    the case where the window started minimized, without the activation
    that ``SW_RESTORE`` would otherwise cause.
    """
    if os.name != "nt":
        raise OSError("SetWindowPos is Windows-only")
    user32 = _user32()
    ok = user32.SetWindowPos(
        int(hwnd),
        0,
        int(x),
        int(y),
        int(width),
        int(height),
        SWP_NOACTIVATE | SWP_NOZORDER,
    )
    user32.ShowWindow(hwnd, SW_SHOWNOACTIVATE)
    return bool(ok)


# Matches "f1".."f24" (Win32 defines VK_F1..VK_F24 as a contiguous range).
_FUNCTION_KEY_RE = re.compile(r"^f([1-9]|1[0-9]|2[0-4])$")


def capture_screen_png_bytes() -> bytes:
    """Grab the primary monitor as PNG bytes via mss + Pillow."""
    import mss
    from PIL import Image

    with mss.mss() as sct:
        mon = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
        shot = sct.grab(mon)
        img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
        img.thumbnail((1280, 720))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()


# PW_RENDERFULLCONTENT — added in Windows 8.1, needed for windows that use
# DirectComposition/hardware-accelerated content (the plain PrintWindow(0)
# flag often produces a blank/black result for those otherwise).
_PW_RENDERFULLCONTENT = 0x00000002
# A rendered CAD/UI window has real pixel variance (toolbars, a 3D viewport,
# text); a PrintWindow call that silently produced nothing comes back as a
# single flat color. Below this stddev (over a 0-255 grayscale channel), the
# result is treated as "PrintWindow didn't actually render anything" rather
# than trusted at face value.
_BLANK_CAPTURE_STDDEV_THRESHOLD = 1.0


def _capture_window_via_printwindow(hwnd: int, width: int, height: int) -> Any | None:
    """Best-effort: ``hwnd``'s own rendered content via ``user32.PrintWindow``
    — the window paints ITSELF into an offscreen bitmap on request, so this
    works regardless of Z-order or on-screen occlusion (another window, even
    a fullscreen game, sitting visually on top of it doesn't matter at all)
    and never touches focus/activation/Z-order — unlike a
    ``SetForegroundWindow`` "flick", which risks silently no-op'ing under
    Windows' foreground-lock rules, and — worse — can kick an occluding
    EXCLUSIVE-fullscreen app (a game) out of that mode with no clean way to
    restore it, for a real UX disruption in exchange for an unreliable fix.

    Returns a Pillow ``Image`` or ``None`` (never raises) if ``PrintWindow``
    itself reports failure, or its result looks blank — some GPU-accelerated
    window content still doesn't come through this API even with
    ``PW_RENDERFULLCONTENT``, so this is verified, not just assumed.
    """
    if width <= 0 or height <= 0:
        return None
    try:
        import win32gui
        import win32ui
        from PIL import Image, ImageStat
    except Exception:  # noqa: BLE001 — pywin32/Pillow unavailable is a caller-visible fallback, not a crash
        return None

    window_dc = mem_dc = save_dc = bitmap = None
    try:
        window_dc = win32gui.GetWindowDC(hwnd)
        mem_dc = win32ui.CreateDCFromHandle(window_dc)
        save_dc = mem_dc.CreateCompatibleDC()
        bitmap = win32ui.CreateBitmap()
        bitmap.CreateCompatibleBitmap(mem_dc, width, height)
        save_dc.SelectObject(bitmap)

        rendered = ctypes.windll.user32.PrintWindow(hwnd, save_dc.GetSafeHdc(), _PW_RENDERFULLCONTENT)
        if not rendered:
            return None

        info = bitmap.GetInfo()
        bits = bitmap.GetBitmapBits(True)
        img = Image.frombuffer(
            "RGB", (info["bmWidth"], info["bmHeight"]), bits, "raw", "BGRX", 0, 1
        )
        if ImageStat.Stat(img.convert("L")).stddev[0] < _BLANK_CAPTURE_STDDEV_THRESHOLD:
            return None
        return img
    except Exception:  # noqa: BLE001 — best-effort; any GDI failure just falls back to the region-grab
        return None
    finally:
        if bitmap is not None:
            win32gui.DeleteObject(bitmap.GetHandle())
        if save_dc is not None:
            save_dc.DeleteDC()
        if mem_dc is not None:
            mem_dc.DeleteDC()
        if window_dc is not None:
            win32gui.ReleaseDC(hwnd, window_dc)


def capture_window_png_bytes(hwnd: int) -> bytes:
    """Grab ``hwnd``'s own contents as PNG bytes — ``PrintWindow`` first (see
    ``_capture_window_via_printwindow``: immune to Z-order/occlusion, never
    touches focus), falling back to an on-screen region grab via mss +
    Pillow (``get_window_rect``) only if that comes back empty/unsupported.

    The mss fallback path is what lets a zero-focus workflow verify a
    window's contents regardless of whether it's focused, in the
    background, or on a secondary monitor, AS LONG AS nothing else is drawn
    on top of it — the PrintWindow path above removes that last caveat for
    windows it works on.
    """
    left, top, right, bottom = get_window_rect(hwnd)
    width, height = max(1, right - left), max(1, bottom - top)

    img = _capture_window_via_printwindow(hwnd, width, height)
    if img is None:
        import mss
        from PIL import Image

        region = {"left": left, "top": top, "width": width, "height": height}
        with mss.mss() as sct:
            shot = sct.grab(region)
            img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")

    img.thumbnail((1280, 720))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def get_secondary_monitor() -> dict[str, int] | None:
    """Return the first non-primary monitor's geometry via ``mss``, or ``None``.

    ``mss().monitors[0]`` is the combined virtual-desktop bounding box, not
    a real monitor — skipped here. Returns ``None`` on a single-monitor
    setup so callers can fall back rather than move a window somewhere
    unreachable.
    """
    import mss

    with mss.mss() as sct:
        monitors = list(sct.monitors[1:])
    for mon in monitors:
        if not mon.get("is_primary"):
            return {
                "left": int(mon["left"]),
                "top": int(mon["top"]),
                "width": int(mon["width"]),
                "height": int(mon["height"]),
            }
    return None


