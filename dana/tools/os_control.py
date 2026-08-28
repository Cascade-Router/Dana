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

import base64
import ctypes
import io
import os
import random
import re
import threading
import time
from ctypes import wintypes
from typing import Any

from dana.middleware.kill_switch import EmergencyKillSwitchTriggered
from dana.paths import CAPTURES_DIR

# Rate limits for physical typing.
_MAX_CHARS_PER_BURST = 400
_MIN_INTERVAL_SEC = 0.5
_MAX_CHARS_PER_SEC = 40.0
_last_keystroke_ts = 0.0
_chars_window: list[tuple[float, int]] = []
_rate_lock = threading.Lock()

# Humanized cadence between scan-code press and release (seconds).
_HUMAN_DELAY_MIN = 0.040
_HUMAN_DELAY_MAX = 0.110

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

# ---------------------------------------------------------------------------
# Win32 SendInput (hardware scan codes) — no pyautogui / pynput
# ---------------------------------------------------------------------------

INPUT_KEYBOARD = 1
INPUT_MOUSE = 0
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008
KEYEVENTF_EXTENDEDKEY = 0x0001
MAPVK_VK_TO_VSC = 0
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_WHEEL = 0x0800
MOUSEEVENTF_HWHEEL = 0x1000
MOUSEEVENTF_ABSOLUTE = 0x8000
MOUSEEVENTF_VIRTUALDESK = 0x4000

# One "notch" of physical mouse-wheel rotation (Win32 WHEEL_DELTA).
WHEEL_DELTA = 120

# Window-management constants (EnumWindows / SetForegroundWindow).
SW_RESTORE = 9
# Zero-focus workspace: show/reposition a window WITHOUT activating it.
SW_SHOWNOACTIVATE = 4
SWP_NOACTIVATE = 0x0010
SWP_NOZORDER = 0x0004

# Virtual-key codes needed for MapVirtualKey → scan code.
VK_SHIFT = 0x10
VK_CONTROL = 0x11
VK_MENU = 0x12  # Alt
VK_RETURN = 0x0D
VK_TAB = 0x09
VK_ESCAPE = 0x1B
VK_SPACE = 0x20
VK_BACK = 0x08
VK_LWIN = 0x5B
VK_DELETE = 0x2E
VK_INSERT = 0x2D
VK_HOME = 0x24
VK_END = 0x23
VK_PRIOR = 0x21  # Page Up
VK_NEXT = 0x22  # Page Down
VK_UP = 0x26
VK_DOWN = 0x28
VK_LEFT = 0x25
VK_RIGHT = 0x27
VK_F1 = 0x70

# Clipboard constants (OpenClipboard / GetClipboardData / SetClipboardData).
CF_UNICODETEXT = 13
GMEM_MOVEABLE = 0x0002

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

# US-QWERTY printable → (vk, needs_shift). Built from VkKeyScanW when possible.
_CHAR_VK: dict[str, tuple[int, bool]] = {}


class KEYBDINPUT(ctypes.Structure):
    _fields_ = (
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    )


class MOUSEINPUT(ctypes.Structure):
    _fields_ = (
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    )


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = (
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    )


class INPUT_UNION(ctypes.Union):
    _fields_ = (("ki", KEYBDINPUT), ("mi", MOUSEINPUT), ("hi", HARDWAREINPUT))


class INPUT(ctypes.Structure):
    _fields_ = (("type", wintypes.DWORD), ("union", INPUT_UNION))


def _user32():
    return ctypes.windll.user32


def _kernel32():
    return ctypes.windll.kernel32


def get_cursor_pos() -> tuple[int, int]:
    """Return current cursor position in screen pixels."""
    pt = wintypes.POINT()
    if not _user32().GetCursorPos(ctypes.byref(pt)):
        raise OSError(f"GetCursorPos failed: {ctypes.GetLastError()}")
    return int(pt.x), int(pt.y)


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


def set_foreground_window(hwnd: int) -> bool:
    """Bring ``hwnd`` to the foreground via ``SetForegroundWindow``.

    Restores the window first (``ShowWindow``/``SW_RESTORE``) if it's
    minimized, since a minimized window can't visibly come to the front.
    Windows may still deny the focus-steal per its foreground-lock rules
    (e.g. the requesting process isn't the current foreground process) —
    that is a normal, expected outcome, not an error condition, so this
    returns ``False`` rather than raising; callers decide how to report it.
    """
    if os.name != "nt":
        raise OSError("SetForegroundWindow is Windows-only")
    user32 = _user32()
    if user32.IsIconic(hwnd):
        user32.ShowWindow(hwnd, SW_RESTORE)
    return bool(user32.SetForegroundWindow(hwnd))


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


def _configure_clipboard_signatures(user32: Any, kernel32: Any) -> None:
    """Set explicit ctypes argtypes/restype for the clipboard APIs.

    Without this, ctypes assumes a 32-bit ``c_int`` return for every
    function, which silently truncates 64-bit handles on Win64 — this is
    idempotent and cheap, so both clipboard functions call it defensively
    rather than relying on shared import-time state.
    """
    user32.OpenClipboard.argtypes = [wintypes.HWND]
    user32.OpenClipboard.restype = wintypes.BOOL
    user32.CloseClipboard.restype = wintypes.BOOL
    user32.EmptyClipboard.restype = wintypes.BOOL
    user32.IsClipboardFormatAvailable.argtypes = [wintypes.UINT]
    user32.IsClipboardFormatAvailable.restype = wintypes.BOOL
    user32.GetClipboardData.argtypes = [wintypes.UINT]
    user32.GetClipboardData.restype = wintypes.HANDLE
    user32.SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]
    user32.SetClipboardData.restype = wintypes.HANDLE
    kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
    kernel32.GlobalAlloc.restype = wintypes.HANDLE
    kernel32.GlobalLock.argtypes = [wintypes.HANDLE]
    kernel32.GlobalLock.restype = ctypes.c_void_p
    kernel32.GlobalUnlock.argtypes = [wintypes.HANDLE]
    kernel32.GlobalUnlock.restype = wintypes.BOOL
    kernel32.GlobalFree.argtypes = [wintypes.HANDLE]
    kernel32.GlobalFree.restype = wintypes.HANDLE


def _open_clipboard_with_retry(user32: Any, *, attempts: int = 5) -> bool:
    """Best-effort retry: another process (clipboard manager, AV) can hold
    the clipboard for a few milliseconds at a time."""
    for _attempt in range(attempts):
        if user32.OpenClipboard(None):
            return True
        time.sleep(0.02)
    return False


def read_clipboard_text() -> str | None:
    """Read plaintext (``CF_UNICODETEXT``) off the system clipboard via raw
    Win32 APIs (``OpenClipboard``/``GetClipboardData``/``CloseClipboard``).

    Returns ``None`` if the clipboard is empty or holds no text format.
    Raises ``OSError`` if the clipboard can't be opened at all or a handle
    can't be locked — both unexpected failure modes, unlike "no text there".
    """
    if os.name != "nt":
        raise OSError("Win32 clipboard access is Windows-only")
    user32 = _user32()
    kernel32 = _kernel32()
    _configure_clipboard_signatures(user32, kernel32)

    if not _open_clipboard_with_retry(user32):
        raise OSError(f"OpenClipboard failed: {ctypes.GetLastError()}")
    try:
        if not user32.IsClipboardFormatAvailable(CF_UNICODETEXT):
            return None
        handle = user32.GetClipboardData(CF_UNICODETEXT)
        if not handle:
            return None
        locked = kernel32.GlobalLock(handle)
        if not locked:
            raise OSError(f"GlobalLock failed: {ctypes.GetLastError()}")
        try:
            return ctypes.wstring_at(locked)
        finally:
            kernel32.GlobalUnlock(handle)
    finally:
        user32.CloseClipboard()


def write_clipboard_text(text: str) -> None:
    """Replace the system clipboard's contents with ``text`` via raw Win32
    APIs (``SetClipboardData`` with ``CF_UNICODETEXT``).

    Allocates a moveable global memory block, copies the UTF-16LE-encoded
    text (plus a null terminator) into it, then hands ownership to the
    clipboard. Per the Win32 contract, the system owns that memory once
    ``SetClipboardData`` succeeds, so this only frees it on the failure
    path, before ownership transfers.
    """
    if os.name != "nt":
        raise OSError("Win32 clipboard access is Windows-only")
    user32 = _user32()
    kernel32 = _kernel32()
    _configure_clipboard_signatures(user32, kernel32)

    payload = str(text or "")
    encoded = payload.encode("utf-16-le") + b"\x00\x00"

    h_mem = kernel32.GlobalAlloc(GMEM_MOVEABLE, len(encoded))
    if not h_mem:
        raise OSError(f"GlobalAlloc failed: {ctypes.GetLastError()}")
    locked = kernel32.GlobalLock(h_mem)
    if not locked:
        kernel32.GlobalFree(h_mem)
        raise OSError(f"GlobalLock failed: {ctypes.GetLastError()}")
    ctypes.memmove(locked, encoded, len(encoded))
    kernel32.GlobalUnlock(h_mem)

    if not _open_clipboard_with_retry(user32):
        kernel32.GlobalFree(h_mem)
        raise OSError(f"OpenClipboard failed: {ctypes.GetLastError()}")
    try:
        user32.EmptyClipboard()
        if not user32.SetClipboardData(CF_UNICODETEXT, h_mem):
            kernel32.GlobalFree(h_mem)
            raise OSError(f"SetClipboardData failed: {ctypes.GetLastError()}")
        # Clipboard now owns h_mem — do not free it ourselves on success.
    finally:
        user32.CloseClipboard()


def _check_kill_switch() -> None:
    """Refuse physical actuation while the F12 panic latch is set.

    Every raw Win32 SendInput wrapper below calls this first, so keystrokes,
    clicks, mouse-down/up, and wheel events all funnel through one gate
    immediately before any hardware input API is touched — regardless of
    which higher-level actuator (or a tool that calls straight into this
    module, like ``execute_os_keystrokes``) invoked them.
    """
    try:
        from dana.middleware.kill_switch import halt_if_requested
    except Exception:  # noqa: BLE001
        return
    if halt_if_requested():
        raise EmergencyKillSwitchTriggered("halted by GLOBAL_HALT_EVENT")


def move_cursor_absolute(x: int, y: int) -> None:
    """Move cursor via SendInput absolute coordinates (0..65535 mapped)."""
    _check_kill_switch()
    if os.name != "nt":
        raise OSError("SendInput mouse move is Windows-only")
    sw, sh = get_screen_size()
    sw = max(1, sw)
    sh = max(1, sh)
    ax = int(max(0, min(sw - 1, int(x))) * 65535 / (sw - 1 if sw > 1 else 1))
    ay = int(max(0, min(sh - 1, int(y))) * 65535 / (sh - 1 if sh > 1 else 1))
    inp = INPUT()
    inp.type = INPUT_MOUSE
    inp.union.mi = MOUSEINPUT(
        dx=ax,
        dy=ay,
        mouseData=0,
        dwFlags=MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE,
        time=0,
        dwExtraInfo=None,
    )
    sent = _user32().SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
    if sent != 1:
        raise OSError(f"SendInput mouse move failed (sent={sent})")


def click_left_sendinput() -> None:
    """Left-click at the current cursor via SendInput."""
    _check_kill_switch()
    if os.name != "nt":
        raise OSError("SendInput mouse click is Windows-only")
    for flags in (MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP):
        inp = INPUT()
        inp.type = INPUT_MOUSE
        inp.union.mi = MOUSEINPUT(
            dx=0,
            dy=0,
            mouseData=0,
            dwFlags=flags,
            time=0,
            dwExtraInfo=None,
        )
        sent = _user32().SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
        if sent != 1:
            raise OSError(f"SendInput mouse click failed (sent={sent})")
        time.sleep(random.uniform(0.02, 0.06))


def mouse_down_sendinput() -> None:
    """Press and hold the left mouse button at the current cursor via SendInput."""
    _check_kill_switch()
    if os.name != "nt":
        raise OSError("SendInput mouse down is Windows-only")
    inp = INPUT()
    inp.type = INPUT_MOUSE
    inp.union.mi = MOUSEINPUT(
        dx=0,
        dy=0,
        mouseData=0,
        dwFlags=MOUSEEVENTF_LEFTDOWN,
        time=0,
        dwExtraInfo=None,
    )
    sent = _user32().SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
    if sent != 1:
        raise OSError(f"SendInput mouse down failed (sent={sent})")


def mouse_up_sendinput() -> None:
    """Release the left mouse button at the current cursor via SendInput."""
    _check_kill_switch()
    if os.name != "nt":
        raise OSError("SendInput mouse up is Windows-only")
    inp = INPUT()
    inp.type = INPUT_MOUSE
    inp.union.mi = MOUSEINPUT(
        dx=0,
        dy=0,
        mouseData=0,
        dwFlags=MOUSEEVENTF_LEFTUP,
        time=0,
        dwExtraInfo=None,
    )
    sent = _user32().SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
    if sent != 1:
        raise OSError(f"SendInput mouse up failed (sent={sent})")


def scroll_wheel_sendinput(*, dx: int = 0, dy: int = 0) -> None:
    """Send one mouse-wheel event via SendInput.

    ``dy`` is vertical wheel delta (positive scrolls up/away from the user,
    negative scrolls down); ``dx`` is horizontal wheel delta (positive
    scrolls right, negative scrolls left) — matches Win32
    ``MOUSEEVENTF_WHEEL``/``MOUSEEVENTF_HWHEEL`` semantics. Deltas are
    typically multiples of ``WHEEL_DELTA`` (120), one notch per call.
    Exactly one of ``dx``/``dy`` should be non-zero; ``dy`` wins if both are
    given, since vertical and horizontal wheel are distinct hardware events.
    """
    _check_kill_switch()
    if os.name != "nt":
        raise OSError("SendInput mouse wheel is Windows-only")
    if dy != 0:
        flags = MOUSEEVENTF_WHEEL
        raw_delta = int(dy)
    elif dx != 0:
        flags = MOUSEEVENTF_HWHEEL
        raw_delta = int(dx)
    else:
        return
    inp = INPUT()
    inp.type = INPUT_MOUSE
    inp.union.mi = MOUSEINPUT(
        dx=0,
        dy=0,
        # mouseData is a DWORD field but Windows reads it as signed; mask the
        # negative delta to its unsigned 32-bit two's-complement bit pattern.
        mouseData=raw_delta & 0xFFFFFFFF,
        dwFlags=flags,
        time=0,
        dwExtraInfo=None,
    )
    sent = _user32().SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
    if sent != 1:
        raise OSError(f"SendInput mouse wheel failed (sent={sent})")


def _human_sleep() -> None:
    time.sleep(random.uniform(_HUMAN_DELAY_MIN, _HUMAN_DELAY_MAX))


def _vk_to_scan(vk: int) -> int:
    scan = int(_user32().MapVirtualKeyW(int(vk) & 0xFF, MAPVK_VK_TO_VSC))
    return scan & 0xFF


def _send_scan(vk: int, *, key_up: bool = False) -> None:
    """Emit one key event via SendInput using hardware scan codes (no VK in packet)."""
    _check_kill_switch()
    scan = _vk_to_scan(vk)
    if scan == 0 and vk not in (0,):
        # Still attempt — some keys map to 0 on exotic layouts.
        scan = int(vk) & 0xFF
    flags = KEYEVENTF_SCANCODE
    if key_up:
        flags |= KEYEVENTF_KEYUP
    if vk in _EXTENDED_VKS:
        flags |= KEYEVENTF_EXTENDEDKEY
    inp = INPUT()
    inp.type = INPUT_KEYBOARD
    # wVk MUST stay 0 when KEYEVENTF_SCANCODE is set — hardware-level path.
    inp.union.ki = KEYBDINPUT(
        wVk=0,
        wScan=scan,
        dwFlags=flags,
        time=0,
        dwExtraInfo=None,
    )
    sent = _user32().SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
    if sent != 1:
        raise OSError(f"SendInput failed (sent={sent}, GetLastError={ctypes.GetLastError()})")


def _tap_vk(vk: int) -> None:
    _send_scan(vk, key_up=False)
    _human_sleep()
    _send_scan(vk, key_up=True)
    _human_sleep()


def _resolve_char(ch: str) -> tuple[int, bool] | None:
    """Return (virtual_key, needs_shift) for a single printable character."""
    if ch in _CHAR_VK:
        return _CHAR_VK[ch]
    # VkKeyScanW: low byte = VK, high byte = shift/ctrl/alt state.
    result = int(_user32().VkKeyScanW(ord(ch)))
    if result == -1 or result == 0xFFFF:
        return None
    vk = result & 0xFF
    shift = bool(result & 0x100)
    _CHAR_VK[ch] = (vk, shift)
    return vk, shift


def _type_char_stealth(ch: str) -> bool:
    if ch == "\n" or ch == "\r":
        _tap_vk(VK_RETURN)
        return True
    if ch == "\t":
        _tap_vk(VK_TAB)
        return True
    if ch == " ":
        _tap_vk(VK_SPACE)
        return True
    if ch == "\b":
        _tap_vk(VK_BACK)
        return True

    resolved = _resolve_char(ch)
    if resolved is None:
        return False
    vk, needs_shift = resolved
    if needs_shift:
        _send_scan(VK_SHIFT, key_up=False)
        _human_sleep()
    _send_scan(vk, key_up=False)
    _human_sleep()
    _send_scan(vk, key_up=True)
    _human_sleep()
    if needs_shift:
        _send_scan(VK_SHIFT, key_up=True)
        _human_sleep()
    return True


_HOTKEY_VK = {
    "ctrl": VK_CONTROL,
    "control": VK_CONTROL,
    "shift": VK_SHIFT,
    "alt": VK_MENU,
    "enter": VK_RETURN,
    "return": VK_RETURN,
    "tab": VK_TAB,
    "esc": VK_ESCAPE,
    "escape": VK_ESCAPE,
    "a": 0x41,
    "c": 0x43,
    "s": 0x53,
    "v": 0x56,
    "z": 0x5A,
    "win": VK_LWIN,
    "windows": VK_LWIN,
    "super": VK_LWIN,
    "space": VK_SPACE,
    "backspace": VK_BACK,
    "back": VK_BACK,
    "delete": VK_DELETE,
    "del": VK_DELETE,
    "insert": VK_INSERT,
    "ins": VK_INSERT,
    "home": VK_HOME,
    "end": VK_END,
    "pageup": VK_PRIOR,
    "pgup": VK_PRIOR,
    "pagedown": VK_NEXT,
    "pgdn": VK_NEXT,
    "up": VK_UP,
    "down": VK_DOWN,
    "left": VK_LEFT,
    "right": VK_RIGHT,
}

# Matches "f1".."f24" (Win32 defines VK_F1..VK_F24 as a contiguous range).
_FUNCTION_KEY_RE = re.compile(r"^f([1-9]|1[0-9]|2[0-4])$")


def type_text_sendinput(text: str) -> dict[str, Any]:
    """Type plaintext via scan-code SendInput with humanized cadence.

    Returns ``{ok, chars_typed, stripped_controls, engine}``.
    """
    if os.name != "nt":
        return {
            "ok": False,
            "error": "SendInput stealth typing is Windows-only",
            "chars_typed": 0,
            "stripped_controls": 0,
            "engine": "sendinput",
        }

    typed = 0
    stripped = 0
    for ch in text:
        # Strip non-printable controls except newline/tab/backspace.
        if ord(ch) < 32 and ch not in ("\n", "\r", "\t", "\b"):
            stripped += 1
            continue
        if ord(ch) == 127:
            stripped += 1
            continue
        try:
            if _type_char_stealth(ch):
                typed += 1
            else:
                stripped += 1
        except OSError as exc:
            return {
                "ok": False,
                "error": str(exc),
                "chars_typed": typed,
                "stripped_controls": stripped,
                "engine": "sendinput_scancode",
            }
    return {
        "ok": True,
        "chars_typed": typed,
        "stripped_controls": stripped,
        "engine": "sendinput_scancode",
        "dry_run": False,
    }


def press_hotkey_sendinput(keys: tuple[str, ...]) -> None:
    """Allowlisted hotkey chord via scan-code SendInput."""
    vks: list[int] = []
    for name in keys:
        vk = _HOTKEY_VK.get(name.lower())
        if vk is None:
            raise ValueError(f"unsupported hotkey part: {name}")
        vks.append(vk)
    # Press modifiers then key, release in reverse.
    for vk in vks:
        _send_scan(vk, key_up=False)
        _human_sleep()
    for vk in reversed(vks):
        _send_scan(vk, key_up=True)
        _human_sleep()


def resolve_key_name(name: str) -> int | None:
    """Resolve a human-readable key name to its Win32 virtual-key code.

    Recognizes everything in ``_HOTKEY_VK`` (modifiers + named/navigation
    keys), function keys (``f1``..``f24``), single letters (``a``-``z``),
    and single digits (``0``-``9``). Returns ``None`` for anything else —
    callers should treat that as an unsupported key rather than guessing.
    Unlike ``_HOTKEY_VK``/``press_hotkey_sendinput``'s narrow, exact-tuple
    ``_ALLOWED_HOTKEYS`` allowlist, this backs the more general
    ``press_key_combo``/``execute_shortcut`` path — safety there comes from
    validating every key resolves before pressing anything, not from a
    closed list of exact combos.
    """
    key = str(name or "").strip().lower()
    if not key:
        return None
    if key in _HOTKEY_VK:
        return _HOTKEY_VK[key]
    m = _FUNCTION_KEY_RE.match(key)
    if m:
        return VK_F1 + (int(m.group(1)) - 1)
    if len(key) == 1:
        if "a" <= key <= "z":
            return ord(key.upper())
        if key.isdigit():
            return ord(key)
    return None


def press_key_combo(keys: list[str]) -> None:
    """Press ``keys`` down in order, then release them in reverse order.

    Resolves every name via ``resolve_key_name`` and validates the whole
    combo *before* pressing anything — fails closed with ``ValueError``
    rather than risk pressing some keys and not others (e.g. a stuck
    modifier) on a bad combo.
    """
    if not keys:
        raise ValueError("press_key_combo requires at least one key")
    resolved: list[int] = []
    for name in keys:
        vk = resolve_key_name(name)
        if vk is None:
            raise ValueError(f"unsupported key: {name!r}")
        resolved.append(vk)
    for vk in resolved:
        _send_scan(vk, key_up=False)
        _human_sleep()
    for vk in reversed(resolved):
        _send_scan(vk, key_up=True)
        _human_sleep()


def _dry_run() -> bool:
    from dana.security.dry_run import is_dry_run_enabled

    return is_dry_run_enabled()


def _rate_limit_ok(n_chars: int) -> tuple[bool, str]:
    """Gate keystroke bursts.

    Humanized SendInput already runs ~9–25 chars/sec, so we must not reject a
    burst merely because ``n_chars > _MAX_CHARS_PER_SEC`` (that blocked every
    slide comment longer than 40 chars before any key was pressed).
    """
    global _last_keystroke_ts, _chars_window
    now = time.monotonic()
    with _rate_lock:
        if now - _last_keystroke_ts < _MIN_INTERVAL_SEC:
            return False, f"rate_limited: wait {_MIN_INTERVAL_SEC:.1f}s between bursts"
        if n_chars > _MAX_CHARS_PER_BURST:
            return False, f"rate_limited: max {_MAX_CHARS_PER_BURST} chars per burst"
        _chars_window = [(t, n) for t, n in _chars_window if now - t < 1.0]
        recent = sum(n for _, n in _chars_window)
        # Only block if prior bursts in the last second already saturated the
        # rolling budget; the current burst is paced by humanized delays.
        if recent >= _MAX_CHARS_PER_SEC:
            return False, f"rate_limited: max {_MAX_CHARS_PER_SEC:.0f} chars/sec"
        _chars_window.append((now, n_chars))
        _last_keystroke_ts = now
    return True, ""


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
        import win32con
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


def _vision_describe(png: bytes, *, prompt: str = "") -> str:
    """Best-effort vision summary via Cascade MoA vision model (Ollama)."""
    try:
        from dana.cascade_router import extract_vision_context

        return extract_vision_context(png, prompt=prompt)
    except Exception:
        pass

    b64 = base64.b64encode(png).decode("ascii")
    ask = (prompt or "Describe the main UI elements and readable text on this screen.").strip()
    try:
        import json
        import urllib.request

        from dana.cascade_router import vision_model_name

        model = vision_model_name()
        payload = {
            "model": model,
            "prompt": ask,
            "images": [b64],
            "stream": False,
            "keep_alive": 0,
        }
        req = urllib.request.Request(
            os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
            + "/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        text = str(data.get("response") or "").strip()
        if text:
            return text
    except Exception:
        pass

    try:
        from PIL import Image

        img = Image.open(io.BytesIO(png))
        w, h = img.size
        return (
            f"Screen capture {w}x{h} PNG ({len(png)} bytes). "
            "Vision model unavailable — describe UI from YOLO/spatial context if present."
        )
    except Exception as exc:  # noqa: BLE001
        return f"Screen captured ({len(png)} bytes) but analysis failed: {exc}"


def capture_and_analyze_screen(*, prompt: str = "", save_copy: bool = True) -> str:
    """Tool entry: screenshot + vision summary (observation string for ReAct)."""
    try:
        png = capture_screen_png_bytes()
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: capture_and_analyze_screen failed: {exc}"

    path_note = ""
    if save_copy:
        try:
            CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
            out = CAPTURES_DIR / "last_screen_capture.png"
            out.write_bytes(png)
            path_note = f" saved={out}"
        except Exception:
            path_note = ""

    summary = _vision_describe(png, prompt=prompt)
    return (
        f"OK: capture_and_analyze_screen bytes={len(png)}{path_note}\n"
        f"VISION: {summary}"
    )


def execute_os_keystrokes(
    text: str = "",
    *,
    hotkey: str = "",
    interval: float = 0.02,
) -> str:
    """Rate-limited stealth typing / allowlisted hotkey via SendInput scan codes.

    ``interval`` is ignored — cadence is randomized 40–110 ms (humanized).
    """
    del interval  # humanized delays replace fixed interval
    hotkey = (hotkey or "").strip().lower()
    text = text or ""

    if hotkey:
        keys = tuple(k.strip() for k in hotkey.replace("+", " ").split() if k.strip())
        if keys not in _ALLOWED_HOTKEYS:
            return (
                f"ERROR: hotkey {hotkey!r} not allowlisted. "
                f"Allowed: {sorted('+'.join(k) for k in _ALLOWED_HOTKEYS)}"
            )
        ok, reason = _rate_limit_ok(len(keys))
        if not ok:
            return f"ERROR: {reason}"
        if _dry_run():
            return f"OK: execute_os_keystrokes dry_run hotkey={'+'.join(keys)} engine=sendinput_scancode"
        try:
            press_hotkey_sendinput(keys)
            return f"OK: execute_os_keystrokes hotkey={'+'.join(keys)} engine=sendinput_scancode"
        except EmergencyKillSwitchTriggered as exc:
            return f"HALTED: execute_os_keystrokes — {exc}"
        except Exception as exc:  # noqa: BLE001
            return f"ERROR: execute_os_keystrokes hotkey failed: {exc}"

    if not str(text).strip():
        return "ERROR: missing text (or hotkey)"

    ok, reason = _rate_limit_ok(len(text))
    if not ok:
        return f"ERROR: {reason}"

    if _dry_run():
        return (
            f"OK: execute_os_keystrokes dry_run chars={len(text)} "
            f"engine=sendinput_scancode"
        )

    result = type_text_sendinput(str(text))
    if not result.get("ok"):
        error = result.get("error") or ""
        if "GLOBAL_HALT_EVENT" in str(error):
            return f"HALTED: execute_os_keystrokes — {error}"
        return f"ERROR: execute_os_keystrokes blocked/failed: {error}"
    return (
        f"OK: execute_os_keystrokes typed chars={result.get('chars_typed', 0)} "
        f"stripped={result.get('stripped_controls', 0)} "
        f"engine={result.get('engine', 'sendinput_scancode')}"
    )
