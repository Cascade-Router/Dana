"""OS Computer Use — screen capture + vision + stealth SendInput keystrokes.

Tools:
  capture_and_analyze_screen — mss screenshot → Cascade MoA vision summary
  execute_os_keystrokes      — hardware scan-code SendInput (no pyautogui/pynput)

Closed-loop upgrade (Stage 6.1):
  ``dana.operators.ghost_typist.GhostTypistOperator`` / ``type_stealth_text``
  wraps this SendInput backend with chunked Sense-Evaluate-Act visual guards.

Safety:
  - DONNA_OS_DRY_RUN=1 skips real input.
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
import threading
import time
from ctypes import wintypes
from typing import Any

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

# Virtual-key codes needed for MapVirtualKey → scan code.
VK_SHIFT = 0x10
VK_CONTROL = 0x11
VK_MENU = 0x12  # Alt
VK_RETURN = 0x0D
VK_TAB = 0x09
VK_ESCAPE = 0x1B
VK_SPACE = 0x20
VK_BACK = 0x08

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


def get_cursor_pos() -> tuple[int, int]:
    """Return current cursor position in screen pixels."""
    pt = wintypes.POINT()
    if not _user32().GetCursorPos(ctypes.byref(pt)):
        raise OSError(f"GetCursorPos failed: {ctypes.GetLastError()}")
    return int(pt.x), int(pt.y)


def get_screen_size() -> tuple[int, int]:
    user32 = _user32()
    return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))


def move_cursor_absolute(x: int, y: int) -> None:
    """Move cursor via SendInput absolute coordinates (0..65535 mapped)."""
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
}


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


def _dry_run() -> bool:
    return os.environ.get("DONNA_OS_DRY_RUN", "").strip().lower() in ("1", "true", "yes")


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
        return f"ERROR: execute_os_keystrokes blocked/failed: {result.get('error')}"
    return (
        f"OK: execute_os_keystrokes typed chars={result.get('chars_typed', 0)} "
        f"stripped={result.get('stripped_controls', 0)} "
        f"engine={result.get('engine', 'sendinput_scancode')}"
    )
