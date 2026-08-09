"""Stage 7.4 — Yield to Human (physical input soft-interrupt).

Low-level WH_MOUSE_LL / WH_KEYBOARD_LL hooks ignore injected (Dana) events
and stamp ``LAST_PHYSICAL_INPUT_TIME`` on real human mouse/keyboard activity.
Operators call ``yield_check()`` before Act to pause until 3s of quiet.
"""

from __future__ import annotations

import ctypes
import os
import threading
import time
from ctypes import wintypes
from typing import Callable

# Thread-safe last physical (non-injected) input timestamp.
_LOCK = threading.Lock()
LAST_PHYSICAL_INPUT_TIME: float = 0.0

YIELD_QUIET_S = 3.0
_LISTENER_STARTED = False
_LISTENER_LOCK = threading.Lock()

# Win32 constants
WH_KEYBOARD_LL = 13
WH_MOUSE_LL = 14
WM_QUIT = 0x0012
LLKHF_INJECTED = 0x00000010
LLMHF_INJECTED = 0x00000001
LLMHF_LOWER_IL_INJECTED = 0x00000002

HC_ACTION = 0

# Keep callback refs alive for the process lifetime.
_keyboard_proc = None
_mouse_proc = None
_hook_kb = None
_hook_ms = None
_hook_thread: threading.Thread | None = None


def _log(msg: str) -> None:
    try:
        from dana.logging import log

        log("HumanYield", msg)
    except Exception:  # noqa: BLE001
        print(f"[HumanYield] {msg}", flush=True)


def note_physical_input(*, source: str = "test") -> None:
    """Stamp physical input time (tests / non-Windows fallbacks)."""
    global LAST_PHYSICAL_INPUT_TIME
    with _LOCK:
        LAST_PHYSICAL_INPUT_TIME = time.time()
    _ = source


def last_physical_input_age_s() -> float:
    """Seconds since last physical input (``inf`` if never)."""
    with _LOCK:
        ts = float(LAST_PHYSICAL_INPUT_TIME)
    if ts <= 0.0:
        return float("inf")
    return max(0.0, time.time() - ts)


def _should_yield(*, quiet_s: float = YIELD_QUIET_S) -> bool:
    with _LOCK:
        ts = float(LAST_PHYSICAL_INPUT_TIME)
    if ts <= 0.0:
        return False
    return (time.time() - ts) < float(quiet_s)


def yield_check(
    *,
    operator: str = "operator",
    quiet_s: float = YIELD_QUIET_S,
    sleep_s: float = 0.1,
    on_pause: Callable[[], None] | None = None,
    on_resume: Callable[[], None] | None = None,
) -> bool:
    """Block while human input is recent; return True if a yield occurred.

    Does not fail the task — operators simply wait then continue Act.
    Respects the Stage 7.2 kill switch (breaks out without waiting out the quiet).
    """
    if not _should_yield(quiet_s=quiet_s):
        return False

    paused_emitted = False
    try:
        while _should_yield(quiet_s=quiet_s):
            try:
                from dana.middleware.kill_switch import halt_if_requested

                if halt_if_requested():
                    break
            except Exception:  # noqa: BLE001
                pass
            if not paused_emitted:
                paused_emitted = True
                msg = "[OPERATOR PAUSED: Yielding to human input...]"
                _log(f"{operator}: {msg}")
                if on_pause is not None:
                    try:
                        on_pause()
                    except Exception:  # noqa: BLE001
                        pass
                else:
                    _emit_pause_toast(msg)
            time.sleep(max(0.05, float(sleep_s)))
    finally:
        if paused_emitted:
            msg = "[OPERATOR RESUMED]"
            _log(f"{operator}: {msg}")
            if on_resume is not None:
                try:
                    on_resume()
                except Exception:  # noqa: BLE001
                    pass
            else:
                _emit_resume_toast(msg)
    return paused_emitted


def _emit_pause_toast(body: str) -> None:
    try:
        from dana.middleware.toast_notify import show_silent_toast_async

        show_silent_toast_async("Dana Operator", body)
    except Exception:  # noqa: BLE001
        pass


def _emit_resume_toast(body: str) -> None:
    try:
        from dana.middleware.toast_notify import show_silent_toast_async

        show_silent_toast_async("Dana Operator", body)
    except Exception:  # noqa: BLE001
        pass


def _mark_if_physical(flags: int, *, injected_mask: int) -> None:
    if int(flags) & int(injected_mask):
        return
    note_physical_input(source="hook")


class KBDLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [
        ("vkCode", wintypes.DWORD),
        ("scanCode", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class MSLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [
        ("pt", wintypes.POINT),
        ("mouseData", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


LowLevelKeyboardProc = ctypes.WINFUNCTYPE(
    ctypes.c_long,
    ctypes.c_int,
    wintypes.WPARAM,
    wintypes.LPARAM,
)
LowLevelMouseProc = ctypes.WINFUNCTYPE(
    ctypes.c_long,
    ctypes.c_int,
    wintypes.WPARAM,
    wintypes.LPARAM,
)


def _call_next(n_code: int, w_param: int, l_param: int) -> int:
    user32 = ctypes.windll.user32
    return int(
        user32.CallNextHookEx(
            None,
            int(n_code),
            wintypes.WPARAM(w_param),
            wintypes.LPARAM(l_param),
        )
    )


def _keyboard_hook(n_code: int, w_param: int, l_param: int) -> int:
    try:
        if int(n_code) == HC_ACTION and l_param:
            info = ctypes.cast(l_param, ctypes.POINTER(KBDLLHOOKSTRUCT)).contents
            _mark_if_physical(int(info.flags), injected_mask=LLKHF_INJECTED)
    except Exception:  # noqa: BLE001
        pass
    return _call_next(n_code, w_param, l_param)


def _mouse_hook(n_code: int, w_param: int, l_param: int) -> int:
    try:
        if int(n_code) == HC_ACTION and l_param:
            info = ctypes.cast(l_param, ctypes.POINTER(MSLLHOOKSTRUCT)).contents
            injected = LLMHF_INJECTED | LLMHF_LOWER_IL_INJECTED
            _mark_if_physical(int(info.flags), injected_mask=injected)
    except Exception:  # noqa: BLE001
        pass
    return _call_next(n_code, w_param, l_param)


def _hook_message_loop() -> None:
    global _keyboard_proc, _mouse_proc, _hook_kb, _hook_ms
    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32

    _keyboard_proc = LowLevelKeyboardProc(_keyboard_hook)
    _mouse_proc = LowLevelMouseProc(_mouse_hook)

    _hook_kb = user32.SetWindowsHookExW(
        WH_KEYBOARD_LL,
        _keyboard_proc,
        kernel32.GetModuleHandleW(None),
        0,
    )
    _hook_ms = user32.SetWindowsHookExW(
        WH_MOUSE_LL,
        _mouse_proc,
        kernel32.GetModuleHandleW(None),
        0,
    )
    if not _hook_kb or not _hook_ms:
        err = int(ctypes.GetLastError())
        _log(f"SetWindowsHookExW failed last_error={err}")
        return

    _log("WH_KEYBOARD_LL + WH_MOUSE_LL armed (filter injected)")
    msg = wintypes.MSG()
    while True:
        ret = int(user32.GetMessageW(ctypes.byref(msg), None, 0, 0))
        if ret <= 0:
            break
        user32.TranslateMessage(ctypes.byref(msg))
        user32.DispatchMessageW(ctypes.byref(msg))

    if _hook_kb:
        user32.UnhookWindowsHookEx(_hook_kb)
    if _hook_ms:
        user32.UnhookWindowsHookEx(_hook_ms)


def start_human_yield_listener() -> bool:
    """Start the LL hook message-pump thread (idempotent, Windows-only)."""
    global _LISTENER_STARTED, _hook_thread
    with _LISTENER_LOCK:
        if _LISTENER_STARTED:
            return True
        if os.environ.get("DANA_DISABLE_HUMAN_YIELD", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            _log("listener disabled via DANA_DISABLE_HUMAN_YIELD")
            return False
        if os.name != "nt":
            _log("listener skipped (non-Windows); note_physical_input still works")
            _LISTENER_STARTED = True
            return False

        t = threading.Thread(
            target=_hook_message_loop,
            name="DanaHumanYield",
            daemon=True,
        )
        t.start()
        _hook_thread = t
        _LISTENER_STARTED = True
        return True


def reset_physical_input_clock() -> None:
    """Tests: clear the physical-input stamp so yield_check is a no-op."""
    global LAST_PHYSICAL_INPUT_TIME
    with _LOCK:
        LAST_PHYSICAL_INPUT_TIME = 0.0
