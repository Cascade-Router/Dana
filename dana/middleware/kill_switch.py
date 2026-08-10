"""Stage 7.2 — Human panic button (hardware kill switch).

Global hotkey listener that:
  1. Cancels pending/running Blackboard ``action_queue`` rows
  2. Sets ``GLOBAL_HALT_EVENT`` so Ghost Typist / Navigation SEA loops abort
"""

from __future__ import annotations

import os
import threading
from typing import Any

# Latched until ``clear_global_halt()`` — in-flight operators must stop immediately.
GLOBAL_HALT_EVENT = threading.Event()


class EmergencyKillSwitchTriggered(OSError):
    """Raised by the lowest-level Win32 SendInput wrappers when the panic
    latch (``GLOBAL_HALT_EVENT``) is set, so physical actuation is refused
    before any hardware input API is touched.

    Subclasses ``OSError`` rather than a bespoke base so it is caught for
    free by every existing ``except OSError`` / ``except Exception`` handler
    already wrapping Win32 SendInput calls throughout ``dana.tools`` —
    callers that want to distinguish a halt from an ordinary hardware
    failure can still do so via ``isinstance``.
    """

_LISTENER_STARTED = False
_LISTENER_LOCK = threading.Lock()
_DEFAULT_HOTKEY = "f12"


def is_halted() -> bool:
    return GLOBAL_HALT_EVENT.is_set()


def clear_global_halt() -> None:
    """Allow operators to run again after a panic latch."""
    GLOBAL_HALT_EVENT.clear()


def _log(msg: str) -> None:
    try:
        from dana.logging import log

        log("KillSwitch", msg)
    except Exception:  # noqa: BLE001
        print(f"[KillSwitch] {msg}", flush=True)


def cancel_action_queue(*, db_path: Any = None) -> int:
    """UPDATE pending/running/in_progress rows → ``cancelled``. Return count."""
    from dana.memory.blackboard import cancel_open_actions

    return cancel_open_actions(
        db_path=db_path,
        reason="halted by GLOBAL_HALT_EVENT",
    )


def trigger_halt(*, db_path: Any = None, reason: str = "hotkey") -> dict[str, Any]:
    """Panic sequence: latch GLOBAL_HALT_EVENT + flush the action queue."""
    GLOBAL_HALT_EVENT.set()
    cancelled = 0
    try:
        cancelled = cancel_action_queue(db_path=db_path)
    except Exception as exc:  # noqa: BLE001
        _log(f"action_queue cancel failed: {exc}")
    _log(
        f"PANIC latched reason={reason!r} cancelled_actions={cancelled} "
        f"(clear_global_halt to resume operators)"
    )
    return {"ok": True, "cancelled": cancelled, "reason": reason}


def _resolve_hotkey() -> str:
    raw = (os.environ.get("DANA_KILL_HOTKEY") or "").strip()
    return raw or _DEFAULT_HOTKEY


def _hotkey_callback() -> None:
    try:
        trigger_halt(reason=f"hotkey:{_resolve_hotkey()}")
    except Exception as exc:  # noqa: BLE001
        _log(f"halt callback error: {exc}")


def start_kill_switch_listener(*, hotkey: str | None = None) -> bool:
    """Register a global OS hotkey hook on a daemon thread. Idempotent."""
    global _LISTENER_STARTED
    with _LISTENER_LOCK:
        if _LISTENER_STARTED:
            return True
        if os.environ.get("DANA_DISABLE_KILL_SWITCH", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            _log("listener disabled via DANA_DISABLE_KILL_SWITCH")
            return False

        key = (hotkey or _resolve_hotkey()).strip() or _DEFAULT_HOTKEY

        def _run() -> None:
            try:
                import keyboard  # type: ignore[import-untyped]

                keyboard.add_hotkey(key, _hotkey_callback, suppress=False)
                _log(f"global hotkey armed: {key}")
                # Block this daemon thread forever (hook lives in keyboard's thread).
                keyboard.wait()
            except Exception as exc:  # noqa: BLE001
                _log(f"keyboard listener failed ({exc}); panic API still usable")

        t = threading.Thread(
            target=_run,
            name="DanaKillSwitch",
            daemon=True,
        )
        t.start()
        _LISTENER_STARTED = True
        return True


def halt_if_requested() -> bool:
    """True when operators should break out of their SEA loop immediately."""
    return GLOBAL_HALT_EVENT.is_set()
