"""Foundational Keyboard Actuator — safe, rate-limited stealth typing.

Pairs with ``dana.tools.mouse_actuator.MouseActuator`` to complete the basic
UI interaction loop: click to focus, then type. No pyautogui/pynput — reuses
the existing hardware SendInput backend in ``dana.tools.os_control``
(``type_text_sendinput``), which already emits per-character scan-code events
with a randomized 40-110ms humanized press/release cadence — the same
"human-like keystroke cadence" this module is asked to provide. Re-adding a
second, different per-keystroke delay here would just fight that established
cadence, so this module adds its own protection at the *call* level instead
(one typing actuation per ``_MIN_ACTUATION_INTERVAL_S``), mirroring
``mouse_actuator``'s per-call rate limit rather than duplicating its
per-character one.

Safety:
  - ``DONNA_OS_DRY_RUN=1`` skips the real SendInput call but still runs every
    validation/rate-limit check.
  - Rate-limited to one physical typing actuation per
    ``_MIN_ACTUATION_INTERVAL_S`` (module-wide).
  - Best-effort ``dana.middleware.kill_switch`` check immediately before the
    physical typing call.
  - Refuses empty/whitespace-only text and caps a single call at
    ``_MAX_CHARS_PER_CALL`` so one tool call can't dump unbounded text.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

# One physical typing actuation per this many seconds, module-wide.
_MIN_ACTUATION_INTERVAL_S = 0.35
_MAX_CHARS_PER_CALL = 2000

_rate_lock = threading.Lock()
_last_actuation_ts = 0.0


def _dry_run() -> bool:
    return os.environ.get("DONNA_OS_DRY_RUN", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _rate_limit_ok() -> tuple[bool, str]:
    """Module-wide gate: refuse a second actuation inside the cooldown window."""
    global _last_actuation_ts
    now = time.monotonic()
    with _rate_lock:
        elapsed = now - _last_actuation_ts
        if elapsed < _MIN_ACTUATION_INTERVAL_S:
            wait = _MIN_ACTUATION_INTERVAL_S - elapsed
            return False, f"rate_limited: wait {wait:.2f}s between actuations"
        _last_actuation_ts = now
    return True, ""


@dataclass
class KeyboardActuator:
    """``text`` -> rate-limited, failsafe-checked stealth typing.

    ``type_fn`` defaults to ``dana.tools.os_control.type_text_sendinput``;
    tests inject a stub so the safety pipeline runs with no real hardware
    input.
    """

    type_fn: Callable[[str], dict[str, Any]] | None = None
    events: list[dict[str, Any]] = field(default_factory=list)

    def _type(self, text: str) -> dict[str, Any]:
        if self.type_fn is not None:
            return self.type_fn(text)
        from dana.tools.os_control import type_text_sendinput

        return type_text_sendinput(text)

    def type_text(self, text: str) -> dict[str, Any]:
        """Type ``text`` via the stealth SendInput backend.

        Returns a result dict (``ok``, and on success ``chars_typed``/
        ``dry_run``; on failure ``error`` and optionally ``halted``). Never
        raises for expected failure modes (empty/oversized text, rate limit,
        kill switch, backend failure) so it is safe to call directly from a
        tool wrapper.
        """
        self.events.clear()
        raw = text if isinstance(text, str) else str(text or "")
        if not raw.strip():
            return {"ok": False, "error": "empty text"}
        if len(raw) > _MAX_CHARS_PER_CALL:
            return {
                "ok": False,
                "error": (
                    f"text too long ({len(raw)} > {_MAX_CHARS_PER_CALL} "
                    "chars per call)"
                ),
            }

        ok, reason = _rate_limit_ok()
        if not ok:
            return {"ok": False, "error": reason}

        if _dry_run():
            self.events.append({"event": "dry_run_type", "chars": len(raw)})
            return {"ok": True, "chars_typed": len(raw), "dry_run": True}

        try:
            from dana.middleware.kill_switch import halt_if_requested

            if halt_if_requested():
                self.events.append({"event": "halt"})
                return {
                    "ok": False,
                    "halted": True,
                    "error": "halted by GLOBAL_HALT_EVENT",
                }
        except Exception:  # noqa: BLE001
            pass

        result = self._type(raw)
        if not isinstance(result, dict):
            result = {"ok": bool(result)}
        if not result.get("ok", True):
            self.events.append({"event": "type_failed", "error": result.get("error")})
            return {
                "ok": False,
                "error": result.get("error") or "type_text_sendinput failed",
            }
        chars_typed = int(result.get("chars_typed", len(raw)))
        self.events.append({"event": "type", "chars": chars_typed})
        return {"ok": True, "chars_typed": chars_typed, "dry_run": False}


def type_text(text: str) -> dict[str, Any]:
    """Module-level convenience wrapper around a default ``KeyboardActuator``."""
    return KeyboardActuator().type_text(text)


__all__ = ("KeyboardActuator", "type_text")
