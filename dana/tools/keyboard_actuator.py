"""Foundational Keyboard Actuator — safe, rate-limited stealth typing and shortcuts.

Pairs with ``dana.tools.mouse_actuator.MouseActuator`` to complete the basic
UI interaction loop: click to focus, then type. No pyautogui/pynput — reuses
the existing Win32 ``SendInput`` (ctypes) backend in ``dana.tools.os_control``
(``type_text_sendinput``), which already emits per-character scan-code events
with a randomized 40-110ms humanized press/release cadence — the same
"human-like keystroke cadence" this module is asked to provide. Re-adding a
second, different per-keystroke delay here would just fight that established
cadence, so this module adds its own protection at the *call* level instead
(one typing actuation per ``dana.tools.rate_limiter`` cooldown window),
mirroring ``mouse_actuator``'s per-call rate limit rather than duplicating
its per-character one.

Milestone 2 adds ``execute_shortcut`` alongside ``type_text``: parses a
``"ctrl+c"``-style combo string and presses it via
``dana.tools.os_control.press_key_combo`` — the general-purpose sibling of
the same file's narrowly allowlisted ``execute_os_keystrokes`` hotkey path,
meant for select-all/copy/paste and window-switching combos so clipboard
extraction can stand in for lossy vision OCR.

Safety:
  - ``DANA_OS_DRY_RUN=1`` skips the real SendInput call but still runs every
    validation/rate-limit check.
  - Rate-limited via ``dana.tools.rate_limiter`` (shared across every
    actuator, both methods included).
  - Best-effort ``dana.middleware.kill_switch`` check immediately before the
    physical typing/shortcut call.
  - Refuses empty/whitespace-only text and caps a single call at
    ``_MAX_CHARS_PER_CALL`` so one tool call can't dump unbounded text.
  - ``execute_shortcut`` caps a combo at ``_MAX_COMBO_KEYS`` keys and
    validates every key name resolves to a known VK before pressing
    anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from dana.tools.rate_limiter import get_limiter

_MAX_CHARS_PER_CALL = 2000
_MAX_COMBO_KEYS = 4

_limiter = get_limiter("keyboard")


def _dry_run() -> bool:
    from dana.security.dry_run import is_dry_run_enabled

    return is_dry_run_enabled()


def _rate_limit_ok() -> tuple[bool, str]:
    """Module-wide gate: refuse a second actuation inside the cooldown window.

    Shared implementation: see ``dana.tools.rate_limiter``.
    """
    return _limiter.check_and_mark()


@dataclass
class KeyboardActuator:
    """``text`` -> rate-limited, failsafe-checked stealth typing.

    ``type_fn`` defaults to ``dana.tools.os_control.type_text_sendinput``;
    tests inject a stub so the safety pipeline runs with no real hardware
    input.
    """

    type_fn: Callable[[str], dict[str, Any]] | None = None
    combo_fn: Callable[[list[str]], None] | None = None
    events: list[dict[str, Any]] = field(default_factory=list)

    def _type(self, text: str) -> dict[str, Any]:
        if self.type_fn is not None:
            return self.type_fn(text)
        from dana.tools.os_control import type_text_sendinput

        return type_text_sendinput(text)

    def _combo(self, keys: list[str]) -> None:
        if self.combo_fn is not None:
            self.combo_fn(keys)
            return
        from dana.tools.os_control import press_key_combo

        press_key_combo(keys)

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

    def execute_shortcut(self, combo_string: str) -> dict[str, Any]:
        """Parse ``combo_string`` (e.g. ``"ctrl+c"``, ``"alt+tab"``) and press
        it via ``press_key_combo``: all keys down in order, released in
        reverse order.

        Validates every key name resolves to a known Win32 VK *before* the
        rate-limit/dry-run/kill-switch checks (matching the other
        actuators' "validate first" convention) — an unresolvable key
        aborts with no motion at all, dry-run or not.

        Returns a result dict (``ok``, and on success ``keys``/``dry_run``;
        on failure ``error`` and optionally ``halted``). Never raises for
        expected failure modes (empty/malformed/oversized combo,
        unresolvable key, rate limit, kill switch, backend failure).
        """
        self.events.clear()
        raw = str(combo_string or "").strip()
        if not raw:
            return {"ok": False, "error": "execute_shortcut requires a non-empty combo_string"}

        parts = [p.strip().lower() for p in raw.split("+") if p.strip()]
        if not parts:
            return {"ok": False, "error": f"could not parse combo_string {combo_string!r}"}
        if len(parts) > _MAX_COMBO_KEYS:
            return {
                "ok": False,
                "error": f"combo has too many keys ({len(parts)} > {_MAX_COMBO_KEYS})",
            }

        from dana.tools.os_control import resolve_key_name

        unresolved = [p for p in parts if resolve_key_name(p) is None]
        if unresolved:
            return {"ok": False, "error": f"unsupported key(s): {unresolved}"}

        ok, reason = _rate_limit_ok()
        if not ok:
            return {"ok": False, "error": reason}

        if _dry_run():
            self.events.append({"event": "dry_run_shortcut", "keys": parts})
            return {"ok": True, "keys": parts, "dry_run": True}

        try:
            from dana.middleware.kill_switch import halt_if_requested

            if halt_if_requested():
                self.events.append({"event": "halt", "keys": parts})
                return {
                    "ok": False,
                    "halted": True,
                    "error": "halted by GLOBAL_HALT_EVENT",
                    "keys": parts,
                }
        except Exception:  # noqa: BLE001
            pass

        try:
            self._combo(parts)
        except Exception as exc:  # noqa: BLE001
            self.events.append({"event": "shortcut_failed", "error": str(exc)})
            return {"ok": False, "error": f"execute_shortcut failed: {exc}", "keys": parts}

        self.events.append({"event": "shortcut", "keys": parts})
        return {"ok": True, "keys": parts, "dry_run": False}


def type_text(text: str) -> dict[str, Any]:
    """Module-level convenience wrapper around a default ``KeyboardActuator``."""
    return KeyboardActuator().type_text(text)


def execute_shortcut(combo_string: str) -> dict[str, Any]:
    """Module-level convenience wrapper around a default ``KeyboardActuator``."""
    return KeyboardActuator().execute_shortcut(combo_string)


__all__ = ("KeyboardActuator", "type_text", "execute_shortcut")
