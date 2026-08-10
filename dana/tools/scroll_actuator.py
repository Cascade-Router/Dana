"""Foundational Scroll Actuator — safe, rate-limited mouse-wheel navigation.

Completes the basic UI interaction loop alongside ``mouse_actuator`` (click)
and ``keyboard_actuator`` (type): reveals off-screen elements by scrolling.
No pyautogui/pynput — reuses the existing Win32 ``SendInput`` (ctypes)
backend in ``dana.tools.os_control`` (``scroll_wheel_sendinput``), sending one wheel
notch (``WHEEL_DELTA``) per tick with a small randomized inter-tick delay,
matching the humanized cadence used elsewhere in the OS-control surface
(clicking, typing).

Safety:
  - ``DANA_OS_DRY_RUN=1`` skips the real SendInput calls but still runs
    every validation/rate-limit check.
  - Rate-limited via ``dana.tools.rate_limiter`` (shared across every
    actuator) — one scroll actuation (i.e. one ``scroll()`` call, which may
    itself send several wheel ticks) per cooldown window.
  - Best-effort ``dana.middleware.kill_switch`` check immediately before the
    physical wheel events.
  - Rejects unknown directions and caps a single call at
    ``_MAX_TICKS_PER_CALL`` ticks so one tool call can't spin the wheel
    unbounded.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from dana.tools.os_control import WHEEL_DELTA
from dana.tools.rate_limiter import get_limiter

_MAX_TICKS_PER_CALL = 20

# direction -> (dx_sign, dy_sign) in wheel-notch units.
_DIRECTION_SIGN: dict[str, tuple[int, int]] = {
    "up": (0, 1),
    "down": (0, -1),
    "left": (-1, 0),
    "right": (1, 0),
}

_limiter = get_limiter("scroll")


def _dry_run() -> bool:
    from dana.security.dry_run import is_dry_run_enabled

    return is_dry_run_enabled()


def _rate_limit_ok() -> tuple[bool, str]:
    """Module-wide gate: refuse a second actuation inside the cooldown window.

    Shared implementation: see ``dana.tools.rate_limiter``.
    """
    return _limiter.check_and_mark()


@dataclass
class ScrollActuator:
    """``(direction, ticks)`` -> rate-limited, failsafe-checked wheel events.

    ``scroll_fn`` defaults to ``dana.tools.os_control.scroll_wheel_sendinput``;
    tests inject a stub so the safety pipeline runs with no real hardware
    input. ``scroll_fn`` is called once per tick as ``scroll_fn(dx, dy)``
    with signed wheel-delta units (multiples of ``WHEEL_DELTA``).
    """

    scroll_fn: Callable[[int, int], None] | None = None
    events: list[dict[str, Any]] = field(default_factory=list)

    def _scroll_tick(self, dx: int, dy: int) -> None:
        if self.scroll_fn is not None:
            self.scroll_fn(dx, dy)
            return
        from dana.tools.os_control import scroll_wheel_sendinput

        scroll_wheel_sendinput(dx=dx, dy=dy)

    def scroll(self, direction: str, ticks: int = 3) -> dict[str, Any]:
        """Scroll ``direction`` by ``ticks`` wheel notches.

        Returns a result dict (``ok``, and on success ``direction``/
        ``ticks``/``dry_run``; on failure ``error`` and optionally
        ``halted``/``ticks_completed``). Never raises for expected failure
        modes (bad direction/ticks, rate limit, kill switch, backend
        failure) so it is safe to call directly from a tool wrapper.
        """
        self.events.clear()
        d = str(direction or "").strip().lower()
        if d not in _DIRECTION_SIGN:
            return {
                "ok": False,
                "error": (
                    f"unknown direction {direction!r}; expected one of "
                    f"{sorted(_DIRECTION_SIGN)}"
                ),
            }
        try:
            n = int(ticks)
        except (TypeError, ValueError):
            return {"ok": False, "error": f"invalid ticks {ticks!r}"}
        if n <= 0:
            return {"ok": False, "error": f"ticks must be positive, got {n}"}
        if n > _MAX_TICKS_PER_CALL:
            return {
                "ok": False,
                "error": f"ticks too large ({n} > {_MAX_TICKS_PER_CALL} per call)",
            }

        ok, reason = _rate_limit_ok()
        if not ok:
            return {"ok": False, "error": reason}

        if _dry_run():
            self.events.append({"event": "dry_run_scroll", "direction": d, "ticks": n})
            return {"ok": True, "direction": d, "ticks": n, "dry_run": True}

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

        sx, sy = _DIRECTION_SIGN[d]
        for i in range(n):
            try:
                self._scroll_tick(sx * WHEEL_DELTA, sy * WHEEL_DELTA)
            except Exception as exc:  # noqa: BLE001
                self.events.append(
                    {"event": "scroll_failed", "error": str(exc), "ticks_completed": i}
                )
                return {
                    "ok": False,
                    "error": f"scroll failed after {i} tick(s): {exc}",
                    "ticks_completed": i,
                }
            if i < n - 1:
                time.sleep(random.uniform(0.02, 0.05))
        self.events.append({"event": "scroll", "direction": d, "ticks": n})
        return {"ok": True, "direction": d, "ticks": n, "dry_run": False}


def scroll(direction: str, ticks: int = 3) -> dict[str, Any]:
    """Module-level convenience wrapper around a default ``ScrollActuator``."""
    return ScrollActuator().scroll(direction, ticks)


__all__ = ("ScrollActuator", "scroll")
