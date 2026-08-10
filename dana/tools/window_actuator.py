"""Foundational Window Actuator — safe, rate-limited foreground-window switching.

Milestone 2 (Workspace Orchestration): before click/type/scroll/drag can
reliably act on the right application, Dānā needs to confirm — and if
necessary change — which window is actually in the foreground. This module
resolves a window by regex-matching its title against the live desktop
window list, then brings it to the front via the raw Win32 primitives in
``dana.tools.os_control`` (``get_active_windows``, ``set_foreground_window``).

Safety:
  - ``DANA_OS_DRY_RUN=1`` skips the real ``SetForegroundWindow`` call but
    still runs every validation/matching/rate-limit check, so dry-run
    exercises the full safety path.
  - Rate-limited via ``dana.tools.rate_limiter`` (shared across every
    actuator).
  - Best-effort ``dana.middleware.kill_switch`` check immediately before the
    physical foreground switch.
  - Fails closed on an invalid regex, no matching window, or a Windows-level
    focus-steal denial — never raises for these expected outcomes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from dana.tools.rate_limiter import get_limiter

_limiter = get_limiter("window")


def _dry_run() -> bool:
    from dana.security.dry_run import is_dry_run_enabled

    return is_dry_run_enabled()


def _rate_limit_ok() -> tuple[bool, str]:
    """Module-wide gate: refuse a second actuation inside the cooldown window.

    Shared implementation: see ``dana.tools.rate_limiter``.
    """
    return _limiter.check_and_mark()


@dataclass
class WindowActuator:
    """``title_regex`` -> best-matching visible window -> rate-limited foreground switch.

    ``list_windows_fn``/``focus_fn`` default to the real
    ``dana.tools.os_control`` backend; tests inject stubs so the
    matching/safety pipeline can be exercised with no real windows on the
    live desktop.
    """

    list_windows_fn: Callable[[], list[dict[str, Any]]] | None = None
    focus_fn: Callable[[int], bool] | None = None
    events: list[dict[str, Any]] = field(default_factory=list)

    def _list_windows(self) -> list[dict[str, Any]]:
        if self.list_windows_fn is not None:
            return self.list_windows_fn()
        from dana.tools.os_control import get_active_windows

        return get_active_windows()

    def _focus(self, hwnd: int) -> bool:
        if self.focus_fn is not None:
            return self.focus_fn(hwnd)
        from dana.tools.os_control import set_foreground_window

        return set_foreground_window(hwnd)

    def focus_by_title(self, title_regex: str) -> dict[str, Any]:
        """Bring the best-matching visible window to the foreground.

        Matching: ``title_regex`` is compiled case-insensitively and
        searched against every visible window's title. Among matches, a
        window whose full title matches exactly (``re.fullmatch``) wins;
        otherwise the first partial match wins, which is also the topmost
        window in Win32 Z-order (``get_active_windows`` preserves it) — a
        reasonable "most relevant" default when several windows share a
        substring (e.g. several browser tabs).

        Returns a result dict (``ok``, and on success ``window``/
        ``dry_run``; on failure ``error`` and optionally ``halted``). Never
        raises for expected failure modes (bad regex, no match, rate limit,
        kill switch, focus-steal denial) — those are reported in the return
        value so a tool wrapper can turn them into an observation string.
        """
        self.events.clear()

        pattern_str = str(title_regex or "").strip()
        if not pattern_str:
            return {"ok": False, "error": "focus_by_title requires a non-empty title_regex"}

        try:
            pattern = re.compile(pattern_str, re.IGNORECASE)
        except re.error as exc:
            return {"ok": False, "error": f"invalid title_regex {pattern_str!r}: {exc}"}

        try:
            windows = self._list_windows()
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"could not list windows: {exc}"}

        matches = [w for w in windows if pattern.search(str(w.get("title") or ""))]
        if not matches:
            return {"ok": False, "error": f"no window title matched {pattern_str!r}"}

        exact = [w for w in matches if pattern.fullmatch(str(w.get("title") or ""))]
        best = exact[0] if exact else matches[0]

        ok, reason = _rate_limit_ok()
        if not ok:
            return {"ok": False, "error": reason, "window": best}

        if _dry_run():
            self.events.append({"event": "dry_run_focus", "window": best})
            return {"ok": True, "window": best, "dry_run": True}

        try:
            from dana.middleware.kill_switch import halt_if_requested

            if halt_if_requested():
                self.events.append({"event": "halt", "window": best})
                return {
                    "ok": False,
                    "halted": True,
                    "error": "halted by GLOBAL_HALT_EVENT",
                    "window": best,
                }
        except Exception:  # noqa: BLE001
            pass

        try:
            focused = self._focus(int(best["hwnd"]))
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"SetForegroundWindow failed: {exc}", "window": best}

        if not focused:
            return {
                "ok": False,
                "error": (
                    f"SetForegroundWindow reported failure for hwnd={best.get('hwnd')} "
                    f"title={best.get('title')!r} (Windows may have denied the focus-steal)"
                ),
                "window": best,
            }

        self.events.append({"event": "focus", "window": best})
        return {"ok": True, "window": best, "dry_run": False}


def focus_by_title(title_regex: str) -> dict[str, Any]:
    """Module-level convenience wrapper around a default ``WindowActuator``."""
    return WindowActuator().focus_by_title(title_regex)


__all__ = ("WindowActuator", "focus_by_title")
