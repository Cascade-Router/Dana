"""Foundational Mouse Actuator — safe, rate-limited clicks on vision bounding boxes.

Pairs with ``dana.vision.geometry`` to turn a raw ``[xmin, ymin, xmax, ymax]`` box (from
a vision model or an OS accessibility tree) into a single validated on-screen
click: inset off the dead-space border, take the centroid, optionally rescale
for display-resolution mismatches, then move and click.

The box is generic — whatever coordinate space the caller supplies, rescaled
via ``source_resolution``/``target_resolution`` before the centroid is used.
In practice the caller is almost always ``dana.tools.vision``'s hybrid Win32
UI-Automation + Florence-2 grounding pipeline
(``dana.vision.hybrid_grounding.HybridVisionGrounding``), which returns boxes
in Florence's normalized ``[0, 1000]`` coordinate space — this module never
assumes that convention itself, it just rescales whatever box+resolution
pair it's handed.

No pyautogui/pynput — every physical action bottoms out in Win32
``SendInput`` (ctypes) via ``dana.tools.os_control`` (the same backend
``execute_os_keystrokes`` and ``dana.operators.nav_and_click`` already use),
so every physical input in this codebase goes through one Windows-only,
stealth-cadence code path.

This module is intentionally the *foundational* primitive: a single
validated move+click. ``dana.operators.nav_and_click.NavigationOperator``
builds the more elaborate closed-loop Bezier-path servo (drift repathing,
kill-switch polling mid-path, human-yield checks) on top of the same
``os_control`` backend — reach for that when you need a servo loop, and for
this module when you just need "click this box, safely, once."

Safety:
  - ``DANA_OS_DRY_RUN=1`` skips the real SendInput call but still runs every
    validation/rate-limit check, so dry-run exercises the full safety path.
  - Failsafe: aborts with no motion at all if the computed point falls
    outside the live screen bounds (``dana.tools.os_control.get_screen_size``)
    — a bad bbox or a wrong resolution mapping can never fling the cursor
    off-screen or into an unintended monitor.
  - Rate-limited to one physical actuation per ``_MIN_ACTUATION_INTERVAL_S``
    (module-wide) to prevent a runaway tool-call loop from hammering clicks.
  - Best-effort ``dana.middleware.kill_switch`` check immediately before the
    physical click, matching the existing navigation operator's contract.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from dana.vision.geometry import BBox, get_centroid, inset_bbox, normalize_coordinates

# One physical actuation per this many seconds, module-wide.
_MIN_ACTUATION_INTERVAL_S = 0.35

_rate_lock = threading.Lock()
_last_actuation_ts = 0.0


def _dry_run() -> bool:
    return os.environ.get("DANA_OS_DRY_RUN", "").strip().lower() in {
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
class MouseActuator:
    """``bbox`` -> centroid -> failsafe bounds check -> rate-limited move + click.

    ``move_fn``/``click_fn``/``screen_size_fn`` default to the real
    ``dana.tools.os_control`` SendInput backend; tests inject stubs so the
    geometry/safety pipeline can be exercised with no real hardware input.
    """

    move_fn: Callable[[int, int], None] | None = None
    click_fn: Callable[[], None] | None = None
    screen_size_fn: Callable[[], tuple[int, int]] | None = None
    events: list[dict[str, Any]] = field(default_factory=list)

    def _screen_size(self) -> tuple[int, int]:
        if self.screen_size_fn is not None:
            return self.screen_size_fn()
        from dana.tools.os_control import get_screen_size

        return get_screen_size()

    def _move(self, x: int, y: int) -> None:
        if self.move_fn is not None:
            self.move_fn(x, y)
            return
        from dana.tools.os_control import move_cursor_absolute

        move_cursor_absolute(x, y)

    def _click(self) -> None:
        if self.click_fn is not None:
            self.click_fn()
            return
        from dana.tools.os_control import click_left_sendinput

        click_left_sendinput()

    def click_bbox(
        self,
        bbox: BBox,
        *,
        padding_percent: float = 10.0,
        source_resolution: tuple[float, float] | None = None,
        target_resolution: tuple[float, float] | None = None,
    ) -> dict[str, Any]:
        """Move to and left-click the safe centroid of ``bbox``.

        Returns a result dict (``ok``, and on success ``point``/``dry_run``;
        on failure ``error`` and optionally ``halted``). Never raises for
        expected failure modes (malformed bbox, out-of-bounds point, rate
        limit, kill switch) — those are reported in the return value so a
        tool wrapper can turn them into an observation string.
        """
        self.events.clear()

        try:
            safe_box = inset_bbox(bbox, padding_percent)
            x, y = get_centroid(safe_box)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"invalid bbox {bbox!r}: {exc}"}

        if source_resolution is not None and target_resolution is not None:
            try:
                x, y = normalize_coordinates(x, y, source_resolution, target_resolution)
            except Exception as exc:  # noqa: BLE001
                return {"ok": False, "error": f"invalid resolution scaling: {exc}"}

        try:
            screen_w, screen_h = self._screen_size()
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"could not read screen size: {exc}"}

        if not (0 <= x < screen_w and 0 <= y < screen_h):
            return {
                "ok": False,
                "error": (
                    f"failsafe: target ({x:.1f}, {y:.1f}) is outside the live "
                    f"screen bounds ({screen_w}x{screen_h}); aborted with no motion"
                ),
            }

        point = (int(round(x)), int(round(y)))

        ok, reason = _rate_limit_ok()
        if not ok:
            return {"ok": False, "error": reason, "point": point}

        if _dry_run():
            self.events.append({"event": "dry_run_click", "point": point})
            return {"ok": True, "point": point, "dry_run": True}

        try:
            from dana.middleware.kill_switch import halt_if_requested

            if halt_if_requested():
                self.events.append({"event": "halt", "point": point})
                return {
                    "ok": False,
                    "halted": True,
                    "error": "halted by GLOBAL_HALT_EVENT",
                    "point": point,
                }
        except Exception:  # noqa: BLE001
            pass

        self._move(*point)
        # Brief settle before the physical click, matching the human cadence
        # used elsewhere in the OS-control surface (nav_and_click's pre-click
        # pause; os_control's humanized keystroke delays).
        time.sleep(0.05)
        self._click()
        self.events.append({"event": "click", "point": point})
        return {"ok": True, "point": point, "dry_run": False}


def click_target_bbox(
    bbox: BBox,
    *,
    padding_percent: float = 10.0,
    source_resolution: tuple[float, float] | None = None,
    target_resolution: tuple[float, float] | None = None,
) -> str:
    """Tool entry point: safely click the centroid of a target bounding box."""
    try:
        from dana.ui.status_bus import emit_state_change

        emit_state_change("executing", tool="click_target_bbox")
    except Exception:  # noqa: BLE001
        pass

    result = MouseActuator().click_bbox(
        bbox,
        padding_percent=padding_percent,
        source_resolution=source_resolution,
        target_resolution=target_resolution,
    )
    if result.get("halted"):
        return f"HALTED: click_target_bbox — {result.get('error')}"
    if not result.get("ok"):
        return f"ERROR: click_target_bbox failed: {result.get('error')}"
    return (
        f"OK: click_target_bbox point={result.get('point')} "
        f"dry_run={result.get('dry_run')}"
    )


__all__ = ("MouseActuator", "click_target_bbox")
