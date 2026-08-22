"""Foundational Drag Actuator — safe, rate-limited drag-and-drop between two
vision bounding boxes.

Completes the basic UI interaction loop alongside ``mouse_actuator`` (click),
``keyboard_actuator`` (type), and ``scroll_actuator`` (wheel): grabs an
element at a source bounding box and releases it over a destination bounding
box. No pyautogui/pynput — reuses the existing Win32 ``SendInput`` (ctypes)
backend in ``dana.tools.os_control`` (``move_cursor_absolute``, ``mouse_down_sendinput``,
``mouse_up_sendinput``), sending move-to-source -> mouse down -> a short
sequence of human-cadenced intermediate moves -> mouse up at the destination.

Safety:
  - ``DANA_OS_DRY_RUN=1`` skips the real SendInput calls but still runs
    every validation/rate-limit check, so dry-run exercises the full safety
    path.
  - Failsafe: aborts with no motion at all if either computed point (source
    or destination) falls outside the live screen bounds
    (``dana.tools.os_control.get_screen_size``).
  - Rate-limited via ``dana.tools.rate_limiter`` (shared across every
    actuator) — one physical actuation (i.e. one whole drag) per cooldown
    window.
  - Best-effort ``dana.middleware.kill_switch`` check immediately before the
    physical mouse-down, matching the other actuators' contract.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from dana.logging import log_debug
from dana.tools.rate_limiter import get_limiter
from dana.vision.geometry import BBox, get_centroid, inset_bbox, normalize_coordinates

# Intermediate waypoints between source and destination (excludes the source
# point itself, includes the destination as the final waypoint) — enough to
# read as a smooth, human-cadenced drag rather than a single hard teleport.
_DRAG_WAYPOINTS = 6

_limiter = get_limiter("drag")


def _dry_run() -> bool:
    from dana.security.dry_run import is_dry_run_enabled

    return is_dry_run_enabled()


def _rate_limit_ok() -> tuple[bool, str]:
    """Module-wide gate: refuse a second actuation inside the cooldown window.

    Shared implementation: see ``dana.tools.rate_limiter``.
    """
    return _limiter.check_and_mark()


def _lerp_waypoints(
    start: tuple[int, int], end: tuple[int, int], n: int
) -> list[tuple[int, int]]:
    """Return ``n`` linearly-interpolated points from (excl.) ``start`` to (incl.) ``end``."""
    if n <= 1:
        return [end]
    sx, sy = start
    ex, ey = end
    return [
        (
            int(round(sx + (ex - sx) * (i / n))),
            int(round(sy + (ey - sy) * (i / n))),
        )
        for i in range(1, n + 1)
    ]


@dataclass
class DragActuator:
    """``(source_bbox, dest_bbox)`` -> centroids -> failsafe bounds check ->
    rate-limited move + down + smooth move + up.

    ``move_fn``/``down_fn``/``up_fn``/``screen_size_fn`` default to the real
    ``dana.tools.os_control`` SendInput backend; tests inject stubs so the
    geometry/safety pipeline can be exercised with no real hardware input.
    """

    move_fn: Callable[[int, int], None] | None = None
    down_fn: Callable[[], None] | None = None
    up_fn: Callable[[], None] | None = None
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

    def _down(self) -> None:
        if self.down_fn is not None:
            self.down_fn()
            return
        from dana.tools.os_control import mouse_down_sendinput

        mouse_down_sendinput()

    def _up(self) -> None:
        if self.up_fn is not None:
            self.up_fn()
            return
        from dana.tools.os_control import mouse_up_sendinput

        mouse_up_sendinput()

    def _resolve_point(
        self,
        bbox: BBox,
        *,
        padding_percent: float,
        source_resolution: tuple[float, float] | None,
        target_resolution: tuple[float, float] | None,
        label: str,
    ) -> tuple[float, float] | dict[str, Any]:
        """Inset+centroid+rescale ``bbox``, or an error dict on failure."""
        try:
            safe_box = inset_bbox(bbox, padding_percent)
            x, y = get_centroid(safe_box)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"invalid {label} bbox {bbox!r}: {exc}"}

        if source_resolution is not None and target_resolution is not None:
            try:
                x, y = normalize_coordinates(x, y, source_resolution, target_resolution)
            except Exception as exc:  # noqa: BLE001
                return {"ok": False, "error": f"invalid {label} resolution scaling: {exc}"}
        return x, y

    def drag_bbox(
        self,
        source_bbox: BBox,
        dest_bbox: BBox,
        *,
        padding_percent: float = 10.0,
        source_resolution: tuple[float, float] | None = None,
        target_resolution: tuple[float, float] | None = None,
    ) -> dict[str, Any]:
        """Drag from the safe centroid of ``source_bbox`` to that of ``dest_bbox``.

        Returns a result dict (``ok``, and on success ``source_point``/
        ``dest_point``/``dry_run``; on failure ``error`` and optionally
        ``halted``). Never raises for expected failure modes (malformed
        bbox, out-of-bounds point, rate limit, kill switch) — those are
        reported in the return value so a tool wrapper can turn them into an
        observation string.
        """
        self.events.clear()

        src = self._resolve_point(
            source_bbox,
            padding_percent=padding_percent,
            source_resolution=source_resolution,
            target_resolution=target_resolution,
            label="source",
        )
        if isinstance(src, dict):
            return src
        dst = self._resolve_point(
            dest_bbox,
            padding_percent=padding_percent,
            source_resolution=source_resolution,
            target_resolution=target_resolution,
            label="destination",
        )
        if isinstance(dst, dict):
            return dst

        try:
            screen_w, screen_h = self._screen_size()
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"could not read screen size: {exc}"}

        sx, sy = src
        dxp, dyp = dst
        if not (0 <= sx < screen_w and 0 <= sy < screen_h):
            return {
                "ok": False,
                "error": (
                    f"failsafe: source ({sx:.1f}, {sy:.1f}) is outside the live "
                    f"screen bounds ({screen_w}x{screen_h}); aborted with no motion"
                ),
            }
        if not (0 <= dxp < screen_w and 0 <= dyp < screen_h):
            return {
                "ok": False,
                "error": (
                    f"failsafe: destination ({dxp:.1f}, {dyp:.1f}) is outside the live "
                    f"screen bounds ({screen_w}x{screen_h}); aborted with no motion"
                ),
            }

        source_point = (int(round(sx)), int(round(sy)))
        dest_point = (int(round(dxp)), int(round(dyp)))

        ok, reason = _rate_limit_ok()
        if not ok:
            return {
                "ok": False,
                "error": reason,
                "source_point": source_point,
                "dest_point": dest_point,
            }

        if _dry_run():
            self.events.append(
                {
                    "event": "dry_run_drag",
                    "source_point": source_point,
                    "dest_point": dest_point,
                }
            )
            return {
                "ok": True,
                "source_point": source_point,
                "dest_point": dest_point,
                "dry_run": True,
            }

        try:
            from dana.middleware.kill_switch import halt_if_requested

            if halt_if_requested():
                self.events.append({"event": "halt", "point": source_point})
                return {
                    "ok": False,
                    "halted": True,
                    "error": "halted by GLOBAL_HALT_EVENT",
                    "source_point": source_point,
                    "dest_point": dest_point,
                }
        except Exception:  # noqa: BLE001
            pass

        self._move(*source_point)
        self.events.append({"event": "move", "point": source_point})
        # Brief settle before pressing down, matching the human cadence used
        # elsewhere in the OS-control surface (mouse_actuator's pre-click
        # pause; os_control's humanized keystroke delays).
        time.sleep(0.05)
        self._down()
        self.events.append({"event": "down", "point": source_point})

        waypoints = _lerp_waypoints(source_point, dest_point, _DRAG_WAYPOINTS)
        for i, point in enumerate(waypoints):
            self._move(*point)
            self.events.append({"event": "move", "point": point})
            if i < len(waypoints) - 1:
                time.sleep(random.uniform(0.02, 0.05))

        time.sleep(0.05)
        self._up()
        self.events.append({"event": "up", "point": dest_point})

        return {
            "ok": True,
            "source_point": source_point,
            "dest_point": dest_point,
            "dry_run": False,
        }


def drag_target_bbox(
    source_bbox: BBox,
    dest_bbox: BBox,
    *,
    padding_percent: float = 10.0,
    source_resolution: tuple[float, float] | None = None,
    target_resolution: tuple[float, float] | None = None,
) -> str:
    """Tool entry point: safely drag from one bounding box's centroid to another's."""
    log_debug("Actuator", "executing tool=drag_target_bbox")

    result = DragActuator().drag_bbox(
        source_bbox,
        dest_bbox,
        padding_percent=padding_percent,
        source_resolution=source_resolution,
        target_resolution=target_resolution,
    )
    if result.get("halted"):
        return f"HALTED: drag_target_bbox — {result.get('error')}"
    if not result.get("ok"):
        return f"ERROR: drag_target_bbox failed: {result.get('error')}"
    return (
        f"OK: drag_target_bbox source={result.get('source_point')} "
        f"dest={result.get('dest_point')} dry_run={result.get('dry_run')}"
    )


__all__ = ("DragActuator", "drag_target_bbox")
