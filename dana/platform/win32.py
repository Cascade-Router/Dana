"""Real Win32 window actuation + real FreeCAD IPC — local desktop execution.

Thin ``BaseControlPlane``/``BaseCADEngine`` adapters over the already-tested
low-level code in :mod:`dana.tools.os_control` (raw ``ctypes`` Win32 calls)
and :mod:`dana.plugins.freecad.engine` (``FreeCADCmd`` subprocess IPC) — no
actuation logic is duplicated here, only translated into the shared
interface shape.
"""

from __future__ import annotations

import json
from typing import Any

from dana.platform.base import BaseCADEngine, BaseControlPlane

_CAD_WINDOW_HINTS = ("freecad", "autocad", "acad")


class Win32ControlPlane(BaseControlPlane):
    def resync_workspace(self) -> dict[str, Any]:
        from dana.tools.os_control import (
            get_active_windows,
            get_secondary_monitor,
            move_window_no_activate,
        )

        monitor = get_secondary_monitor()
        if monitor is None:
            return {"ok": True, "moved": [], "note": "single monitor — nothing to resync"}

        width = min(1280, monitor["width"])
        height = min(800, monitor["height"])
        x, y = monitor["left"] + 40, monitor["top"] + 40

        moved: list[dict[str, Any]] = []
        for win in get_active_windows():
            title = str(win.get("title") or "").lower()
            if not any(hint in title for hint in _CAD_WINDOW_HINTS):
                continue
            ok = move_window_no_activate(int(win["hwnd"]), x, y, width, height)
            moved.append({"hwnd": win["hwnd"], "title": win["title"], "moved": ok})
        return {"ok": True, "moved": moved}

    def prevent_focus_steal(self) -> dict[str, Any]:
        from dana.tools.os_control import get_active_windows

        windows = get_active_windows()
        foreground = windows[0] if windows else None  # EnumWindows returns Z-order, topmost first
        return {"ok": True, "foreground": foreground}

    def get_active_display(self) -> dict[str, Any]:
        from dana.tools.os_control import get_screen_size, get_secondary_monitor

        width, height = get_screen_size()
        return {
            "ok": True,
            "primary": {"left": 0, "top": 0, "width": width, "height": height},
            "secondary": get_secondary_monitor(),
        }


class RealFreeCADEngine(BaseCADEngine):
    def create_box(
        self,
        length: float,
        width: float,
        height: float,
        name: str = "Box",
        placement: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> dict[str, Any]:
        from dana.plugins.freecad import engine

        return json.loads(engine.create_box(length, width, height, name, placement=placement))

    def create_cylinder(
        self,
        radius: float,
        height: float,
        name: str = "Cylinder",
        placement: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> dict[str, Any]:
        from dana.plugins.freecad import engine

        return json.loads(engine.create_cylinder(radius, height, name, placement=placement))

    def apply_boolean(
        self, operation: str, base_path: str, tool_path: str, name: str | None = None
    ) -> dict[str, Any]:
        from dana.plugins.freecad import engine

        return json.loads(engine.apply_boolean(operation, base_path, tool_path, name))

    def create_extrusion(
        self, profile_points: list[list[float]], height: float, name: str = "Extrusion"
    ) -> dict[str, Any]:
        from dana.plugins.freecad import engine

        # create_extruded_polyline names its own Part::Feature internally —
        # `name` isn't threaded through to the underlying script (matches
        # its existing behavior; not something to silently change here).
        return json.loads(engine.create_extruded_polyline(profile_points, height))

    def create_pyramid(
        self,
        length: float,
        width: float,
        height: float,
        name: str = "Pyramid",
        placement: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> dict[str, Any]:
        from dana.plugins.freecad import engine

        return json.loads(engine.create_pyramid(length, width, height, name, placement=placement))

    def create_star_prism(
        self,
        points: int,
        outer_radius: float,
        inner_radius: float,
        height: float,
        name: str = "StarPrism",
        placement: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> dict[str, Any]:
        from dana.plugins.freecad import engine

        return json.loads(
            engine.create_star_prism(points, outer_radius, inner_radius, height, name, placement=placement)
        )

    def export_mesh_stl(self, source_path: str, name: str | None = None) -> dict[str, Any]:
        from dana.plugins.freecad import engine

        return json.loads(engine.export_mesh_stl(source_path, name))


__all__ = ("RealFreeCADEngine", "Win32ControlPlane")
