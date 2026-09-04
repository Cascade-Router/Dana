"""Real Win32 window actuation + real FreeCAD IPC — local desktop execution.

Thin ``BaseControlPlane``/``BaseCADEngine`` adapters over the already-tested
low-level code in :mod:`dana.tools.os_control` (raw ``ctypes`` Win32 calls)
and :mod:`dana.plugins.freecad.engine` (``FreeCADCmd`` subprocess IPC) — no
actuation logic is duplicated here, only translated into the shared
interface shape.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from dana.platform.base import BaseCADEngine, BaseControlPlane

# Exact owning-process executable names, NOT a title substring — a title
# substring match (the previous approach) false-positives on any unrelated
# window whose title text happens to mention one of these words, e.g. a code
# editor with this repo open (a title like "engine.py - dana/plugins/freecad
# - Visual Studio Code" contains "freecad") gets forcibly relocated to the
# secondary monitor right along with the real FreeCAD window. Matching the
# actual process image name scopes this to the real CAD application
# regardless of what any window happens to have in its title bar — same
# convention dana.plugins.freecad.engine._is_freecad_gui_running already
# uses for "is FreeCAD running" (exact "freecad.exe", not a substring).
_CAD_WINDOW_PROCESS_NAMES = frozenset({"freecad.exe", "acad.exe"})


def _process_exe_name(pid: int) -> str:
    """Best-effort: the owning process's executable filename, lowercased.
    Returns "" if the pid is gone/inaccessible by the time we look it up —
    never raises, since a window can close between EnumWindows and here."""
    try:
        import psutil

        return (psutil.Process(pid).name() or "").lower()
    except Exception:  # noqa: BLE001
        return ""


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
            if _process_exe_name(int(win["pid"])) not in _CAD_WINDOW_PROCESS_NAMES:
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
        self, operation: str, base_object: str, tool_object: str, name: str | None = None
    ) -> dict[str, Any]:
        from dana.plugins.freecad import engine

        return json.loads(engine.apply_boolean(operation, base_object, tool_object, name))

    def apply_edge_operation(
        self,
        operation: str,
        target_object: str,
        value: float,
        face_centroid: tuple[float, float, float] | None = None,
        name: str | None = None,
    ) -> dict[str, Any]:
        from dana.plugins.freecad import engine

        return json.loads(
            engine.apply_edge_operation(operation, target_object, value, face_centroid=face_centroid, name=name)
        )

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

    def create_polygon(
        self,
        sides: int,
        radius: float,
        height: float,
        name: str = "Polygon",
        placement: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> dict[str, Any]:
        from dana.plugins.freecad import engine

        return json.loads(engine.create_polygon(sides, radius, height, name, placement=placement))

    def export_mesh_stl(
        self, source_path: str, name: str | None = None, target_object: str | None = None
    ) -> dict[str, Any]:
        from dana.plugins.freecad import engine

        return json.loads(engine.export_mesh_stl(source_path, name, target_object=target_object))

    def modify_parameter(
        self, target_object: str, parameter_name: str, new_value: float | Sequence[float]
    ) -> dict[str, Any]:
        from dana.plugins.freecad import engine

        return json.loads(engine.modify_parameter(target_object, parameter_name, new_value))

    def get_bounding_box(self, target_path: str, target_object: str | None = None) -> dict[str, Any]:
        from dana.plugins.freecad import engine

        return json.loads(engine.get_bounding_box(target_path, target_object=target_object))

    def inspect_spatial_properties(self, target_path: str, target_object: str | None = None) -> dict[str, Any]:
        from dana.plugins.freecad import engine

        return json.loads(engine.inspect_spatial_properties(target_path, target_object=target_object))

    def create_pipe(
        self,
        pipe_radius: float,
        path_type: str,
        length_or_angle: float,
        name: str = "Pipe",
        placement: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> dict[str, Any]:
        from dana.plugins.freecad import engine

        return json.loads(
            engine.create_pipe(pipe_radius, path_type, length_or_angle, name, placement=placement)
        )

    def align_objects(
        self,
        source_path: str,
        target_path: str,
        alignment_type: str,
        source_object: str | None = None,
        target_object: str | None = None,
    ) -> dict[str, Any]:
        from dana.plugins.freecad import engine

        return json.loads(
            engine.align_objects(
                source_path, target_path, alignment_type, source_object=source_object, target_object=target_object
            )
        )

    def export_model(
        self,
        target_paths: list[str],
        format: str,
        filename: str,
        target_objects: list[str] | None = None,
    ) -> dict[str, Any]:
        from dana.plugins.freecad import engine

        return json.loads(engine.export_model(target_paths, format, filename, target_objects=target_objects))

    def create_assembly_mate(
        self,
        fixed_path: str,
        moving_path: str,
        mate_type: str,
        mate_params: dict[str, Any] | None = None,
        fixed_object: str | None = None,
        moving_object: str | None = None,
    ) -> dict[str, Any]:
        from dana.plugins.freecad import engine

        return json.loads(
            engine.create_assembly_mate(
                fixed_path, moving_path, mate_type, mate_params, fixed_object=fixed_object, moving_object=moving_object
            )
        )

    def create_sketch_extrude(
        self,
        segments: list[dict[str, Any]],
        height: float,
        start: tuple[float, float] = (0.0, 0.0),
        plane: str = "XY",
        name: str = "Sketch",
        placement: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> dict[str, Any]:
        from dana.plugins.freecad import engine

        return json.loads(
            engine.create_sketch_extrude(segments, height, start=start, plane=plane, name=name, placement=placement)
        )

    def create_feature_on_face(
        self,
        object_name: str,
        face: str,
        shape: str,
        u: float,
        v: float,
        extent: float,
        operation: str,
        radius: float | None = None,
        width: float | None = None,
        length: float | None = None,
        name: str | None = None,
    ) -> dict[str, Any]:
        from dana.plugins.freecad import engine

        return json.loads(
            engine.create_feature_on_face(
                object_name, face, shape, u, v, extent, operation,
                radius=radius, width=width, length=length, name=name,
            )
        )

    def batch_pattern_array(
        self,
        source_path: str,
        pattern_type: str,
        *,
        source_object: str | None = None,
        count_x: int = 1,
        count_y: int = 1,
        spacing_x: float | None = None,
        spacing_y: float | None = None,
        count: int = 1,
        radius: float = 0.0,
        name: str = "Pattern",
    ) -> dict[str, Any]:
        from dana.plugins.freecad import engine

        return json.loads(
            engine.batch_pattern_array(
                source_path,
                pattern_type,
                source_object=source_object,
                count_x=count_x,
                count_y=count_y,
                spacing_x=spacing_x,
                spacing_y=spacing_y,
                count=count,
                radius=radius,
                name=name,
            )
        )


__all__ = ("RealFreeCADEngine", "Win32ControlPlane")
