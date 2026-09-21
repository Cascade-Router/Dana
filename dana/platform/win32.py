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

    def query_topology(self, part_name: str) -> dict:
        """Returns topological face data for the given part."""
        from dana.plugins.freecad.engine import query_topology as engine_query_topology
        return json.loads(engine_query_topology(part_name))
    
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
        self,
        operation: str,
        base_object: str | None = None,
        tool_object: str | None = None,
        name: str | None = None,
        objects: list[str] | None = None,
    ) -> dict[str, Any]:
        from dana.plugins.freecad import engine

        return json.loads(engine.apply_boolean(operation, base_object, tool_object, name, objects=objects))

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
        self,
        target_object: str,
        parameter_name: str,
        new_value: float | Sequence[float],
        yaw: float | None = None,
        pitch: float | None = None,
        roll: float | None = None,
    ) -> dict[str, Any]:
        from dana.plugins.freecad import engine

        return json.loads(
            engine.modify_parameter(target_object, parameter_name, new_value, yaw=yaw, pitch=pitch, roll=roll)
        )

    def get_bounding_box(self, target_path: str, target_object: str | None = None) -> dict[str, Any]:
        from dana.plugins.freecad import engine

        return json.loads(engine.get_bounding_box(target_path, target_object=target_object))

    def inspect_spatial_properties(self, target_path: str, target_object: str | None = None) -> dict[str, Any]:
        from dana.plugins.freecad import engine

        return json.loads(engine.inspect_spatial_properties(target_path, target_object=target_object))

    def query_topology(self, part_name: str) -> dict[str, Any]:
        from dana.plugins.freecad import engine

        return json.loads(engine.query_topology(part_name))

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

    def create_helix(
        self,
        coil_radius: float,
        pitch: float,
        height: float,
        pipe_radius: float,
        name: str = "Helix",
        placement: tuple[float, float, float] = (0.0, 0.0, 0.0),
        angle_offset: float = 0.0,
    ) -> dict[str, Any]:
        from dana.plugins.freecad import engine

        return json.loads(
            engine.create_helix(
                coil_radius, pitch, height, pipe_radius, name, placement=placement, angle_offset=angle_offset
            )
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

    def create_sketch(
        self,
        name: str,
        plane: str,
        geometry: list[dict[str, Any]],
    ) -> dict[str, Any]:
        from dana.plugins.freecad import engine

        return json.loads(engine.create_sketch(name, plane, geometry))

    def apply_sketch_constraint(
        self,
        sketch_name: str,
        constraint_type: str,
        geometry_indices: list[int],
        value: float | None = None,
    ) -> dict[str, Any]:
        from dana.plugins.freecad import engine

        return json.loads(
            engine.apply_sketch_constraint(sketch_name, constraint_type, geometry_indices, value=value)
        )

    def create_pad(
        self,
        sketch_name: str,
        length: float,
        symmetric_to_plane: bool = False,
        reversed_direction: bool = False,
    ) -> dict[str, Any]:
        from dana.plugins.freecad import engine

        return json.loads(
            engine.create_pad(
                sketch_name, length, symmetric_to_plane=symmetric_to_plane, reversed_direction=reversed_direction
            )
        )

    def create_pocket(
        self,
        sketch_name: str,
        depth: float,
        through_all: bool = False,
        symmetric_to_plane: bool = False,
        reversed_direction: bool = False,
    ) -> dict[str, Any]:
        from dana.plugins.freecad import engine

        return json.loads(
            engine.create_pocket(
                sketch_name,
                depth,
                through_all=through_all,
                symmetric_to_plane=symmetric_to_plane,
                reversed_direction=reversed_direction,
            )
        )

    def create_polar_pattern(
        self,
        feature_name: str,
        occurrences: int,
        angle: float = 360.0,
        axis: str = "Z",
        reversed_direction: bool = False,
    ) -> dict[str, Any]:
        from dana.plugins.freecad import engine

        return json.loads(
            engine.create_polar_pattern(
                feature_name, occurrences, angle=angle, axis=axis, reversed_direction=reversed_direction
            )
        )

    def create_linear_pattern(
        self,
        feature_name: str,
        occurrences: int,
        length: float,
        direction: str = "X",
        reversed_direction: bool = False,
    ) -> dict[str, Any]:
        from dana.plugins.freecad import engine

        return json.loads(
            engine.create_linear_pattern(
                feature_name, occurrences, length, direction=direction, reversed_direction=reversed_direction
            )
        )

    def create_sweep(self, profile_sketch: str, path_sketch: str, frenet: bool = True) -> dict[str, Any]:
        from dana.plugins.freecad import engine

        return json.loads(engine.create_sweep(profile_sketch, path_sketch, frenet=frenet))

    def create_loft(
        self,
        cross_section_sketches: list[str],
        ruled: bool = False,
        closed: bool = False,
    ) -> dict[str, Any]:
        from dana.plugins.freecad import engine

        return json.loads(engine.create_loft(cross_section_sketches, ruled=ruled, closed=closed))

    def create_assembly(self, name: str) -> dict[str, Any]:
        from dana.plugins.freecad import engine

        return json.loads(engine.create_assembly(name))

    def add_parts_to_assembly(self, assembly_name: str, part_names: list[str]) -> dict[str, Any]:
        from dana.plugins.freecad import engine

        return json.loads(engine.add_parts_to_assembly(assembly_name, part_names))

    def position_assembly_part(
        self,
        part_name: str,
        placement_x: float = 0.0,
        placement_y: float = 0.0,
        placement_z: float = 0.0,
        yaw: float = 0.0,
        pitch: float = 0.0,
        roll: float = 0.0,
    ) -> dict[str, Any]:
        from dana.plugins.freecad import engine

        return json.loads(
            engine.position_assembly_part(
                part_name,
                placement_x=placement_x,
                placement_y=placement_y,
                placement_z=placement_z,
                yaw=yaw,
                pitch=pitch,
                roll=roll,
            )
        )

    def apply_assembly_constraint(
        self,
        assembly_name: str,
        part1_name: str,
        part1_element: str,
        part2_name: str,
        part2_element: str,
        constraint_type: str,
        offset: float = 0.0,
        uv_tensor: Sequence[float] | None = None,
        world_fractions: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        from dana.plugins.freecad import engine

        return json.loads(
            engine.apply_assembly_constraint(
                assembly_name,
                part1_name,
                part1_element,
                part2_name,
                part2_element,
                constraint_type,
                offset=offset,
                uv_tensor=uv_tensor,
                world_fractions=world_fractions,
            )
        )

    def anchor_assembly_root(self, assembly_name: str, part_name: str) -> dict[str, Any]:
        from dana.plugins.freecad import engine

        return json.loads(engine.anchor_assembly_root(assembly_name, part_name))

    def define_kinematic_joint(
        self,
        assembly_name: str,
        child_link: str,
        parent_link: str = "base_link",
        joint_type: str = "fixed",
        axis: Sequence[float] = (0.0, 0.0, 1.0),
        joint_name: str | None = None,
        limit_lower: float | None = None,
        limit_upper: float | None = None,
        limit_effort: float | None = None,
        limit_velocity: float | None = None,
    ) -> dict[str, Any]:
        from dana.plugins.freecad import engine

        return json.loads(
            engine.define_kinematic_joint(
                assembly_name,
                child_link,
                parent_link=parent_link,
                joint_type=joint_type,
                axis=axis,
                joint_name=joint_name,
                limit_lower=limit_lower,
                limit_upper=limit_upper,
                limit_effort=limit_effort,
                limit_velocity=limit_velocity,
            )
        )

    def validate_assembly_collisions(self, assembly_name: str) -> dict[str, Any]:
        from dana.plugins.freecad import engine

        return json.loads(engine.validate_assembly_collisions(assembly_name))

    def export_assembly_to_urdf(
        self,
        assembly_name: str,
        export_directory: str | None = None,
        density_kg_m3: float | None = None,
    ) -> dict[str, Any]:
        from dana.plugins.freecad import engine

        return json.loads(engine.export_assembly_to_urdf(assembly_name, export_directory, density_kg_m3))

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
