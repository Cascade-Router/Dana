"""Simulated telemetry + headless ``trimesh`` geometry — cloud/sandboxed execution.

Used whenever there's no real Win32 desktop or FreeCAD binary to talk to
(Hugging Face Spaces, CI, any non-Windows/non-FreeCAD host). Every response
carries ``"driver": "mock"`` and a human-readable ``"note"`` so a caller —
or a UI rendering the result — can never mistake simulated output for a
real actuation, mirroring the labeling convention already used in
``hf_space/hf_sandbox``.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from dana.platform.base import BaseCADEngine, BaseControlPlane

_MOCK_NOTE_CONTROL = "mocked — no Windows/Win32 APIs in this container"
_MOCK_NOTE_CAD = "mocked — headless trimesh geometry, no FreeCADCmd binary in this container"

_MOCK_WINDOWS: list[dict[str, Any]] = [
    {"hwnd": 1001, "title": "FreeCAD 1.0 — DanaModel.FCStd", "pid": 4021},
    {"hwnd": 1002, "title": "Dana — Live Trace", "pid": 3110},
]

# Mirrors dana.core.react_dispatch's own _OBJECT_PATH_REGISTRY: a plain
# module-level dict (not an instance attribute) since a fresh
# MockFreeCADEngine() is constructed on every dispatch in real usage — apply_
# boolean/modify_parameter now take object NAMES (matching RealFreeCADEngine's
# shared-session interface), and _mesh_output_path's random tempfile name
# means a name alone can't be resolved back to its .stl path without this.
_MOCK_OBJECT_REGISTRY: dict[str, str] = {}


def _bbox(mesh: Any) -> list[float]:
    lo, hi = mesh.bounds
    return [float(v) for v in (*lo, *hi)]


def _mesh_output_path(name: str) -> Path:
    fd, raw_path = tempfile.mkstemp(suffix=".stl", prefix=f"dana_mock_{name}_")
    os.close(fd)
    return Path(raw_path)


def _fan_triangulated_extrusion(points: list[list[float]], height: float) -> Any:
    """Extrude a closed 2D (XY) polygon ``height`` units along Z, headless.

    Fans both caps from the polygon's CENTROID rather than from vertex 0 —
    a star polygon's concave notches aren't visible from an outer spike
    vertex, so a vertex-0 fan would self-intersect there; every boundary
    point of a symmetric/convex/star-shaped polygon *is* visible from its
    centroid, so this works for a plain square footprint and an N-point
    star alike with no shapely/triangle dependency.
    """
    import numpy as np
    import trimesh

    pts = [(float(x), float(y)) for x, y in points]
    if pts[0] == pts[-1]:
        pts = pts[:-1]
    n = len(pts)
    cx = sum(p[0] for p in pts) / n
    cy = sum(p[1] for p in pts) / n

    # bottom block is [n perimeter verts, 1 center] = n+1 entries, so the
    # top block's perimeter verts start at index n+1, not n.
    bottom = np.array([[x, y, 0.0] for x, y in pts] + [[cx, cy, 0.0]])
    top = np.array([[x, y, float(height)] for x, y in pts] + [[cx, cy, float(height)]])
    vertices = np.vstack([bottom, top])
    bottom_center = n
    top_offset = n + 1
    top_center = top_offset + n

    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.append([bottom_center, j, i])
        faces.append([top_center, top_offset + i, top_offset + j])
        faces.append([i, j, top_offset + j])
        faces.append([i, top_offset + j, top_offset + i])
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=True)


def _star_polygon_vertices(points: int, outer_radius: float, inner_radius: float) -> list[list[float]]:
    import math

    n = points * 2
    vertices = []
    for i in range(n):
        angle = (math.pi / points) * i - (math.pi / 2)
        radius = outer_radius if i % 2 == 0 else inner_radius
        vertices.append([radius * math.cos(angle), radius * math.sin(angle)])
    return vertices


class MockControlPlane(BaseControlPlane):
    def resync_workspace(self) -> dict[str, Any]:
        moved = [
            {"hwnd": w["hwnd"], "title": w["title"], "moved": True}
            for w in _MOCK_WINDOWS
            if "freecad" in w["title"].lower()
        ]
        return {"ok": True, "moved": moved, "driver": "mock", "note": _MOCK_NOTE_CONTROL}

    def prevent_focus_steal(self) -> dict[str, Any]:
        return {
            "ok": True,
            "foreground": _MOCK_WINDOWS[0],
            "driver": "mock",
            "note": _MOCK_NOTE_CONTROL,
        }

    def get_active_display(self) -> dict[str, Any]:
        return {
            "ok": True,
            "primary": {"left": 0, "top": 0, "width": 1920, "height": 1080},
            "secondary": {"left": 1920, "top": 0, "width": 1920, "height": 1080},
            "driver": "mock",
            "note": _MOCK_NOTE_CONTROL,
        }


class MockFreeCADEngine(BaseCADEngine):
    """Headless stand-in for :class:`dana.platform.win32.RealFreeCADEngine`.

    Every ``path`` returned is a real ``.stl`` file on disk (so a
    ``gr.Model3D`` viewer can load it), backed by ``trimesh`` primitives
    instead of an actual ``.FCStd`` FreeCAD document — each object still gets
    its OWN ``.stl`` file (unlike the real engine's shared session document),
    but ``apply_boolean``/``modify_parameter`` still take object NAMES to
    match ``RealFreeCADEngine``'s interface, resolved via the module-level
    ``_MOCK_OBJECT_REGISTRY`` populated by ``create_box``/``create_cylinder``.
    """

    def create_box(
        self,
        length: float,
        width: float,
        height: float,
        name: str = "Box",
        placement: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> dict[str, Any]:
        import trimesh

        dims = {"length": float(length), "width": float(width), "height": float(height)}
        mesh = trimesh.creation.box(extents=[dims["length"], dims["width"], dims["height"]])
        mesh.apply_translation(-mesh.centroid)
        mesh.apply_translation(placement)
        out_path = _mesh_output_path(name)
        mesh.export(out_path)
        _MOCK_OBJECT_REGISTRY[name] = str(out_path)
        return {
            "ok": True,
            "name": name,
            "type": "Part::Box",
            "bounding_box": _bbox(mesh),
            "dimensions": dims,
            "placement": list(placement),
            "path": str(out_path),
            "gui_shown": False,
            "driver": "mock",
            "note": _MOCK_NOTE_CAD,
        }

    def create_cylinder(
        self,
        radius: float,
        height: float,
        name: str = "Cylinder",
        placement: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> dict[str, Any]:
        import trimesh

        dims = {"radius": float(radius), "height": float(height)}
        mesh = trimesh.creation.cylinder(radius=dims["radius"], height=dims["height"])
        mesh.apply_translation(-mesh.centroid)
        mesh.apply_translation(placement)
        out_path = _mesh_output_path(name)
        mesh.export(out_path)
        _MOCK_OBJECT_REGISTRY[name] = str(out_path)
        return {
            "ok": True,
            "name": name,
            "type": "Part::Cylinder",
            "bounding_box": _bbox(mesh),
            "dimensions": dims,
            "placement": list(placement),
            "path": str(out_path),
            "gui_shown": False,
            "driver": "mock",
            "note": _MOCK_NOTE_CAD,
        }

    def apply_boolean(
        self, operation: str, base_object: str, tool_object: str, name: str | None = None
    ) -> dict[str, Any]:
        import trimesh

        op = (operation or "").strip().lower()
        mesh_ops = {"cut": "difference", "union": "union", "intersect": "intersection"}
        feature_types = {"cut": "Part::Cut", "union": "Part::MultiFuse", "intersect": "Part::MultiCommon"}
        default_names = {"cut": "Cut", "union": "Fusion", "intersect": "Common"}
        if op not in mesh_ops:
            return {"ok": False, "error": f"apply_boolean: unknown operation '{operation}' — must be cut, union, or intersect"}

        base_path = _MOCK_OBJECT_REGISTRY.get(base_object)
        tool_path = _MOCK_OBJECT_REGISTRY.get(tool_object)
        if not base_path:
            return {"ok": False, "error": f"apply_boolean: no object named {base_object!r} in this session"}
        if not tool_path:
            return {"ok": False, "error": f"apply_boolean: no object named {tool_object!r} in this session"}
        base = Path(base_path)
        tool = Path(tool_path)
        if not base.is_file():
            return {"ok": False, "error": f"apply_boolean: base_path not found: {base_path}"}
        if not tool.is_file():
            return {"ok": False, "error": f"apply_boolean: tool_path not found: {tool_path}"}

        resolved_name = name or default_names[op]
        base_mesh = trimesh.load(base, force="mesh")
        tool_mesh = trimesh.load(tool, force="mesh")
        try:
            mesh = getattr(base_mesh, mesh_ops[op])(tool_mesh)
            engine_note = _MOCK_NOTE_CAD
        except BaseException:  # noqa: BLE001 — boolean engine unavailable in this container
            mesh = base_mesh
            engine_note = f"{_MOCK_NOTE_CAD}; boolean engine unavailable, returned base unmodified"

        out_path = _mesh_output_path(resolved_name)
        mesh.export(out_path)
        _MOCK_OBJECT_REGISTRY[resolved_name] = str(out_path)
        return {
            "ok": True,
            "name": resolved_name,
            "type": feature_types[op],
            "operation": op,
            "bounding_box": _bbox(mesh),
            "path": str(out_path),
            "gui_shown": False,
            "driver": "mock",
            "note": engine_note,
        }

    def apply_edge_operation(
        self,
        operation: str,
        target_path: str,
        value: float,
        face_centroid: tuple[float, float, float] | None = None,
        name: str | None = None,
        target_object: str | None = None,
    ) -> dict[str, Any]:
        # target_object is accepted for interface parity with the real
        # engine but unused here: every mock object already lives in its
        # OWN dedicated mesh file (see _MOCK_OBJECT_REGISTRY), so
        # target_path alone is never ambiguous the way a shared
        # Session_Active.FCStd path can be for the real FreeCAD driver.
        import trimesh

        op = (operation or "").strip().lower()
        feature_types = {"fillet": "Part::Fillet", "chamfer": "Part::Chamfer"}
        default_names = {"fillet": "Fillet", "chamfer": "Chamfer"}
        if op not in feature_types:
            return {"ok": False, "error": f"apply_edge_operation: unknown operation '{operation}' — must be fillet or chamfer"}

        target = Path(target_path)
        if not target.is_file():
            return {"ok": False, "error": f"apply_edge_operation: target_path not found: {target_path}"}

        resolved_name = name or default_names[op]
        face_targeted = face_centroid is not None
        # Safe stub: trimesh has no generic edge-rounding/beveling operation,
        # so this returns the target's own mesh unmodified under the new
        # name/type rather than attempting to simulate real fillet/chamfer
        # geometry — callers relying on the ok/path/name/type/bounding_box
        # contract (mesh export, the object registry, HITL summaries) still
        # get a consistent result end-to-end in this headless container.
        mesh = trimesh.load(target, force="mesh")
        out_path = _mesh_output_path(resolved_name)
        mesh.export(out_path)
        return {
            "ok": True,
            "name": resolved_name,
            "type": feature_types[op],
            "operation": op,
            "face_targeted": face_targeted,
            "bounding_box": _bbox(mesh),
            "path": str(out_path),
            "gui_shown": False,
            "driver": "mock",
            "note": f"{_MOCK_NOTE_CAD}; edge {op} not geometrically simulated, returned target unmodified",
        }

    def create_extrusion(
        self, profile_points: list[list[float]], height: float, name: str = "Extrusion"
    ) -> dict[str, Any]:
        if len(profile_points) < 3:
            return {"ok": False, "error": "create_extrusion requires at least 3 profile points"}

        mesh = _fan_triangulated_extrusion(profile_points, height)
        dims = {"height": float(height), "profile_points": len(profile_points)}
        out_path = _mesh_output_path(name)
        mesh.export(out_path)
        return {
            "ok": True,
            "name": name,
            "type": "Part::Feature",
            "bounding_box": _bbox(mesh),
            "dimensions": dims,
            "path": str(out_path),
            "gui_shown": False,
            "driver": "mock",
            "note": _MOCK_NOTE_CAD,
        }

    def create_pyramid(
        self,
        length: float,
        width: float,
        height: float,
        name: str = "Pyramid",
        placement: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> dict[str, Any]:
        import numpy as np
        import trimesh

        length_f, width_f, height_f = float(length), float(width), float(height)
        vertices = np.array(
            [
                [-length_f / 2, -width_f / 2, 0.0],
                [length_f / 2, -width_f / 2, 0.0],
                [length_f / 2, width_f / 2, 0.0],
                [-length_f / 2, width_f / 2, 0.0],
                [0.0, 0.0, height_f],
            ]
        )
        faces = [
            [0, 2, 1],
            [0, 3, 2],  # base, facing -Z
            [0, 1, 4],
            [1, 2, 4],
            [2, 3, 4],
            [3, 0, 4],  # four triangular sides
        ]
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
        mesh.apply_translation(placement)
        dims = {"length": length_f, "width": width_f, "height": height_f}
        out_path = _mesh_output_path(name)
        mesh.export(out_path)
        return {
            "ok": True,
            "name": name,
            "type": "Part::Feature",
            "bounding_box": _bbox(mesh),
            "dimensions": dims,
            "placement": list(placement),
            "path": str(out_path),
            "gui_shown": False,
            "driver": "mock",
            "note": _MOCK_NOTE_CAD,
        }

    def create_star_prism(
        self,
        points: int,
        outer_radius: float,
        inner_radius: float,
        height: float,
        name: str = "StarPrism",
        placement: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> dict[str, Any]:
        if int(points) < 3:
            return {"ok": False, "error": "create_star_prism requires at least 3 points"}

        vertices2d = _star_polygon_vertices(int(points), float(outer_radius), float(inner_radius))
        mesh = _fan_triangulated_extrusion(vertices2d, float(height))
        mesh.apply_translation(placement)
        dims = {
            "points": int(points),
            "outer_radius": float(outer_radius),
            "inner_radius": float(inner_radius),
            "height": float(height),
        }
        out_path = _mesh_output_path(name)
        mesh.export(out_path)
        return {
            "ok": True,
            "name": name,
            "type": "Part::Feature",
            "bounding_box": _bbox(mesh),
            "dimensions": dims,
            "placement": list(placement),
            "path": str(out_path),
            "gui_shown": False,
            "driver": "mock",
            "note": _MOCK_NOTE_CAD,
        }

    def export_mesh_stl(
        self, source_path: str, name: str | None = None, target_object: str | None = None
    ) -> dict[str, Any]:
        # target_object unused — see apply_edge_operation's matching note.
        import trimesh

        source = Path(source_path)
        if not source.is_file():
            return {"ok": False, "error": f"export_mesh_stl: source_path not found: {source_path}"}
        if source.suffix.lower() == ".stl":
            out_path = _mesh_output_path(name or source.stem)
            out_path.write_bytes(source.read_bytes())
        else:
            mesh = trimesh.load(source, force="mesh")
            out_path = _mesh_output_path(name or source.stem)
            mesh.export(out_path)
        return {
            "ok": True,
            "source_path": str(source),
            "path": str(out_path),
            "driver": "mock",
            "note": _MOCK_NOTE_CAD,
        }

    def modify_parameter(
        self, target_object: str, parameter_name: str, new_value: float | Sequence[float]
    ) -> dict[str, Any]:
        target_path = _MOCK_OBJECT_REGISTRY.get(target_object)
        if not target_path:
            return {"ok": False, "error": f"modify_parameter: no object named {target_object!r} in this session"}
        target = Path(target_path)
        if not target.is_file():
            return {"ok": False, "error": f"modify_parameter: target_path not found: {target_path}"}
        param = (parameter_name or "").strip()
        if not param:
            return {"ok": False, "error": "modify_parameter requires a non-empty parameter_name"}
        if param.lower() in ("placement", "placement.base"):
            try:
                components = [float(component) for component in new_value]
            except (TypeError, ValueError):
                return {
                    "ok": False,
                    "error": (
                        f"modify_parameter: {param} new_value must be a 3-number [x, y, z] or "
                        f"6-number [x, y, z, yaw, pitch, roll] vector, got {new_value!r}"
                    ),
                }
            if len(components) not in (3, 6):
                return {
                    "ok": False,
                    "error": (
                        f"modify_parameter: {param} new_value must have 3 elements [x, y, z] or "
                        f"6 elements [x, y, z, yaw, pitch, roll] (degrees), got {len(components)}"
                    ),
                }
            resolved_value: float | list[float] = components
        else:
            try:
                resolved_value = float(new_value)
            except (TypeError, ValueError):
                return {"ok": False, "error": f"modify_parameter: new_value must be a number, got {new_value!r}"}
        # Safe stub: a headless mesh has no named "Length"/"Height"/"Radius"
        # properties to setattr onto (there's no parametric object behind
        # it, just triangles), so this can't resize the mesh for real —
        # returns the target unmodified under the same name/path so callers
        # relying on the ok/path/name contract still get a consistent result.
        return {
            "ok": True,
            "name": target_object,
            "path": str(target),
            "parameter_name": param,
            "new_value": resolved_value,
            "driver": "mock",
            "note": f"{_MOCK_NOTE_CAD}; parameter not geometrically applied, returned target unmodified",
        }

    def get_bounding_box(self, target_path: str, target_object: str | None = None) -> dict[str, Any]:
        # target_object unused — see apply_edge_operation's matching note.
        import trimesh

        target = Path(target_path)
        if not target.is_file():
            return {"ok": False, "error": f"get_bounding_box: target_path not found: {target_path}"}
        mesh = trimesh.load(target, force="mesh")
        x_min, y_min, z_min, x_max, y_max, z_max = _bbox(mesh)
        return {
            "ok": True,
            "path": str(target),
            "x_min": x_min,
            "y_min": y_min,
            "z_min": z_min,
            "x_max": x_max,
            "y_max": y_max,
            "z_max": z_max,
            "driver": "mock",
            "note": _MOCK_NOTE_CAD,
        }

    def inspect_spatial_properties(self, target_path: str, target_object: str | None = None) -> dict[str, Any]:
        # target_object unused — see apply_edge_operation's matching note.
        import trimesh

        target = Path(target_path)
        if not target.is_file():
            return {"ok": False, "error": f"inspect_spatial_properties: target_path not found: {target_path}"}
        mesh = trimesh.load(target, force="mesh")
        watertight = bool(mesh.is_watertight)
        return {
            "ok": True,
            "path": str(target),
            "volume": float(mesh.volume) if watertight else 0.0,
            "area": float(mesh.area),
            "center_of_mass": [float(v) for v in mesh.centroid],
            "is_valid": watertight,
            "face_count": int(len(mesh.faces)),
            "edge_count": int(len(mesh.edges_unique)),
            "vertex_count": int(len(mesh.vertices)),
            "driver": "mock",
            "note": _MOCK_NOTE_CAD,
        }

    def create_pipe(
        self,
        pipe_radius: float,
        path_type: str,
        length_or_angle: float,
        name: str = "Pipe",
        placement: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> dict[str, Any]:
        pt = (path_type or "").strip().lower()
        if pt not in ("straight", "arc"):
            return {"ok": False, "error": f"create_pipe: unknown path_type '{path_type}' — must be straight or arc"}
        try:
            radius_f = float(pipe_radius)
            value_f = float(length_or_angle)
        except (TypeError, ValueError):
            return {"ok": False, "error": "create_pipe: pipe_radius and length_or_angle must be numbers"}
        if radius_f <= 0 or value_f <= 0:
            return {"ok": False, "error": "create_pipe: pipe_radius and length_or_angle must be positive numbers"}

        dims = {"pipe_radius": radius_f, "path_type": pt, "length_or_angle": value_f}
        if pt == "straight":
            # A straight pipe is geometrically just a cylinder — real,
            # correct mock geometry, not a stub.
            import trimesh

            mesh = trimesh.creation.cylinder(radius=radius_f, height=value_f, sections=32)
            mesh.apply_translation([0.0, 0.0, value_f / 2.0])  # base-at-origin, matching the real engine
            mesh.apply_translation(placement)
            out_path = _mesh_output_path(name)
            mesh.export(out_path)
            return {
                "ok": True,
                "name": name,
                "type": "Part::Sweep",
                "bounding_box": _bbox(mesh),
                "dimensions": dims,
                "placement": list(placement),
                "path": str(out_path),
                "gui_shown": False,
                "driver": "mock",
                "note": _MOCK_NOTE_CAD,
            }

        # Safe stub: a partial-torus elbow isn't one of trimesh's built-in
        # creation primitives, so the curved-arc case isn't simulated
        # geometrically here — returns a placeholder result under the
        # ok/path/name/type contract so callers (mesh export, the object
        # registry, HITL summaries) still work end-to-end.
        out_path = _mesh_output_path(name)
        import trimesh

        placeholder = trimesh.creation.cylinder(radius=radius_f, height=radius_f * 2, sections=32)
        placeholder.apply_translation(placement)
        placeholder.export(out_path)
        return {
            "ok": True,
            "name": name,
            "type": "Part::Sweep",
            "bounding_box": _bbox(placeholder),
            "dimensions": dims,
            "placement": list(placement),
            "path": str(out_path),
            "gui_shown": False,
            "driver": "mock",
            "note": f"{_MOCK_NOTE_CAD}; arc sweep not geometrically simulated, returned a placeholder",
        }

    def align_objects(
        self,
        source_path: str,
        target_path: str,
        alignment_type: str,
        source_object: str | None = None,
        target_object: str | None = None,
    ) -> dict[str, Any]:
        # source_object/target_object unused — see apply_edge_operation's matching note.
        import trimesh

        align = (alignment_type or "").strip().lower()
        valid_types = ("top_center", "bottom_center", "flush_left", "flush_right")
        if align not in valid_types:
            return {
                "ok": False,
                "error": f"align_objects: unknown alignment_type '{alignment_type}' — must be one of {', '.join(valid_types)}",
            }

        source = Path(source_path)
        target = Path(target_path)
        if not source.is_file():
            return {"ok": False, "error": f"align_objects: source_path not found: {source_path}"}
        if not target.is_file():
            return {"ok": False, "error": f"align_objects: target_path not found: {target_path}"}

        source_mesh = trimesh.load(source, force="mesh")
        target_mesh = trimesh.load(target, force="mesh")
        s_min, s_max = source_mesh.bounds
        t_min, t_max = target_mesh.bounds
        scx, scy, scz = (s_min + s_max) / 2.0
        tcx, tcy, tcz = (t_min + t_max) / 2.0

        if align == "top_center":
            delta = [tcx - scx, tcy - scy, t_max[2] - s_min[2]]
        elif align == "bottom_center":
            delta = [tcx - scx, tcy - scy, t_min[2] - s_max[2]]
        elif align == "flush_left":
            delta = [t_min[0] - s_min[0], tcy - scy, tcz - scz]
        else:  # flush_right
            delta = [t_max[0] - s_max[0], tcy - scy, tcz - scz]

        source_mesh.apply_translation(delta)
        source_mesh.export(source)
        return {
            "ok": True,
            "name": source.stem,
            "path": str(source),
            "alignment_type": align,
            # Best-effort in mock mode: the translation just applied to the
            # mesh, not a tracked absolute Placement.Base like the real
            # FreeCAD engine reports (there's no separate placement state
            # here beyond the mesh's own vertex positions).
            "placement": [float(v) for v in delta],
            "bounding_box": _bbox(source_mesh),
            "gui_shown": False,
            "driver": "mock",
            "note": _MOCK_NOTE_CAD,
        }

    def create_assembly_mate(
        self,
        fixed_path: str,
        moving_path: str,
        mate_type: str,
        mate_params: dict[str, Any] | None = None,
        fixed_object: str | None = None,
        moving_object: str | None = None,
    ) -> dict[str, Any]:
        # fixed_object/moving_object unused — see apply_edge_operation's matching note.
        import trimesh

        # Reuses the real engine's pure delta-math helper directly — same
        # justification as batch_pattern_array's reuse of _pattern_offsets:
        # plain arithmetic, no FreeCAD import at module scope, so it's exactly
        # as safe to call from this headless driver as duplicating it here.
        from dana.plugins.freecad.engine import _MATE_TYPES, _mate_delta

        fixed = Path(fixed_path)
        moving = Path(moving_path)
        if not fixed.is_file():
            return {"ok": False, "error": f"create_assembly_mate: fixed_path not found: {fixed_path}"}
        if not moving.is_file():
            return {"ok": False, "error": f"create_assembly_mate: moving_path not found: {moving_path}"}
        mt = (mate_type or "").strip().lower()
        if mt not in _MATE_TYPES:
            return {
                "ok": False,
                "error": f"create_assembly_mate: unknown mate_type '{mate_type}' — "
                f"must be one of {', '.join(sorted(_MATE_TYPES))}",
            }

        fixed_mesh = trimesh.load(fixed, force="mesh")
        moving_mesh = trimesh.load(moving, force="mesh")
        f_min, f_max = fixed_mesh.bounds
        m_min, m_max = moving_mesh.bounds
        fixed_bbox = {"x_min": f_min[0], "y_min": f_min[1], "z_min": f_min[2], "x_max": f_max[0], "y_max": f_max[1], "z_max": f_max[2]}
        moving_bbox = {"x_min": m_min[0], "y_min": m_min[1], "z_min": m_min[2], "x_max": m_max[0], "y_max": m_max[1], "z_max": m_max[2]}

        try:
            delta = _mate_delta(mt, dict(mate_params or {}), fixed_bbox, moving_bbox)
        except ValueError as exc:
            return {"ok": False, "error": f"create_assembly_mate: {exc}"}

        moving_mesh.apply_translation(delta)
        moving_mesh.export(moving)
        return {
            "ok": True,
            "name": moving.stem,
            "path": str(moving),
            "mate_type": mt,
            "fixed_object": str(fixed),
            "placement": [float(v) for v in delta],
            "bounding_box": _bbox(moving_mesh),
            "gui_shown": False,
            "driver": "mock",
            "note": _MOCK_NOTE_CAD,
        }

    def create_sketch_extrude(
        self,
        segments: list[dict[str, Any]],
        height: float,
        start: tuple[float, float] = (0.0, 0.0),
        plane: str = "XY",
        name: str = "Sketch",
        placement: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> dict[str, Any]:
        if not segments:
            return {"ok": False, "error": "create_sketch_extrude requires at least one segment"}

        # Safe stub: trimesh's fan-triangulated extrusion (reused from
        # _fan_triangulated_extrusion above) only handles straight-edged
        # polygons and only ever builds in the XY plane — a real rounded
        # arc and a non-XY work plane aren't geometrically simulated here,
        # matching the same honest-approximation philosophy the fillet/
        # chamfer and arc-pipe stubs above use.
        has_arc = any(str(seg.get("type", "line")).lower() == "arc" for seg in segments)
        points = [[float(start[0]), float(start[1])]]
        for seg in segments:
            to = seg["to"]
            points.append([float(to[0]), float(to[1])])

        mesh = _fan_triangulated_extrusion(points, float(height))
        mesh.apply_translation(placement)
        dims = {"height": float(height), "plane": str(plane).upper(), "segment_count": len(segments)}
        out_path = _mesh_output_path(name)
        mesh.export(out_path)

        caveats = []
        if has_arc:
            caveats.append("arc segments approximated as straight chords")
        if str(plane).upper() != "XY":
            caveats.append("non-XY planes aren't applied to mock geometry (profile always built in XY)")
        note = _MOCK_NOTE_CAD if not caveats else f"{_MOCK_NOTE_CAD}; " + "; ".join(caveats)

        return {
            "ok": True,
            "name": name,
            "type": "Part::Feature",
            "bounding_box": _bbox(mesh),
            "dimensions": dims,
            "placement": list(placement),
            "path": str(out_path),
            "gui_shown": False,
            "driver": "mock",
            "note": note,
        }

    def batch_pattern_array(
        self,
        source_path: str,
        pattern_type: str,
        *,
        count_x: int = 1,
        count_y: int = 1,
        spacing_x: float | None = None,
        spacing_y: float | None = None,
        count: int = 1,
        radius: float = 0.0,
        name: str = "Pattern",
    ) -> dict[str, Any]:
        import math

        import trimesh

        # Reuses the real engine's pure offset-math helper directly — it's
        # plain arithmetic with no FreeCAD import at module scope (see
        # dana.plugins.freecad.engine's docstring), so it's exactly as safe
        # to call from this headless driver as duplicating the formula here.
        from dana.plugins.freecad.engine import _PATTERN_TYPES, _pattern_offsets

        source = Path(source_path)
        if not source.is_file():
            return {"ok": False, "error": f"batch_pattern_array: source_path not found: {source_path}"}
        pt = (pattern_type or "").strip().lower()
        if pt not in _PATTERN_TYPES:
            return {
                "ok": False,
                "error": f"batch_pattern_array: unknown pattern_type '{pattern_type}' — must be linear, grid, or circular",
            }

        base_mesh = trimesh.load(source, force="mesh")
        sx = spacing_x if spacing_x is not None else float(base_mesh.extents[0])
        sy = spacing_y if spacing_y is not None else float(base_mesh.extents[1])
        offsets = _pattern_offsets(
            pt, count_x=count_x, count_y=count_y, spacing_x=sx, spacing_y=sy, count=count, radius=radius
        )

        copies = []
        for dx, dy, dz, rot in offsets:
            copy = base_mesh.copy()
            if rot:
                copy.apply_transform(trimesh.transformations.rotation_matrix(math.radians(rot), [0, 0, 1]))
            copy.apply_translation([dx, dy, dz])
            copies.append(copy)
        combined = trimesh.util.concatenate(copies) if len(copies) > 1 else copies[0]

        out_path = _mesh_output_path(name)
        combined.export(out_path)
        return {
            "ok": True,
            "name": name,
            "type": "Part::Compound",
            "bounding_box": _bbox(combined),
            "dimensions": {"pattern_type": pt, "copy_count": len(offsets)},
            "path": str(out_path),
            "gui_shown": False,
            "driver": "mock",
            "note": _MOCK_NOTE_CAD,
        }

    def export_model(
        self,
        target_paths: list[str],
        format: str,
        filename: str,
        target_objects: list[str] | None = None,
    ) -> dict[str, Any]:
        # target_objects unused — see apply_edge_operation's matching note.
        fmt = (format or "").strip().lower()
        if fmt not in ("stl", "step"):
            return {"ok": False, "error": f"export_model: unknown format '{format}' — must be stl or step"}
        paths = [Path(p) for p in (target_paths or [])]
        if not paths:
            return {"ok": False, "error": "export_model requires at least one target path"}
        missing = [str(p) for p in paths if not p.is_file()]
        if missing:
            return {"ok": False, "error": f"export_model: target path(s) not found: {missing}"}

        if fmt == "step":
            # Honest failure rather than a fake file: trimesh has no B-rep/
            # STEP writer (triangle-mesh formats only), so this genuinely
            # cannot be produced without the real FreeCAD engine's
            # Part.export — writing something mislabeled .step would be
            # worse than a clear error for a manufacturing/CAD-interchange
            # export a user might actually try to open elsewhere.
            return {
                "ok": False,
                "error": (
                    "export_model: STEP export isn't supported by the mock (trimesh) engine — "
                    "trimesh has no B-rep/STEP writer, only triangle-mesh formats. Needs the "
                    "real FreeCAD engine (Part.export)."
                ),
            }

        import trimesh

        meshes = [trimesh.load(p, force="mesh") for p in paths]
        combined = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
        out_path = _mesh_output_path(filename or "export")
        combined.export(out_path)
        return {
            "ok": True,
            "format": fmt,
            "path": str(out_path),
            "target_count": len(paths),
            "driver": "mock",
            "note": _MOCK_NOTE_CAD,
        }


__all__ = ("MockControlPlane", "MockFreeCADEngine")
