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
from pathlib import Path
from typing import Any

from dana.platform.base import BaseCADEngine, BaseControlPlane

_MOCK_NOTE_CONTROL = "mocked — no Windows/Win32 APIs in this container"
_MOCK_NOTE_CAD = "mocked — headless trimesh geometry, no FreeCADCmd binary in this container"

_MOCK_WINDOWS: list[dict[str, Any]] = [
    {"hwnd": 1001, "title": "FreeCAD 1.0 — DanaModel.FCStd", "pid": 4021},
    {"hwnd": 1002, "title": "Dana — Live Trace", "pid": 3110},
]


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
    instead of an actual ``.FCStd`` FreeCAD document.
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

    def apply_boolean_cut(self, base_path: str, tool_path: str, name: str = "Cut") -> dict[str, Any]:
        import trimesh

        base = Path(base_path)
        tool = Path(tool_path)
        if not base.is_file():
            return {"ok": False, "error": f"apply_boolean_cut: base_path not found: {base_path}"}
        if not tool.is_file():
            return {"ok": False, "error": f"apply_boolean_cut: tool_path not found: {tool_path}"}

        base_mesh = trimesh.load(base, force="mesh")
        tool_mesh = trimesh.load(tool, force="mesh")
        try:
            mesh = base_mesh.difference(tool_mesh)
            engine_note = _MOCK_NOTE_CAD
        except BaseException:  # noqa: BLE001 — boolean engine unavailable in this container
            mesh = base_mesh
            engine_note = f"{_MOCK_NOTE_CAD}; boolean engine unavailable, returned base unmodified"

        out_path = _mesh_output_path(name)
        mesh.export(out_path)
        return {
            "ok": True,
            "name": name,
            "type": "Part::Cut",
            "bounding_box": _bbox(mesh),
            "path": str(out_path),
            "gui_shown": False,
            "driver": "mock",
            "note": engine_note,
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

    def export_mesh_stl(self, source_path: str, name: str | None = None) -> dict[str, Any]:
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


__all__ = ("MockControlPlane", "MockFreeCADEngine")
