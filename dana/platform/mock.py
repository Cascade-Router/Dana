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
        self, length: float, width: float, height: float, name: str = "Box"
    ) -> dict[str, Any]:
        import trimesh

        dims = {"length": float(length), "width": float(width), "height": float(height)}
        mesh = trimesh.creation.box(extents=[dims["length"], dims["width"], dims["height"]])
        mesh.apply_translation(-mesh.centroid)
        out_path = _mesh_output_path(name)
        mesh.export(out_path)
        return {
            "ok": True,
            "name": name,
            "type": "Part::Box",
            "bounding_box": _bbox(mesh),
            "dimensions": dims,
            "path": str(out_path),
            "gui_shown": False,
            "driver": "mock",
            "note": _MOCK_NOTE_CAD,
        }

    def create_cylinder(self, radius: float, height: float, name: str = "Cylinder") -> dict[str, Any]:
        import trimesh

        dims = {"radius": float(radius), "height": float(height)}
        mesh = trimesh.creation.cylinder(radius=dims["radius"], height=dims["height"])
        mesh.apply_translation(-mesh.centroid)
        out_path = _mesh_output_path(name)
        mesh.export(out_path)
        return {
            "ok": True,
            "name": name,
            "type": "Part::Cylinder",
            "bounding_box": _bbox(mesh),
            "dimensions": dims,
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

        import numpy as np
        import trimesh

        pts = [(float(x), float(y)) for x, y in profile_points]
        if pts[0] == pts[-1]:
            pts = pts[:-1]
        n = len(pts)
        # Fan triangulation from vertex 0 — exact for convex/star-shaped
        # profiles (covers the default square footprint this mock exists
        # to unblock); no shapely/triangle dependency needed for that case.
        bottom = np.array([[x, y, 0.0] for x, y in pts])
        top = np.array([[x, y, float(height)] for x, y in pts])
        vertices = np.vstack([bottom, top])
        faces = []
        for i in range(1, n - 1):
            faces.append([0, i + 1, i])
            faces.append([n, n + i, n + i + 1])
        for i in range(n):
            j = (i + 1) % n
            faces.append([i, j, n + j])
            faces.append([i, n + j, n + i])
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)

        dims = {"height": float(height), "profile_points": n}
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
