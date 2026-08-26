"""Spatial Awareness — read a generated mesh's real geometric properties so
Dana can mathematically place URDF joint origins/mating surfaces instead of
guessing coordinates.

Pure-Python via ``trimesh`` (no FreeCADCmd subprocess, no CAD engine
abstraction needed) — the same library ``dana.platform.mock``'s
``MockFreeCADEngine`` already uses for every mesh operation in the headless
cloud container, so this works identically there and on a real desktop
install; ``.stl`` (and anything else trimesh understands) loads the same
way in both.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def query_geometry_properties(mesh_path: str) -> dict[str, Any]:
    """Read ``mesh_path``'s real bounding box, dimensions, and centroid.

    Returns ``{"ok": True, "bounding_box": [...], "dimensions": {...},
    "centroid": [...]}`` on success, or ``{"ok": False, "error": "..."}``
    for a missing file or a mesh trimesh can't parse — never raises.
    """
    path = Path(mesh_path)
    if not path.is_file():
        return {"ok": False, "error": f"query_geometry_properties: mesh_path not found: {mesh_path}"}

    try:
        import trimesh

        # force="mesh" — same guard dana.platform.mock uses everywhere it
        # calls trimesh.load: a multi-object file (e.g. an STL/OBJ holding
        # several disjoint shells) would otherwise come back as a
        # trimesh.Scene, which has no .bounds/.centroid of its own.
        mesh = trimesh.load(path, force="mesh")
        x_min, y_min, z_min = (float(v) for v in mesh.bounds[0])
        x_max, y_max, z_max = (float(v) for v in mesh.bounds[1])
        centroid = [float(v) for v in mesh.centroid]
    except Exception as exc:  # noqa: BLE001 — any trimesh/parse failure degrades to a clean error, never a crash
        return {"ok": False, "error": f"query_geometry_properties: failed to read {mesh_path}: {exc}"}

    return {
        "ok": True,
        "bounding_box": [x_min, y_min, z_min, x_max, y_max, z_max],
        "dimensions": {
            "length": x_max - x_min,
            "width": y_max - y_min,
            "height": z_max - z_min,
        },
        "centroid": centroid,
    }
