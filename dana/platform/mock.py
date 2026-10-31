"""Simulated telemetry + headless ``trimesh`` geometry — cloud/sandboxed execution.

Used whenever there's no real Win32 desktop or FreeCAD binary to talk to
(Hugging Face Spaces, CI, any non-Windows/non-FreeCAD host). Every response
carries ``"driver": "mock"`` and a human-readable ``"note"`` so a caller —
or a UI rendering the result — can never mistake simulated output for a
real actuation, mirroring the labeling convention already used in
``hf_space/hf_sandbox``.
"""

from __future__ import annotations

import copy
import json
import math
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from dana.platform.base import BaseCADEngine, BaseControlPlane
from dana.session_context import get_session_id

_MOCK_NOTE_CONTROL = "mocked — no Windows/Win32 APIs in this container"
_MOCK_NOTE_CAD = "mocked — headless trimesh geometry, no FreeCADCmd binary in this container"

_MOCK_WINDOWS: list[dict[str, Any]] = [
    {"hwnd": 1001, "title": "FreeCAD 1.0 — DanaModel.FCStd", "pid": 4021},
    {"hwnd": 1002, "title": "Dana — Live Trace", "pid": 3110},
]

# Mirrors dana.core.react_dispatch's own _OBJECT_PATH_REGISTRY (see that
# module's matching comment for the full reasoning): a plain module-level
# dict (not an instance attribute) since a fresh MockFreeCADEngine() is
# constructed on every dispatch in real usage — apply_boolean/
# modify_parameter now take object NAMES (matching RealFreeCADEngine's
# shared-session interface), and _mesh_output_path's random tempfile name
# means a name alone can't be resolved back to its .stl path without this.
# Nested per session_id for the exact same reason react_dispatch.py's
# registry is: two chat sessions each naming an object "Box" would
# otherwise clobber each other's entry in one shared global dict. Always
# go through _mock_object_registry() below, never this dict directly.
_MOCK_OBJECT_REGISTRY: dict[str, dict[str, str]] = {}

# create_sketch's mock stores each sketch's RAW geometry list here (not just
# a bare mesh path) — create_pad/create_pocket need the actual line/circle/
# arc data to fake a real extrusion/cut via _mock_profile_from_geometry
# below; _MOCK_OBJECT_REGISTRY alone (name -> mesh file path) has already
# lost that by the time a mesh is on disk. Nested per session_id, same
# reasoning as _MOCK_OBJECT_REGISTRY above. Always go through
# _mock_sketch_registry() below, never this dict directly.
_MOCK_SKETCH_GEOMETRY: dict[str, dict[str, list[dict[str, Any]]]] = {}

# The active PartDesign::Body's current "tip" mesh path, one shared body per
# session — mirrors the real engine's own "find/create ONE body, reuse it
# across create_pad/create_pocket calls" containment logic (see
# dana.plugins.freecad.engine._PARTDESIGN_BODY_SNIPPET), just tracked as a
# plain path here since a headless mock has no real Body/Tip object at all.
# Always go through _mock_body_tip_registry() below, never this dict directly.
_MOCK_BODY_TIP: dict[str, dict[str, str]] = {}

# create_assembly's own member list per assembly name — a headless mock has
# no real App::Part.Group to move objects into (every mock object is its
# own standalone mesh file, never a shared multi-object document), so
# add_parts_to_assembly instead tracks membership here and rebuilds the
# assembly's own placeholder mesh as the concatenation of every current
# member's mesh each time it's called — an accumulating, idempotent list
# (adding the same part twice is a no-op), same "real geometry, cheaply
# faked" philosophy as batch_pattern_array's own mock. Always go through
# _mock_assembly_registry() below, never this dict directly.
_MOCK_ASSEMBLY_MEMBERS: dict[str, dict[str, list[str]]] = {}

# define_kinematic_joint's own state, keyed the same way
# dana.plugins.freecad.engine's real DanaKinematicJoints custom property
# is: assembly_name -> {child_link_name: joint_def}. A headless mock has
# no real App::Part custom property to stash this on, so it's tracked here
# instead — purely bookkeeping (export_assembly_to_urdf itself is NOT
# supported by this driver, see its own docstring below, so nothing ever
# reads this back into an actual URDF under mock), same "recorded but not
# geometrically applied" honesty as apply_assembly_constraint's own mock.
# Always go through _mock_kinematic_joints_registry() below, never this
# dict directly.
_MOCK_KINEMATIC_JOINTS: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}

# apply_boolean's lineage, per session: result name -> {"operands": [...],
# "applied": bool}. generate_assembly_bom reads it to leave consumed
# operands out (the real engine's consumed-operand filter, which reads the
# same thing off Part::Cut's Base/Tool links) and to refuse a result whose
# boolean never ran ("applied": False). apply_boolean now fails outright
# instead of recording such a result, so that check is only a backstop.
_MOCK_BOOLEAN_RESULTS: dict[str, dict[str, dict[str, Any]]] = {}

# The placement each object's mesh is currently baked at, per session:
# name -> [x, y, z, yaw, pitch, roll]. Mock meshes are stored already
# transformed, so position_assembly_part needs the old placement to move a
# mesh to a new one (absolute, like the real Placement). Objects without an
# entry are at the origin.
_MOCK_PLACEMENTS: dict[str, dict[str, list[float]]] = {}


def _mock_sketch_registry() -> dict[str, list[dict[str, Any]]]:
    """THIS session's own slice of ``_MOCK_SKETCH_GEOMETRY``."""
    return _MOCK_SKETCH_GEOMETRY.setdefault(get_session_id(), {})


def _mock_body_tip_registry() -> dict[str, str]:
    """THIS session's own slice of ``_MOCK_BODY_TIP``."""
    return _MOCK_BODY_TIP.setdefault(get_session_id(), {})


def _mock_assembly_registry() -> dict[str, list[str]]:
    """THIS session's own slice of ``_MOCK_ASSEMBLY_MEMBERS``."""
    return _MOCK_ASSEMBLY_MEMBERS.setdefault(get_session_id(), {})


def _mock_boolean_registry() -> dict[str, dict[str, Any]]:
    """THIS session's own slice of ``_MOCK_BOOLEAN_RESULTS``."""
    return _MOCK_BOOLEAN_RESULTS.setdefault(get_session_id(), {})


def _mock_placement_registry() -> dict[str, list[float]]:
    """THIS session's own slice of ``_MOCK_PLACEMENTS``."""
    return _MOCK_PLACEMENTS.setdefault(get_session_id(), {})


def _mock_kinematic_joints_registry() -> dict[str, dict[str, dict[str, Any]]]:
    """THIS session's own slice of ``_MOCK_KINEMATIC_JOINTS``."""
    return _MOCK_KINEMATIC_JOINTS.setdefault(get_session_id(), {})


def _mock_profile_from_geometry(geometry: list[dict[str, Any]]) -> tuple[list[list[float]] | None, str]:
    """Best-effort 2D profile reconstruction from a ``create_sketch``
    geometry list, for ``create_pad``/``create_pocket``'s own mock extrude/
    cut — honest, non-rigorous approximations only, same "safe stub, clearly
    noted" philosophy as ``create_sketch_extrude``'s own mock above:

    - A chain of ONLY line segments becomes its own polygon outline
      (assumes each entry's "end" leads into the next entry's "start", the
      same closed-profile assumption ``create_sketch_extrude``'s mock makes
      for its own ``segments`` argument).
    - A SINGLE circle becomes a many-sided regular-polygon approximation.
    - Anything else (arcs, multiple circles, a line/circle mix) falls back
      to the geometry's own 2D bounding box as a rectangle — the same "flat
      placeholder" honesty ``create_sketch``'s own mock already uses,
      surfaced to the caller as a non-empty caveat string.

    Returns ``(points, caveat)`` — ``points`` is ``None`` only when
    ``geometry`` yields no usable coordinates at all (e.g. an empty list).
    """
    import math

    kinds = [str(g.get("type", "")).strip().lower() for g in geometry]
    if kinds and all(k == "line" for k in kinds):
        points = [[float(geometry[0]["start"][0]), float(geometry[0]["start"][1])]]
        for g in geometry:
            end = g["end"]
            points.append([float(end[0]), float(end[1])])
        return points, ""
    if len(geometry) == 1 and kinds[0] == "circle":
        cx, cy = geometry[0]["center"]
        radius = float(geometry[0]["radius"])
        sides = 32
        points = [
            [
                float(cx) + radius * math.cos(2 * math.pi * i / sides),
                float(cy) + radius * math.sin(2 * math.pi * i / sides),
            ]
            for i in range(sides)
        ]
        return points, ""

    xs: list[float] = []
    ys: list[float] = []
    for g in geometry:
        kind = str(g.get("type", "")).strip().lower()
        if kind == "line":
            xs += [float(g["start"][0]), float(g["end"][0])]
            ys += [float(g["start"][1]), float(g["end"][1])]
        elif kind in ("circle", "arc"):
            cx, cy = g["center"]
            radius = float(g["radius"])
            xs += [float(cx) - radius, float(cx) + radius]
            ys += [float(cy) - radius, float(cy) + radius]
    if not xs:
        return None, "empty geometry"
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    return (
        [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
        "profile approximated as its own bounding-box rectangle (mixed/arc geometry not faithfully reconstructed in mock mode)",
    )


# Mirrors dana.plugins.freecad.ir's _FACE_NORMAL/_FACE_U_AXIS/_FACE_V_AXIS
# tables (the Universal CAD IR's create_feature_on_face composite resolver)
# for create_feature_on_face's mock — same values, duplicated rather than
# shared, matching this file's existing convention of keeping its own small
# operation-name lookup tables independent of engine.py's (see
# apply_boolean's mesh_ops/feature_types/default_names below, each a mock-
# local re-derivation of engine.py's _BOOLEAN_FEATURE_TYPE/_DEFAULT_BOOLEAN_NAME).
_MOCK_FACE_NORMAL: dict[str, tuple[float, float, float]] = {
    "top": (0.0, 0.0, 1.0),
    "bottom": (0.0, 0.0, -1.0),
    "front": (0.0, -1.0, 0.0),
    "back": (0.0, 1.0, 0.0),
    "right": (1.0, 0.0, 0.0),
    "left": (-1.0, 0.0, 0.0),
}
_MOCK_FACE_U_AXIS: dict[str, tuple[float, float, float]] = {
    "top": (1.0, 0.0, 0.0),
    "bottom": (1.0, 0.0, 0.0),
    "front": (1.0, 0.0, 0.0),
    "back": (-1.0, 0.0, 0.0),
    "right": (0.0, 1.0, 0.0),
    "left": (0.0, -1.0, 0.0),
}
_MOCK_FACE_V_AXIS: dict[str, tuple[float, float, float]] = {
    "top": (0.0, 1.0, 0.0),
    "bottom": (0.0, -1.0, 0.0),
    "front": (0.0, 0.0, 1.0),
    "back": (0.0, 0.0, 1.0),
    "right": (0.0, 0.0, 1.0),
    "left": (0.0, 0.0, 1.0),
}
# Which bbox axis (0=x, 1=y, 2=z) and extreme (True=max, False=min) is this
# face's own coordinate — the other two axes always use the bbox center.
_MOCK_FACE_EXTREME_AXIS: dict[str, tuple[int, bool]] = {
    "top": (2, True),
    "bottom": (2, False),
    "front": (1, False),
    "back": (1, True),
    "right": (0, True),
    "left": (0, False),
}
_MOCK_FACE_FEATURE_CLEARANCE = 0.5


def _mock_object_registry() -> dict[str, str]:
    """THIS session's own slice of ``_MOCK_OBJECT_REGISTRY``."""
    return _MOCK_OBJECT_REGISTRY.setdefault(get_session_id(), {})


def _bbox(mesh: Any) -> list[float]:
    lo, hi = mesh.bounds
    return [float(v) for v in (*lo, *hi)]


def _mesh_output_path(name: str, *, ext: str = "glb") -> Path:
    """Every create_*/apply_*/... call below routes its own live-preview
    mesh through here. Defaults to ``.glb`` (GLTF Binary) rather than
    ``.stl`` — the live viewer/WebSocket bandwidth format switch — since
    every one of those callers just wants a small, servable preview mesh
    and trimesh's ``mesh.export()`` dispatches purely off this path's own
    extension with no other code change needed at any call site.

    ``export_model`` is the ONE exception: its explicit user-facing
    "download as STL" request passes ``ext="stl"`` to override this
    default back to a genuine ``.stl`` — that's a real 3D-printing/CAD
    interchange file a caller asked for by name, not a preview, and must
    never silently become a ``.glb`` instead.

    Written under the session's own ``freecad_output/sessions/<id>/mock/``
    (not the system temp dir), so deleting the session removes them and a
    restarted server still finds a resumed session's meshes. Each call gets
    a fresh file: the rebuild's rollback restores registry entries that
    point at the previous ones. The ``mock/`` subfolder keeps these
    intermediate meshes out of the session's artifact listing, which only
    scans the top level.
    """
    from dana.plugins.freecad import engine

    out_dir = engine._session_dir() / "mock"
    out_dir.mkdir(exist_ok=True)
    fd, raw_path = tempfile.mkstemp(suffix=f".{ext}", prefix=f"dana_mock_{engine._safe_name(name)}_", dir=out_dir)
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


def _regular_polygon_vertices(sides: int, radius: float) -> list[list[float]]:
    """Mirrors dana.plugins.freecad.engine's own helper of the same name —
    duplicated, not imported, since that module assumes a real FreeCADCmd
    binary is reachable and this one deliberately doesn't."""
    import math

    return [
        [
            radius * math.cos((2 * math.pi / sides) * i - math.pi / 2),
            radius * math.sin((2 * math.pi / sides) * i - math.pi / 2),
        ]
        for i in range(sides)
    ]


# --- mesh analysis shared by query_topology / check_printability / collisions ---

# Adjacent triangles closer than this (dihedral angle) belong to the same
# face. A trimesh cylinder's 32 sections are 11.25° apart, so its side is one
# curved face; a box's 90° edges keep its six faces apart. A prism with 24 or
# more sides reads as curved, which is also how a slicer would see it.
_MOCK_SMOOTH_ANGLE_RAD = math.radians(15.0)
# All of a planar face's triangle normals agree to within ~0.5°.
_MOCK_PLANAR_DOT = 0.99996
# Collision overlap below this (mm³) is float32 noise on coincident faces in
# manifold3d's boolean, not an interpenetration.
_MOCK_COLLISION_VOLUME_EPSILON = 1e-3
_MOCK_CURVED_FACE_WARNING = (
    "face is not planar ({surface_type}) -- a single normal vector is not well-defined across a "
    "curved surface; use a Concentric-style axis/center reference (apply_assembly_constraint's "
    "'Concentric') instead of this face's normal"
)
_MOCK_TOPOLOGY_NOTE = (
    "mock topology: faces are the mesh's smooth regions (triangles meeting at under 15°), so face "
    "and edge numbering is stable for this mesh but not FreeCAD's own Face/Edge numbering"
)


def _union_find_labels(count: int, pairs: Any) -> list[int]:
    """Connected-component label per node: a pure-Python union-find (scipy and
    networkx aren't base requirements, so trimesh.graph isn't usable here)."""
    parent = list(range(count))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in pairs:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return [find(i) for i in range(count)]


def _axis_label(vector: Any) -> str | None:
    """"+Z", "-X", ... when ``vector`` is a world axis (within ~0.5°)."""
    for i, axis in enumerate("XYZ"):
        if abs(float(vector[i])) >= _MOCK_PLANAR_DOT:
            return ("+" if vector[i] > 0 else "-") + axis
    return None


def _mesh_topology(mesh: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Faces and edges of a triangle mesh, in query_topology's face schema.

    Faces are the connected smooth regions; a region whose triangle normals
    all agree is planar (``Part::GeomPlane``, with the real tool's normal and
    u/v span fields), one whose normals are all perpendicular to a common
    axis is reported as ``Part::GeomCylinder``, anything else as
    ``Part::GeomBSplineSurface``. Edges are the creases between two faces.
    Ordering is canonical: planar faces by normal axis (-X, +X, -Y, +Y, -Z,
    +Z, as on a FreeCAD Part::Box) then position, then curved faces by area.
    """
    import numpy as np

    count = len(mesh.faces)
    adjacency = np.asarray(mesh.face_adjacency)
    angles = np.asarray(mesh.face_adjacency_angles)
    labels = _union_find_labels(count, adjacency[angles < _MOCK_SMOOTH_ANGLE_RAD]) if count else []
    groups: dict[int, list[int]] = {}
    for tri, label in enumerate(labels):
        groups.setdefault(label, []).append(tri)

    normals = np.asarray(mesh.face_normals)
    areas = np.asarray(mesh.area_faces)
    centers = np.asarray(mesh.triangles_center)
    regions = []
    for label, tris in groups.items():
        idx = np.asarray(tris)
        weights = areas[idx]
        area = float(weights.sum())
        if area <= 0.0:
            continue
        centroid = (centers[idx] * weights[:, None]).sum(axis=0) / area
        mean_normal = (normals[idx] * weights[:, None]).sum(axis=0)
        norm = float(np.linalg.norm(mean_normal))
        unit = mean_normal / norm if norm > 0 else mean_normal
        planar = norm > 0 and bool((normals[idx] @ unit >= _MOCK_PLANAR_DOT).all())
        region: dict[str, Any] = {"label": label, "area": area, "centroid": centroid, "is_planar": planar}
        if planar:
            # u = the first world axis not parallel to the normal, projected
            # into the plane; v = normal x u. Spans are the face's vertex
            # extent along each.
            ref = next(np.eye(3)[i] for i in range(3) if abs(unit[i]) < 0.9)
            u_dir = ref - (ref @ unit) * unit
            u_dir /= np.linalg.norm(u_dir)
            v_dir = np.cross(unit, u_dir)
            points = mesh.vertices[np.unique(mesh.faces[idx])]
            u_proj, v_proj = points @ u_dir, points @ v_dir
            region.update(
                surface_type="Part::GeomPlane",
                normal=unit,
                direction=_axis_label(unit),
                u_span=float(u_proj.max() - u_proj.min()),
                v_span=float(v_proj.max() - v_proj.min()),
                u_dir=u_dir,
                v_dir=v_dir,
            )
        else:
            # Normals all perpendicular to one axis -> the normal matrix has
            # a (near-)zero singular value along that axis.
            singular = np.linalg.svd(normals[idx], compute_uv=False)
            cylindrical = len(singular) == 3 and singular[2] <= 0.02 * singular[0]
            region["surface_type"] = "Part::GeomCylinder" if cylindrical else "Part::GeomBSplineSurface"
        regions.append(region)

    def order(region: dict[str, Any]) -> tuple:
        c = [round(float(v), 6) for v in region["centroid"]]
        if region["is_planar"]:
            axis = int(np.argmax(np.abs(region["normal"])))
            return (0, axis, 0 if region["normal"][axis] < 0 else 1, c[axis], c)
        return (1, -round(region["area"], 6), 0, 0.0, c)

    regions.sort(key=order)
    face_names = {r["label"]: f"Face{i + 1}" for i, r in enumerate(regions)}

    faces = []
    for region in regions:
        planar = region["is_planar"]
        faces.append(
            {
                "face_index": face_names[region["label"]],
                "area": region["area"],
                "is_planar": planar,
                "surface_type": region["surface_type"],
                "normal": [float(v) for v in region["normal"]] if planar else None,
                "normal_warning": None
                if planar
                else _MOCK_CURVED_FACE_WARNING.format(surface_type=region["surface_type"]),
                "centroid": [float(v) for v in region["centroid"]],
                "direction": region.get("direction"),
                "u_span_physical_mm": region["u_span"] if planar else None,
                "v_span_physical_mm": region["v_span"] if planar else None,
                "u_direction_vector": [float(v) for v in region["u_dir"]] if planar else None,
                "v_direction_vector": [float(v) for v in region["v_dir"]] if planar else None,
            }
        )

    crease_segments: dict[tuple[str, str], list[Any]] = {}
    edge_vertices = np.asarray(mesh.face_adjacency_edges)
    for (fa, fb), verts in zip(adjacency, edge_vertices):
        a, b = face_names.get(labels[fa]), face_names.get(labels[fb])
        if a is None or b is None or a == b:
            continue
        key = tuple(sorted((a, b), key=lambda n: int(n[4:])))
        crease_segments.setdefault(key, []).append(verts)
    edges = []
    for key in sorted(crease_segments, key=lambda k: (int(k[0][4:]), int(k[1][4:]))):
        segments = np.asarray(crease_segments[key])
        points = mesh.vertices[np.unique(segments)]
        length = float(np.linalg.norm(mesh.vertices[segments[:, 0]] - mesh.vertices[segments[:, 1]], axis=1).sum())
        center = points.mean(axis=0)
        spread = np.linalg.svd(points - center, compute_uv=False) if len(points) > 1 else np.zeros(1)
        straight = len(points) == 2 or (len(spread) > 1 and spread[1] <= 1e-6 * max(spread[0], 1e-12))
        edge: dict[str, Any] = {
            "edge_index": f"Edge{len(edges) + 1}",
            "faces": list(key),
            "kind": "line" if straight else "curve",
            "length": length,
            "center": [float(v) for v in center],
        }
        if straight:
            direction = points[-1] - points[0]
            direction /= np.linalg.norm(direction) or 1.0
            proj = (points - center) @ direction
            edge["start"] = [float(v) for v in center + direction * proj.min()]
            edge["end"] = [float(v) for v in center + direction * proj.max()]
        edges.append(edge)
    return faces, edges


def _mesh_edge_use(mesh: Any) -> Any:
    """How many triangles use each unique edge (2 everywhere on a closed
    manifold; 1 on an open edge; 3+ on a non-manifold one)."""
    import numpy as np

    return np.bincount(np.asarray(mesh.edges_unique_inverse), minlength=len(mesh.edges_unique))


def _to_manifold(mesh: Any) -> Any:
    """``mesh`` as a manifold3d solid; ``ValueError`` if it isn't one (open
    or non-manifold), the mock's equivalent of OCCT refusing a bad shape."""
    import manifold3d
    import numpy as np

    solid = manifold3d.Manifold(
        manifold3d.Mesh(
            vert_properties=np.asarray(mesh.vertices, dtype=np.float32),
            tri_verts=np.asarray(mesh.faces, dtype=np.uint32),
        )
    )
    status = solid.status()
    if status != manifold3d.Error.NoError:
        raise ValueError(f"not a closed manifold solid ({status.name})")
    return solid


def _sweep_circle(
    path: Sequence[Sequence[float]], tangents: Sequence[Sequence[float]], radius: float, *, sections: int = 24
) -> Any:
    """A solid tube: a circle of ``radius`` swept along the polyline ``path``
    (``tangents``: the path's exact direction at each point, so the end caps
    sit square to it), capped at both ends — what Part::Sweep with Solid=True
    builds from a Part::Circle profile. Rotation-minimizing frames (double
    reflection), so the cross-section never twists along a helix."""
    import numpy as np
    import trimesh

    points = np.asarray(path, dtype=float)
    n = len(points)
    tangents = np.asarray(tangents, dtype=float)
    tangents /= np.linalg.norm(tangents, axis=1)[:, None]
    ref = np.array([0.0, 0.0, 1.0]) if abs(tangents[0][2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    normal = ref - (ref @ tangents[0]) * tangents[0]
    frames = [normal / np.linalg.norm(normal)]
    for i in range(n - 1):
        step = points[i + 1] - points[i]
        c1 = step @ step
        r_l = frames[i] - (2.0 / c1) * (step @ frames[i]) * step
        t_l = tangents[i] - (2.0 / c1) * (step @ tangents[i]) * step
        v2 = tangents[i + 1] - t_l
        c2 = v2 @ v2
        frames.append(r_l - (2.0 / c2) * (v2 @ r_l) * v2 if c2 > 1e-18 else r_l)
    normals = np.asarray(frames)
    binormals = np.cross(tangents, normals)

    theta = np.linspace(0.0, 2.0 * math.pi, sections, endpoint=False)
    ring = np.cos(theta)[None, :, None] * normals[:, None, :] + np.sin(theta)[None, :, None] * binormals[:, None, :]
    vertices = (points[:, None, :] + radius * ring).reshape(-1, 3)
    start, end = len(vertices), len(vertices) + 1
    vertices = np.vstack([vertices, points[0], points[-1]])

    faces = []
    for i in range(n - 1):
        for j in range(sections):
            a, b = i * sections + j, i * sections + (j + 1) % sections
            c, d = a + sections, b + sections
            faces += [[a, b, d], [a, d, c]]
    last = (n - 1) * sections
    for j in range(sections):
        k = (j + 1) % sections
        faces.append([start, k, j])
        faces.append([end, last + j, last + k])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    if mesh.volume < 0:
        mesh.invert()
    return mesh


def _placement_matrix(x: float, y: float, z: float, yaw: float, pitch: float, roll: float) -> Any:
    """4x4 for FreeCAD's ``App.Placement(App.Vector(x, y, z),
    App.Rotation(yaw, pitch, roll))``: Rz(yaw) · Ry(pitch) · Rx(roll), then
    the translation."""
    import trimesh

    tf = trimesh.transformations
    matrix = (
        tf.rotation_matrix(math.radians(yaw), [0, 0, 1])
        @ tf.rotation_matrix(math.radians(pitch), [0, 1, 0])
        @ tf.rotation_matrix(math.radians(roll), [1, 0, 0])
    )
    matrix[:3, 3] = [x, y, z]
    return matrix


def _rebuild_assembly_mesh(assembly: str) -> tuple[Any, Path]:
    """Re-renders an assembly's preview mesh as the concatenation of its
    CURRENT members' meshes (idempotent: a part added twice is listed once)."""
    import trimesh

    meshes = []
    for member_name in _mock_assembly_registry().get(assembly, []):
        member_path = _mock_object_registry().get(member_name)
        if member_path and Path(member_path).is_file():
            meshes.append(trimesh.load(member_path, force="mesh"))
    combined = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
    out_path = _mesh_output_path(assembly)
    combined.export(out_path)
    _mock_object_registry()[assembly] = str(out_path)
    return combined, out_path


def _resolve_mock_name(name: str | None) -> str | None:
    """The registered object ``name`` refers to: exact match, else a unique
    case-insensitive match (the real engine also falls back to Label and
    case-insensitive Name)."""
    wanted = (name or "").strip()
    registry = _mock_object_registry()
    if wanted in registry:
        return wanted
    matches = [n for n in registry if n.lower() == wanted.lower()]
    return matches[0] if len(matches) == 1 else None


def _register_mock_object(name: str, path: Path, placement: Sequence[float] = (0.0, 0.0, 0.0), yaw: float = 0.0) -> None:
    """Records a freshly created object and the placement its mesh was baked
    at, so position_assembly_part can move it later."""
    _mock_object_registry()[name] = str(path)
    _mock_placement_registry()[name] = [*(float(v) for v in placement), float(yaw), 0.0, 0.0]


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


def _mock_pattern_sources(op: str, feature_name: Any) -> tuple[list[str], list[Path]] | dict[str, Any]:
    """(feature names, their mesh paths) for a pattern call, or an error
    payload. Same name/list normalization as the real engine."""
    from dana.plugins.freecad.engine import pattern_feature_names

    names = pattern_feature_names(feature_name)
    if isinstance(names, str):
        return {"ok": False, "error": f"{op}: {names}"}
    sources: list[Path] = []
    for name in names:
        source_path = _mock_object_registry().get(name)
        if not source_path:
            return {
                "ok": False,
                "error": (
                    f"{op}: no feature named {name!r} in this session — "
                    "create it first with create_freecad_pad or create_freecad_pocket"
                ),
            }
        source = Path(source_path)
        if not source.is_file():
            return {"ok": False, "error": f"{op}: source path not found: {source_path}"}
        sources.append(source)
    return names, sources


class MockFreeCADEngine(BaseCADEngine):
    """Headless stand-in for :class:`dana.platform.win32.RealFreeCADEngine`.

    Every ``path`` returned is a real ``.glb`` mesh on disk (so a
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
        # Part::Box's placement is its minimum corner, not its center: a
        # model reasoning in FreeCAD terms (a live run did) otherwise cuts
        # a half-depth "through" hole on the mock.
        mesh.apply_translation(-mesh.bounds[0])
        mesh.apply_translation(placement)
        out_path = _mesh_output_path(name)
        mesh.export(out_path)
        _register_mock_object(name, out_path, placement)
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
        # Part::Cylinder's placement is the center of its base circle, axis +Z.
        mesh.apply_translation([0.0, 0.0, dims["height"] / 2.0])
        mesh.apply_translation(placement)
        out_path = _mesh_output_path(name)
        mesh.export(out_path)
        _register_mock_object(name, out_path, placement)
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
        self,
        operation: str,
        base_object: str | None = None,
        tool_object: str | None = None,
        name: str | None = None,
        objects: list[str] | None = None,
    ) -> dict[str, Any]:
        import trimesh

        op = (operation or "").strip().lower()
        mesh_ops = {"cut": "difference", "union": "union", "intersect": "intersection"}
        feature_types = {"cut": "Part::Cut", "union": "Part::MultiFuse", "intersect": "Part::MultiCommon"}
        default_names = {"cut": "Cut", "union": "Fusion", "intersect": "Common"}
        if op not in mesh_ops:
            return {"ok": False, "error": f"apply_boolean: unknown operation '{operation}' — must be cut, union, or intersect"}

        clean_objects = [str(o).strip() for o in objects if str(o).strip()] if objects else []
        if clean_objects:
            if op == "cut":
                return {
                    "ok": False,
                    "error": "apply_boolean: 'objects' is not valid for 'cut' — cut always takes base_object/tool_object",
                }
            if len(clean_objects) < 2:
                return {"ok": False, "error": "apply_boolean: 'objects' must name at least 2 objects to fuse/intersect"}
            object_names = clean_objects
        elif base_object and tool_object:
            object_names = [base_object, tool_object]
        else:
            return {"ok": False, "error": "apply_boolean: give either base_object+tool_object, or objects (2+ names)"}

        paths: list[Path] = []
        for obj_name in object_names:
            obj_path = _mock_object_registry().get(obj_name)
            if not obj_path:
                return {"ok": False, "error": f"apply_boolean: no object named {obj_name!r} in this session"}
            path = Path(obj_path)
            if not path.is_file():
                return {"ok": False, "error": f"apply_boolean: path not found for {obj_name!r}: {obj_path}"}
            paths.append(path)

        resolved_name = name or default_names[op]
        meshes = [trimesh.load(p, force="mesh") for p in paths]
        try:
            # Sequential pairwise reduction — N-ary fuse/intersect isn't a
            # single trimesh call, but folding mesh_ops[op] across every
            # entry (base_object/tool_object's own 2-element case included)
            # produces the same result a real Part::MultiFuse/MultiCommon's
            # own N-ary Shapes list would.
            mesh = meshes[0]
            for other in meshes[1:]:
                mesh = getattr(mesh, mesh_ops[op])(other)
        except Exception as exc:  # noqa: BLE001 — trimesh raises various types when no boolean backend is installed
            # This used to return ok=True with the base mesh unmodified, so a
            # "successful" cut silently left the part uncut. trimesh's
            # booleans need the manifold3d backend (requirements.txt).
            return {
                "ok": False,
                "error": (
                    f"apply_boolean: the mock CAD driver could not compute the {op} "
                    f"({type(exc).__name__}: {exc}). Mesh booleans need the manifold3d package; "
                    "install it with `pip install manifold3d`."
                ),
            }

        out_path = _mesh_output_path(resolved_name)
        mesh.export(out_path)
        _mock_object_registry()[resolved_name] = str(out_path)
        _mock_boolean_registry()[resolved_name] = {"operands": object_names, "applied": True}
        return {
            "ok": True,
            "name": resolved_name,
            "type": feature_types[op],
            "operation": op,
            "bounding_box": _bbox(mesh),
            "path": str(out_path),
            "gui_shown": False,
            "driver": "mock",
            "note": _MOCK_NOTE_CAD,
        }

    def apply_edge_operation(
        self,
        operation: str,
        target_object: str,
        value: float,
        face_centroid: tuple[float, float, float] | None = None,
        name: str | None = None,
    ) -> dict[str, Any]:
        import trimesh

        op = (operation or "").strip().lower()
        feature_types = {"fillet": "Part::Fillet", "chamfer": "Part::Chamfer"}
        default_names = {"fillet": "Fillet", "chamfer": "Chamfer"}
        if op not in feature_types:
            return {"ok": False, "error": f"apply_edge_operation: unknown operation '{operation}' — must be fillet or chamfer"}

        target_path = _mock_object_registry().get(target_object)
        if not target_path:
            return {"ok": False, "error": f"apply_edge_operation: no object named {target_object!r} in this session"}
        target = Path(target_path)
        if not target.is_file():
            return {"ok": False, "error": f"apply_edge_operation: target_path not found: {target_path}"}

        resolved_name = name or default_names[op]
        face_targeted = face_centroid is not None
        # The geometry isn't simulated, but feasibility is: OCC cannot round
        # or bevel an edge by half or more of the shortest edge it meets
        # (two fillets would overlap), and fails such a call. The solid's
        # smallest extent stands in for that edge -- exact for the boxes and
        # plates these ops mostly target, conservative otherwise.
        mesh = trimesh.load(target, force="mesh")
        smallest = float(min(mesh.extents))
        if not value or float(value) <= 0:
            return {"ok": False, "error": f"apply_edge_operation: {op} size must be positive, got {value!r}"}
        if float(value) >= smallest / 2.0:
            return {
                "ok": False,
                "error": (
                    f"BRep_API: command not done -- a {float(value):g} mm {op} cannot fit on "
                    f"{target_object!r}, whose shortest edge is {smallest:g} mm (needs < {smallest / 2.0:g} mm)"
                ),
            }
        # Safe stub: trimesh has no generic edge-rounding/beveling operation,
        # so this returns the target's own mesh unmodified under the new
        # name/type rather than attempting to simulate real fillet/chamfer
        # geometry — callers relying on the ok/path/name/type/bounding_box
        # contract (mesh export, the object registry, HITL summaries) still
        # get a consistent result end-to-end in this headless container.
        out_path = _mesh_output_path(resolved_name)
        mesh.export(out_path)
        _mock_object_registry()[resolved_name] = str(out_path)
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
        _register_mock_object(name, out_path)
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
        _register_mock_object(name, out_path, placement)
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
        _register_mock_object(name, out_path, placement)
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

    def create_polygon(
        self,
        sides: int,
        radius: float,
        height: float,
        name: str = "Polygon",
        placement: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> dict[str, Any]:
        if int(sides) < 3:
            return {"ok": False, "error": "create_polygon requires at least 3 sides"}

        vertices2d = _regular_polygon_vertices(int(sides), float(radius))
        mesh = _fan_triangulated_extrusion(vertices2d, float(height))
        mesh.apply_translation(placement)
        dims = {"sides": int(sides), "radius": float(radius), "height": float(height)}
        out_path = _mesh_output_path(name)
        mesh.export(out_path)
        _register_mock_object(name, out_path, placement)
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
        # Name kept (matches RealFreeCADEngine's interface/BaseCADEngine) even
        # though this now always writes .glb, not .stl — see _mesh_output_path's
        # own docstring for why the live-preview format switched to GLTF
        # Binary (WebSocket/hosting bandwidth) without renaming the method.
        import trimesh

        source = Path(source_path)
        if not source.is_file():
            return {"ok": False, "error": f"export_mesh_stl: source_path not found: {source_path}"}
        if source.suffix.lower() == ".glb":
            # Fast path: every mock create_*/apply_* call already writes
            # .glb via _mesh_output_path's own default, so this is the
            # common case — a byte-identical copy under the new name,
            # never a lossy trimesh round-trip.
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
        self,
        target_object: str,
        parameter_name: str,
        new_value: float | Sequence[float],
        yaw: float | None = None,
        pitch: float | None = None,
        roll: float | None = None,
    ) -> dict[str, Any]:
        target_path = _mock_object_registry().get(target_object)
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
                        f"modify_parameter: {param} new_value must be a 3-number [x, y, z] "
                        f"vector, got {new_value!r}"
                    ),
                }
            if len(components) != 3:
                return {
                    "ok": False,
                    "error": (
                        f"modify_parameter: {param} new_value must have exactly 3 elements [x, y, z] "
                        f"(mm) — pass rotation via the separate yaw/pitch/roll parameters instead of "
                        f"packing it into this vector, got {len(components)}"
                    ),
                }
            if yaw is None and pitch is None and roll is None:
                resolved_value: float | list[float] = components
            else:
                resolved_value = components + [float(yaw or 0.0), float(pitch or 0.0), float(roll or 0.0)]
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

    def query_topology(self, part_name: str) -> dict[str, Any]:
        """The real tool's per-face schema, measured on the object's mesh
        (see ``_mesh_topology``), plus ``edges`` and whole-solid properties
        (``bounding_box``, ``volume``, ``area``, ``center_of_mass``,
        ``is_watertight``)."""
        import trimesh

        resolved = _resolve_mock_name(part_name)
        if resolved is None:
            return {"ok": False, "error": f"query_topology: no object named {part_name!r} in this session"}
        path = Path(_mock_object_registry()[resolved])
        if not path.is_file():
            return {"ok": False, "error": f"query_topology: mesh for {resolved!r} not found: {path}"}
        mesh = trimesh.load(path, force="mesh")
        faces, edges = _mesh_topology(mesh)
        x_min, y_min, z_min, x_max, y_max, z_max = _bbox(mesh)
        watertight = bool(mesh.is_watertight)
        # "part_name", never "name": see the real engine's query_topology —
        # dispatch would otherwise register this read as a new object.
        return {
            "ok": True,
            "part_name": resolved,
            "path": str(path),
            "face_count": len(faces),
            "faces": faces,
            "edge_count": len(edges),
            "edges": edges,
            "bounding_box": {
                "x_min": x_min, "y_min": y_min, "z_min": z_min,
                "x_max": x_max, "y_max": y_max, "z_max": z_max,
                "xlen": x_max - x_min, "ylen": y_max - y_min, "zlen": z_max - z_min,
            },
            "volume": abs(float(mesh.volume)) if watertight else None,
            "area": float(mesh.area),
            "center_of_mass": [float(v) for v in (mesh.center_mass if watertight else mesh.centroid)],
            "is_watertight": watertight,
            "driver": "mock",
            "note": f"{_MOCK_NOTE_CAD}; {_MOCK_TOPOLOGY_NOTE}",
        }

    def _default_print_target(self) -> str | None:
        """The real engine's default: the session's PartDesign Body if there
        is one, else the last top-level solid — here the newest registered
        object that isn't a sketch, an assembly container or a consumed
        boolean operand."""
        if _mock_body_tip_registry().get("tip"):
            return "Body"
        skip = set(_mock_sketch_registry()) | set(_mock_assembly_registry())
        skip |= {op for entry in _mock_boolean_registry().values() for op in entry["operands"]}
        candidates = [n for n in _mock_object_registry() if n not in skip]
        return candidates[-1] if candidates else None

    def check_printability(
        self,
        object_name: str | None = None,
        build_volume_mm: Sequence[float] | None = None,
        max_overhang_deg: float = 45.0,
    ) -> dict[str, Any]:
        """The real report (``printability.build_report``, same schema and
        orientation analysis) on the object's mesh: trimesh supplies the
        mesh checks FreeCAD's Mesh module does on the real engine, except
        self-intersection, which isn't checked here (reported ``None``).
        The STL lands where the real one does, ready for slice_stl_to_gcode."""
        import numpy as np
        import trimesh

        from dana.plugins.freecad import engine, printability
        from dana.plugins.os import file_system
        from dana.session_context import session_scoped_dir

        if not 0.0 < float(max_overhang_deg) < 90.0:
            return {"ok": False, "error": "check_printability: max_overhang_deg must be between 0 and 90"}
        try:
            printer = printability.printer_profile(build_volume_mm)
        except ValueError as exc:
            return {"ok": False, "error": f"check_printability: {exc}"}

        if (object_name or "").strip():
            target = _resolve_mock_name(object_name)
            if target is None:
                return {"ok": False, "error": f"check_printability: no object named {object_name!r} in this session"}
            path = _mock_object_registry()[target]
        else:
            target = self._default_print_target()
            if target is None:
                return {"ok": False, "error": "check_printability: nothing built in this session yet — build a part first"}
            path = _mock_body_tip_registry()["tip"] if target == "Body" else _mock_object_registry()[target]
        if not Path(path).is_file():
            return {"ok": False, "error": f"check_printability: mesh for {target!r} not found: {path}"}
        mesh = trimesh.load(path, force="mesh")

        edge_use = _mesh_edge_use(mesh)
        watertight = bool(mesh.is_watertight)
        checks = {
            "shape_valid": len(mesh.faces) > 0,
            "closed_solid": watertight and abs(float(mesh.volume)) > 0.0,
            "mesh_solid": watertight,
            "non_manifold": bool((edge_use > 2).any()),
            "self_intersections": None,
            "open_edges": int((edge_use == 1).sum()),
            "components": len(set(_union_find_labels(len(mesh.faces), mesh.face_adjacency))),
            # Mesh-only checks (FreeCAD's shape check covers these on the real engine).
            "winding_consistent": bool(mesh.is_winding_consistent),
            "positive_volume": watertight and float(mesh.volume) > 0.0,
            "degenerate_faces": int((~mesh.nondegenerate_faces()).sum()),
        }
        triangles = np.asarray(mesh.triangles)
        lo, hi = triangles.min(axis=1), triangles.max(axis=1)
        solid = [
            printability.SolidFacet(tuple(map(float, n)), float(a), tuple(map(float, l)), tuple(map(float, h)))
            for n, a, l, h in zip(mesh.face_normals, mesh.area_faces, lo, hi)
        ]
        report = printability.build_report(
            target=target,
            checks=checks,
            bbox=_bbox(mesh),
            facets=(printability.Facet(f.normal[2], f.area, f.lo[2], f.hi[2]) for f in solid),
            printer=printer,
            max_overhang_deg=float(max_overhang_deg),
            solid_facets=solid,
        )
        stl_path = session_scoped_dir(file_system._SANDBOX_ROOT / "prints") / f"{engine._safe_name(target)}.stl"
        mesh.export(stl_path)
        return {
            "ok": True,
            **report,
            "stl_path": str(stl_path),
            "driver": "mock",
            "note": f"{_MOCK_NOTE_CAD}; mesh checks by trimesh (winding, volume sign, degenerate "
            "faces, edges); self-intersection not checked",
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
        import trimesh

        if pt == "straight":
            # A straight sweep of a circle is exactly a cylinder.
            mesh = trimesh.creation.cylinder(radius=radius_f, height=value_f, sections=32)
            mesh.apply_translation([0.0, 0.0, value_f / 2.0])  # base-at-origin, matching the real engine
        else:
            # The real engine's arc: the profile circle sits at (bend, 0, 0)
            # in the XZ plane and sweeps value_f degrees counter-clockwise
            # around Z, bend radius from ir.pipe_arc_radius.
            from dana.plugins.freecad.ir import pipe_arc_radius

            bend = pipe_arc_radius(radius_f)
            angle = math.radians(min(value_f, 360.0))
            steps = max(8, math.ceil(math.degrees(angle) / 5.0))
            params = [angle * i / steps for i in range(steps + 1)]
            path = [[bend * math.cos(t), bend * math.sin(t), 0.0] for t in params]
            mesh = _sweep_circle(path, [[-math.sin(t), math.cos(t), 0.0] for t in params], radius_f, sections=32)
            dims["arc_radius"] = bend
        mesh.apply_translation(placement)
        out_path = _mesh_output_path(name)
        mesh.export(out_path)
        _register_mock_object(name, out_path, placement)
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
        try:
            coil_radius_f = float(coil_radius)
            pitch_f = float(pitch)
            height_f = float(height)
            pipe_radius_f = float(pipe_radius)
            angle_offset_f = float(angle_offset)
        except (TypeError, ValueError):
            return {"ok": False, "error": "create_helix: coil_radius, pitch, height, and pipe_radius must be numbers"}
        if coil_radius_f <= 0 or pitch_f <= 0 or height_f <= 0 or pipe_radius_f <= 0:
            return {"ok": False, "error": "create_helix: coil_radius, pitch, height, and pipe_radius must all be positive numbers"}
        if pipe_radius_f >= coil_radius_f:
            return {
                "ok": False,
                "error": "create_helix: pipe_radius must be smaller than coil_radius (the tube can't be wider than the coil itself)",
            }

        dims = {
            "coil_radius": coil_radius_f,
            "pitch": pitch_f,
            "height": height_f,
            "pipe_radius": pipe_radius_f,
            "turns": height_f / pitch_f,
            "angle_offset": angle_offset_f,
        }
        # Part::Helix's path (right-handed, starting at (coil_radius, 0, 0))
        # with the pipe's circle swept along it; angle_offset is a rigid
        # rotation about Z, as in the real engine.
        import trimesh

        steps = max(16, math.ceil(dims["turns"] * 48))
        omega = 2.0 * math.pi / pitch_f
        heights = [height_f * i / steps for i in range(steps + 1)]
        path = [[coil_radius_f * math.cos(omega * z), coil_radius_f * math.sin(omega * z), z] for z in heights]
        tangents = [
            [-coil_radius_f * omega * math.sin(omega * z), coil_radius_f * omega * math.cos(omega * z), 1.0]
            for z in heights
        ]
        mesh = _sweep_circle(path, tangents, pipe_radius_f)
        if angle_offset_f != 0.0:
            mesh.apply_transform(trimesh.transformations.rotation_matrix(math.radians(angle_offset_f), [0, 0, 1]))
        mesh.apply_translation(placement)
        out_path = _mesh_output_path(name)
        mesh.export(out_path)
        _register_mock_object(name, out_path, placement, yaw=angle_offset_f)
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

    def create_sketch(
        self,
        name: str,
        plane: str,
        geometry: list[dict[str, Any]],
    ) -> dict[str, Any]:
        import trimesh

        plane_u = (plane or "XY").strip().upper()
        if plane_u not in ("XY", "XZ", "YZ"):
            return {"ok": False, "error": f"create_sketch: unknown plane '{plane}' — must be XY, XZ, or YZ"}
        if not geometry:
            return {"ok": False, "error": "create_sketch requires a non-empty geometry list"}

        xs: list[float] = []
        ys: list[float] = []
        geometry_summary: list[dict[str, Any]] = []
        for i, item in enumerate(geometry):
            kind = str(item.get("type", "")).strip().lower()
            try:
                if kind == "line":
                    start, end = item["start"], item["end"]
                    xs += [float(start[0]), float(end[0])]
                    ys += [float(start[1]), float(end[1])]
                elif kind in ("circle", "arc"):
                    cx, cy = item["center"]
                    r = float(item["radius"])
                    xs += [float(cx) - r, float(cx) + r]
                    ys += [float(cy) - r, float(cy) + r]
                else:
                    return {
                        "ok": False,
                        "error": f"create_sketch: geometry[{i}] has unknown type {item.get('type')!r}",
                    }
            except (KeyError, TypeError, ValueError) as exc:
                return {"ok": False, "error": f"create_sketch: malformed geometry[{i}] — {exc}"}
            geometry_summary.append({"index": i, "type": kind})

        # Safe stub: a headless mesh has no real Sketcher solver behind it —
        # this just visualizes the raw geometry's own 2D bounding box as a
        # paper-thin slab so a 3D viewer has SOMETHING to show, same honest-
        # approximation philosophy as create_sketch_extrude's own mock above.
        width = max(max(xs) - min(xs), 1e-6)
        depth = max(max(ys) - min(ys), 1e-6)
        mesh = trimesh.creation.box(extents=[width, depth, 1e-3])
        mesh.apply_translation([(max(xs) + min(xs)) / 2.0, (max(ys) + min(ys)) / 2.0, 0.0])
        out_path = _mesh_output_path(name)
        mesh.export(out_path)
        _mock_object_registry()[name] = str(out_path)
        # create_pad/create_pocket need the RAW geometry (not just this bare
        # placeholder mesh) to fake a real extrude/cut later — see
        # _mock_profile_from_geometry's own docstring.
        _mock_sketch_registry()[name] = list(geometry)
        return {
            "ok": True,
            "name": name,
            "type": "Sketcher::SketchObject",
            "dimensions": {"plane": plane_u, "geometry": geometry_summary},
            "bounding_box": _bbox(mesh),
            "path": str(out_path),
            "gui_shown": False,
            "driver": "mock",
            "note": f"{_MOCK_NOTE_CAD}; sketch geometry not constrained/solved, shown as a flat placeholder slab",
        }

    def apply_sketch_constraint(
        self,
        sketch_name: str,
        constraint_type: str,
        geometry_indices: list[int],
        value: float | None = None,
    ) -> dict[str, Any]:
        valid_types = {"Coincident", "Horizontal", "Vertical", "Distance", "Radius"}
        if constraint_type not in valid_types:
            return {
                "ok": False,
                "error": (
                    f"apply_sketch_constraint: unknown constraint_type '{constraint_type}' — "
                    f"must be one of {sorted(valid_types)}"
                ),
            }
        target_path = _mock_object_registry().get(sketch_name)
        if not target_path:
            return {"ok": False, "error": f"apply_sketch_constraint: no object named {sketch_name!r} in this session"}
        target = Path(target_path)
        if not target.is_file():
            return {"ok": False, "error": f"apply_sketch_constraint: target_path not found: {target_path}"}
        expected_counts = {"Coincident": 4, "Horizontal": 1, "Vertical": 1, "Distance": 1, "Radius": 1}
        expected = expected_counts[constraint_type]
        if len(geometry_indices) != expected:
            return {
                "ok": False,
                "error": (
                    f"apply_sketch_constraint: '{constraint_type}' requires exactly "
                    f"{expected} geometry_indices, got {len(geometry_indices)}"
                ),
            }
        if constraint_type in ("Distance", "Radius") and value is None:
            return {"ok": False, "error": f"apply_sketch_constraint: '{constraint_type}' requires a numeric value"}
        # Safe stub: a headless mesh has no real Sketcher constraint solver
        # behind it — same "recorded but not geometrically applied" contract
        # as modify_parameter's own mock above.
        return {
            "ok": True,
            "name": sketch_name,
            "type": "Sketcher::SketchObject",
            "dimensions": {
                "constraint_type": constraint_type,
                "geometry_indices": [int(i) for i in geometry_indices],
                "value": float(value) if value is not None else None,
            },
            "path": str(target),
            "driver": "mock",
            "note": f"{_MOCK_NOTE_CAD}; constraint not geometrically solved, sketch mesh unmodified",
        }

    def create_pad(
        self,
        sketch_name: str,
        length: float,
        symmetric_to_plane: bool = False,
        reversed_direction: bool = False,
    ) -> dict[str, Any]:
        try:
            length_f = float(length)
        except (TypeError, ValueError):
            return {"ok": False, "error": "create_pad: length must be a number"}
        if length_f <= 0:
            return {"ok": False, "error": "create_pad: length must be a positive number"}
        geometry = _mock_sketch_registry().get(sketch_name)
        if geometry is None:
            return {"ok": False, "error": f"create_pad: no sketch named {sketch_name!r} in this session"}
        points, caveat = _mock_profile_from_geometry(geometry)
        if points is None:
            return {"ok": False, "error": f"create_pad: sketch {sketch_name!r} has no usable geometry"}

        mesh = _fan_triangulated_extrusion(points, length_f)
        if symmetric_to_plane:
            mesh.apply_translation([0.0, 0.0, -length_f / 2.0])
        elif reversed_direction:
            mesh.apply_translation([0.0, 0.0, -length_f])
        resolved_name = "Pad"
        out_path = _mesh_output_path(resolved_name)
        mesh.export(out_path)
        _mock_object_registry()[resolved_name] = str(out_path)
        # One shared active-body tip per session, mirroring the real
        # engine's own find/create/reuse Body logic — see create_pocket
        # below, which cuts into whatever this most recently pointed at.
        _mock_body_tip_registry()["tip"] = str(out_path)
        note = _MOCK_NOTE_CAD if not caveat else f"{_MOCK_NOTE_CAD}; {caveat}"
        return {
            "ok": True,
            "name": resolved_name,
            "type": "PartDesign::Pad",
            "dimensions": {
                "length": length_f,
                "symmetric_to_plane": bool(symmetric_to_plane),
                "reversed_direction": bool(reversed_direction),
            },
            "bounding_box": _bbox(mesh),
            "path": str(out_path),
            "gui_shown": False,
            "driver": "mock",
            "note": note,
        }

    def create_pocket(
        self,
        sketch_name: str,
        depth: float,
        through_all: bool = False,
        symmetric_to_plane: bool = False,
        reversed_direction: bool = False,
    ) -> dict[str, Any]:
        import trimesh

        try:
            depth_f = float(depth)
        except (TypeError, ValueError):
            return {"ok": False, "error": "create_pocket: depth must be a number"}
        if depth_f <= 0:
            return {"ok": False, "error": "create_pocket: depth must be a positive number"}
        geometry = _mock_sketch_registry().get(sketch_name)
        if geometry is None:
            return {"ok": False, "error": f"create_pocket: no sketch named {sketch_name!r} in this session"}
        tip_path = _mock_body_tip_registry().get("tip")
        if not tip_path or not Path(tip_path).is_file():
            return {
                "ok": False,
                "error": "create_pocket: no active body solid to cut into — create one with create_freecad_pad first",
            }
        points, caveat = _mock_profile_from_geometry(geometry)
        if points is None:
            return {"ok": False, "error": f"create_pocket: sketch {sketch_name!r} has no usable geometry"}

        base_mesh = trimesh.load(tip_path, force="mesh")
        lo, hi = base_mesh.bounds
        cut_depth = float(depth_f) if not through_all else float(hi[2] - lo[2]) + 10.0
        tool_mesh = _fan_triangulated_extrusion(points, cut_depth)
        # Anchor the cutting tool against the body's own Z span rather than
        # the sketch's local Z=0 plane — same "figure out where it should
        # sit relative to the target" honesty create_feature_on_face's own
        # mock already applies (there against a resolved face; here against
        # the active body's own bounds).
        if symmetric_to_plane:
            tool_mesh.apply_translation([0.0, 0.0, -cut_depth / 2.0])
        elif reversed_direction:
            tool_mesh.apply_translation([0.0, 0.0, -cut_depth])
        try:
            result_mesh = base_mesh.difference(tool_mesh)
            engine_note = _MOCK_NOTE_CAD
        except BaseException:  # noqa: BLE001 — boolean engine unavailable in this container
            result_mesh = base_mesh
            engine_note = f"{_MOCK_NOTE_CAD}; boolean engine unavailable, returned base unmodified"
        if caveat:
            engine_note = f"{engine_note}; {caveat}"
        resolved_name = "Pocket"
        out_path = _mesh_output_path(resolved_name)
        result_mesh.export(out_path)
        _mock_object_registry()[resolved_name] = str(out_path)
        _mock_body_tip_registry()["tip"] = str(out_path)
        return {
            "ok": True,
            "name": resolved_name,
            "type": "PartDesign::Pocket",
            "dimensions": {
                "depth": depth_f,
                "through_all": bool(through_all),
                "symmetric_to_plane": bool(symmetric_to_plane),
                "reversed_direction": bool(reversed_direction),
            },
            "bounding_box": _bbox(result_mesh),
            "path": str(out_path),
            "gui_shown": False,
            "driver": "mock",
            "note": engine_note,
        }

    def create_polar_pattern(
        self,
        feature_name: str | list[str],
        occurrences: int,
        angle: float = 360.0,
        axis: str = "Z",
        reversed_direction: bool = False,
    ) -> dict[str, Any]:
        import math

        import trimesh

        axis_vectors = {"X": [1.0, 0.0, 0.0], "Y": [0.0, 1.0, 0.0], "Z": [0.0, 0.0, 1.0]}
        axis_u = (axis or "Z").strip().upper()
        if axis_u not in axis_vectors:
            return {"ok": False, "error": f"create_polar_pattern: unknown axis '{axis}' — must be X, Y, or Z"}
        try:
            occurrences_i = int(occurrences)
        except (TypeError, ValueError):
            return {"ok": False, "error": "create_polar_pattern: occurrences must be an integer"}
        if occurrences_i < 2:
            return {"ok": False, "error": "create_polar_pattern: occurrences must be at least 2"}
        try:
            angle_f = float(angle)
        except (TypeError, ValueError):
            return {"ok": False, "error": "create_polar_pattern: angle must be a number"}
        if angle_f <= 0:
            return {"ok": False, "error": "create_polar_pattern: angle must be a positive number"}

        features = _mock_pattern_sources("create_polar_pattern", feature_name)
        if isinstance(features, dict):
            return features
        names, sources = features
        meshes = [trimesh.load(source, force="mesh") for source in sources]
        base_mesh = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
        # Safe stub: a headless mesh has no real PartDesign::Body/Origin
        # behind it — this rotates about the GLOBAL origin along the chosen
        # axis (a body's own principal axes always pass through its own
        # placement origin, which for this shared session is the global
        # origin), evenly spaced at angle/occurrences per step, same
        # "honest, non-rigorous approximation" philosophy as
        # batch_pattern_array's own mock above.
        step = angle_f / occurrences_i
        if reversed_direction:
            step = -step
        axis_vec = axis_vectors[axis_u]
        copies = []
        for i in range(occurrences_i):
            copy = base_mesh.copy()
            if i:
                copy.apply_transform(trimesh.transformations.rotation_matrix(math.radians(step * i), axis_vec))
            copies.append(copy)
        combined = trimesh.util.concatenate(copies) if len(copies) > 1 else copies[0]

        resolved_name = "PolarPattern"
        out_path = _mesh_output_path(resolved_name)
        combined.export(out_path)
        _mock_object_registry()[resolved_name] = str(out_path)
        _mock_body_tip_registry()["tip"] = str(out_path)
        return {
            "ok": True,
            "name": resolved_name,
            "type": "PartDesign::PolarPattern",
            "dimensions": {
                "occurrences": occurrences_i,
                "angle": angle_f,
                "axis": axis_u,
                "reversed_direction": bool(reversed_direction),
                "features": names,
            },
            "bounding_box": _bbox(combined),
            "path": str(out_path),
            "gui_shown": False,
            "driver": "mock",
            "note": f"{_MOCK_NOTE_CAD}; array rotated about the global origin, not a real Body placement offset",
        }

    def create_linear_pattern(
        self,
        feature_name: str | list[str],
        occurrences: int,
        length: float,
        direction: str = "X",
        reversed_direction: bool = False,
    ) -> dict[str, Any]:
        import trimesh

        direction_vectors = {"X": [1.0, 0.0, 0.0], "Y": [0.0, 1.0, 0.0], "Z": [0.0, 0.0, 1.0]}
        direction_u = (direction or "X").strip().upper()
        if direction_u not in direction_vectors:
            return {
                "ok": False,
                "error": f"create_linear_pattern: unknown direction '{direction}' — must be X, Y, or Z",
            }
        try:
            occurrences_i = int(occurrences)
        except (TypeError, ValueError):
            return {"ok": False, "error": "create_linear_pattern: occurrences must be an integer"}
        if occurrences_i < 2:
            return {"ok": False, "error": "create_linear_pattern: occurrences must be at least 2"}
        try:
            length_f = float(length)
        except (TypeError, ValueError):
            return {"ok": False, "error": "create_linear_pattern: length must be a number"}
        if length_f <= 0:
            return {"ok": False, "error": "create_linear_pattern: length must be a positive number"}

        features = _mock_pattern_sources("create_linear_pattern", feature_name)
        if isinstance(features, dict):
            return features
        names, sources = features
        meshes = [trimesh.load(source, force="mesh") for source in sources]
        base_mesh = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
        # Safe stub: total span from first to last copy, inclusive — same
        # "honest, non-rigorous approximation" philosophy as
        # create_polar_pattern's own mock above.
        spacing = length_f / (occurrences_i - 1)
        if reversed_direction:
            spacing = -spacing
        dvec = direction_vectors[direction_u]
        copies = []
        for i in range(occurrences_i):
            copy = base_mesh.copy()
            if i:
                copy.apply_translation([dvec[0] * spacing * i, dvec[1] * spacing * i, dvec[2] * spacing * i])
            copies.append(copy)
        combined = trimesh.util.concatenate(copies) if len(copies) > 1 else copies[0]

        resolved_name = "LinearPattern"
        out_path = _mesh_output_path(resolved_name)
        combined.export(out_path)
        _mock_object_registry()[resolved_name] = str(out_path)
        _mock_body_tip_registry()["tip"] = str(out_path)
        return {
            "ok": True,
            "name": resolved_name,
            "type": "PartDesign::LinearPattern",
            "dimensions": {
                "occurrences": occurrences_i,
                "length": length_f,
                "direction": direction_u,
                "reversed_direction": bool(reversed_direction),
                "features": names,
            },
            "bounding_box": _bbox(combined),
            "path": str(out_path),
            "gui_shown": False,
            "driver": "mock",
            "note": _MOCK_NOTE_CAD,
        }

    def create_sweep(self, profile_sketch: str, path_sketch: str, frenet: bool = True) -> dict[str, Any]:
        import math

        profile = (profile_sketch or "").strip()
        path = (path_sketch or "").strip()
        if not profile:
            return {"ok": False, "error": "create_sweep requires profile_sketch"}
        if not path:
            return {"ok": False, "error": "create_sweep requires path_sketch"}
        if profile == path:
            return {
                "ok": False,
                "error": "create_sweep: profile_sketch and path_sketch must be two different sketches",
            }

        profile_geometry = _mock_sketch_registry().get(profile)
        if profile_geometry is None:
            return {"ok": False, "error": f"create_sweep: no sketch named {profile!r} in this session"}
        path_geometry = _mock_sketch_registry().get(path)
        if path_geometry is None:
            return {"ok": False, "error": f"create_sweep: no sketch named {path!r} in this session"}

        profile_points, profile_caveat = _mock_profile_from_geometry(profile_geometry)
        path_points, path_caveat = _mock_profile_from_geometry(path_geometry)
        if profile_points is None or path_points is None:
            return {"ok": False, "error": "create_sweep: one of the sketches has no usable geometry"}

        caveats = [c for c in (profile_caveat, path_caveat) if c]
        # Safe stub: a headless mesh has no real Frenet-frame sweep kernel —
        # a two-point (single straight segment) path is swept as a real,
        # correct straight extrusion of the profile (exactly what a straight
        # sweep actually is), same "real geometry when it's this cheap"
        # philosophy as create_pipe's own straight-path case; anything with
        # more than one path segment falls back to that same straight
        # extrusion using the path's own overall span as the length, honestly
        # noted as an approximation — same "placeholder, clearly labeled"
        # philosophy as create_pipe's own curved-arc case.
        if len(path_points) == 2:
            (x1, y1), (x2, y2) = path_points
            length = math.hypot(x2 - x1, y2 - y1)
        else:
            xs = [p[0] for p in path_points]
            ys = [p[1] for p in path_points]
            length = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
            caveats.append(
                "path has more than one segment — swept as a straight extrusion along the path's "
                "own overall span, not the real curved/multi-segment path"
            )
        mesh = _fan_triangulated_extrusion(profile_points, max(length, 1e-6))

        resolved_name = "Sweep"
        out_path = _mesh_output_path(resolved_name)
        mesh.export(out_path)
        _mock_object_registry()[resolved_name] = str(out_path)
        _mock_body_tip_registry()["tip"] = str(out_path)
        note = _MOCK_NOTE_CAD if not caveats else f"{_MOCK_NOTE_CAD}; " + "; ".join(caveats)
        return {
            "ok": True,
            "name": resolved_name,
            "type": "PartDesign::AdditivePipe",
            "dimensions": {"frenet": bool(frenet)},
            "bounding_box": _bbox(mesh),
            "path": str(out_path),
            "gui_shown": False,
            "driver": "mock",
            "note": note,
        }

    def create_loft(
        self,
        cross_section_sketches: list[str],
        ruled: bool = False,
        closed: bool = False,
    ) -> dict[str, Any]:
        import trimesh

        names = [str(s).strip() for s in cross_section_sketches if str(s).strip()]
        if len(names) < 2:
            return {"ok": False, "error": "create_loft requires at least 2 cross_section_sketches"}
        if len(set(names)) != len(names):
            return {"ok": False, "error": "create_loft: cross_section_sketches must all be distinct sketch names"}

        profiles: list[list[list[float]]] = []
        for n in names:
            geometry = _mock_sketch_registry().get(n)
            if geometry is None:
                return {"ok": False, "error": f"create_loft: no sketch named {n!r} in this session"}
            points, _caveat = _mock_profile_from_geometry(geometry)
            if points is None:
                return {"ok": False, "error": f"create_loft: sketch {n!r} has no usable geometry"}
            profiles.append(points)

        # Safe stub: trimesh has no real NURBS/B-rep loft kernel — approximates
        # the loft as a THIN slab per cross-section, stacked at evenly-spaced
        # Z heights (a "stepped" stand-in for a smooth blend, not a true
        # topological loft), same honest-approximation philosophy as
        # create_pipe's own curved-arc placeholder.
        spacing = 10.0
        slabs = []
        for i, points in enumerate(profiles):
            slab = _fan_triangulated_extrusion(points, max(spacing * 0.2, 1e-6))
            slab.apply_translation([0.0, 0.0, spacing * i])
            slabs.append(slab)
        combined = trimesh.util.concatenate(slabs) if len(slabs) > 1 else slabs[0]

        resolved_name = "Loft"
        out_path = _mesh_output_path(resolved_name)
        combined.export(out_path)
        _mock_object_registry()[resolved_name] = str(out_path)
        _mock_body_tip_registry()["tip"] = str(out_path)
        return {
            "ok": True,
            "name": resolved_name,
            "type": "PartDesign::AdditiveLoft",
            "dimensions": {"cross_section_count": len(names), "ruled": bool(ruled), "closed": bool(closed)},
            "bounding_box": _bbox(combined),
            "path": str(out_path),
            "gui_shown": False,
            "driver": "mock",
            "note": f"{_MOCK_NOTE_CAD}; loft approximated as stacked cross-section slabs, not a real smooth blend",
        }

    def create_assembly(self, name: str) -> dict[str, Any]:
        import trimesh

        resolved_name = (name or "").strip()
        if not resolved_name:
            return {"ok": False, "error": "create_assembly requires a non-empty name"}

        # Safe stub: an App::Part is a pure organizational container with no
        # geometry of its own — this tiny marker box is purely so the
        # ok/path/name/bounding_box contract every other create_* result
        # already gives callers (mesh export, the object registry, HITL
        # summaries) still holds, same "placeholder, honestly labeled"
        # philosophy as create_sketch's own flat slab.
        mesh = trimesh.creation.box(extents=[1e-3, 1e-3, 1e-3])
        out_path = _mesh_output_path(resolved_name)
        mesh.export(out_path)
        _mock_object_registry()[resolved_name] = str(out_path)
        _mock_assembly_registry()[resolved_name] = []
        return {
            "ok": True,
            "name": resolved_name,
            "type": "App::Part",
            "bounding_box": _bbox(mesh),
            "path": str(out_path),
            "gui_shown": False,
            "driver": "mock",
            "note": f"{_MOCK_NOTE_CAD}; empty assembly container shown as a tiny marker box until parts are added",
        }

    def add_parts_to_assembly(self, assembly_name: str, part_names: list[str]) -> dict[str, Any]:
        assembly = (assembly_name or "").strip()
        if not assembly:
            return {"ok": False, "error": "add_parts_to_assembly requires assembly_name"}
        names = [str(p).strip() for p in part_names if str(p).strip()]
        if not names:
            return {"ok": False, "error": "add_parts_to_assembly requires a non-empty part_names list"}
        if assembly not in _mock_object_registry():
            return {"ok": False, "error": f"add_parts_to_assembly: no object named {assembly!r} in this session"}
        members = _mock_assembly_registry().setdefault(assembly, [])
        unknown = [n for n in names if n not in _mock_object_registry()]
        if unknown:
            return {"ok": False, "error": f"add_parts_to_assembly: no object(s) named {unknown} in this session"}
        for n in names:
            if n not in members:
                members.append(n)

        combined, out_path = _rebuild_assembly_mesh(assembly)
        return {
            "ok": True,
            "name": assembly,
            "type": "App::Part",
            "dimensions": {"part_names": names},
            "bounding_box": _bbox(combined),
            "path": str(out_path),
            "gui_shown": False,
            "driver": "mock",
            "note": _MOCK_NOTE_CAD,
        }

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
        target = (part_name or "").strip()
        if not target:
            return {"ok": False, "error": "position_assembly_part requires part_name"}
        target_path = _mock_object_registry().get(target)
        if not target_path:
            return {"ok": False, "error": f"position_assembly_part: no object named {target!r} in this session"}
        if not Path(target_path).is_file():
            return {"ok": False, "error": f"position_assembly_part: target_path not found: {target_path}"}
        try:
            x, y, z = float(placement_x), float(placement_y), float(placement_z)
            yaw_f, pitch_f, roll_f = float(yaw), float(pitch), float(roll)
        except (TypeError, ValueError):
            return {
                "ok": False,
                "error": "position_assembly_part: placement_x/y/z and yaw/pitch/roll must all be numbers",
            }
        # Absolute, like the real Placement: undo the placement the mesh is
        # baked at, then apply the new one, and refresh every assembly
        # preview that shows this part.
        import numpy as np
        import trimesh

        placements = _mock_placement_registry()
        old = placements.get(target, [0.0] * 6)
        mesh = trimesh.load(target_path, force="mesh")
        mesh.apply_transform(
            _placement_matrix(x, y, z, yaw_f, pitch_f, roll_f) @ np.linalg.inv(_placement_matrix(*old))
        )
        out_path = _mesh_output_path(target)
        mesh.export(out_path)
        _mock_object_registry()[target] = str(out_path)
        placements[target] = [x, y, z, yaw_f, pitch_f, roll_f]
        for assembly, members in _mock_assembly_registry().items():
            if target in members:
                _rebuild_assembly_mesh(assembly)
        return {
            "ok": True,
            "name": target,
            "path": str(out_path),
            "bounding_box": _bbox(mesh),
            "dimensions": {"placement": [x, y, z], "yaw": yaw_f, "pitch": pitch_f, "roll": roll_f},
            "driver": "mock",
            "note": _MOCK_NOTE_CAD,
        }

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
        # Same "recorded but not geometrically applied" contract as
        # position_assembly_part's own mock above — this driver has no
        # real BRep Face/Edge data to validate part1_element/part2_element
        # against (the whole point of the real tool), so it can only
        # confirm both parts exist and the constraint_type is one of the
        # 5 real values, never that the referenced geometry is real.
        # uv_tensor (Continuous Parametric UV Placement) is accepted for
        # signature parity with the real engine but, same reasoning, never
        # actually resolved against real Face geometry here — just echoed
        # back in dimensions (defaulted to [0.5, 0.5] the same way the real
        # engine defaults an omitted uv_tensor to face-center). world_fractions
        # is likewise accepted for signature parity only and just echoed
        # back verbatim (None when omitted) — this driver has no real face
        # geometry to resolve it against either.
        valid_types = {"Coincident", "Concentric", "Parallel", "Perpendicular", "Distance"}
        if constraint_type not in valid_types:
            return {
                "ok": False,
                "error": f"apply_assembly_constraint: constraint_type must be one of {sorted(valid_types)}",
            }
        registry = _mock_object_registry()
        missing = [n for n in (part1_name, part2_name) if n not in registry]
        if missing:
            return {"ok": False, "error": f"apply_assembly_constraint: unknown part(s) {missing}"}
        return {
            "ok": True,
            "name": part2_name,
            "dimensions": {
                "part1": f"{part1_name}.{part1_element}",
                "part2": f"{part2_name}.{part2_element}",
                "constraint_type": constraint_type,
                "offset": offset,
                "uv_tensor": list(uv_tensor) if uv_tensor is not None else [0.5, 0.5],
                "world_fractions": dict(world_fractions) if world_fractions is not None else None,
            },
            "driver": "mock",
            "note": f"{_MOCK_NOTE_CAD}; Face/Edge references not validated, no real geometric transform applied",
        }

    def anchor_assembly_root(self, assembly_name: str, part_name: str) -> dict[str, Any]:
        assembly = (assembly_name or "").strip()
        part = (part_name or "").strip()
        if not assembly or not part:
            return {"ok": False, "error": "anchor_assembly_root requires assembly_name and part_name"}
        registry = _mock_object_registry()
        if part not in registry:
            return {"ok": False, "error": f"anchor_assembly_root: unknown part {part!r}"}
        return {
            "ok": True,
            "name": part,
            "dimensions": {"placement": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0]},
            "path": registry.get(part),
            "driver": "mock",
            "note": f"{_MOCK_NOTE_CAD}; DanaAnchored not enforced by other mock tools, since none of "
            "them geometrically move the mesh anyway",
        }

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
        assembly = (assembly_name or "").strip()
        child = (child_link or "").strip()
        if not assembly:
            return {"ok": False, "error": "define_kinematic_joint requires assembly_name"}
        if not child:
            return {"ok": False, "error": "define_kinematic_joint requires child_link"}
        parent = (parent_link or "base_link").strip() or "base_link"
        if child == parent:
            return {"ok": False, "error": "define_kinematic_joint: child_link and parent_link cannot be the same part"}
        valid_types = {"fixed", "revolute", "continuous", "prismatic"}
        jtype = (joint_type or "fixed").strip().lower()
        if jtype not in valid_types:
            return {"ok": False, "error": f"define_kinematic_joint: joint_type must be one of {sorted(valid_types)}"}
        members = _mock_assembly_registry().get(assembly)
        if members is None:
            return {"ok": False, "error": f"define_kinematic_joint: unknown assembly_name {assembly!r}"}
        if child not in members:
            return {
                "ok": False,
                "error": f"define_kinematic_joint: {child!r} is not a member of assembly {assembly!r}",
            }
        if parent != "base_link" and parent not in members:
            return {
                "ok": False,
                "error": f"define_kinematic_joint: {parent!r} is not a member of assembly {assembly!r}",
            }
        try:
            axis_vec = [float(v) for v in axis]
        except (TypeError, ValueError):
            return {"ok": False, "error": "define_kinematic_joint: axis must be 3 numbers"}
        if len(axis_vec) != 3:
            return {"ok": False, "error": "define_kinematic_joint: axis must have exactly 3 elements [x, y, z]"}

        joints = _mock_kinematic_joints_registry().setdefault(assembly, {})
        joints[child] = {
            "parent": parent,
            "type": jtype,
            "axis": axis_vec,
            "joint_name": (joint_name or "").strip() or None,
            "limit_lower": limit_lower,
            "limit_upper": limit_upper,
            "limit_effort": limit_effort,
            "limit_velocity": limit_velocity,
        }
        return {
            "ok": True,
            "name": assembly,
            "dimensions": {"parent_link": parent, "child_link": child, "joint_type": jtype, "axis": axis_vec},
            "driver": "mock",
            "note": f"{_MOCK_NOTE_CAD}; joint recorded, used by export_assembly_to_urdf",
        }

    def validate_assembly_collisions(self, assembly_name: str) -> dict[str, Any]:
        """The real engine's audit on the members' meshes, same result
        schema: an axis-aligned bounding-box broadphase, then manifold3d's
        exact mesh intersection volume for pairs whose boxes meet and its
        ``min_gap`` for the clearance. Consumed boolean operands are left
        out, as on the real engine. A member that isn't a closed manifold
        mesh can't be checked: its pairs go to ``collision_check_errors``
        and the call fails rather than calling the assembly collision-free."""
        import numpy as np
        import trimesh

        assembly = _resolve_mock_name(assembly_name)
        members = _mock_assembly_registry().get(assembly or "")
        if members is None:
            return {"ok": False, "error": f"validate_assembly_collisions: no assembly named {assembly_name!r} in this session"}
        consumed = {op for entry in _mock_boolean_registry().values() for op in entry["operands"]}
        registry = _mock_object_registry()
        names = [m for m in members if m not in consumed and m in registry and Path(registry[m]).is_file()]
        meshes = {n: trimesh.load(registry[n], force="mesh") for n in names}
        solids: dict[str, Any] = {}
        solid_errors: dict[str, str] = {}
        for n, mesh in meshes.items():
            try:
                solids[n] = _to_manifold(mesh)
            except Exception as exc:  # noqa: BLE001 — manifold3d raises various types on bad input
                solid_errors[n] = str(exc)

        collisions, clearances, check_errors = [], [], []
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                lo_a, hi_a = meshes[a].bounds
                lo_b, hi_b = meshes[b].bounds
                boxes_meet = bool(np.all(lo_a <= hi_b) and np.all(lo_b <= hi_a))
                bad = solid_errors.get(a) or solid_errors.get(b)
                if bad:
                    stage = "intersection" if boxes_meet else "distance"
                    check_errors.append({"part_a": a, "part_b": b, "stage": stage, "error": bad})
                    continue
                overlap = 0.0
                if boxes_meet:
                    try:
                        overlap = float((solids[a] ^ solids[b]).volume())
                    except Exception as exc:  # noqa: BLE001
                        check_errors.append({"part_a": a, "part_b": b, "stage": "intersection", "error": str(exc)})
                        continue
                if overlap > _MOCK_COLLISION_VOLUME_EPSILON:
                    collisions.append({"part_a": a, "part_b": b, "overlap_volume": overlap})
                    distance = 0.0
                else:
                    reach = float(np.linalg.norm(np.maximum(hi_a, hi_b) - np.minimum(lo_a, lo_b))) + 1.0
                    try:
                        distance = float(solids[a].min_gap(solids[b], reach))
                    except Exception as exc:  # noqa: BLE001
                        check_errors.append({"part_a": a, "part_b": b, "stage": "distance", "error": str(exc)})
                        continue
                clearances.append({"part_a": a, "part_b": b, "distance_mm": distance})

        result = {
            "name": assembly,
            "collisions": collisions,
            "has_collisions": bool(collisions),
            "checked_members": len(names),
            "clearances": clearances,
            "path": registry.get(assembly, ""),
            "driver": "mock",
            "note": f"{_MOCK_NOTE_CAD}; overlap and clearance computed on the members' meshes",
        }
        if check_errors:
            pairs = ", ".join(f"{e['part_a']}/{e['part_b']}" for e in check_errors)
            return {
                "ok": False,
                "error": (
                    f"validate_assembly_collisions could not check {len(check_errors)} part pair(s) "
                    f"({pairs}): their geometry is not a valid solid, so the assembly cannot be "
                    "confirmed collision-free. Inspect those parts with inspect_spatial_properties "
                    "and fix or rebuild them, then validate again."
                ),
                **result,
                "collision_check_errors": check_errors,
            }
        return {"ok": True, **result}

    def export_assembly_to_urdf(
        self,
        assembly_name: str,
        export_directory: str | None = None,
        density_kg_m3: float | None = None,
    ) -> dict[str, Any]:
        """The real engine's export on the members' meshes: each member's
        mesh is written in its own local frame (its placement undone) to
        ``<out>/meshes/<name>.stl``, its placement relative to its joint
        parent (define_kinematic_joint, else the assembly) becomes the link
        origin, and its exact mesh volume, center of mass and inertia tensor
        (about the center of mass, unit density, like FreeCAD's
        MatrixOfInertia) give the ``<inertial>``. The XML is written by the
        same ``urdf_builder.export_assembly_parts_to_urdf`` the real engine
        calls. Consumed boolean operands are left out, as there."""
        import numpy as np
        import trimesh

        from dana.plugins.freecad import engine
        from dana.tools.urdf_builder import ROOT_LINK_NAME, export_assembly_parts_to_urdf

        assembly = _resolve_mock_name(assembly_name)
        members = _mock_assembly_registry().get(assembly or "")
        if members is None:
            return {"ok": False, "error": f"export_assembly_to_urdf: no assembly named {assembly_name!r} in this session"}
        safe = engine._safe_name(assembly)
        out_dir = Path(export_directory) if (export_directory or "").strip() else engine._export_dir() / f"{safe}_urdf"
        meshes_dir = out_dir / "meshes"
        meshes_dir.mkdir(parents=True, exist_ok=True)

        consumed = {op for entry in _mock_boolean_registry().values() for op in entry["operands"]}
        registry = _mock_object_registry()
        placements = _mock_placement_registry()
        joints = _mock_kinematic_joints_registry().get(assembly, {})

        def world(name: str) -> Any:
            return _placement_matrix(*placements.get(name, [0.0] * 6))

        parts = []
        for name in members:
            path = registry.get(name)
            if name in consumed or not path or not Path(path).is_file():
                continue
            mesh = trimesh.load(path, force="mesh")
            mesh.apply_transform(np.linalg.inv(world(name)))
            mesh.export(meshes_dir / f"{name}.stl")
            jdef = joints.get(name, {})
            parent = jdef.get("parent") or ROOT_LINK_NAME
            if parent != ROOT_LINK_NAME and parent not in registry:
                return {
                    "ok": False,
                    "error": f"export_assembly_to_urdf: {name!r} has a joint to unknown parent {parent!r}",
                }
            relative = world(name) if parent == ROOT_LINK_NAME else np.linalg.inv(world(parent)) @ world(name)
            roll, pitch, yaw = trimesh.transformations.euler_from_matrix(relative, "sxyz")
            part: dict[str, Any] = {
                "name": name,
                "mesh_file": f"meshes/{name}.stl",
                "origin_xyz": [float(v) for v in relative[:3, 3]],
                "origin_rpy": [float(roll), float(pitch), float(yaw)],
                "joint_parent": parent,
                "joint_type": jdef.get("type") or "fixed",
                "joint_axis": jdef.get("axis") or [0.0, 0.0, 1.0],
                "joint_name": jdef.get("joint_name"),
                "limit_lower": jdef.get("limit_lower"),
                "limit_upper": jdef.get("limit_upper"),
                "limit_effort": jdef.get("limit_effort"),
                "limit_velocity": jdef.get("limit_velocity"),
            }
            if mesh.is_watertight and mesh.volume > 0:
                inertia = mesh.moment_inertia
                part.update(
                    volume=float(mesh.volume),
                    center_of_mass=[float(v) for v in mesh.center_mass],
                    inertia={
                        "ixx": float(inertia[0][0]), "ixy": float(inertia[0][1]), "ixz": float(inertia[0][2]),
                        "iyy": float(inertia[1][1]), "iyz": float(inertia[1][2]), "izz": float(inertia[2][2]),
                    },
                )
            parts.append(part)
        if not parts:
            return {"ok": False, "error": f"export_assembly_to_urdf: assembly {assembly!r} has no parts with geometry"}

        density_kwargs = {"density_kg_m3": float(density_kg_m3)} if density_kg_m3 is not None else {}
        result = json.loads(export_assembly_parts_to_urdf(assembly, parts, str(out_dir), **density_kwargs))
        if not result.get("ok"):
            return {"ok": False, "error": f"export_assembly_to_urdf: {result.get('error')}"}
        return {
            "ok": True,
            "name": result.get("name") or safe,
            "type": "urdf",
            "path": result.get("path"),
            "link_count": result.get("link_count"),
            "joint_count": result.get("joint_count"),
            "meshes_dir": str(meshes_dir),
            "driver": "mock",
            "note": f"{_MOCK_NOTE_CAD}; links and inertia from the members' meshes",
        }

    def generate_assembly_bom(
        self, assembly_name: str, material: str = "PLA", part_materials: dict[str, str] | None = None
    ) -> dict[str, Any]:
        import trimesh

        from dana.plugins.manufacturing.bom_exporter import build_bom

        assembly = (assembly_name or "").strip()
        members = _mock_assembly_registry().get(assembly)
        if members is None:
            return {"ok": False, "error": f"generate_assembly_bom: no assembly named {assembly!r} in this session"}
        # Unlike export_assembly_to_urdf, a BOM only needs each part's volume,
        # which a closed trimesh mesh gives exactly (a box primitive is
        # exact; curved primitives are their tessellation). An open mesh has
        # no well-defined volume, so it is refused rather than guessed.
        booleans = _mock_boolean_registry()
        consumed = {op for entry in booleans.values() for op in entry["operands"]}
        uncut = [n for n in members if n not in consumed and booleans.get(n, {}).get("applied") is False]
        if uncut:
            return {
                "ok": False,
                "error": f"generate_assembly_bom: the mock boolean for {uncut} never ran (trimesh's manifold3d "
                "backend is not installed), so its mesh is the uncut base — install manifold3d or use a real "
                "FreeCAD engine to cost it",
                "driver": "mock",
            }
        parts, open_meshes = [], []
        for name in members:
            if name in consumed:
                continue  # an input to a boolean result, not a separate part
            path = _mock_object_registry().get(name)
            if not path or not Path(path).is_file():
                continue
            mesh = trimesh.load(path, force="mesh")
            if not mesh.is_watertight:
                open_meshes.append(name)
                continue
            parts.append(
                {
                    "name": name,
                    "volume_mm3": abs(float(mesh.volume)),
                    "center_of_mass_mm": [float(v) for v in mesh.center_mass],
                    "mesh": mesh,
                }
            )
        if open_meshes:
            return {
                "ok": False,
                "error": f"generate_assembly_bom: mock mesh(es) for {open_meshes} are not closed, so their "
                "volume is undefined — a real FreeCAD engine is required to cost them",
                "driver": "mock",
            }
        # Same ground-contact rule as the real engine: vertices within 0.5 mm
        # of the assembly's lowest Z (mesh vertices, since there is no BRep).
        footprint: list[list[float]] = []
        if parts:
            ground = min(float(p["mesh"].bounds[0][2]) for p in parts)
            for p in parts:
                vertices = p.pop("mesh").vertices
                footprint += [[float(v[0]), float(v[1])] for v in vertices if v[2] <= ground + 0.5]
        result = build_bom(assembly, parts, material, part_materials, footprint_xy=footprint)
        if result.get("ok"):
            result["driver"] = "mock"
            result["note"] = f"{_MOCK_NOTE_CAD}; volumes measured from the mock driver's trimesh meshes"
        return result

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
        import numpy as np
        import trimesh

        op = (operation or "").strip().lower()
        if op not in ("cut", "add"):
            return {"ok": False, "error": f"create_feature_on_face: unknown operation '{operation}' — must be 'cut' or 'add'"}
        shape_key = (shape or "").strip().lower()
        if shape_key not in ("circle", "rectangle"):
            return {"ok": False, "error": f"create_feature_on_face: unknown shape '{shape}' — must be 'circle' or 'rectangle'"}
        face_key = (face or "").strip().lower()
        if face_key not in _MOCK_FACE_NORMAL:
            return {"ok": False, "error": f"create_feature_on_face: unknown face '{face}' — must be one of {sorted(_MOCK_FACE_NORMAL)}"}
        if shape_key == "circle" and (radius is None or float(radius) <= 0):
            return {"ok": False, "error": "create_feature_on_face requires a positive radius for shape='circle'"}
        if shape_key == "rectangle" and (
            width is None or length is None or float(width) <= 0 or float(length) <= 0
        ):
            return {"ok": False, "error": "create_feature_on_face requires positive width/length for shape='rectangle'"}

        base_path = _mock_object_registry().get(object_name)
        if not base_path:
            return {"ok": False, "error": f"create_feature_on_face: no object named {object_name!r} in this session"}
        base = Path(base_path)
        if not base.is_file():
            return {"ok": False, "error": f"create_feature_on_face: object path not found: {base_path}"}
        base_mesh = trimesh.load(base, force="mesh")

        # Same face-resolution math as the real engine's _face_axes +
        # _FACE_ORIGIN_EXPR, against this mesh's own bounds instead of a
        # FreeCAD BoundBox — no genuine flat-face verification here (unlike
        # the real driver's Part.Plane check), since a headless trimesh
        # object has no reliable per-face surface-type introspection; this
        # mock always assumes the requested face is flat, matching every
        # other mock stub's "best-effort, not geometrically rigorous" style.
        lo, hi = (np.array(bound) for bound in base_mesh.bounds)
        center = (lo + hi) / 2.0
        axis_i, use_max = _MOCK_FACE_EXTREME_AXIS[face_key]
        face_origin = center.copy()
        face_origin[axis_i] = hi[axis_i] if use_max else lo[axis_i]
        normal = np.array(_MOCK_FACE_NORMAL[face_key])
        u_axis = np.array(_MOCK_FACE_U_AXIS[face_key])
        v_axis = np.array(_MOCK_FACE_V_AXIS[face_key])
        center_point = face_origin + u_axis * float(u) + v_axis * float(v)

        clearance = _MOCK_FACE_FEATURE_CLEARANCE
        push_sign = 1.0 if op == "cut" else -1.0
        extrude_sign = -1.0 if op == "cut" else 1.0
        span_start = push_sign * clearance
        span_end = span_start + extrude_sign * (float(extent) + clearance)
        mid_offset = (span_start + span_end) / 2.0
        total_height = abs(span_end - span_start)

        if shape_key == "circle":
            tool_mesh = trimesh.creation.cylinder(radius=float(radius), height=total_height)
        else:
            tool_mesh = trimesh.creation.box(extents=(float(width), float(length), total_height))
        tool_mesh.apply_translation((0.0, 0.0, mid_offset))
        rotation = np.eye(4)
        rotation[:3, 0] = u_axis
        rotation[:3, 1] = v_axis
        rotation[:3, 2] = normal
        tool_mesh.apply_transform(rotation)
        tool_mesh.apply_translation(center_point)

        mesh_ops = {"cut": "difference", "add": "union"}
        feature_types = {"cut": "Part::Cut", "add": "Part::MultiFuse"}
        default_names = {"cut": "Cut", "add": "Fusion"}
        try:
            result_mesh = getattr(base_mesh, mesh_ops[op])(tool_mesh)
            engine_note = _MOCK_NOTE_CAD
        except BaseException:  # noqa: BLE001 — boolean engine unavailable in this container
            result_mesh = base_mesh
            engine_note = f"{_MOCK_NOTE_CAD}; boolean engine unavailable, returned base unmodified"

        resolved_name = name or default_names[op]
        out_path = _mesh_output_path(resolved_name)
        result_mesh.export(out_path)
        _mock_object_registry()[resolved_name] = str(out_path)
        return {
            "ok": True,
            "name": resolved_name,
            "type": feature_types[op],
            "operation": "cut" if op == "cut" else "union",
            "bounding_box": _bbox(result_mesh),
            "dimensions": {
                "face": face_key, "shape": shape_key, "u": float(u), "v": float(v),
                "extent": float(extent), "operation": op,
            },
            "path": str(out_path),
            "gui_shown": False,
            "driver": "mock",
            "note": engine_note,
        }

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
        import math

        import trimesh

        # source_object (real engine's by-name fix — see
        # dana.plugins.freecad.ir's "pattern" kind) is accepted for call-site
        # signature compatibility only: this mock never has more than one
        # object per mesh file, so there's no "wrong object" heuristic to fix
        # here in the first place.
        del source_object

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
        # A genuine .stl — this is the user-facing "download as STL" export,
        # not a live-preview mesh.
        # Same place and name the real engine's export_model uses.
        from dana.plugins.freecad import engine

        out_path = engine._export_dir() / f"{engine._safe_name(filename or 'export')}.stl"
        combined.export(out_path)
        return {
            "ok": True,
            "format": fmt,
            "path": str(out_path),
            "target_count": len(paths),
            "driver": "mock",
            "note": _MOCK_NOTE_CAD,
        }


# Session persistence (dana.api.session_state): the mock driver's per-session
# state, so a restarted mock server (HF Space, CI) keeps a resumed session's
# objects. Meshes are files on disk; restore keeps only objects whose mesh
# file still exists.
_PERSISTED_MOCK_REGISTRIES: dict[str, dict[str, Any]] = {
    "objects": _MOCK_OBJECT_REGISTRY,
    "sketches": _MOCK_SKETCH_GEOMETRY,
    "body_tip": _MOCK_BODY_TIP,
    "assemblies": _MOCK_ASSEMBLY_MEMBERS,
    "joints": _MOCK_KINEMATIC_JOINTS,
    "booleans": _MOCK_BOOLEAN_RESULTS,
    "placements": _MOCK_PLACEMENTS,
}


def snapshot_mock_session(session_id: str) -> dict[str, Any]:
    return {
        key: copy.deepcopy(registry[session_id])
        for key, registry in _PERSISTED_MOCK_REGISTRIES.items()
        if session_id in registry
    }


def replace_mock_session(session_id: str, data: dict[str, Any]) -> None:
    """Sets this session's mock state to exactly ``data`` (a
    ``snapshot_mock_session`` result; ``{}`` clears it) — the parametric
    rebuild's scratch reset and rollback."""
    for key, registry in _PERSISTED_MOCK_REGISTRIES.items():
        if key in data:
            registry[session_id] = copy.deepcopy(data[key])
        else:
            registry.pop(session_id, None)


def restore_mock_session(session_id: str, data: Any) -> set[str]:
    """Restores ``snapshot_mock_session`` output and returns the names of the
    objects that survived (their mesh file still exists)."""
    if isinstance(data, dict):
        for key, registry in _PERSISTED_MOCK_REGISTRIES.items():
            if isinstance(data.get(key), dict):
                registry[session_id] = copy.deepcopy(data[key])
    objects = _MOCK_OBJECT_REGISTRY.get(session_id, {})
    for name in [n for n, mesh in objects.items() if not (isinstance(mesh, str) and Path(mesh).is_file())]:
        objects.pop(name)
    return set(objects)


__all__ = (
    "MockControlPlane",
    "MockFreeCADEngine",
    "replace_mock_session",
    "restore_mock_session",
    "snapshot_mock_session",
)
