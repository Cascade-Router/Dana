"""Pre-print sanity check for a session part: is it a closed, manifold solid,
does it fit the printer, and will it print without supports as oriented.

Two halves:

* ``build_report`` — pure Python. Turns mesh-check flags plus per-facet
  ``(normal_z, area, z_min, z_max)`` data into the agent-facing report
  ``{"printable", "requires_supports", "warnings", ...}``. No FreeCAD needed,
  so it's tested directly on synthetic meshes.
* ``check_printability`` — the real engine side. One FreeCADCmd script exports
  the target's shape to a binary STL, reloads it with ``Mesh.Mesh(stl)``, and
  records FreeCAD's own mesh checks (``isSolid``, non-manifolds,
  self-intersections, open edges) and every facet, then ``build_report`` runs
  on that.

The overhang check is a geometric heuristic, not a slicer: a downward-facing
facet more than ``max_overhang_deg`` from vertical (default 45°) that isn't
resting on the build plate needs support. The part is checked as oriented in
the document (+Z up, lowest point on the bed). Small horizontal holes and short
bridges also count as overhangs here even though most printers bridge them,
which is why the report gives the overhang area and Z range, not only a flag.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from dana.plugins.freecad.engine import (
    _OK_MARKER,
    _RESOLVE_OBJECT_SNIPPET,
    _error,
    _ok,
    _run_freecad_script,
    _session_document_path,
)


@dataclass(frozen=True)
class PrinterProfile:
    name: str
    x_mm: float
    y_mm: float
    z_mm: float


# Prusa MK4 / i3-class bed — the spec's default.
DEFAULT_PRINTER = PrinterProfile("Prusa MK4", 250.0, 210.0, 220.0)
DEFAULT_MAX_OVERHANG_DEG = 45.0
# Overhang area below this (mm²) is tessellation noise (a sliver facet on a
# curved edge), not something a slicer would generate support for.
MIN_OVERHANG_AREA_MM2 = 1.0
# How close (mm) a facet must be to the part's lowest Z to count as resting
# on the build plate.
BED_CONTACT_TOLERANCE_MM = 0.01
# Mesh export accuracy: chord deviation (mm) and angular deflection (rad).
_LINEAR_DEFLECTION_MM = 0.05
_ANGULAR_DEFLECTION_RAD = math.radians(5.0)


@dataclass(frozen=True)
class Facet:
    normal_z: float
    area: float
    z_min: float
    z_max: float


def find_overhangs(
    facets: Iterable[Facet], *, bed_z: float, max_overhang_deg: float = DEFAULT_MAX_OVERHANG_DEG
) -> dict[str, Any]:
    """Downward-facing facets steeper than ``max_overhang_deg`` from vertical,
    excluding those lying on the bed. A facet's angle from vertical is
    ``asin(-normal_z)`` for a downward normal, so it overhangs when
    ``normal_z < -sin(max_overhang_deg)``."""
    limit = -math.sin(math.radians(max_overhang_deg))
    area = 0.0
    count = 0
    z_lo = math.inf
    z_hi = -math.inf
    worst = 0.0
    for f in facets:
        if f.normal_z >= limit or f.z_max <= bed_z + BED_CONTACT_TOLERANCE_MM:
            continue
        area += f.area
        count += 1
        z_lo = min(z_lo, f.z_min)
        z_hi = max(z_hi, f.z_max)
        worst = max(worst, math.degrees(math.asin(min(1.0, -f.normal_z))))
    return {
        "facet_count": count,
        "area_mm2": round(area, 3),
        "z_range_mm": [round(z_lo, 3), round(z_hi, 3)] if count else None,
        "worst_angle_from_vertical_deg": round(worst, 1) if count else None,
    }


def fit_build_volume(size: Sequence[float], printer: PrinterProfile) -> str:
    """"fits", "fits_rotated" (only with a 90° turn about Z), or "too_large"."""
    x, y, z = size
    if z > printer.z_mm:
        return "too_large"
    if x <= printer.x_mm and y <= printer.y_mm:
        return "fits"
    if y <= printer.x_mm and x <= printer.y_mm:
        return "fits_rotated"
    return "too_large"


def build_report(
    *,
    target: str,
    checks: dict[str, Any],
    bbox: Sequence[float],
    facets: Iterable[Facet],
    printer: PrinterProfile = DEFAULT_PRINTER,
    max_overhang_deg: float = DEFAULT_MAX_OVERHANG_DEG,
) -> dict[str, Any]:
    """The agent-facing report. ``checks`` holds the mesh/shape flags:
    ``shape_valid``, ``closed_solid``, ``mesh_solid`` (watertight),
    ``non_manifold`` (bool), ``self_intersections`` (bool),
    ``open_edges`` (count), ``components`` (count). ``bbox`` is
    ``[x_min, y_min, z_min, x_max, y_max, z_max]``."""
    warnings: list[str] = []
    blocking = False

    if not checks.get("shape_valid", True):
        blocking = True
        warnings.append("The CAD shape is invalid (FreeCAD's shape check failed); repair the model first.")
    if not checks.get("closed_solid", True):
        blocking = True
        warnings.append("The part is not a closed solid (open shell or bare faces); a slicer can't fill it.")
    if not checks.get("mesh_solid", True):
        blocking = True
        warnings.append("The exported mesh is not watertight.")
    if checks.get("open_edges"):
        blocking = True
        warnings.append(f"The mesh has {checks['open_edges']} open (naked) edge(s).")
    if checks.get("non_manifold"):
        blocking = True
        warnings.append("The mesh has non-manifold edges or points.")
    if checks.get("self_intersections"):
        blocking = True
        warnings.append("The mesh intersects itself.")
    if (checks.get("components") or 1) > 1:
        warnings.append(f"The part is {checks['components']} separate bodies; each prints as its own island.")

    size = [bbox[3] - bbox[0], bbox[4] - bbox[1], bbox[5] - bbox[2]]
    fit = fit_build_volume(size, printer)
    if fit == "too_large":
        blocking = True
        warnings.append(
            f"The part ({size[0]:.1f} x {size[1]:.1f} x {size[2]:.1f} mm) exceeds the "
            f"{printer.name} build volume ({printer.x_mm:g} x {printer.y_mm:g} x {printer.z_mm:g} mm)."
        )
    elif fit == "fits_rotated":
        warnings.append(f"The part only fits the {printer.name} bed rotated 90° about Z.")

    overhangs = find_overhangs(facets, bed_z=bbox[2], max_overhang_deg=max_overhang_deg)
    requires_supports = overhangs["area_mm2"] >= MIN_OVERHANG_AREA_MM2
    if requires_supports:
        z_lo, z_hi = overhangs["z_range_mm"]
        warnings.append(
            f"{overhangs['area_mm2']:.1f} mm² of downward-facing surface is steeper than "
            f"{max_overhang_deg:g}° from vertical (Z {z_lo:.1f}-{z_hi:.1f} mm, worst "
            f"{overhangs['worst_angle_from_vertical_deg']:.0f}°); it needs supports, a bridge, or reorienting."
        )

    return {
        "printable": not blocking,
        "requires_supports": requires_supports,
        "warnings": warnings,
        "target": target,
        "dimensions_mm": [round(v, 3) for v in size],
        "build_volume": {**asdict(printer), "fit": fit},
        "overhangs": {**overhangs, "max_overhang_deg": max_overhang_deg},
        "checks": checks,
    }


# --- real engine --------------------------------------------------------------------

_CHECK_SCRIPT = (
    "import FreeCAD as App\nimport Mesh\nimport MeshPart\nimport Part\nimport json\n\n"
    + _RESOLVE_OBJECT_SNIPPET
    + """
doc = App.openDocument({session_path!r})
_name = {object_name!r}
if _name:
    target = resolve_object(doc, _name)
    if target is None:
        raise RuntimeError("Object not found: " + _name)
else:
    _bodies = [o for o in doc.Objects if o.TypeId == "PartDesign::Body"]
    _solids = [o for o in doc.Objects if o.isDerivedFrom("Part::Feature") and o.Visibility
               and not o.Shape.isNull() and o.Shape.Solids and not o.InList]
    target = _bodies[-1] if _bodies else (_solids[-1] if _solids else None)
    if target is None:
        raise RuntimeError("no PartDesign Body or visible solid in the session document — pass object_name")
shape = target.Shape
if shape.isNull():
    raise RuntimeError(target.Name + " has no shape to check")

checks = {{"shape_valid": bool(shape.isValid())}}
# Every face belongs to a solid and every shell is closed (no stray faces).
checks["closed_solid"] = (
    bool(shape.Solids)
    and all(_s.isClosed() for _s in shape.Shells)
    and len(shape.Faces) <= sum(len(_s.Faces) for _s in shape.Solids)
)

_mesh = MeshPart.meshFromShape(
    Shape=shape, LinearDeflection={linear!r}, AngularDeflection={angular!r}, Relative=False
)
_mesh.write({stl_path!r})  # binary STL
mesh = Mesh.Mesh({stl_path!r})
checks["mesh_solid"] = bool(mesh.isSolid())
checks["non_manifold"] = bool(mesh.hasNonManifolds())
checks["self_intersections"] = bool(mesh.hasSelfIntersections())
checks["components"] = int(mesh.countComponents())
_edge_use = {{}}
for _f in mesh.Facets:
    _p = _f.PointIndices
    for _e in ((_p[0], _p[1]), (_p[1], _p[2]), (_p[2], _p[0])):
        _k = (min(_e), max(_e))
        _edge_use[_k] = _edge_use.get(_k, 0) + 1
checks["open_edges"] = sum(1 for _n in _edge_use.values() if _n == 1)
checks["facet_count"] = int(mesh.CountFacets)

_bb = mesh.BoundBox
facets = [
    [_f.Normal.z, _f.Area, min(_v[2] for _v in _f.Points), max(_v[2] for _v in _f.Points)]
    for _f in mesh.Facets
]
# A file, not stdout: FreeCADCmd's stdout capture can drop output at exit.
with open({data_path!r}, "w", encoding="utf-8") as _out:
    json.dump({{"target": target.Name, "checks": checks,
               "bbox": [_bb.XMin, _bb.YMin, _bb.ZMin, _bb.XMax, _bb.YMax, _bb.ZMax], "facets": facets}}, _out)
print({marker!r})
"""
)


def check_printability(
    object_name: str | None = None,
    *,
    printer: PrinterProfile = DEFAULT_PRINTER,
    max_overhang_deg: float = DEFAULT_MAX_OVERHANG_DEG,
) -> str:
    """Run the pre-print check on ``object_name`` (default: the session's last
    PartDesign Body, else its last visible top-level solid). Read-only — the
    session document isn't saved. Returns the engine's usual JSON string."""
    if not 0.0 < float(max_overhang_deg) < 90.0:
        return _error("check_printability: max_overhang_deg must be between 0 and 90")
    session_path = _session_document_path()
    if not session_path.is_file():
        return _error("check_printability: no session document yet — build a part first")
    stl_path = str(session_path.with_name("printability_check.stl"))
    fd, data_path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        script = _CHECK_SCRIPT.format(
            session_path=str(session_path),
            object_name=(object_name or "").strip(),
            linear=_LINEAR_DEFLECTION_MM,
            angular=_ANGULAR_DEFLECTION_RAD,
            stl_path=stl_path,
            data_path=data_path,
            marker=_OK_MARKER,
        )
        result = _run_freecad_script(script)
        if not result["ok"]:
            return _error(f"check_printability failed: {result['error']}")
        with open(data_path, encoding="utf-8") as fh:
            data = json.load(fh)
    finally:
        try:
            os.unlink(data_path)
        except OSError:
            pass
    report = build_report(
        target=data["target"],
        checks=data["checks"],
        bbox=data["bbox"],
        facets=(Facet(*f) for f in data["facets"]),
        printer=printer,
        max_overhang_deg=float(max_overhang_deg),
    )
    return _ok(**report, stl_path=stl_path)


def printer_profile(build_volume_mm: Sequence[float] | None) -> PrinterProfile:
    """``DEFAULT_PRINTER``, or a custom ``[x, y, z]`` build volume."""
    if not build_volume_mm:
        return DEFAULT_PRINTER
    x, y, z = (float(v) for v in build_volume_mm)
    if min(x, y, z) <= 0:
        raise ValueError("build_volume_mm values must be positive")
    return PrinterProfile("custom printer", x, y, z)


__all__ = (
    "DEFAULT_PRINTER",
    "Facet",
    "PrinterProfile",
    "build_report",
    "check_printability",
    "find_overhangs",
    "fit_build_volume",
    "printer_profile",
)
