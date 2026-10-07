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

``evaluate_orientations`` repeats the overhang and build-volume checks for the
six principal poses (each of the part's ±X/±Y/±Z axes pointing up, i.e. as the
build-plate normal) and recommends one, with the rotation that gets there and
a remediation hint. Only those six are tried, not arbitrary angles.
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
    _safe_name,
    _session_document_path,
)
from dana.plugins.os import file_system
from dana.session_context import session_scoped_dir


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


@dataclass(frozen=True)
class SolidFacet:
    """A facet in the part's own frame: unit ``normal`` (x, y, z), ``area``,
    and its vertices' per-axis minimum ``lo`` and maximum ``hi``."""

    normal: tuple[float, float, float]
    area: float
    lo: tuple[float, float, float]
    hi: tuple[float, float, float]


# (part axis pointing up, axis index, sign, rotation that brings it to +Z).
# Rotations are right-handed about the part's own axes, so e.g. rotating
# +90° about X turns the part's +Y up. Listed current-pose first.
_ORIENTATIONS: tuple[tuple[str, int, int, tuple[str, float] | None], ...] = (
    ("+Z", 2, 1, None),
    ("-Z", 2, -1, ("X", 180.0)),
    ("+X", 0, 1, ("Y", -90.0)),
    ("-X", 0, -1, ("Y", 90.0)),
    ("+Y", 1, 1, ("X", 90.0)),
    ("-Y", 1, -1, ("X", -90.0)),
)
# A facet this close to straight down counts toward bed contact.
_BED_CONTACT_NORMAL = -0.999


def _evaluate_pose(
    facets: Sequence[SolidFacet], axis: int, sign: int, printer: PrinterProfile, max_overhang_deg: float
) -> dict[str, Any]:
    def up(f: SolidFacet) -> Facet:
        lo, hi = (f.lo[axis], f.hi[axis]) if sign > 0 else (-f.hi[axis], -f.lo[axis])
        return Facet(sign * f.normal[axis], f.area, lo, hi)

    projected = [up(f) for f in facets]
    bed = min(f.z_min for f in projected)
    height = max(f.z_max for f in projected) - bed
    footprint = [max(f.hi[i] for f in facets) - min(f.lo[i] for f in facets) for i in range(3) if i != axis]
    fit = fit_build_volume((*footprint, height), printer)
    overhangs = find_overhangs(projected, bed_z=bed, max_overhang_deg=max_overhang_deg)
    contact = sum(
        f.area for f in projected if f.normal_z < _BED_CONTACT_NORMAL and f.z_max <= bed + BED_CONTACT_TOLERANCE_MM
    )
    requires_supports = overhangs["area_mm2"] >= MIN_OVERHANG_AREA_MM2
    return {
        "fit": fit,
        "requires_supports": requires_supports,
        "support_free": fit != "too_large" and not requires_supports,
        "overhang_area_mm2": overhangs["area_mm2"],
        "height_mm": round(height, 3),
        "bed_contact_mm2": round(contact, 3),
    }


def _describe_pose(pose: dict[str, Any]) -> str:
    rotation = pose["rotation"]
    if rotation is None:
        return "as modelled"
    return (
        f"rotated {rotation['degrees']:+g}° about the part's {rotation['axis']} axis "
        f"(its {pose['up_axis']} side up, {pose['bed_face']} side on the bed)"
    )


def evaluate_orientations(
    facets: Sequence[SolidFacet],
    *,
    printer: PrinterProfile = DEFAULT_PRINTER,
    max_overhang_deg: float = DEFAULT_MAX_OVERHANG_DEG,
    mesh_ok: bool = True,
) -> dict[str, Any]:
    """Score the six principal poses and pick one.

    Returns ``current_orientation_printable`` (as modelled it fits the bed
    and needs no supports), ``recommended_orientation`` (one of
    ``orientations``: the current pose whenever that is already
    support-free, so a fine part is never told to move; otherwise the best by
    fit, then supports, then least overhang, then most bed contact, then
    lowest height), and ``remediation_hint`` (None when nothing needs doing).
    ``mesh_ok=False`` (an open or non-manifold mesh) makes every pose
    unprintable: no rotation fixes that.
    """
    poses = []
    for up_axis, axis, sign, rotation in _ORIENTATIONS:
        pose = {
            "up_axis": up_axis,
            "bed_face": ("-" if sign > 0 else "+") + up_axis[1],
            "rotation": None if rotation is None else {"axis": rotation[0], "degrees": rotation[1]},
            "is_current": rotation is None,
            **_evaluate_pose(facets, axis, sign, printer, max_overhang_deg),
        }
        poses.append(pose)
    current = poses[0]

    if current["support_free"]:
        best = current
    else:
        best = min(
            poses,
            key=lambda p: (
                p["fit"] == "too_large",
                p["requires_supports"],
                p["overhang_area_mm2"],
                -p["bed_contact_mm2"],
                p["height_mm"],
                not p["is_current"],
            ),
        )

    if not mesh_ok:
        hint = "Repair the mesh first (see warnings); no orientation fixes an open or non-manifold mesh."
    elif current["support_free"]:
        hint = None
    elif best["fit"] == "too_large":
        hint = (
            f"Too large for the {printer.name} ({printer.x_mm:g} x {printer.y_mm:g} x {printer.z_mm:g} mm) "
            "in every principal orientation; scale it down or split it into parts."
        )
    elif best["support_free"]:
        turned = best["fit"] == "fits_rotated"
        if current["fit"] == "too_large":
            fits = " and it fits the bed" + (" if turned 90° on it" if turned else "")
        else:
            fits = ", turned 90° on the bed" if turned else ""
        hint = f"Print it {_describe_pose(best)}: no supports needed{fits} ({best['height_mm']:g} mm tall)."
    else:
        least = (
            f"least support is {_describe_pose(best)}, {best['overhang_area_mm2']:g} mm² of overhang"
            + ("" if best["is_current"] else f" vs {current['overhang_area_mm2']:g} mm² as modelled")
        )
        hint = (
            f"Needs supports in every principal orientation; {least}. Enable supports in the slicer, or redesign "
            f"the overhangs to {max_overhang_deg:g}° or less from vertical (e.g. chamfers instead of flat ledges), "
            "or split the part."
        )

    return {
        "current_orientation_printable": mesh_ok and current["support_free"],
        "recommended_orientation": best,
        "remediation_hint": hint,
        "orientations": poses,
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
    solid_facets: Sequence[SolidFacet] | None = None,
) -> dict[str, Any]:
    """The agent-facing report. ``checks`` holds the mesh/shape flags:
    ``shape_valid``, ``closed_solid``, ``mesh_solid`` (watertight),
    ``non_manifold`` (bool), ``self_intersections`` (bool),
    ``open_edges`` (count), ``components`` (count). ``bbox`` is
    ``[x_min, y_min, z_min, x_max, y_max, z_max]``. With ``solid_facets``
    (the same facets in full 3D) the report also carries
    ``evaluate_orientations``' fields."""
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
    mesh_ok = not blocking  # everything so far is a defect no rotation can fix

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

    report = {
        "printable": not blocking,
        "requires_supports": requires_supports,
        "warnings": warnings,
        "target": target,
        "dimensions_mm": [round(v, 3) for v in size],
        "build_volume": {**asdict(printer), "fit": fit},
        "overhangs": {**overhangs, "max_overhang_deg": max_overhang_deg},
        "checks": checks,
    }
    if solid_facets:
        report.update(
            evaluate_orientations(solid_facets, printer=printer, max_overhang_deg=max_overhang_deg, mesh_ok=mesh_ok)
        )
    return report


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
# [nx, ny, nz, area, x_min, y_min, z_min, x_max, y_max, z_max] per facet.
facets = [
    [_f.Normal.x, _f.Normal.y, _f.Normal.z, _f.Area]
    + [min(_v[_i] for _v in _f.Points) for _i in range(3)]
    + [max(_v[_i] for _v in _f.Points) for _i in range(3)]
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
    # Inside the agent workspace sandbox (not freecad_output/), so the next
    # steps — slice_stl_to_gcode, then dispatch_to_printer, both sandboxed —
    # can use the exported STL and the G-code sliced next to it.
    prints_dir = session_scoped_dir(file_system._SANDBOX_ROOT / "prints")
    tmp_stl = prints_dir / f"_check_{os.getpid()}.stl"
    stl_path = str(tmp_stl)
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
            tmp_stl.unlink(missing_ok=True)
            return _error(f"check_printability failed: {result['error']}")
        with open(data_path, encoding="utf-8") as fh:
            data = json.load(fh)
    finally:
        try:
            os.unlink(data_path)
        except OSError:
            pass
    final_stl = prints_dir / f"{_safe_name(data['target'])}.stl"
    os.replace(tmp_stl, final_stl)
    stl_path = str(final_stl)
    solid = [SolidFacet((f[0], f[1], f[2]), f[3], (f[4], f[5], f[6]), (f[7], f[8], f[9])) for f in data["facets"]]
    report = build_report(
        target=data["target"],
        checks=data["checks"],
        bbox=data["bbox"],
        facets=(Facet(s.normal[2], s.area, s.lo[2], s.hi[2]) for s in solid),
        printer=printer,
        max_overhang_deg=float(max_overhang_deg),
        solid_facets=solid,
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
    "SolidFacet",
    "build_report",
    "check_printability",
    "evaluate_orientations",
    "find_overhangs",
    "fit_build_volume",
    "printer_profile",
)
