"""Standard parametric part generator — builds exact standard-hardware
solids (a NEMA 17 motor placeholder, a socket head cap screw, a ball
bearing) instead of leaving the LLM to sketch/extrude hallucinated
dimensions for them.

``insert_standard_part`` reuses ``dana.plugins.freecad.engine``'s stateless
FreeCADCmd script-runner directly (``_run_freecad_script``/``_output_path``/
``_BBOX_PRINT``/``_PLACEMENT_SNIPPET``) rather than going through the
``BaseCADEngine`` platform abstraction (``dana.platform.base/win32/mock``) —
this is a FreeCAD-plugin-specific generator built on ``engineering_standards
.py``'s exact dimension tables, the same architectural role as
``py_export.py``/``engineering_standards.py`` themselves, not a primitive
that needs a headless/non-Windows stand-in. Every call is still a single
stateless FreeCADCmd subprocess: file-in (none needed here — every part is
built from scratch) / file-out (one ``.FCStd``), exactly like every
``create_*`` function in ``engine.py``.
"""

from __future__ import annotations

from typing import Any

from dana.plugins.freecad.engine import (
    _BBOX_PRINT,
    _OK_MARKER,
    _PLACEMENT_SNIPPET,
    _auto_show,
    _dry_run_result,
    _error,
    _ok,
    _output_path,
    _run_freecad_script,
)
from dana.plugins.freecad.engineering_standards import (
    get_bearing_geometry,
    get_nema17_dimensions,
    parse_screw_spec,
)
from dana.platform.factory import IS_HF_SPACE
from dana.security.dry_run import is_dry_run_enabled

_PART_TYPES = frozenset({"nema17_motor", "socket_head_screw", "ball_bearing"})

# A square-bodied placeholder (matching NEMA 17's actual square mounting
# face) + a cylindrical pilot boss + a shaft — a Part::Compound rather than
# a fused solid, since these three features aren't meant to be machined as
# one part; they're separate real components co-located for clearance
# checking and assembly mating.
_NEMA17_SCRIPT = """\
import FreeCAD as App
import Part

body_w = {body_width}
body_d = {body_depth}
boss_r = {boss_diameter} / 2.0
boss_h = {boss_depth}
shaft_r = {shaft_diameter} / 2.0
shaft_len = {shaft_length}

body = Part.makeBox(body_w, body_w, body_d, App.Vector(-body_w / 2.0, -body_w / 2.0, 0.0))
boss = Part.makeCylinder(boss_r, boss_h, App.Vector(0.0, 0.0, body_d))
shaft = Part.makeCylinder(shaft_r, shaft_len, App.Vector(0.0, 0.0, body_d + boss_h))

doc = App.newDocument("DanaModel")
obj = doc.addObject("Part::Feature", {name!r})
obj.Shape = Part.makeCompound([body, boss, shaft])
""" + _PLACEMENT_SNIPPET + """\
doc.recompute()
doc.saveAs({out_path!r})
""" + _BBOX_PRINT + """\
print("{marker} path=" + {out_path!r})
"""

# Shank + head as a compound (a screw isn't a single convex solid, and the
# two features are visually/functionally distinct) — the shank sits below
# the origin plane, head above, matching how a screw threads INTO a surface
# at Z=0 with its head standing proud of it.
_SCREW_SCRIPT = """\
import FreeCAD as App
import Part

shank = Part.makeCylinder({nominal_diameter} / 2.0, {length})
head = Part.makeCylinder({head_diameter} / 2.0, {head_height}, App.Vector(0.0, 0.0, {length}))

doc = App.newDocument("DanaModel")
obj = doc.addObject("Part::Feature", {name!r})
obj.Shape = Part.makeCompound([shank, head])
""" + _PLACEMENT_SNIPPET + """\
doc.recompute()
doc.saveAs({out_path!r})
""" + _BBOX_PRINT + """\
print("{marker} path=" + {out_path!r})
"""

# A genuine hollow ring (outer cylinder minus a concentric bore) — unlike
# the other two part types this IS one real solid, built with a single
# boolean cut directly on raw Part shapes (no addObject/Part::Cut feature
# history needed since nothing downstream needs to re-parametrize it).
_BEARING_SCRIPT = """\
import FreeCAD as App
import Part

outer = Part.makeCylinder({outer_diameter} / 2.0, {width})
bore = Part.makeCylinder({bore_diameter} / 2.0, {width})
ring = outer.cut(bore)

doc = App.newDocument("DanaModel")
obj = doc.addObject("Part::Feature", {name!r})
obj.Shape = ring
""" + _PLACEMENT_SNIPPET + """\
doc.recompute()
doc.saveAs({out_path!r})
""" + _BBOX_PRINT + """\
print("{marker} path=" + {out_path!r})
"""


def _resolve_nema17(name: str | None) -> tuple[str, str, dict[str, Any], dict[str, float]]:
    dims = get_nema17_dimensions()
    resolved_name = name or "NEMA17Motor"
    fmt_kwargs = {
        "body_width": dims["body_width_mm"],
        "body_depth": dims["typical_body_depth_mm"],
        "boss_diameter": dims["pilot_boss_diameter_mm"],
        "boss_depth": dims["pilot_boss_depth_mm"],
        "shaft_diameter": dims["shaft_diameter_mm"],
        "shaft_length": dims["default_shaft_length_mm"],
    }
    return resolved_name, _NEMA17_SCRIPT, fmt_kwargs, dims


def _resolve_screw(specification: str, name: str | None) -> tuple[str, str, dict[str, Any], dict[str, float]]:
    geo = parse_screw_spec(specification)  # raises ValueError on a bad spec
    resolved_name = name or f"Screw_{(specification or '').strip().upper()}"
    fmt_kwargs = {
        "nominal_diameter": geo["nominal_diameter_mm"],
        "length": geo["length_mm"],
        "head_diameter": geo["head_diameter_mm"],
        "head_height": geo["head_height_mm"],
    }
    return resolved_name, _SCREW_SCRIPT, fmt_kwargs, geo


def _resolve_bearing(specification: str, name: str | None) -> tuple[str, str, dict[str, Any], dict[str, float]]:
    geo = get_bearing_geometry(specification)  # raises ValueError on an unknown designation
    resolved_name = name or f"Bearing_{(specification or '').strip()}"
    fmt_kwargs = {
        "outer_diameter": geo["outer_diameter_mm"],
        "bore_diameter": geo["bore_diameter_mm"],
        "width": geo["width_mm"],
    }
    return resolved_name, _BEARING_SCRIPT, fmt_kwargs, geo


def insert_standard_part(
    part_type: str,
    specification: str = "",
    name: str | None = None,
    placement: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> str:
    """Generate an exact standard-hardware solid from
    ``engineering_standards.py``'s dimension tables — never a hallucinated
    guess. ``part_type``:

    - ``"nema17_motor"``: a NEMA 17 motor placeholder (square body + pilot
      boss + shaft), ``specification`` unused.
    - ``"socket_head_screw"``: ``specification`` is a spec string like
      ``"M3x12"`` (M<diameter>x<length_mm>).
    - ``"ball_bearing"``: ``specification`` is a bearing designation like
      ``"608"``.
    """
    if IS_HF_SPACE:
        # Unlike every create_* op in engine.py, this function never goes
        # through get_cad_engine()'s Mock/Real switch (see this module's own
        # docstring) — it always shells out to a real FreeCADCmd subprocess.
        # Gated here, at the shell-out itself, so it's closed regardless of
        # which caller reaches it.
        return _error("insert_standard_part is disabled in the hosted cloud demo — it requires the real FreeCAD engine.")
    pt = (part_type or "").strip().lower()
    if pt not in _PART_TYPES:
        return _error(f"insert_standard_part: unknown part_type '{part_type}' — must be one of {', '.join(sorted(_PART_TYPES))}")
    placement = (float(placement[0]), float(placement[1]), float(placement[2]))

    try:
        if pt == "nema17_motor":
            resolved_name, script_template, fmt_kwargs, dims_out = _resolve_nema17(name)
        elif pt == "socket_head_screw":
            resolved_name, script_template, fmt_kwargs, dims_out = _resolve_screw(specification, name)
        else:  # ball_bearing
            resolved_name, script_template, fmt_kwargs, dims_out = _resolve_bearing(specification, name)
    except ValueError as exc:
        return _error(f"insert_standard_part: {exc}")

    if is_dry_run_enabled():
        return _dry_run_result(
            "insert_standard_part", part_type=pt, name=resolved_name, dimensions=dims_out, placement=list(placement)
        )

    out_path = _output_path(resolved_name, ext="FCStd")
    script = script_template.format(
        name=resolved_name, placement=placement, out_path=str(out_path), marker=_OK_MARKER, **fmt_kwargs
    )
    result = _run_freecad_script(script)
    if not result["ok"]:
        return _error(f"insert_standard_part failed: {result['error']}")
    return _ok(
        name=resolved_name,
        type="Part::Feature",
        part_type=pt,
        bounding_box=result.get("bounding_box"),
        dimensions=dims_out,
        placement=list(placement),
        path=str(out_path),
        gui_shown=_auto_show(out_path),
    )


__all__ = ("insert_standard_part",)
