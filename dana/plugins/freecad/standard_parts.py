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

import re
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
from dana.plugins.freecad.fasteners_bootstrap import ensure_fasteners_workbench
from dana.plugins.freecad.engineering_standards import (
    get_bearing_geometry,
    get_nema17_dimensions,
    parse_screw_spec,
)
from dana.platform.factory import IS_HF_SPACE
from dana.security.dry_run import is_dry_run_enabled

_PART_TYPES = frozenset({"nema17_motor", "socket_head_screw", "ball_bearing", "fastener"})

# Loose enough to cover real designations across standards bodies
# ("ISO4017", "DIN912", "ANSI-B18.2.1") and thread sizes ("M6", "M8x1.25")
# without a hand-maintained enum — FreeCAD's own Fasteners workbench is the
# actual source of truth for which combinations exist, not this module.
# Restricting to this charset is a clean-error convenience (a malformed
# designation fails here with an actionable message instead of a cryptic
# FreeCAD-side one) — NOT an injection guard: every value below is embedded
# into the generated script via `!r` (repr), which is injection-safe for
# any string regardless of content.
_FASTENER_TOKEN_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")

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


# Generates via FreeCAD's own community "Fasteners" workbench
# (https://github.com/shaise/FreeCAD_FastenersWB) instead of this module's
# own hand-built compound-of-primitives approach the other three part types
# use above — real ISO/DIN/ANSI thread geometry (including actual helical
# threads, not a cylinder placeholder), so it needs the workbench installed
# in the target FreeCAD environment.
#
# There is no top-level `Fasteners.makeFastener(type, size, length)` — that
# was this call's original, never-verified guess. Verified against the real
# installed workbench (FreeCAD 1.1.3 + shaise/FreeCAD_FastenersWB, both via
# a direct FreeCADCmd probe and reading FastenersCmd.py/ScrewMaker.py): the
# addon has no `__init__.py` (an implicit namespace package) and its
# internal modules do flat sibling imports (`import FastenerBase`, not
# `from fasteners import FastenerBase`), so BOTH the Mod directory AND the
# `fasteners/` (or `Fasteners/`) directory inside it must be on sys.path —
# not just the Mod directory alone. The actual generation path mirrors what
# FastenersCmd.py's own "Add Screw" GUI command does when invoked with no
# attach-to selection: build a Part::FeaturePython object, hand it to
# FastenersCmd.FSScrewObject(obj, type, None) (None = freestanding, not
# attached to a face — the GUI command instead loops over a live selection,
# which doesn't exist headlessly), then set Diameter and Length as
# ``App::PropertyEnumeration`` STRING values (never a raw float — confirmed
# live: setting Length before a Diameter-triggered recompute, or as '30.0'
# instead of '30', both raise "not part of the enumeration"). Diameter must
# be recomputed before Length is set: FSScrewObject only refreshes Length's
# valid enum options (ScrewMaker.GetAllLengths) reactively inside its own
# execute() hook, which a doc.recompute() call is what actually triggers.
# `FreeCADGui` importing cleanly in headless FreeCADCmd (confirmed live,
# despite FastenerBase.py importing it at module level) is what makes this
# whole approach viable without a display.
_FASTENER_SCRIPT = """\
import sys
import os
import glob
_dana_mod_path = os.environ.get('DANA_FREECAD_MOD_PATH')
if _dana_mod_path:
    if _dana_mod_path not in sys.path:
        sys.path.append(_dana_mod_path)
    for _cand in glob.glob(os.path.join(_dana_mod_path, '*')):
        if os.path.isdir(_cand) and os.path.basename(_cand).lower() == 'fasteners' and _cand not in sys.path:
            sys.path.append(_cand)
import FreeCAD as App
import Part

try:
    import FastenersCmd
except ImportError:
    # flush=True: subprocess.run's captured stdout is a pipe, not a TTY —
    # CPython block-buffers writes to a pipe rather than line-buffering
    # them. FreeCAD's own sys.exit() handling terminates the process
    # without running normal Python interpreter shutdown/flush, so an
    # un-flushed message here is silently lost (confirmed directly against
    # this project's own installed FreeCADCmd — the message never reached
    # the parent process's captured stdout without this).
    print(
        "FASTENERS_WORKBENCH_MISSING: The FreeCAD Fasteners workbench is not "
        "installed in this FreeCAD environment. Install it via Tools -> Addon "
        "Manager -> 'Fasteners' (by shaise), then restart FreeCADCmd.",
        flush=True,
    )
    sys.exit(1)

doc = App.newDocument("DanaModel")
try:
    obj = doc.addObject("Part::FeaturePython", {name!r})
    FastenersCmd.FSScrewObject(obj, {fastener_type!r}, None)
    obj.Diameter = {size!r}
    doc.recompute()  # refresh Length's enum options (if any) to match the new Diameter
    if hasattr(obj, "Length"):
        # Not every fastener_type has one — nuts/washers have no Length
        # property at all (confirmed live: ISO4032 raises AttributeError on
        # a bare obj.Length assignment), so `length` is genuinely a no-op
        # for those, exactly as insert_standard_part's own docstring already
        # documents ("length ... unused for nuts, but still required").
        obj.Length = {length_token!r}
    doc.recompute()  # generate the actual geometry at this Diameter/Length
    if obj.Shape is None or obj.Shape.isNull():
        raise RuntimeError("generated fastener has an empty/null Shape")
except Exception as exc:
    print("FASTENERS_API_ERROR: " + str(exc), flush=True)
    sys.exit(1)
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


def _resolve_fastener(
    fastener_type: str, size: str, length: float | None, name: str | None
) -> tuple[str, str, dict[str, Any], dict[str, float | str]]:
    # FreeCAD's Fasteners workbench keys its type registry on an exact,
    # space-free, uppercase token (e.g. "ISO4017", not "iso4017" or
    # "ISO 4017") — normalize before validating/using it so an LLM-emitted
    # variant resolves to the same canonical designation instead of either
    # failing the token check (a stray space) or silently mismatching the
    # workbench's registry (wrong case).
    ft = str(fastener_type or "").replace(" ", "").upper()
    sz = (size or "").strip()
    if not ft or not _FASTENER_TOKEN_RE.match(ft):
        raise ValueError(
            f"fastener_type must be a standard designation like 'ISO4017' or 'DIN912', got {fastener_type!r}"
        )
    if not sz or not _FASTENER_TOKEN_RE.match(sz):
        raise ValueError(f"size must be a thread designation like 'M6' or 'M8x1.25', got {size!r}")
    if length is None:
        raise ValueError("length (mm) is required for part_type='fastener'")
    try:
        length_f = float(length)
    except (TypeError, ValueError):
        raise ValueError(f"length must be a number, got {length!r}") from None
    if length_f <= 0:
        raise ValueError("length must be a positive number")

    resolved_name = name or f"Fastener_{ft}_{sz}"
    # The Fasteners workbench's Length property is an App::PropertyEnumeration
    # of plain integer-looking strings ('16', '20', '25', '30', .... see
    # ScrewMaker.GetAllLengths) — never a float string like '30.0', which
    # FreeCAD rejects outright as "not part of the enumeration". Formatted
    # here (Dana's own process) rather than in the FreeCADCmd script so the
    # script template only ever does a plain {length_token!r} substitution.
    length_token = str(int(length_f)) if length_f == int(length_f) else str(length_f)
    fmt_kwargs = {"fastener_type": ft, "size": sz, "length_token": length_token}
    dims_out: dict[str, float | str] = {"fastener_type": ft, "size": sz, "length_mm": length_f}
    return resolved_name, _FASTENER_SCRIPT, fmt_kwargs, dims_out


def insert_standard_part(
    part_type: str,
    specification: str = "",
    name: str | None = None,
    placement: tuple[float, float, float] = (0.0, 0.0, 0.0),
    fastener_type: str = "",
    size: str = "",
    length: float | None = None,
) -> str:
    """Generate an exact standard-hardware solid — never a hallucinated
    guess. ``part_type``:

    - ``"nema17_motor"``: a NEMA 17 motor placeholder (square body + pilot
      boss + shaft), from ``engineering_standards.py``'s dimension table;
      ``specification`` unused.
    - ``"socket_head_screw"``: from ``engineering_standards.py``'s
      dimension table; ``specification`` is a spec string like ``"M3x12"``
      (M<diameter>x<length_mm>).
    - ``"ball_bearing"``: from ``engineering_standards.py``'s dimension
      table; ``specification`` is a bearing designation like ``"608"``.
    - ``"fastener"``: real ISO/DIN/ANSI hardware (hex bolts, nuts, socket
      screws, ...) via FreeCAD's own Fasteners workbench — ``fastener_type``
      is a standard designation (e.g. ``"ISO4017"`` for a hex bolt,
      ``"ISO4032"`` for a hex nut), ``size`` is a thread designation (e.g.
      ``"M6"``, ``"M8"``), ``length`` is the fastener length in mm (unused
      for nuts, but still required — pass any positive value).
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
        elif pt == "ball_bearing":
            resolved_name, script_template, fmt_kwargs, dims_out = _resolve_bearing(specification, name)
        else:  # fastener
            resolved_name, script_template, fmt_kwargs, dims_out = _resolve_fastener(fastener_type, size, length, name)
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

    extra_env: dict[str, str] | None = None
    if pt == "fastener":
        # Auto-provision the Fasteners workbench into FreeCAD's user Mod
        # directory if it isn't already installed there (see
        # fasteners_bootstrap.py), then hand its path to the FreeCADCmd
        # subprocess via DANA_FREECAD_MOD_PATH — NOT PYTHONPATH, which
        # FreeCADCmd.exe's embedded Python interpreter ignores on Windows
        # (confirmed live). _FASTENER_SCRIPT's own preamble reads this var
        # and does the sys.path.append itself instead (also locating the
        # actual fasteners/ subdirectory itself, since the workbench's
        # internal modules need that on sys.path too — see the script's own
        # comment). The script's `import FastenersCmd` (with its
        # FASTENERS_WORKBENCH_MISSING fallback message) degrades to exactly
        # the same clean error whether or not this resolves anything.
        mod_dir = ensure_fasteners_workbench()
        if mod_dir is not None:
            extra_env = {"DANA_FREECAD_MOD_PATH": str(mod_dir)}

    result = _run_freecad_script(script, extra_env=extra_env)
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
