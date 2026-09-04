"""Frontier 4 — "Show Your Work": render a ``CadCallLog`` into a standalone,
human-editable FreeCAD macro (``.py``).

``dana.plugins.freecad.engine`` executes every tool call in its own throwaway
``FreeCADCmd`` subprocess against its own temp ``.FCStd`` document — real
isolation, but not something a professional engineer wants to read. This
module instead renders the SAME logical sequence of FreeCAD API calls against
ONE shared ``doc``, with later steps referencing earlier objects by their
FreeCAD ``Name`` via ``doc.getObject(...)`` — the natural shape a human would
actually write by hand in the FreeCAD GUI's macro editor.

Each ``CadCallRecord`` supplies both the arguments a call was made with and
the engine's own result payload (resolved default names, coerced dimensions,
absolute placements) — the ``_build_*`` functions below only normalize that
data into template-ready fields; the actual FreeCAD API call *shape* per
operation lives in ``templates/macro_export.py.jinja2``.

Universal CAD IR migration: ``build_replay_steps`` now checks
``dana.plugins.freecad.ir``'s registry FIRST for each record's tool_id,
falling back to this module's own ``_STEP_BUILDERS``/``_build_*`` only for
a tool_id ``ir.py`` hasn't been taught yet — the SAME adapter/fallback
pattern ``engine.py``'s real-time execution uses (see that module's own
migration notes). A kind migrated to ``ir.py`` is thereby replayed through
the SAME step-dict shape/renderer engine.py and the Composite Skill
Compiler use for it — one point of truth, not three. The ``_build_*``
functions below stay exactly as they were for every kind NOT yet migrated;
this module still renders ``render_macro_script``'s OWN output through
``templates/macro_export.py.jinja2`` (not yet retired) either way — only
WHICH function builds each individual step dict changed, not the "one
shared document, human-readable macro" rendering ``render_macro_script``
itself still owns.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jinja2

from dana.paths import DANA_WORKSPACE
from dana.plugins.freecad import ir
from dana.plugins.freecad.call_log import CadCallLog, CadCallRecord

# Reused so the exported script's pipe-elbow bend radius always matches
# whatever the live engine actually built — duplicating these two floats
# instead of importing would silently drift the day engine.py's tuning changes.
from dana.plugins.freecad.engine import (
    _PIPE_ARC_BEND_RADIUS_MULTIPLIER as _ARC_BEND_MULTIPLIER,
    _PIPE_ARC_MIN_BEND_RADIUS as _ARC_MIN_RADIUS,
)

# Same reuse rationale — the blueprint step's view directions/positions and
# template file paths must match techdraw_export.py's own layout exactly.
from dana.plugins.freecad.techdraw_export import _PAGE_TEMPLATES, _VIEW_LAYOUT

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_TEMPLATE_NAME = "macro_export.py.jinja2"
_EXPORT_DIR = DANA_WORKSPACE / "exports"

# Mirrors dana.core.react_dispatch._EXTRUSION_DEFAULT_HALF_WIDTH — the square
# footprint dispatch synthesizes when a face-anchored extrusion call gave it
# only a clicked point, never real profile geometry. Kept as a literal rather
# than an import: dana.plugins.freecad is a lower layer than dana.core and
# shouldn't depend back on the ReAct dispatch module for one constant.
_EXTRUSION_DEFAULT_HALF_WIDTH = 10.0

_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9_-]")

# Re-exported from dana.plugins.freecad.ir (this function ORIGINATED here,
# moved once ir.py's own from_record kind-builders needed it too — see
# ir.safe_var_name's own docstring for why that direction, not the
# reverse, avoids a circular import). Kept as a module-level name so every
# _build_* function below (and any external caller matching this file's
# pre-migration import) is unaffected.
_safe_var_name = ir.safe_var_name


def _placement(result: dict[str, Any]) -> tuple[float, float, float]:
    raw = result.get("placement") or (0.0, 0.0, 0.0)
    x, y, z = raw
    return (float(x), float(y), float(z))


def _build_box(rec: CadCallRecord, index: int) -> dict[str, Any]:
    dims = rec.result.get("dimensions") or {}
    name = str(rec.result.get("name", "Box"))
    return {
        "kind": "box",
        "var": _safe_var_name(name, index),
        "name": name,
        "length": float(dims.get("length", 0.0)),
        "width": float(dims.get("width", 0.0)),
        "height": float(dims.get("height", 0.0)),
        "placement": _placement(rec.result),
    }


def _build_cylinder(rec: CadCallRecord, index: int) -> dict[str, Any]:
    dims = rec.result.get("dimensions") or {}
    name = str(rec.result.get("name", "Cylinder"))
    return {
        "kind": "cylinder",
        "var": _safe_var_name(name, index),
        "name": name,
        "radius": float(dims.get("radius", 0.0)),
        "height": float(dims.get("height", 0.0)),
        "placement": _placement(rec.result),
    }


def _build_pyramid(rec: CadCallRecord, index: int) -> dict[str, Any]:
    dims = rec.result.get("dimensions") or {}
    name = str(rec.result.get("name", "Pyramid"))
    return {
        "kind": "pyramid",
        "var": _safe_var_name(name, index),
        "name": name,
        "length": float(dims.get("length", 0.0)),
        "width": float(dims.get("width", 0.0)),
        "height": float(dims.get("height", 0.0)),
        "placement": _placement(rec.result),
    }


def _build_star_prism(rec: CadCallRecord, index: int) -> dict[str, Any]:
    dims = rec.result.get("dimensions") or {}
    name = str(rec.result.get("name", "StarPrism"))
    return {
        "kind": "star_prism",
        "var": _safe_var_name(name, index),
        "name": name,
        "points": int(dims.get("points", 5)),
        "outer_radius": float(dims.get("outer_radius", 0.0)),
        "inner_radius": float(dims.get("inner_radius", 0.0)),
        "height": float(dims.get("height", 0.0)),
        "placement": _placement(rec.result),
    }


def _build_polygon(rec: CadCallRecord, index: int) -> dict[str, Any]:
    dims = rec.result.get("dimensions") or {}
    name = str(rec.result.get("name", "Polygon"))
    return {
        "kind": "polygon",
        "var": _safe_var_name(name, index),
        "name": name,
        "sides": int(dims.get("sides", 6)),
        "radius": float(dims.get("radius", 0.0)),
        "height": float(dims.get("height", 0.0)),
        "placement": _placement(rec.result),
    }


def _build_extrusion(rec: CadCallRecord, index: int) -> dict[str, Any]:
    profile = rec.arguments.get("profile_points")
    if not profile:
        # Face-anchored call: dana.core.react_dispatch synthesized a square
        # footprint around the clicked point rather than passing real profile
        # geometry (no explicit points ever reached these logged arguments) —
        # same fallback math it used live, so the replay matches what was
        # actually built, not an empty/default extrusion.
        position = rec.arguments.get("target_position") or (0.0, 0.0, 0.0)
        x, y = float(position[0]), float(position[1])
        half = _EXTRUSION_DEFAULT_HALF_WIDTH
        profile = [[x - half, y - half], [x + half, y - half], [x + half, y + half], [x - half, y + half]]
    dims = rec.result.get("dimensions") or {}
    name = str(rec.result.get("name", "ExtrudedPolyline"))
    return {
        "kind": "extrusion",
        "var": _safe_var_name(name, index),
        "name": name,
        "points": [[float(p[0]), float(p[1])] for p in profile],
        "height": float(dims.get("height", rec.arguments.get("height", 0.0))),
    }


def _build_pipe(rec: CadCallRecord, index: int) -> dict[str, Any]:
    dims = rec.result.get("dimensions") or {}
    radius = float(dims.get("pipe_radius", 0.0))
    name = str(rec.result.get("name", "Pipe"))
    return {
        "kind": "pipe",
        "var": _safe_var_name(name, index),
        "name": name,
        "path_type": str(dims.get("path_type", "straight")),
        "pipe_radius": radius,
        "length_or_angle": float(dims.get("length_or_angle", 0.0)),
        "arc_radius": max(radius * _ARC_BEND_MULTIPLIER, _ARC_MIN_RADIUS),
        "placement": _placement(rec.result),
    }


def _build_boolean(rec: CadCallRecord, index: int) -> dict[str, Any]:
    name = str(rec.result.get("name", "Bool"))
    return {
        "kind": "boolean",
        "var": _safe_var_name(name, index),
        "name": name,
        "feature_type": str(rec.result.get("type", "Part::Cut")),
        "operation": str(rec.result.get("operation", "cut")),
        "base_object": str(rec.arguments.get("base_object", "")),
        "tool_object": str(rec.arguments.get("tool_object", "")),
    }


def _build_edge_operation(rec: CadCallRecord, index: int) -> dict[str, Any]:
    centroid = rec.arguments.get("face_centroid")
    name = str(rec.result.get("name", "Edge"))
    return {
        "kind": "edge_operation",
        "var": _safe_var_name(name, index),
        "name": name,
        "feature_type": str(rec.result.get("type", "Part::Fillet")),
        "target_object": str(rec.arguments.get("target_object", "")),
        "value": float(rec.arguments.get("value", 0.0)),
        "centroid": tuple(float(v) for v in centroid) if centroid else None,
    }


def _build_modify_parameter(rec: CadCallRecord, index: int) -> dict[str, Any]:
    return {
        "kind": "modify_parameter",
        "target_object": str(rec.arguments.get("target_object", "")),
        "parameter_name": str(rec.result.get("parameter_name", rec.arguments.get("parameter_name", ""))),
        "new_value": float(rec.result.get("new_value", rec.arguments.get("new_value", 0.0))),
    }


def _build_align(rec: CadCallRecord, index: int) -> dict[str, Any]:
    return {
        "kind": "align",
        "source_object": str(rec.arguments.get("source_object", "")),
        "placement": _placement(rec.result),
    }


def _build_mate(rec: CadCallRecord, index: int) -> dict[str, Any]:
    # Same shape as _build_align: create_assembly_mate translates the
    # moving object's Placement.Base to an absolute final position, exactly
    # like align_freecad_objects — the only difference is which pure
    # function (engine._mate_delta vs. engine._alignment_delta) computed it.
    return {
        "kind": "mate",
        "moving_object": str(rec.arguments.get("moving_obj", "")),
        "fixed_object": str(rec.arguments.get("fixed_obj", "")),
        "mate_type": str(rec.result.get("mate_type", "")),
        "placement": _placement(rec.result),
    }


def _build_standard_part(rec: CadCallRecord, index: int) -> dict[str, Any]:
    dims = rec.result.get("dimensions") or {}
    part_type = str(rec.result.get("part_type", ""))
    name = str(rec.result.get("name", "StandardPart"))
    step: dict[str, Any] = {
        "kind": "standard_part",
        "var": _safe_var_name(name, index),
        "name": name,
        "part_type": part_type,
        "placement": _placement(rec.result),
    }
    if part_type == "nema17_motor":
        step.update(
            body_width=float(dims.get("body_width_mm", 0.0)),
            body_depth=float(dims.get("typical_body_depth_mm", 0.0)),
            boss_diameter=float(dims.get("pilot_boss_diameter_mm", 0.0)),
            boss_depth=float(dims.get("pilot_boss_depth_mm", 0.0)),
            shaft_diameter=float(dims.get("shaft_diameter_mm", 0.0)),
            shaft_length=float(dims.get("default_shaft_length_mm", 0.0)),
        )
    elif part_type == "socket_head_screw":
        step.update(
            nominal_diameter=float(dims.get("nominal_diameter_mm", 0.0)),
            length=float(dims.get("length_mm", 0.0)),
            head_diameter=float(dims.get("head_diameter_mm", 0.0)),
            head_height=float(dims.get("head_height_mm", 0.0)),
        )
    else:  # ball_bearing
        step.update(
            outer_diameter=float(dims.get("outer_diameter_mm", 0.0)),
            bore_diameter=float(dims.get("bore_diameter_mm", 0.0)),
            width=float(dims.get("width_mm", 0.0)),
        )
    return step


def _build_blueprint(rec: CadCallRecord, index: int) -> dict[str, Any]:
    views = list(rec.result.get("views") or [])
    page_size = str(rec.result.get("page_size", "a4")).lower()
    view_specs = [
        (
            name,
            _VIEW_LAYOUT[name.lower()]["direction"],
            _VIEW_LAYOUT[name.lower()]["xdirection"],
            _VIEW_LAYOUT[name.lower()]["slot"][0],
            _VIEW_LAYOUT[name.lower()]["slot"][1],
        )
        for name in views
    ]
    return {
        "kind": "blueprint",
        "target_object": str(rec.arguments.get("object_name", "")),
        "dxf_name": str(rec.result.get("name", "Blueprint")),
        "template_parts": _PAGE_TEMPLATES.get(page_size, _PAGE_TEMPLATES["a4"]),
        "view_specs": view_specs,
    }


def _build_export(rec: CadCallRecord, index: int) -> dict[str, Any]:
    targets = rec.arguments.get("target_objects") or []
    return {
        "kind": "export",
        "targets": [str(t) for t in targets],
        "format": str(rec.result.get("format", rec.arguments.get("format", "stl"))),
        "path": str(rec.result.get("path", "")),
    }


# Every geometry-mutating (or geometry-delivering, for export) tool_id this
# exporter knows how to replay — deliberately narrower than the full
# TOOL_HANDLERS registry in dana.core.react_dispatch: status/vision/camera/
# window tools (system_state, manipulate_camera, resync_workspace, ...) never
# touch FreeCAD geometry, so they're recorded in the log but skipped here.
_STEP_BUILDERS: dict[str, Callable[[CadCallRecord, int], dict[str, Any]]] = {
    "create_freecad_box": _build_box,
    "create_freecad_cylinder": _build_cylinder,
    "create_freecad_pyramid": _build_pyramid,
    "create_freecad_star_prism": _build_star_prism,
    "create_freecad_polygon": _build_polygon,
    "create_freecad_extrusion": _build_extrusion,
    "create_freecad_pipe": _build_pipe,
    "perform_freecad_boolean": _build_boolean,
    "perform_freecad_edge_operation": _build_edge_operation,
    "modify_freecad_parameter": _build_modify_parameter,
    "align_freecad_objects": _build_align,
    "create_assembly_mate": _build_mate,
    "export_freecad_model": _build_export,
    "insert_standard_part": _build_standard_part,
    "generate_2d_blueprint": _build_blueprint,
}

_ENV = jinja2.Environment(
    loader=jinja2.FileSystemLoader(str(_TEMPLATE_DIR)),
    trim_blocks=True,
    lstrip_blocks=True,
)
_ENV.filters["pyrepr"] = repr


def build_replay_steps(
    records: list[CadCallRecord], *, start_index: int = 1
) -> tuple[list[dict[str, Any]], list[str]]:
    """Turns an ordered list of ``CadCallRecord`` into the template-ready
    ``steps``/``skipped`` shape ``render_macro_script`` renders — pulled out
    as its own public function so a second consumer (``dana.plugins.freecad
    .skill_compiler``, which compiles a completed plan's own call-log slice
    into a reusable parameterized tool) can reuse the EXACT SAME
    record-to-step normalization ``_STEP_BUILDERS`` already implements,
    rather than re-deriving it against ``dana.core.react_dispatch
    .TOOL_HANDLERS``'s raw arguments/result payloads a second time.

    ``start_index`` lets a caller number steps starting from where its own
    slice began in the FULL session log (e.g. the call right after
    ``create_plan`` succeeded), rather than always restarting at 1 — purely
    cosmetic (feeds ``step["index"]``/the rendered "# Step N" comment and
    ``_safe_var_name``'s uniqueness suffix), never semantic.

    Universal CAD IR adapter: for each record, ``dana.plugins.freecad.ir``'s
    registry is checked FIRST (``ir.get_ir_kind(rec.tool_id)``) — a
    migrated tool_id's step dict comes from its ``IRKindSpec.from_record``
    (the SAME builder ``engine.py``'s real-time execution and
    ``skill_compiler.py``'s compiler use for that tool_id), never this
    module's own ``_STEP_BUILDERS``. Falls back to ``_STEP_BUILDERS`` only
    for a tool_id ``ir.py`` hasn't been taught yet — the fallback routing
    is per-TOOL_ID, decided fresh for every record, so a session mixing
    migrated and not-yet-migrated tool calls replays correctly either way.
    """
    steps: list[dict[str, Any]] = []
    skipped: list[str] = []
    for offset, rec in enumerate(records, start=start_index):
        if not rec.ok:
            skipped.append(f"Step {offset}: {rec.tool_id} failed — {rec.error}")
            continue
        ir_kind = ir.get_ir_kind(rec.tool_id)
        if ir_kind is not None:
            step = ir_kind.from_record(rec, offset)
        else:
            builder = _STEP_BUILDERS.get(rec.tool_id)
            if builder is None:
                skipped.append(f"Step {offset}: {rec.tool_id} (not a FreeCAD geometry operation)")
                continue
            step = builder(rec, offset)
        step["index"] = offset
        step["tool_id"] = rec.tool_id
        steps.append(step)
    return steps, skipped


def render_macro_script(log: CadCallLog, *, document_name: str = "DanaModel") -> str:
    """Render ``log`` into a standalone FreeCAD macro's full source text."""
    steps, skipped = build_replay_steps(log.records)
    template = _ENV.get_template(_TEMPLATE_NAME)
    return template.render(document_name=document_name, steps=steps, skipped=skipped)


def write_macro_script(
    log: CadCallLog, filename: str = "dana_session_macro", *, document_name: str = "DanaModel"
) -> str:
    """Render ``log`` and write it to ``DANA_WORKSPACE/exports/<filename>.py``."""
    _EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = _SAFE_FILENAME_RE.sub("_", filename or "dana_session_macro").strip("_") or "dana_session_macro"
    out_path = _EXPORT_DIR / f"{safe_name}.py"
    out_path.write_text(render_macro_script(log, document_name=document_name), encoding="utf-8")
    return str(out_path)


__all__ = ("build_replay_steps", "render_macro_script", "write_macro_script")
