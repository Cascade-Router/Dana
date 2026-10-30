"""Zero-token parametric rebuild: change a dimension, rebuild the session's
model through FreeCADCmd, no LLM call.

The source of truth is the session's ``CadCallLog``: every successful
geometry call is turned back into Universal CAD IR steps (the same
``ir.get_ir_kind(...).from_record`` / ``ir.unroll_composite`` builders live
execution, macro export and skill compilation use), the user's overrides are
patched into those step dicts, and the whole sequence is rendered once with
``render_ir_script(doc_mode="session")`` into a FRESH scratch document. Only
when that succeeds is the scratch document swapped over
``Session_Active.FCStd`` (after a ``.bak`` copy), so an invalid dimension
never touches the live model. Replaying into a fresh document keeps every
object name identical (no FreeCAD auto-suffixing), so later steps' by-name
references, the topology DAG and the object registry stay valid.

Editable fields come from ``skill_compiler.DEFAULT_PARAMETER_FIELDS`` (plus
the face-feature profile's own dimensions) via ``classify_parameters``.

Refuses, instead of silently dropping geometry, when the log contains a
geometry call that has no IR replay (raw scripts, assembly tools, legacy
primitives) — those are reported as ``unsupported``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from dana.plugins.freecad import ir
from dana.plugins.freecad.call_log import CadCallLog, CadCallRecord
from dana.plugins.freecad.py_export import build_replay_steps, composite_replay_steps
from dana.plugins.freecad.skill_compiler import DEFAULT_PARAMETER_FIELDS, classify_parameters

# Editable fields per step kind: the skill compiler's defaults, plus the
# create_freecad_feature_on_face profile (radius/width/length/extent/u/v)
# and the swept/faceted primitives.
_EDITABLE_FIELDS: dict[str, tuple[str, ...]] = {
    **DEFAULT_PARAMETER_FIELDS,
    # Positions and a rotation on top of the compiler's dimensional defaults.
    "face_profile": (*DEFAULT_PARAMETER_FIELDS["face_profile"], "u", "v"),
    "helix": (*DEFAULT_PARAMETER_FIELDS["helix"], "angle_offset"),
}

# Fields that may be zero or negative (positions on a face, a rotation);
# every other editable field is a size or count and must be positive.
_SIGNED_FIELDS = frozenset({"u", "v", "angle_offset"})
_COUNT_FIELDS = frozenset({"occurrences"})

# Kinds whose edge/face selection is by an absolute centroid captured at the
# original call: changing an upstream dimension can move the selection.
_CENTROID_DRESSUP_KINDS = frozenset({"edge_operation"})
DRESSUP_WARNING = "Downstream fillet/chamfer may shift"


@dataclass
class RebuildPlan:
    steps: list[dict[str, Any]]
    # Parallel to ``steps``: the call-log index each step came from.
    record_of_step: list[int]
    unsupported: list[str]


def _composite_steps(rec: CadCallRecord, offset: int) -> list[dict[str, Any]]:
    spec = ir.get_composite_ir(rec.tool_id)
    assert spec is not None
    # A deterministic scope keeps the private intermediates' names stable
    # from one rebuild to the next.
    return composite_replay_steps(rec, spec, start_index=offset * 10, scope=f"rb{offset}")


def plan_rebuild(records: list[CadCallRecord], geometry_tool_ids: frozenset[str]) -> RebuildPlan:
    """IR steps for every successful geometry call in ``records``.
    ``geometry_tool_ids`` names the tools that change geometry; one of those
    without an IR replay makes the log unsupported."""
    steps: list[dict[str, Any]] = []
    record_of_step: list[int] = []
    unsupported: list[str] = []
    for i, rec in enumerate(records):
        if not rec.ok:
            continue
        # Each record owns step indices offset*10 .. offset*10+9, so template
        # variables (derived from the index) never collide across records.
        offset = i + 1
        if ir.get_composite_ir(rec.tool_id) is not None:
            try:
                new_steps = _composite_steps(rec, offset)
            except (TypeError, ValueError):
                unsupported.append(rec.tool_id)
                continue
        elif ir.get_ir_kind(rec.tool_id) is not None:
            new_steps, skipped = build_replay_steps([rec], start_index=offset * 10)
            if skipped:
                unsupported.append(rec.tool_id)
                continue
        else:
            if rec.tool_id in geometry_tool_ids:
                unsupported.append(rec.tool_id)
            continue  # status/export/planning calls don't change geometry
        steps.extend(new_steps)
        record_of_step.extend([i] * len(new_steps))
    return RebuildPlan(steps, record_of_step, sorted(set(unsupported)))


def _unit(step: dict[str, Any], field: str) -> str:
    if field in _COUNT_FIELDS:
        return "count"
    if field == "angle_offset" or (
        step["kind"] == "pipe" and field == "length_or_angle" and step.get("path_type") == "arc"
    ):
        return "deg"
    if step["kind"] == "modify_parameter":
        name = str(step.get("parameter_name") or "").lower()
        return "deg" if "angle" in name else ("mm" if any(k in name for k in ("length", "width", "height", "radius", "depth")) else "")
    return "mm"


def _label(step: dict[str, Any], field: str) -> str:
    if step["kind"] == "modify_parameter":
        return f"{step.get('target_object')}.{step.get('parameter_name')}"
    if step["kind"] == "face_profile":
        return f"{step.get('object_name')} {step.get('face')} feature.{field}"
    if step["kind"] == "edge_operation":
        noun = "radius" if "Fillet" in str(step.get("feature_type")) else "size"
        return f"{step.get('name')}.{noun}"
    return f"{step.get('name')}.{field}"


def parameter_table(records: list[CadCallRecord], geometry_tool_ids: frozenset[str]) -> dict[str, Any]:
    """The editable dimensions of this session's model.

    ``{"params": [{"id", "label", "value", "unit", "min", "warning"?}],
    "unsupported": [tool_id, ...], "reason": str | None}``. ``id`` is
    ``"<call-log index>.<field>"``, stable while the log only grows."""
    plan = plan_rebuild(records, geometry_tool_ids)
    expose: dict[str, list[tuple[int, str]]] = {}
    for pos, step in enumerate(plan.steps):
        for field in _EDITABLE_FIELDS.get(step["kind"], ()):
            value = step.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue  # absent / None (e.g. a circle profile has no width)
            if step["kind"] == "pocket" and field == "depth" and step.get("through_all"):
                continue
            expose[f"{plan.record_of_step[pos]}.{field}"] = [(pos, field)]

    params = []
    for spec in classify_parameters(plan.steps, expose):
        pos, field = spec.locations[0]
        step = plan.steps[pos]
        entry: dict[str, Any] = {
            "id": spec.name,
            "label": _label(step, field),
            "value": spec.default,
            "unit": _unit(step, field),
            "min": None if field in _SIGNED_FIELDS else (1 if field in _COUNT_FIELDS else 0),
        }
        if any(later["kind"] in _CENTROID_DRESSUP_KINDS and later.get("centroid") for later in plan.steps[pos + 1 :]):
            entry["warning"] = DRESSUP_WARNING
        params.append(entry)
    reason = (
        "Rebuild is unavailable: this model includes steps that can't be replayed "
        f"({', '.join(plan.unsupported)}), so rebuilding would drop geometry."
        if plan.unsupported
        else None
    )
    return {"params": params, "unsupported": plan.unsupported, "reason": reason}


def _apply_overrides(plan: RebuildPlan, overrides: dict[str, Any]) -> dict[tuple[int, str], float]:
    """Patches ``overrides`` into the plan's steps; returns them keyed by
    (call-log index, field). Raises ValueError for an unknown id or an
    invalid value."""
    applied: dict[tuple[int, str], float] = {}
    for param_id, raw in overrides.items():
        record_str, _, field = str(param_id).partition(".")
        try:
            record_index = int(record_str)
        except ValueError:
            raise ValueError(f"unknown parameter {param_id!r}") from None
        positions = [
            pos
            for pos, rec_i in enumerate(plan.record_of_step)
            if rec_i == record_index and field in _EDITABLE_FIELDS.get(plan.steps[pos]["kind"], ())
            and isinstance(plan.steps[pos].get(field), (int, float))
        ]
        if not positions:
            raise ValueError(f"unknown parameter {param_id!r}")
        try:
            value = float(raw)
        except (TypeError, ValueError):
            raise ValueError(f"{param_id}: {raw!r} is not a number") from None
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError(f"{param_id}: {raw!r} is not a finite number")
        if field in _COUNT_FIELDS:
            if value < 1 or value != int(value):
                raise ValueError(f"{param_id}: must be a whole number of at least 1")
            value = int(value)
        elif field not in _SIGNED_FIELDS and value <= 0:
            raise ValueError(f"{param_id}: must be greater than 0")
        for pos in positions:
            plan.steps[pos][field] = value
            if plan.steps[pos]["kind"] == "pipe" and field == "pipe_radius":
                plan.steps[pos]["arc_radius"] = ir.pipe_arc_radius(value)  # the bend follows the radius
        applied[(record_index, field)] = value
    return applied


def _write_back(rec: CadCallRecord, field: str, value: float) -> CadCallRecord:
    """The record as if it had been called with ``value``: every place its
    from_record builder may read ``field`` from is updated."""
    arguments = dict(rec.arguments)
    result = dict(rec.result)
    if field in arguments or not result.get("dimensions"):
        arguments[field] = value
    dims = result.get("dimensions")
    if isinstance(dims, dict) and field in dims:
        result["dimensions"] = {**dims, field: value}
    if field in result:
        result[field] = value
    return replace(rec, arguments=arguments, result=result)


def _prepare(
    records: list[CadCallRecord], overrides: dict[str, Any], geometry_tool_ids: frozenset[str]
) -> dict[str, Any]:
    """The checks both rebuild paths share: ``{"plan", "applied"}``, or an
    ``{"ok": False, "error"}`` result to return as is."""
    plan = plan_rebuild(records, geometry_tool_ids)
    if plan.unsupported:
        return {
            "ok": False,
            "error": f"can't rebuild: these steps have no replay ({', '.join(plan.unsupported)})",
            "unsupported": plan.unsupported,
        }
    if not plan.steps:
        return {"ok": False, "error": "nothing to rebuild yet — no geometry has been created this session"}
    try:
        return {"plan": plan, "applied": _apply_overrides(plan, overrides)}
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}


def _patched_records(records: list[CadCallRecord], applied: dict[tuple[int, str], float]) -> list[CadCallRecord]:
    by_record: dict[int, CadCallRecord] = {}
    for (record_index, field), value in applied.items():
        by_record[record_index] = _write_back(by_record.get(record_index, records[record_index]), field, value)
    return [by_record.get(i, rec) for i, rec in enumerate(records)]


def rebuild(
    call_log: CadCallLog,
    overrides: dict[str, Any],
    *,
    session_path: Path,
    geometry_tool_ids: frozenset[str],
    run_script: Any,
    ok_marker: str,
    backup: Any,
    timeout: float,
) -> dict[str, Any]:
    """Rebuilds the model with ``overrides`` applied.

    Success swaps the rebuilt document over ``session_path`` (after a
    ``.bak`` copy via ``backup``) and rewrites the overridden values into
    ``call_log`` so later rebuilds and macro exports match. Failure leaves
    both the document and the log untouched. ``run_script`` is
    ``engine._run_freecad_script`` (one FreeCADCmd subprocess, engine lock).
    """
    records = call_log.records
    prepared = _prepare(records, overrides, geometry_tool_ids)
    if "error" in prepared:
        return prepared
    plan, applied = prepared["plan"], prepared["applied"]

    scratch = session_path.with_name("Session_Rebuild.FCStd")
    scratch.unlink(missing_ok=True)
    final = plan.steps[-1]
    script = ir.render_ir_script(
        plan.steps, doc_mode="session", session_path=str(scratch), final_var=final["var"], marker=ok_marker
    )
    result = run_script(script, timeout=timeout)
    if not result.get("ok") or not scratch.is_file():
        scratch.unlink(missing_ok=True)
        return {"ok": False, "error": f"rebuild failed, model left unchanged: {result.get('error') or 'no document written'}"}

    backup(session_path)
    os.replace(scratch, session_path)

    call_log.replace(_patched_records(records, applied))

    return {
        "ok": True,
        "path": str(session_path),
        "name": result.get("resolved_name") or final.get("name") or final.get("target_object"),
        "bounding_box": result.get("bounding_box"),
        "applied": {f"{i}.{field}": value for (i, field), value in applied.items()},
    }


def rebuild_mock(
    call_log: CadCallLog,
    overrides: dict[str, Any],
    *,
    session_id: str,
    engine: Any,
    handlers: dict[str, Any],
    geometry_tool_ids: frozenset[str],
) -> dict[str, Any]:
    """The headless (trimesh) driver's rebuild — what the Hugging Face Space
    and CI run. There is no FreeCAD document to re-render, so the patched
    call log is replayed through the mock driver's own tool ``handlers``
    (``react_dispatch.TOOL_HANDLERS``) on a cleared mock state for this
    session. A step that fails restores the previous mock state exactly, so
    the model and the log are left as they were."""
    from dana.platform import mock as mock_driver

    records = call_log.records
    prepared = _prepare(records, overrides, geometry_tool_ids)
    if "error" in prepared:
        return prepared
    plan, applied = prepared["plan"], prepared["applied"]
    patched = _patched_records(records, applied)

    saved = mock_driver.snapshot_mock_session(session_id)
    mock_driver.replace_mock_session(session_id, {})
    last: dict[str, Any] = {}
    for i in sorted(set(plan.record_of_step)):
        rec = patched[i]
        try:
            last = handlers[rec.tool_id](dict(rec.arguments), engine, None)
        except Exception as exc:  # noqa: BLE001 — reported, state restored below
            last = {"ok": False, "error": str(exc)}
        if not last.get("ok", True):
            mock_driver.replace_mock_session(session_id, saved)
            return {
                "ok": False,
                "error": f"rebuild failed at {rec.tool_id}, model left unchanged: {last.get('error') or 'failed'}",
            }

    call_log.replace(patched)
    return {
        "ok": True,
        "path": last.get("path"),
        "name": last.get("name"),
        "bounding_box": last.get("bounding_box"),
        "applied": {f"{i}.{field}": value for (i, field), value in applied.items()},
    }


__all__ = ("DRESSUP_WARNING", "RebuildPlan", "parameter_table", "plan_rebuild", "rebuild", "rebuild_mock")
