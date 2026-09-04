"""Composite Skill Compiler — turns a completed Plan-and-Execute FSM run
(``dana.core.react_dispatch``'s Phase 6 gate: ``create_plan`` -> N geometry
calls -> done) into a single, reusable, parameterized tool.

Architectural stance, up front, since it shapes every function below:

1. **Data source is the CadCallLog, not ``_PLAN_STATE_REGISTRY``.**
   ``_PLAN_STATE_REGISTRY`` (dana.core.react_dispatch) holds only
   ``{"has_plan": bool, "plan_text": str}`` — a flat string for the prompt
   anchor, no structured call data at all. ``session["call_log"]`` (a
   ``dana.plugins.freecad.call_log.CadCallLog``) is the actual ordered,
   structured ``(tool_id, arguments, result)`` history — and its own module
   docstring already says it exists "purely for code generation". This
   module is CadCallLog's SECOND consumer, after
   ``dana.plugins.freecad.py_export`` (the "Show Your Work" macro
   exporter) — and reuses that module's ``build_replay_steps`` directly
   rather than re-deriving the tool_id -> step-dict normalization a second
   time.

2. **Script generation goes through ``dana.plugins.freecad.ir`` — the
   Universal CAD IR — never through ``engine.py``'s own isolated
   ``_BOOLEAN_CUT_SCRIPT``/``_EDGE_OP_*`` f-string templates.** Those
   per-call templates each embed their own document-open/save preamble and
   were never designed to be spliced into one another (different variable
   names, different marker/print conventions, some open a FRESH document
   per call). ``ir.py`` (originally sketched here, then promoted to its
   own module once ``py_export.py`` and ``engine.py``'s real-time
   execution both needed the SAME step-dict schema and renderer) is the
   one place "replay an ordered sequence of calls against ONE shared
   document" is solved — a neutral, JSON-safe step-dict IR (not raw script
   text) rendered by ONE canonical Jinja2 template
   (``templates/universal_ir.py.jinja2``). The "safe concatenation" answer
   is: never concatenate script TEXT — concatenate the STRUCTURED
   step-dict IR, then render it once, through the same renderer every
   other IR consumer in this package uses.

3. **Object-name collisions ("MotorBracket001") are prevented, not
   detected.** Every cross-step reference in the replayed script
   (``base_object``, ``tool_object``, ``target_object``, ...) resolves by
   NAME STRING via ``doc.getObject(name)`` — see
   ``templates/universal_ir.py.jinja2``. If a compiled skill re-used its
   ORIGINAL recorded names (e.g. literal ``"MotorBracket"``) on a second
   invocation in the same session document, FreeCAD would silently
   auto-suffix the colliding object to ``"MotorBracket001"`` while every
   later step's ``doc.getObject("MotorBracket")`` lookup kept referencing
   the ORIGINAL (now differently-purposed) object — a silent, hard-to-debug
   correctness bug, not a crash. Tracking FreeCAD's own post-hoc rename
   after each step (reading ``obj.Name`` back and threading the ACTUAL name
   through every subsequent step) would work but adds real complexity and
   is exactly the class of bug this whole system already exists to prevent
   (see ``dana.core.react_dispatch.resolve_living_leaf`` — this is the same
   "topological amnesia" failure mode, self-inflicted by the compiler this
   time). Instead, ``_apply_name_prefix`` below rewrites EVERY object name
   AND every cross-reference field through a fresh, per-call
   ``name_prefix`` BEFORE the script ever runs, so no name FreeCAD sees in
   a given call was ever used before in that document — the collision
   literally cannot occur, by construction, rather than being detected and
   recovered from after the fact.

4. **The compiled tool's generated file is DATA, not logic.** ``run(args)``
   in a generated skill file does not re-implement Jinja2 rendering or
   subprocess execution — it holds only its own frozen step-IR + parameter
   map, and calls back into this module's ``execute_compiled_steps``. A fix
   or improvement to script generation here therefore applies to EVERY
   previously-compiled skill immediately, not just skills compiled after
   the fix — the opposite of baking the full pipeline into each generated
   file.
"""

from __future__ import annotations

import copy
import json
import re
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from dana.plugins.freecad import ir
from dana.plugins.freecad.call_log import CadCallRecord
from dana.plugins.freecad.py_export import build_replay_steps

# Reused rather than reimplemented — see this module's docstring point 2.
# dana.plugins.freecad.py_export already sets this same precedent (importing
# engine.py's "private" constants across submodules of the same package,
# with a comment explaining why duplicating would silently drift).
from dana.plugins.freecad.engine import (
    _OK_MARKER,
    _error,
    _ok,
    _run_freecad_script,
    _session_document_path,
)

_VALID_SKILL_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")

# Fields in each step-kind's dict (see py_export._build_*) that hold a
# STRING NAME referencing an object an EARLIER step created — never a
# parameter candidate (see classify_parameters), always rewritten by
# _apply_name_prefix. Distinct from `name`/`var`, which every step already
# has for the object IT ITSELF creates.
#
# Built ON TOP of ir._COMPOSITE_REFERENCE_FIELDS (never a hand-copied
# duplicate of it) — those two tables silently drifting apart used to be a
# real, live bug: ir.py gained "modify_placement" as an atomic kind before
# this table did, so a compiled skill containing a modify_placement step
# had its own `target_object` reference left un-prefixed after the first
# invocation, quietly pointing at the ORIGINAL (pre-prefix) name instead of
# the current call's renamed one. Extending ir.py's table here, rather than
# each module keeping an independently-maintained copy, is what actually
# closes that class of drift for good — every kind ir.py's atomic registry
# ever grows is covered here automatically, and this table adds ONLY the
# kinds a compiled skill's own recorded call log can contain that no
# hardcoded CompositeIRSpec here ever produces ("align"/"mate"/"blueprint").
# This is ALSO now the reference-field table Recursive Skill Compilation
# passes to ir.unroll_steps when inlining a dependency skill's own steps
# (see _inline_dependency_call below) — it needs to be the superset either
# way, so there is no second table to keep in sync for that either.
_REFERENCE_FIELDS: dict[str, tuple[str, ...]] = {
    **ir._COMPOSITE_REFERENCE_FIELDS,
    "align": ("source_object",),
    "mate": ("moving_object", "fixed_object"),
    "blueprint": ("target_object",),
}

# Default parameter CANDIDATES per step kind — deliberately narrow and
# conservative (see module docstring's framing of this as inherently a
# judgment call, not a solved problem): only genuinely dimensional fields
# a caller would plausibly want to vary per-instance. Explicitly EXCLUDED:
# `placement` (usually a topological/structural choice, not a size),
# every categorical/structural field (`operation`, `feature_type`, `face`,
# `shape`, `part_type`, `kind`), and every _REFERENCE_FIELDS entry (object
# wiring, never caller-facing). ``compile_call_log_to_skill``'s own
# ``expose`` argument is the authoritative override — an agent or human
# that just watched the plan execute has the semantic context ("that 12mm
# was NEMA 17's known clearance hole, not an arbitrary choice") this
# heuristic structurally cannot have; treat this table as the FALLBACK
# suggestion when nothing more specific is given, not an oracle.
DEFAULT_PARAMETER_FIELDS: dict[str, tuple[str, ...]] = {
    "box": ("length", "width", "height"),
    "cylinder": ("radius", "height"),
    "edge_operation": ("value",),
    "modify_parameter": ("new_value",),
}


@dataclass(frozen=True)
class ParameterSpec:
    """One exposed parameter of a compiled tool — ``locations`` is every
    ``(step_index_in_list, field_name)`` this parameter's runtime value
    gets substituted into (usually one location; a caller-supplied
    ``expose`` mapping may alias the SAME parameter into several locations,
    e.g. one "hole_radius" feeding two symmetric holes)."""

    name: str
    json_type: str  # "number" | "string" | "integer"
    default: Any
    locations: tuple[tuple[int, str], ...]
    description: str = ""


def is_valid_skill_name(skill_name: str) -> bool:
    return bool(_VALID_SKILL_NAME.fullmatch(skill_name or ""))


def slice_records_since_plan(records: list[CadCallRecord]) -> list[CadCallRecord]:
    """Everything AFTER the most recent successful ``create_plan`` call —
    the actual candidate sequence for compilation. Returns ``[]`` (never
    raises) if no ``create_plan`` record exists at all, so a caller can
    surface "nothing to compile — this session's plan gate was never
    opened" as a clean ``ok: False`` rather than compiling an arbitrary,
    un-plan-scoped tail of the session.

    Only the LAST ``create_plan`` matters, not the first: a session that
    replans mid-way (a second ``create_plan`` call replaces the plan, same
    as ``dana.plugins.planning.task_board.create_plan``'s own "always
    replaces, never merges" semantics) should compile from the CURRENT
    plan's own steps, not drag in an earlier, abandoned attempt's calls.
    """
    last_plan_index: int | None = None
    for i, rec in enumerate(records):
        if rec.tool_id == "create_plan" and rec.ok:
            last_plan_index = i
    if last_plan_index is None:
        return []
    return records[last_plan_index + 1 :]


def classify_parameters(
    steps: list[dict[str, Any]],
    expose: dict[str, list[tuple[int, str]]] | None = None,
) -> list[ParameterSpec]:
    """Decides which step fields become the compiled tool's parameters.

    ``expose``, when given, is AUTHORITATIVE: ``{parameter_name: [(step_idx,
    field), ...]}``, where ``step_idx`` indexes into ``steps`` (0-based
    list position, matching the ``steps`` argument here — NOT
    ``step["index"]``, the 1-based display/comment number). This is how a
    caller (typically the LLM itself, via ``compile_plan_as_skill``'s own
    ``expose_parameters`` argument — it has the semantic context this
    function's heuristic structurally lacks) overrides or extends the
    default suggestion, including ALIASING two locations to the same
    parameter (e.g. a symmetric pair of holes sharing one ``hole_radius``).

    ``expose=None`` falls back to ``DEFAULT_PARAMETER_FIELDS``: every field
    listed there, on every step whose ``kind`` has an entry, becomes its
    OWN independent parameter (name ``f"{kind}_{step_idx}_{field}"``).
    """
    if expose is not None:
        specs: list[ParameterSpec] = []
        for param_name, locations in expose.items():
            if not locations:
                continue
            step_idx, field = locations[0]
            if not (0 <= step_idx < len(steps)) or field not in steps[step_idx]:
                continue
            default = steps[step_idx][field]
            specs.append(
                ParameterSpec(
                    name=param_name,
                    json_type=_json_type_for(default),
                    default=default,
                    locations=tuple(locations),
                )
            )
        return specs

    specs = []
    for i, step in enumerate(steps):
        candidate_fields = DEFAULT_PARAMETER_FIELDS.get(step["kind"], ())
        for field_name in candidate_fields:
            if field_name not in step:
                continue
            default = step[field_name]
            param_name = f"{step['kind']}_{i}_{field_name}"
            specs.append(
                ParameterSpec(
                    name=param_name,
                    json_type=_json_type_for(default),
                    default=default,
                    locations=((i, field_name),),
                    description=f"{field_name} for step {step['index']} ({step['tool_id']})",
                )
            )
    return specs


def _resolve_location_token(
    by_index: dict[int, tuple[int, dict[str, Any]]], loc_token: str
) -> tuple[int, str] | None:
    """Resolves one bare ``"step_number.field_name"`` location token (no
    ``=``, no ``,``) to ``(list_position, field)``, or ``None`` if it's
    malformed, the step number doesn't exist, or that step has no such
    field. A LIST-POSITION lookup (``step_number - 1``) is deliberately NOT
    used — a record skipped by ``build_replay_steps`` (a failed call, or a
    tool_id ``py_export`` doesn't know how to replay) leaves a GAP in
    ``step["index"]`` values without shifting ``steps``' own list
    positions down to match, so this searches for the step whose OWN
    ``index`` field equals the token's step number instead of assuming the
    two line up.
    """
    if "." not in loc_token:
        return None
    raw_step_number, _, field = loc_token.partition(".")
    try:
        step_number = int(raw_step_number.strip())
    except ValueError:
        return None
    field = field.strip()
    located = by_index.get(step_number)
    if located is None or field not in located[1]:
        return None
    list_pos, _step = located
    return list_pos, field


def validate_expose_tokens(steps: list[dict[str, Any]], tokens: list[str]) -> list[str]:
    """Fix #5 — Skill Compilation Integrity: strict, pre-flight validation of
    the LLM-facing ``expose_parameters`` token list, run BEFORE
    ``parse_expose_tokens`` ever gets a chance to run — this is the gate
    that decides whether compilation goes ahead AT ALL, never a place that
    silently drops a bad half of an otherwise-good token the way
    ``parse_expose_tokens``'s own resolver deliberately does once
    compilation has already been decided to proceed (see that function's
    own docstring for why lenient-drop is the right behavior for IT).

    Every location token the LLM named must resolve to a REAL step (by its
    1-based ``step_number`` — ``ir.py``'s own ``step["index"]``, not list
    position) and a REAL field already present on that step's own recorded
    IR dict — the exact structured record ``execute_compiled_steps`` will
    later replay from the ``CadCallLog``. An invented field, an
    out-of-range step number, or a malformed token is never silently
    ignored or partially applied here; this returns one human-readable
    message per invalid token/location (empty list == every token is
    valid), so ``compile_call_log_to_skill`` can reject the WHOLE
    compilation with a specific, actionable reason instead of silently
    compiling a skill with fewer parameters than the LLM actually asked for.
    """
    by_index = {step["index"]: step for step in steps}
    errors: list[str] = []
    for raw_token in tokens:
        token = raw_token.strip()
        if not token:
            continue

        if "=" in token:
            param_name, _, locations_part = token.partition("=")
            param_name = param_name.strip()
            loc_tokens = [t.strip() for t in locations_part.split(",") if t.strip()]
            if not param_name:
                errors.append(f"'{token}': missing parameter name before '='")
                continue
            if not loc_tokens:
                errors.append(f"'{token}': no locations given after '='")
                continue
        else:
            loc_tokens = [token]

        for loc_token in loc_tokens:
            if "." not in loc_token:
                errors.append(f"'{loc_token}': expected the form 'step_number.field_name'")
                continue
            raw_step_number, _, field = loc_token.partition(".")
            field = field.strip()
            raw_step_number = raw_step_number.strip()
            try:
                step_number = int(raw_step_number)
            except ValueError:
                errors.append(f"'{loc_token}': '{raw_step_number}' is not a valid step number")
                continue
            step = by_index.get(step_number)
            if step is None:
                errors.append(
                    f"'{loc_token}': no executed step numbered {step_number} "
                    f"(valid step numbers: {sorted(by_index)})"
                )
                continue
            if not field or field not in step:
                errors.append(
                    f"'{loc_token}': step {step_number} ({step['tool_id']}) has no field '{field}' — "
                    f"this parameter was never actually recorded for that step"
                )
    return errors


def parse_expose_tokens(steps: list[dict[str, Any]], tokens: list[str]) -> dict[str, list[tuple[int, str]]]:
    """Translates the LLM-facing ``expose_parameters`` string form into the
    0-based list-position ``{parameter_name: [(step_idx, field), ...]}``
    ``classify_parameters`` expects — the Parameter Aliasing Graph's own
    resolver. Each token is one of two shapes:

    - ``"step_number.field_name"`` (bare, e.g. ``"3.radius"``) — one
      independent parameter, auto-named ``f"{kind}_{step_number}_{field}"``
      (matching ``classify_parameters``'s own default-heuristic naming),
      substituted into that ONE location. Unchanged from the original
      single-location form.
    - ``"parameter_name=step_number.field[,step_number.field, ...]"``
      (e.g. ``"bracket_thickness=1.length,2.height"``) — an explicit ALIAS:
      every listed location is driven by the SAME caller-chosen parameter
      name and, at call time, receives the SAME runtime value (see
      ``execute_compiled_steps``'s generated ``run(args)``, whose
      substitution loop already writes one parameter's value into every
      location in its ``_PARAM_MAP`` entry — this resolver is the only
      piece that was missing to actually EXPRESS a multi-location mapping
      from the tool call arguments; the substitution mechanism already
      supported it).

    A malformed or unresolvable LOCATION within an otherwise-good token is
    dropped, not the whole token — ``"thickness=1.length,9.bogus"`` still
    yields ``thickness`` mapped to just ``1.length`` rather than being
    discarded entirely, since a partially-wrong alias losing only its bad
    half is more useful (and more debuggable from the returned parameter
    count) than silently losing the whole intended parameter. A token that
    resolves to ZERO locations is skipped entirely — a parameter controlling
    nothing) rather than raising, so one bad token can't block every other
    token in the same call from compiling as intended.

    Two tokens that name the SAME explicit ``parameter_name`` are merged
    (their location lists concatenated) rather than the second silently
    overwriting the first — so an agent that lists ``"thickness=1.length"``
    and, later in the same array, ``"thickness=2.height"`` gets the same
    two-location alias as writing it in one token, without needing to know
    that in advance.
    """
    by_index = {step["index"]: (pos, step) for pos, step in enumerate(steps)}
    expose: dict[str, list[tuple[int, str]]] = {}
    for raw_token in tokens:
        token = raw_token.strip()
        if not token:
            continue

        if "=" in token:
            param_name, _, locations_part = token.partition("=")
            param_name = param_name.strip()
            loc_tokens = [t.strip() for t in locations_part.split(",") if t.strip()]
        else:
            param_name = None  # auto-named below, once the first location resolves
            loc_tokens = [token]
        if not param_name and "=" in token:
            continue  # "=1.length" with no name before the '=' — malformed, skip

        resolved: list[tuple[int, str]] = []
        for loc_token in loc_tokens:
            location = _resolve_location_token(by_index, loc_token)
            if location is not None:
                resolved.append(location)
        if not resolved:
            continue

        if param_name is None:
            list_pos, field = resolved[0]
            step = steps[list_pos]
            param_name = f"{step['kind']}_{step['index']}_{field}"

        expose.setdefault(param_name, []).extend(resolved)
    return expose


def _json_type_for(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    return "string"


def _apply_name_prefix(steps: list[dict[str, Any]], prefix: str) -> list[dict[str, Any]]:
    """Rewrites every object name this step SEQUENCE defines or references
    through ``f"{prefix}_{original_name}"`` — see module docstring point 3
    for why this must happen on every call (a fresh ``prefix`` per
    invocation), not once at compile time. Pure structural rewrite over the
    step-dict IR, never touching rendered script TEXT — a name that
    happens to also appear as a coordinate or unrelated string elsewhere
    can never be accidentally matched, unlike a text/regex-based rename
    would risk.
    """
    rename_map: dict[str, str] = {}
    for step in steps:
        original = step.get("name")
        if isinstance(original, str) and original:
            rename_map[original] = f"{prefix}_{original}"

    rewritten: list[dict[str, Any]] = []
    for step in steps:
        step = dict(step)
        if isinstance(step.get("name"), str) and step["name"] in rename_map:
            step["name"] = rename_map[step["name"]]
        for field_name in _REFERENCE_FIELDS.get(step["kind"], ()):
            ref = step.get(field_name)
            if isinstance(ref, str) and ref in rename_map:
                step[field_name] = rename_map[ref]
        rewritten.append(step)
    return rewritten


# ---------------------------------------------------------------------------
# Recursive Skill Compilation — a compiled skill (Skill A) whose own recorded
# CadCallLog includes a call to ANOTHER already-compiled skill (Skill B, e.g.
# "create_standard_hole") is compiled by INLINING Skill B's own frozen
# ``_STEPS`` directly into Skill A's, at COMPILE time — never by having Skill
# A's generated ``run(args)`` call Skill B's tool_id back at RUN time.
#
# Why compile-time inlining, not a runtime call:
#   1. Script generation is a ONE ``ir.render_ir_script`` call over ONE flat
#      step list per invocation (see module docstring point 2) — there is no
#      mechanism for splicing a SECOND FreeCADCmd script/document-open into
#      the middle of that, short of two separate subprocess round trips
#      bridged by an object-name handoff, exactly the fragile "several
#      isolated scripts stitched together" shape ir.py was built to retire.
#   2. A compiled skill's own tool_id can be renamed, deleted, or recompiled
#      out from under a caller at any time (dana.core.skill_loader's hot
#      reload). A runtime call would need to re-resolve and re-validate that
#      tool_id EVERY invocation, with a real "tool_not_found" failure mode if
#      it's gone. Inlining needs Skill B to exist only ONCE, at Skill A's OWN
#      compile time — after that, Skill A's ``_STEPS`` are fully
#      self-contained data; nothing about Skill A's own future dispatch ever
#      depends on Skill B existing again. See ``_render_generated_module``'s
#      own docstring for what ``_DEPENDENCIES`` (recorded regardless) is
#      actually for, given inlining already eliminates this failure mode for
#      Skill A's own dispatch.
#   3. Skill B's own frozen ``_STEPS`` is, by construction, ALREADY a flat
#      list of atomic step dicts (it was built by this exact same compiler,
#      recursively — see ``ir.unroll_steps``'s own docstring) — inlining it
#      needs no new Jinja2 template block, just ``ir.unroll_steps`` run a
#      second (or Nth) time over data that happens to have come from a
#      skill file instead of a hardcoded ``CompositeIRSpec``.
# ---------------------------------------------------------------------------

# ``resolve_skill_steps(tool_id)`` -> ``(steps, param_map, dependencies)`` for
# an existing, loaded user skill's own frozen IR, or ``None`` if ``tool_id``
# isn't a known/resolvable user skill at all (nothing to inline; the caller
# falls back to ``build_replay_steps``'s normal native-tool-id handling).
# ``dependencies`` is that skill's OWN already-recorded ``_DEPENDENCIES``
# tuple (empty if it has none) — threaded through so a THIRD-level nesting
# (Skill A inlines Skill B, which itself inlined Skill C) still produces a
# complete, transitively-accurate dependency lineage for Skill A without
# needing to inline Skill C's steps a second time (Skill B already did that
# at ITS OWN compile time; Skill A only ever sees Skill B's already-flat
# ``_STEPS``). Deliberately a plain injected callable, not an import of
# ``dana.core.skill_loader``/``dana.core.react_dispatch`` here — this
# package (``dana.plugins.freecad``) stays a lower layer those two modules
# depend DOWNWARD on, never the reverse (confirmed: neither currently
# imports anything from ``dana.plugins.freecad`` upward into ``dana.core``
# already the other way around); the resolver closure is built and passed
# in by whichever ``dana.core`` caller actually owns the loaded-skills
# registry (see ``compile_call_log_to_skill``'s own docstring for the call
# site sketch).
SkillStepsResolver = Callable[
    [str], "tuple[list[dict[str, Any]], dict[str, list[tuple[int, str]]], tuple[str, ...]] | None"
]


def _substitute_params(
    steps: list[dict[str, Any]], param_map: dict[str, list[tuple[int, str]]], arguments: dict[str, Any]
) -> list[dict[str, Any]]:
    """Applies Skill B's OWN ``_PARAM_MAP`` against the ACTUAL arguments
    Skill A's call log recorded it being invoked with — the exact same
    substitution loop the generated ``run(args)`` performs at RUN time (see
    ``_render_generated_module``'s generated body), run here instead at
    COMPILE time, against the one concrete call being inlined rather than
    whatever arguments a future caller might pass. A parameter genuinely
    absent from ``arguments`` (the recorded call relied on that parameter's
    own default) is left at Skill B's frozen default value, unchanged —
    matching ``run(args)``'s own ``args.get(_param_name)`` treating
    "absent" and "use the default" identically.
    """
    steps = copy.deepcopy(steps)
    for param_name, locations in param_map.items():
        if param_name not in arguments:
            continue
        value = arguments[param_name]
        for step_idx, field in locations:
            steps[step_idx][field] = value
    return steps


def _inline_dependency_call(
    rec: CadCallRecord,
    resolved: tuple[list[dict[str, Any]], dict[str, list[tuple[int, str]]], tuple[str, ...]],
    *,
    start_index: int,
) -> list[dict[str, Any]]:
    """Turns ONE recorded call to an existing user skill into its fully
    UUID-scoped atomic replacement — see the module-level comment above for
    why this is compile-time inlining, never a runtime call back into that
    skill's own ``tool_id``.

    The nested scope token is fresh PER RECORD (``ir.unroll_steps``'s own
    default), never derived from ``rec`` in any reusable way — two separate
    calls to the SAME dependency skill elsewhere in this SAME compilation
    (e.g. two mounting holes) each get their own independent scope, so
    their respective intermediates can never collide with each other,
    exactly like two sibling calls to a hardcoded ``CompositeIRSpec`` would
    (see ``ir.unroll_steps``'s own docstring on UUID collision safety).

    The nested composite's own LAST step — its externally-visible result —
    is renamed to match what THIS session's call log actually recorded it
    as (``rec.result["name"]``), overriding whatever bare name Skill B's own
    frozen ``_STEPS`` used natively. This is what keeps every OTHER step
    Skill A recorded directly (which already reference that SAME recorded
    name — ``build_replay_steps`` populated their own reference fields from
    ``rec.arguments``/``rec.result`` at THEIR OWN record-processing time,
    entirely independent of this function) resolving correctly with no
    further rewriting needed for that link at all.
    """
    dep_steps, dep_param_map, _dep_dependencies = resolved
    substituted = _substitute_params(dep_steps, dep_param_map, rec.arguments)
    inlined = ir.unroll_steps(
        substituted, tool_id=rec.tool_id, start_index=start_index, reference_fields=_REFERENCE_FIELDS
    )
    recorded_name = str(rec.result.get("name") or "")
    if recorded_name:
        inlined[-1] = {**inlined[-1], "name": recorded_name}
    return inlined


def build_steps_with_dependencies(
    records: list[CadCallRecord],
    *,
    resolve_skill_steps: SkillStepsResolver | None = None,
    start_index: int = 1,
) -> tuple[list[dict[str, Any]], list[str], tuple[str, ...]]:
    """Like ``py_export.build_replay_steps``, but recursion-aware: any
    record whose ``tool_id`` resolves via ``resolve_skill_steps`` is INLINED
    (``_inline_dependency_call`` — one record can therefore expand into
    SEVERAL atomic steps) instead of being handed to
    ``build_replay_steps``'s own native-tool_id-only normalization, which
    has no concept of a user skill at all and would otherwise silently
    ``skip`` it as "not a FreeCAD geometry operation" — the exact gap that
    made calling an existing compiled skill from within a NEW plan silently
    lose that whole step today.

    Every OTHER record (native tool_id, or an unresolvable tool_id when
    ``resolve_skill_steps`` is ``None``/returns ``None`` for it) is handed to
    ``build_replay_steps`` ONE RECORD AT A TIME — reusing its exact
    per-record normalization (``ir.get_ir_kind`` then its own
    ``_STEP_BUILDERS`` fallback) rather than re-deriving it a second time
    here, while still letting THIS function own step-index bookkeeping
    across a sequence that can now expand non-uniformly (one record -> one
    step, ONE record -> many steps, or zero for a skip) — something
    ``build_replay_steps``'s own single whole-list pass has no reason to
    support for its OTHER caller (``py_export``'s "Show Your Work" macro
    export, which has no nested-skill concept and stays entirely unchanged).

    Returns ``(steps, skipped, dependencies)`` — ``dependencies`` is the
    SORTED, DEDUPED union of every inlined record's own ``tool_id`` plus
    every dependency THAT skill itself already declared (transitive
    closure), for ``_render_generated_module``'s own ``_DEPENDENCIES``.
    """
    steps: list[dict[str, Any]] = []
    skipped: list[str] = []
    dependencies: set[str] = set()
    next_index = start_index
    for offset, rec in enumerate(records, start=start_index):
        if not rec.ok:
            skipped.append(f"Step {offset}: {rec.tool_id} failed — {rec.error}")
            continue
        resolved = resolve_skill_steps(rec.tool_id) if resolve_skill_steps is not None else None
        if resolved is not None:
            try:
                inlined = _inline_dependency_call(rec, resolved, start_index=next_index)
            except ValueError:
                skipped.append(f"Step {offset}: {rec.tool_id} (dependency skill has no steps to inline)")
                continue
            steps.extend(inlined)
            next_index += len(inlined)
            dependencies.add(rec.tool_id)
            dependencies.update(resolved[2])
            continue
        one_step, one_skipped = build_replay_steps([rec], start_index=next_index)
        steps.extend(one_step)
        skipped.extend(one_skipped)
        next_index += len(one_step)
    return steps, skipped, tuple(sorted(dependencies))


def execute_compiled_steps(steps: list[dict[str, Any]], *, name_prefix: str | None = None) -> dict[str, Any]:
    """Shared runtime entry point every compiled skill's generated
    ``run(args)`` calls into (see module docstring point 4) — renders
    ``steps`` (already parameter-substituted by the caller) against THIS
    session's shared document and executes it via the same
    ``_run_freecad_script`` primitive every other ``dana.plugins.freecad
    .engine`` function uses, returning a PLAIN DICT payload (never a raw
    JSON string) — ``dana.core.react_dispatch.TOOL_HANDLERS`` entries must
    return a dict; ``_ok``/``_error`` themselves return ``json.dumps(...)``
    strings by design (every OTHER ``engine.py`` function is called through
    a platform driver — e.g. ``dana.platform.win32``'s wrappers — whose
    OWN job is ``json.loads()``-ing that string). ``execute_compiled_steps``
    IS that driver layer for a compiled skill, so it parses the result here
    itself rather than passing the raw string straight through — the exact
    bug this comment exists to prevent someone re-introducing.

    ``name_prefix`` defaults to a fresh short UUID hex per call — never
    reused across invocations of the same OR different compiled skills in
    the same document, which is what actually makes the "MotorBracket001"
    collision class structurally impossible (see module docstring point 3)
    rather than merely unlikely.
    """
    if not steps:
        return json.loads(_error("compiled skill has no steps to execute"))
    prefix = name_prefix or uuid.uuid4().hex[:8]
    prefixed = _apply_name_prefix(steps, prefix)

    script = ir.render_ir_script(
        prefixed,
        doc_mode="session",
        session_path=str(_session_document_path()),
        final_var=prefixed[-1]["var"],
        marker=_OK_MARKER,
    )
    result = _run_freecad_script(script)
    if not result["ok"]:
        return json.loads(_error(f"compiled skill execution failed: {result['error']}"))
    return json.loads(
        _ok(
            name=result.get("resolved_name") or prefixed[-1]["name"],
            path=str(_session_document_path()),
            bounding_box=result.get("bounding_box"),
            name_prefix=prefix,
        )
    )


def _render_generated_module(
    skill_name: str,
    description: str,
    steps: list[dict[str, Any]],
    params: list[ParameterSpec],
    dependencies: tuple[str, ...] = (),
) -> str:
    """The compiled tool's own ``.py`` SOURCE BODY — deliberately tiny: two
    literal data structures (``repr()``'d, so no custom serialization
    format to keep in sync) plus a ``run(args)`` that substitutes
    parameters and delegates everything else to
    ``execute_compiled_steps``. See module docstring point 4 for why the
    actual rendering/execution logic is NOT duplicated here.

    Deliberately does NOT define ``TOOL_SCHEMA`` — this is the
    ``python_code`` argument ``dana.core.skill_loader.save_skill`` accepts,
    and ``save_skill`` ALREADY prepends its own ``TOOL_SCHEMA = <json.dumps
    of the validated schema dict>`` line above whatever ``python_code`` it's
    given (see that function's own ``file_contents`` assembly). Defining it
    again here would either shadow that real one with a stale/empty copy or
    silently rely on assignment order — this body has no schema of its own
    at all, by design, so there is nothing to drift.

    ``dependencies`` (Recursive Skill Compilation — see
    ``build_steps_with_dependencies``) is recorded as a bare ``_DEPENDENCIES``
    tuple literal, PURELY as lineage/provenance metadata: ``run()`` never
    reads it, and never calls back into any of those tool_ids at run time —
    ``_STEPS`` above already has every one of THEIR atomic steps inlined, by
    the time this module is generated, so this skill's own dispatch depends
    on nothing here still existing (see the module-level comment above
    ``build_steps_with_dependencies`` for why that's a real, load-bearing
    property, not an incidental one). What ``_DEPENDENCIES`` IS for: a human
    or the SkillsPlugin UI inspecting ``read_skill_source`` can see this
    skill's real ancestry, and a FUTURE attempt to recompile FROM this
    skill's own call log (nesting it a level deeper still) can check each
    declared dependency still resolves and fail with a clear, specific
    "missing dependency" error at THAT compile time, rather than silently
    compiling an incomplete or stale result.
    """
    param_map_repr = repr({p.name: list(p.locations) for p in params})
    steps_repr = repr(steps)
    dependencies_repr = repr(tuple(dependencies))
    param_reads = "\n".join(
        f"    {p.name} = args.get({p.name!r}, {p.default!r})" for p in params
    )
    param_names = ", ".join(p.name for p in params) or "no exposed parameters"
    deps_note = f" Depends on: {', '.join(dependencies)}." if dependencies else ""

    return (
        f'"""Compiled skill: {skill_name}. Auto-generated by '
        f"dana.plugins.freecad.skill_compiler from a completed Plan-and-Execute run.\n"
        f'Exposed parameters: {param_names}.{deps_note}"""\n\n'
        f"from dana.plugins.freecad.skill_compiler import execute_compiled_steps\n\n"
        f"_STEPS = {steps_repr}\n\n"
        f"_PARAM_MAP = {param_map_repr}\n\n"
        f"# Lineage only -- every step above is already fully inlined/self-contained;\n"
        f"# run() below never reads this or calls back into any of these tool_ids.\n"
        f"_DEPENDENCIES = {dependencies_repr}\n\n\n"
        f"def run(args: dict) -> dict:\n"
        f"{param_reads if param_reads else '    pass'}\n"
        f"    import copy\n"
        f"    steps = copy.deepcopy(_STEPS)\n"
        f"    for _param_name, _locations in _PARAM_MAP.items():\n"
        f"        _value = args.get(_param_name)\n"
        f"        if _value is None:\n"
        f"            continue\n"
        f"        for _step_idx, _field in _locations:\n"
        f"            steps[_step_idx][_field] = _value\n"
        f"    name_prefix = args.get('name_prefix')\n"
        f"    return execute_compiled_steps(steps, name_prefix=name_prefix)\n"
    )


def _build_tool_schema(skill_name: str, description: str, params: list[ParameterSpec]) -> dict[str, Any]:
    properties: dict[str, Any] = {
        p.name: {"type": p.json_type, "description": p.description or p.name} for p in params
    }
    properties["name_prefix"] = {
        "type": "string",
        "description": (
            "Optional. Leave unset — a fresh unique prefix is generated automatically so this "
            "compiled tool never collides with objects from a previous call."
        ),
    }
    return {
        "type": "function",
        "function": {
            "name": skill_name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": [p.name for p in params],
            },
        },
    }


def compile_call_log_to_skill(
    records: list[CadCallRecord],
    *,
    skill_name: str,
    description: str,
    expose_tokens: list[str] | None = None,
    resolve_skill_steps: SkillStepsResolver | None = None,
) -> dict[str, Any]:
    """Pure compiler entry point: ``records`` (typically
    ``slice_records_since_plan(session["call_log"].records)``) in,
    ``{"ok": True, "python_code": str, "schema": dict}`` out — exactly the
    two positional inputs ``dana.core.skill_loader.save_skill(skill_name,
    python_code, schema)`` already accepts. Deliberately does NOT write to
    disk, touch ``TOOL_HANDLERS``, or touch any tool registry itself — see
    ``dana.core.react_dispatch._tool_compile_plan_as_skill`` for the
    handler that calls this and then hands the result to
    ``save_skill``/``refresh_user_skills``, reusing that ALREADY-WIRED
    hot-reload pipeline rather than standing up a parallel one.

    ``expose_tokens`` is the LLM-facing ``"step_number.field_name"`` form
    (see ``parse_expose_tokens``) — ``None``/empty falls back to
    ``classify_parameters``'s own default heuristic.

    ``resolve_skill_steps`` (Recursive Skill Compilation) is an OPTIONAL
    injected callable — see ``SkillStepsResolver``'s own docstring for why
    this stays a plain parameter rather than an import of
    ``dana.core.skill_loader``/``dana.core.react_dispatch`` from this
    module. ``None`` (the default) reproduces the exact PRE-recursive
    behavior: a call to an existing user skill found in ``records`` is
    silently ``skip``ped, same as any other tool_id this module doesn't
    recognize. A real caller (``dana.core.react_dispatch
    ._tool_compile_plan_as_skill``) supplies a closure over its own loaded-
    skills registry, e.g.::

        def _resolve_skill_steps(tool_id):
            entry = _LOADED_SKILL_MODULES.get(tool_id)  # populated by refresh_user_skills
            if entry is None:
                return None
            module = entry["module"]
            steps = getattr(module, "_STEPS", None)
            param_map = getattr(module, "_PARAM_MAP", None)
            if steps is None or param_map is None:
                return None  # a user skill file that ISN'T one of this compiler's own outputs
            dependencies = tuple(getattr(module, "_DEPENDENCIES", ()))
            return steps, param_map, dependencies
    """
    if not is_valid_skill_name(skill_name):
        return {"ok": False, "error": f"invalid skill_name {skill_name!r} — must be lowercase snake_case"}
    if not records:
        return {
            "ok": False,
            "error": "nothing to compile — no successful tool calls since the last create_plan",
        }

    steps, skipped, dependencies = build_steps_with_dependencies(
        records, resolve_skill_steps=resolve_skill_steps, start_index=1
    )
    if not steps:
        return {
            "ok": False,
            "error": "no compilable geometry steps found",
            "skipped": skipped,
        }

    if expose_tokens:
        # Fix #5 — Skill Compilation Integrity: reject the WHOLE compilation
        # (never silently drop the bad half — see validate_expose_tokens's
        # own docstring) the instant expose_parameters hallucinates a step
        # number or a field the recorded CadCallLog never actually produced.
        invalid = validate_expose_tokens(steps, expose_tokens)
        if invalid:
            return {
                "ok": False,
                "error": (
                    "compile_plan_as_skill: expose_parameters contains invalid mapping(s) — "
                    "compilation rejected: " + "; ".join(invalid)
                ),
                "invalid_expose_tokens": invalid,
            }

    expose = parse_expose_tokens(steps, expose_tokens) if expose_tokens else None
    params = classify_parameters(steps, expose)
    python_code = _render_generated_module(skill_name, description, steps, params, dependencies)
    schema = _build_tool_schema(skill_name, description, params)
    return {
        "ok": True,
        "skill_name": skill_name,
        "python_code": python_code,
        "schema": schema,
        "parameter_count": len(params),
        "step_count": len(steps),
        "dependencies": list(dependencies),
        "skipped": skipped,
    }


__all__ = (
    "DEFAULT_PARAMETER_FIELDS",
    "ParameterSpec",
    "SkillStepsResolver",
    "build_steps_with_dependencies",
    "classify_parameters",
    "compile_call_log_to_skill",
    "execute_compiled_steps",
    "is_valid_skill_name",
    "parse_expose_tokens",
    "slice_records_since_plan",
    "validate_expose_tokens",
)
