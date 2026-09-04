"""Universal CAD Intermediate Representation — the ONE step-dict schema and
Jinja2 rendering pipeline shared by all three places that currently
generate FreeCAD script text independently:

1. ``dana.plugins.freecad.engine`` — real-time tool execution, historically
   one bespoke triple-quoted f-string script template per tool_id
   (``_BOX_SCRIPT``, ``_BOOLEAN_CUT_SCRIPT``, ``_EDGE_OP_WHOLE_SCRIPT``, ...).
2. ``dana.plugins.freecad.py_export`` — "Show Your Work", replays a whole
   session's ``CadCallLog`` into one standalone macro.
3. ``dana.plugins.freecad.skill_compiler`` — the Composite Skill Compiler,
   compiles a plan's own call-log slice into a reusable parameterized tool.

(2) and (3) already agreed on a step-dict IR (``{"kind": "box", "var": ...,
"name": ..., "length": ..., ...}``) rendered by a Jinja2 template — this
module is that agreement made explicit and SHARED, so registering a new
tool_id's IR support here is what actually closes the "written in three
places" gap, not an incidental byproduct of it.

**Migration status is registry membership, not a separate flag.**
``is_ir_migrated(tool_id)`` is simply ``tool_id in _IR_REGISTRY`` — there is
no second "migrated: true/false" table that could drift out of sync with
what's actually registered below. A tool_id with an ``IRKindSpec`` here is
migrated everywhere at once (engine.py's adapter, py_export's replay,
skill_compiler's compilation); one without still runs on its own legacy
path in each of those three places, unchanged, until someone registers it.

**Why unification was more tractable than it looked**: the step-rendering
template's closing print block (NAME/BBOX/PATH) already had to match
``engine.py``'s own ``_SESSION_RESULT_PRINT`` contract for the Composite
Skill Compiler's output to be dispatch-compatible — it already does, byte
for byte in spirit (see ``templates/universal_ir.py.jinja2``'s footer vs.
``engine.py``'s ``_SESSION_RESULT_PRINT``). The two script-generation
worlds were never as far apart as three separate implementation SITES
suggested; they just never had a shared home.
"""

from __future__ import annotations

import math
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import jinja2

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_TEMPLATE_NAME = "universal_ir.py.jinja2"

_IDENT_INVALID_RE = re.compile(r"\W")


def safe_var_name(name: str, index: int) -> str:
    """A guaranteed-valid, unique local Python identifier for a step's
    newly-created object — the FreeCAD object Name itself may contain
    characters (spaces, punctuation) that aren't valid as a bare variable.

    Lives here (not privately in ``py_export.py``, where it originated)
    because ``from_record`` builders registered below need it too, and
    ``py_export.py`` now imports IT from here — the reverse direction
    would make ``ir.py`` (the shared foundation every other module in this
    package depends on) depend back on one of its own consumers, a real
    circular-import risk given ``py_export.py`` also needs to call INTO
    this module's registry for its fallback-routing (see
    ``build_replay_steps``).
    """
    candidate = _IDENT_INVALID_RE.sub("_", name or "obj").strip("_") or "obj"
    if candidate[0].isdigit():
        candidate = f"_{candidate}"
    return f"{candidate.lower()}_{index}"


_ENV = jinja2.Environment(
    loader=jinja2.FileSystemLoader(str(_TEMPLATE_DIR)),
    trim_blocks=True,
    lstrip_blocks=True,
)
_ENV.filters["pyrepr"] = repr


def render_ir_script(
    steps: list[dict[str, Any]],
    *,
    doc_mode: Literal["session", "new", "standalone"],
    session_path: str | None = None,
    document_name: str = "DanaModel",
    marker: str = "",
    final_var: str | None = None,
    final_path_expr: str | None = None,
) -> str:
    """Renders ``steps`` (already fully resolved — parameter substitution
    and name-collision-safe prefixing, if any, already applied by the
    caller) through the ONE canonical template.

    ``doc_mode="session"`` opens (or starts, if absent) THIS session's
    shared ``Session_Active.FCStd`` and saves back to it — what real-time
    ``engine.py`` execution and a compiled skill's own ``run(args)`` both
    need. ``doc_mode="new"`` always creates a brand-new, never-saved-back
    standalone document — what ``py_export.py``'s one-off human-readable
    macro export needs (``document_name`` is that document's own name in
    that case; irrelevant for ``"session"``, which always names it from
    the real session). ``doc_mode="standalone"`` opens/creates NO document
    up front at all — a single self-contained step owns its ENTIRE document
    lifecycle itself: opening its own source document by path, building a
    brand-new result document, and ``saveAs``-ing it to a path the step
    itself computed. This is the pre-existing "one object, one file"
    architecture ``align_objects`` still uses (as opposed to the newer
    "everything shares Session_Active.FCStd, by name" architecture every
    kind currently registered in ``_IR_REGISTRY`` now follows, including
    "pattern" — see its own comment below on the Document Lifecycle
    Unification that moved it off this mode) — the Universal IR represents
    that older shape faithfully rather than forcing it into the
    session-document shape it doesn't have, should a future one-object-
    per-file tool need it again.

    ``final_path_expr``, required (and ONLY meaningful) when
    ``doc_mode="standalone"`` and ``marker`` is given: a raw PYTHON SOURCE
    EXPRESSION (not data — never ``pyrepr``'d) naming the variable the
    standalone step's own block bound its output path to (e.g.
    ``"_out_path_1"``), so the closing marker block can print the RIGHT
    path — there is no single shared ``_session_path`` to fall back on in
    this mode.

    ``marker``, when given, appends the NAME/BBOX/PATH print block
    (``engine.py``'s own ``_SESSION_RESULT_PRINT`` contract — see this
    module's docstring) so the caller's own ``_run_freecad_script``-style
    stdout parsing keeps working unchanged. Omitted (``""``) for a plain
    macro export that's meant to be opened by a human, not machine-parsed.
    """
    if doc_mode == "session" and not session_path:
        raise ValueError("render_ir_script(doc_mode='session') requires session_path")
    if doc_mode == "standalone" and marker and not final_path_expr:
        raise ValueError("render_ir_script(doc_mode='standalone') requires final_path_expr when marker is set")
    if not steps:
        raise ValueError("render_ir_script requires at least one step")
    template = _ENV.get_template(_TEMPLATE_NAME)
    return template.render(
        doc_mode=doc_mode,
        session_path=session_path,
        document_name=document_name,
        steps=steps,
        final_var=final_var or steps[-1]["var"],
        final_path_expr=final_path_expr,
        marker=marker,
    )


@dataclass(frozen=True)
class IRKindSpec:
    """One CAD operation's complete IR contract.

    ``from_args`` builds this kind's step dict from the tool's OWN native
    Python arguments — what ``engine.py``'s real-time execution calls with
    (e.g. ``create_box(length, width, height, name, placement)``'s own
    parameters). ``from_record`` builds the SAME shape from a logged
    ``CadCallRecord`` (arguments + the engine's own result payload) — what
    replay (``py_export``) and compilation (``skill_compiler``) call with.
    Both MUST produce a dict a single shared Jinja2 template block (keyed
    by ``kind`` in ``templates/universal_ir.py.jinja2``) can render —
    that's the actual unification; these two functions are just its two
    entry points; ``from_record`` takes ``Any`` (not
    ``dana.plugins.freecad.call_log.CadCallRecord`` directly) purely to
    avoid ``ir.py`` importing ``call_log.py`` for a type-only reference
    that would otherwise be this module's only dependency on it.
    """

    kind: str
    tool_id: str
    from_args: Callable[..., dict[str, Any]]
    from_record: Callable[[Any, int], dict[str, Any]]


@dataclass(frozen=True)
class CompositeIRSpec:
    """One LLM-facing tool call that expands to an ORDERED SEQUENCE of >=2
    atomic IR steps — the Hierarchical/Composite IR node. Unlike
    ``IRKindSpec`` (one tool_id == one atomic FreeCAD operation, one entry
    in the Jinja2 template's ``{% elif step.kind == ... %}`` chain),
    ``CompositeIRSpec`` never gets its own template block: every step
    ``resolve_composite`` returns must already be a plain atomic step dict
    in some OTHER registered kind's shape (typically built by calling that
    kind's own ``IRKindSpec.from_args`` — see ``_feature_on_face_composite``
    below for the worked example), so the template renders a composite
    exactly like any other multi-step sequence and needs no awareness that
    "composite" is even a concept.

    ``resolve_composite`` takes this tool's OWN native keyword arguments
    (same convention as ``IRKindSpec.from_args``) and returns the flat,
    ORDERED list of atomic step dicts — WITHOUT ``index``/``var`` (assigned
    uniformly by ``unroll_composite``, the only caller) — needed to realize
    it. **Contract: the LAST element is the composite's own outward-facing
    result** (its ``name`` is the caller's real, externally-referenceable
    name); every earlier element is a private intermediate this composite
    creates and consumes internally — ``unroll_composite`` namespaces those
    (and rewrites every sibling reference to them) through a fresh
    per-invocation scope so they can never collide with an unrelated object
    or a second invocation of the same composite tool_id sharing the same
    document. A step kind that needs to reference one of those intermediates
    (e.g. a boolean step's ``tool_object``) does so by the SAME bare local
    label ``resolve_composite`` gave that intermediate's own ``name`` —
    ``unroll_composite`` resolves the rewrite structurally, not by string
    convention.
    """

    tool_id: str
    resolve_composite: Callable[..., list[dict[str, Any]]]


_IR_REGISTRY: dict[str, IRKindSpec | CompositeIRSpec] = {}


def register_ir_kind(spec: IRKindSpec) -> None:
    _IR_REGISTRY[spec.tool_id] = spec


def register_composite_ir(spec: CompositeIRSpec) -> None:
    _IR_REGISTRY[spec.tool_id] = spec


def get_ir_kind(tool_id: str) -> IRKindSpec | None:
    spec = _IR_REGISTRY.get(tool_id)
    return spec if isinstance(spec, IRKindSpec) else None


def get_composite_ir(tool_id: str) -> CompositeIRSpec | None:
    spec = _IR_REGISTRY.get(tool_id)
    return spec if isinstance(spec, CompositeIRSpec) else None


def is_ir_migrated(tool_id: str) -> bool:
    return tool_id in _IR_REGISTRY


def migrated_tool_ids() -> frozenset[str]:
    return frozenset(_IR_REGISTRY)


# Reference fields (see ir.py's atomic kinds above) that hold a STRING NAME
# pointing at a sibling step this SAME composite invocation also produced —
# THE DEFAULT for both ``unroll_composite`` (hardcoded composites) and
# ``unroll_steps`` (its general primitive). Deliberately narrow: composite
# unrolling must run during real-time ``engine.py`` execution too, not only
# inside ``skill_compiler``, so the common case (unrolling one of THIS
# module's own registered ``CompositeIRSpec``s, which only ever emit kinds
# already covered here) never needs to import anything from a downstream
# consumer just to get a correct default. ``face_profile`` is intentionally
# ABSENT — its own ``object_name`` always points at an object an EARLIER,
# unrelated tool call created, never at a sibling step this composite itself
# defined, so it must never be rewritten.
#
# skill_compiler.py's own broader ``_REFERENCE_FIELDS`` table is a strict
# SUPERSET (adds "align"/"mate"/"blueprint" — kinds no hardcoded
# ``CompositeIRSpec`` here ever produces, but a compiled skill's OWN
# recorded call log genuinely can) — Recursive Skill Compilation passes that
# broader table explicitly via ``unroll_steps``'s own ``reference_fields``
# argument when inlining a dependency skill's steps, rather than this
# module importing skill_compiler's table (which would invert the package's
# dependency direction) or skill_compiler silently relying on THIS narrower
# table and mis-handling an align/mate/blueprint step it inlines.
_COMPOSITE_REFERENCE_FIELDS: dict[str, tuple[str, ...]] = {
    "boolean": ("base_object", "tool_object"),
    "edge_operation": ("target_object",),
    "modify_parameter": ("target_object",),
    "modify_placement": ("target_object",),
    # Document Lifecycle Unification: "pattern" now resolves its
    # source_object by name against the SAME shared document every other
    # kind above does (see the "pattern" kind's own comment below), so it
    # needs the identical by-name-reference treatment — e.g. Recursive
    # Skill Compilation inlining a dependency skill whose own frozen steps
    # pattern an object THAT SAME dependency created earlier.
    "pattern": ("source_object",),
}


def unroll_steps(
    raw_steps: list[dict[str, Any]],
    *,
    tool_id: str = "",
    start_index: int = 1,
    scope: str | None = None,
    reference_fields: dict[str, tuple[str, ...]] | None = None,
) -> list[dict[str, Any]]:
    """THE unrolling primitive — UUID-scopes and re-indexes an already-
    resolved flat list of atomic step dicts into a form that can be spliced
    into a larger step sequence without colliding with anything else in it.
    Factored out of ``unroll_composite`` (which is now a thin wrapper over
    this — see its own docstring) so it can ALSO drive Recursive Skill
    Compilation: a compiled skill's own frozen ``_STEPS`` is, by
    construction, already a flat list of atomic step dicts (every kind in it
    came from ``ir.get_ir_kind(...).from_record``/``py_export``'s
    ``_STEP_BUILDERS``, never a nested composite of its own — see
    ``dana.plugins.freecad.skill_compiler``'s own module docstring point 2),
    so ``dana.plugins.freecad.skill_compiler``'s recursive inliner can call
    this SAME primitive directly on a dependency skill's ``_STEPS``, with NO
    ``CompositeIRSpec``/registry entry involved at all — a dynamically
    compiled skill is a composite node the ``_IR_REGISTRY`` was never meant
    to hold one entry per instance of (it can be created/deleted at runtime,
    unboundedly many of them), not a gap this primitive itself has.

    Unrolling happens HERE, in Python, before ``steps`` ever reaches
    ``templates/universal_ir.py.jinja2`` — not as recursive Jinja2 macro
    resolution — for the same reason every OTHER step-dict rewrite in this
    package (parameter substitution, ``skill_compiler``'s own name-prefix
    rewrite) already happens in Python: the template's own documented
    contract is "steps have already been through every Python-level rewrite
    the caller needed... this template only renders" (see this module's and
    the template's docstrings). Recursive macro resolution would duplicate
    that rewrite logic a second time in a language with no real namespacing
    or ``uuid`` primitive of its own, purely to avoid one Python function
    call — worse on every axis: harder to unit-test (can't inspect the
    resolved step list without invoking Jinja), harder to debug, and no
    actual UUID-collision-safety benefit, since the template would still
    need Python-supplied randomness for the scope token either way. It also
    means RECURSIVE nesting (a compiled skill inlining a call to ANOTHER
    compiled skill, which inlines a call to a THIRD) needs no new template
    concept at all: each level is resolved to plain atomics in Python before
    the next level ever runs its own ``unroll_steps`` pass over the result,
    so the template only ever sees a flat list of kinds it already renders.

    UUID collision safety: every step in ``raw_steps`` except the LAST (see
    ``CompositeIRSpec``'s contract — a dependency skill's own frozen
    ``_STEPS`` follows the identical convention, since
    ``execute_compiled_steps``'s runtime prefixing already treats its own
    last step as the externally-visible result) has its own ``name``
    rewritten to ``f"_ir_{scope}_{name}"``, where ``scope`` defaults to a
    fresh ``uuid.uuid4().hex[:8]`` PER CALL — matching
    ``skill_compiler.execute_compiled_steps``'s own default-prefix
    convention, and, critically, INDEPENDENT per call site: two separate
    nested calls to the SAME dependency skill within one compilation (e.g.
    two mounting holes) each get their OWN fresh ``scope``, so their
    respective intermediates never collide with EACH OTHER either, not just
    with the outer skill's own names. Every reference field
    (``reference_fields``, defaulting to ``_COMPOSITE_REFERENCE_FIELDS`` —
    pass ``skill_compiler``'s own broader table explicitly when unrolling a
    dependency skill's steps, since those can contain kinds
    ``_COMPOSITE_REFERENCE_FIELDS`` alone doesn't cover, e.g. ``"align"``/
    ``"mate"``/``"blueprint"``) that pointed at one of those local names is
    rewritten identically, so cross-step wiring survives the rename. A field
    that DOESN'T match the rename map (e.g. ``face_profile.object_name``, or
    a dependency skill's step referencing an object the OUTER skill created
    directly, outside the dependency's own call) is left untouched — the
    rename is keyed purely by "was this name DEFINED by one of THESE steps",
    never by field identity, so a reference reaching OUT of this scope can
    never be accidentally caught by it. This is exactly what keeps nested
    scopes from bleeding into each other or into the global document: a
    rename map built from nothing but this call's OWN ``raw_steps`` cannot,
    by construction, contain an outer or sibling scope's names.
    """
    if not raw_steps:
        raise ValueError(f"unroll_steps({tool_id!r}) resolved to zero steps")
    ref_fields = reference_fields if reference_fields is not None else _COMPOSITE_REFERENCE_FIELDS

    scope = scope or uuid.uuid4().hex[:8]
    rename_map: dict[str, str] = {}
    for raw in raw_steps[:-1]:
        local_name = raw.get("name")
        if isinstance(local_name, str) and local_name:
            rename_map[local_name] = f"_ir_{scope}_{local_name}"

    resolved: list[dict[str, Any]] = []
    for offset, raw in enumerate(raw_steps):
        step = dict(raw)
        if step.get("name") in rename_map:
            step["name"] = rename_map[step["name"]]
        for field_name in ref_fields.get(step["kind"], ()):
            ref = step.get(field_name)
            if isinstance(ref, str) and ref in rename_map:
                step[field_name] = rename_map[ref]
        index = start_index + offset
        step["index"] = index
        step.setdefault("var", safe_var_name(step.get("name") or step["kind"], index))
        if tool_id:
            step["tool_id"] = tool_id
        resolved.append(step)
    return resolved


def unroll_composite(
    spec: CompositeIRSpec,
    args: dict[str, Any],
    *,
    start_index: int = 1,
    scope: str | None = None,
) -> list[dict[str, Any]]:
    """The one place a HARDCODED ``CompositeIRSpec`` (a fixed, tool_id-keyed
    entry in ``_IR_REGISTRY``, e.g. ``create_freecad_feature_on_face``) ever
    becomes concrete atomic steps. Every consumer (engine.py's real-time
    execution, py_export's replay, skill_compiler's compilation) calls this,
    never ``spec.resolve_composite`` directly, so the UUID-scoping and
    index/var assignment happen exactly once, uniformly, regardless of which
    of the three call sites is asking.

    A THIN wrapper over ``unroll_steps`` (see its own docstring for the full
    unrolling contract this delegates to unchanged) — this function's own
    remaining job is exactly one thing ``unroll_steps`` itself can't do:
    calling ``spec.resolve_composite(**args)`` to turn the composite's OWN
    native keyword arguments into its raw step list in the first place. A
    dynamically compiled skill has no such callable (its raw steps are
    already-resolved data sitting in a generated module, not a function to
    invoke) — that's the whole reason ``unroll_steps`` exists as its own
    public primitive rather than staying inlined here.
    """
    raw_steps = spec.resolve_composite(**args)
    return unroll_steps(raw_steps, tool_id=spec.tool_id, start_index=start_index, scope=scope)


def _placement_of(result: dict[str, Any]) -> tuple[float, float, float]:
    raw = result.get("placement") or (0.0, 0.0, 0.0)
    x, y, z = raw
    return (float(x), float(y), float(z))


# ---------------------------------------------------------------------
# Registered kinds. Each entry here is the ENTIRE IR contract for its
# tool_id — engine.py's adapter, py_export's replay fallback, and
# skill_compiler's compiler all read the SAME spec. Adding a new tool_id's
# IR support is exactly: write its from_args/from_record pair, add its
# rendering block to templates/universal_ir.py.jinja2, register() it here.
# Nothing else changes in any of the three consuming modules.
#
# Named functions, not lambdas: `var` is derived from the same `name`
# expression the `"name"` field itself uses (via safe_var_name), which
# reads far worse as a one-expression lambda than as two ordinary
# statements.
# ---------------------------------------------------------------------


def _box_from_args(
    *, name: str, length: float, width: float, height: float,
    placement: tuple[float, float, float] = (0.0, 0.0, 0.0), var: str = "obj", index: int = 1,
) -> dict[str, Any]:
    return {
        "kind": "box", "var": var, "name": name,
        "length": float(length), "width": float(width), "height": float(height),
        "placement": tuple(float(v) for v in placement),
        "index": index, "tool_id": "create_freecad_box",
    }


def _box_from_record(rec: Any, index: int) -> dict[str, Any]:
    name = str(rec.result.get("name", "Box"))
    dims = rec.result.get("dimensions") or {}
    return {
        "kind": "box", "var": safe_var_name(name, index), "name": name,
        "length": float(dims.get("length", 0.0)), "width": float(dims.get("width", 0.0)),
        "height": float(dims.get("height", 0.0)), "placement": _placement_of(rec.result),
        "index": index, "tool_id": "create_freecad_box",
    }


register_ir_kind(
    IRKindSpec(kind="box", tool_id="create_freecad_box", from_args=_box_from_args, from_record=_box_from_record)
)


def _cylinder_from_args(
    *, name: str, radius: float, height: float,
    placement: tuple[float, float, float] = (0.0, 0.0, 0.0), var: str = "obj", index: int = 1,
) -> dict[str, Any]:
    return {
        "kind": "cylinder", "var": var, "name": name,
        "radius": float(radius), "height": float(height),
        "placement": tuple(float(v) for v in placement),
        "index": index, "tool_id": "create_freecad_cylinder",
    }


def _cylinder_from_record(rec: Any, index: int) -> dict[str, Any]:
    name = str(rec.result.get("name", "Cylinder"))
    dims = rec.result.get("dimensions") or {}
    return {
        "kind": "cylinder", "var": safe_var_name(name, index), "name": name,
        "radius": float(dims.get("radius", 0.0)), "height": float(dims.get("height", 0.0)),
        "placement": _placement_of(rec.result),
        "index": index, "tool_id": "create_freecad_cylinder",
    }


register_ir_kind(
    IRKindSpec(
        kind="cylinder",
        tool_id="create_freecad_cylinder",
        from_args=_cylinder_from_args,
        from_record=_cylinder_from_record,
    )
)


def _boolean_from_args(
    *, name: str, operation: str, feature_type: str, base_object: str, tool_object: str,
    var: str = "obj", index: int = 1,
) -> dict[str, Any]:
    return {
        "kind": "boolean", "var": var, "name": name, "operation": operation, "feature_type": feature_type,
        "base_object": base_object, "tool_object": tool_object,
        "index": index, "tool_id": "perform_freecad_boolean",
    }


def _boolean_from_record(rec: Any, index: int) -> dict[str, Any]:
    name = str(rec.result.get("name", "Bool"))
    return {
        "kind": "boolean", "var": safe_var_name(name, index), "name": name,
        "operation": str(rec.result.get("operation", "cut")),
        "feature_type": str(rec.result.get("type", "Part::Cut")),
        "base_object": str(rec.arguments.get("base_object", "")),
        "tool_object": str(rec.arguments.get("tool_object", "")),
        "index": index, "tool_id": "perform_freecad_boolean",
    }


register_ir_kind(
    IRKindSpec(
        kind="boolean",
        tool_id="perform_freecad_boolean",
        from_args=_boolean_from_args,
        from_record=_boolean_from_record,
    )
)


def _edge_operation_from_args(
    *, name: str, feature_type: str, target_object: str, value: float,
    centroid: tuple[float, float, float] | None = None, var: str = "obj", index: int = 1,
) -> dict[str, Any]:
    return {
        "kind": "edge_operation", "var": var, "name": name, "feature_type": feature_type,
        "target_object": target_object, "value": float(value),
        "centroid": tuple(float(v) for v in centroid) if centroid else None,
        "index": index, "tool_id": "perform_freecad_edge_operation",
    }


def _edge_operation_from_record(rec: Any, index: int) -> dict[str, Any]:
    name = str(rec.result.get("name", "Edge"))
    centroid = rec.arguments.get("face_centroid")
    return {
        "kind": "edge_operation", "var": safe_var_name(name, index), "name": name,
        "feature_type": str(rec.result.get("type", "Part::Fillet")),
        "target_object": str(rec.arguments.get("target_object", "")),
        "value": float(rec.arguments.get("value", 0.0)),
        "centroid": tuple(float(v) for v in centroid) if centroid else None,
        "index": index, "tool_id": "perform_freecad_edge_operation",
    }


register_ir_kind(
    IRKindSpec(
        kind="edge_operation",
        tool_id="perform_freecad_edge_operation",
        from_args=_edge_operation_from_args,
        from_record=_edge_operation_from_record,
    )
)


def _modify_parameter_from_args(
    *, target_object: str, parameter_name: str, new_value: Any, var: str = "obj", index: int = 1,
) -> dict[str, Any]:
    return {
        "kind": "modify_parameter", "var": var, "target_object": target_object,
        "parameter_name": parameter_name, "new_value": new_value,
        "index": index, "tool_id": "modify_freecad_parameter",
    }


def _modify_parameter_from_record(rec: Any, index: int) -> dict[str, Any]:
    target_object = str(rec.arguments.get("target_object", ""))
    return {
        "kind": "modify_parameter", "var": safe_var_name(target_object, index),
        "target_object": target_object,
        "parameter_name": str(rec.result.get("parameter_name", rec.arguments.get("parameter_name", ""))),
        "new_value": rec.result.get("new_value", rec.arguments.get("new_value", 0.0)),
        "index": index, "tool_id": "modify_freecad_parameter",
    }


register_ir_kind(
    IRKindSpec(
        kind="modify_parameter",
        tool_id="modify_freecad_parameter",
        from_args=_modify_parameter_from_args,
        from_record=_modify_parameter_from_record,
    )
)

# modify_placement — the vector counterpart to modify_parameter's scalar
# setattr: a plain `setattr(obj, name, value)` can't express "replace
# Placement with a new Vector base, optionally preserving the object's
# existing Rotation" (Placement isn't a settable number). ``rotation`` is
# ``None`` for the 3-value [x, y, z] case (translate only — the template
# reads the object's OWN CURRENT Placement.Rotation at render time so a
# move never silently discards prior orientation) or a resolved
# (yaw, pitch, roll) triple for the 6-value case (translate AND replace
# rotation via FreeCAD's own Euler convention). A raw source-expression
# string (engine.py's old ``rotation_expr``) is deliberately NOT part of
# this step dict — every other IR kind here carries plain data through
# ``pyrepr``, never pre-built Python source text; the template branches on
# ``step.rotation`` itself instead.
def _modify_placement_from_args(
    *, target_object: str, x: float, y: float, z: float,
    rotation: tuple[float, float, float] | None = None, var: str = "obj", index: int = 1,
) -> dict[str, Any]:
    return {
        "kind": "modify_placement", "var": var, "target_object": target_object,
        "x": float(x), "y": float(y), "z": float(z),
        "rotation": tuple(float(v) for v in rotation) if rotation is not None else None,
        "index": index, "tool_id": "modify_placement",
    }


def _modify_placement_from_record(rec: Any, index: int) -> dict[str, Any]:
    target_object = str(rec.arguments.get("target_object", ""))
    raw = rec.result.get("new_value", rec.arguments.get("new_value")) or [0.0, 0.0, 0.0]
    vals = [float(v) for v in raw]
    return {
        "kind": "modify_placement", "var": safe_var_name(target_object, index),
        "target_object": target_object,
        "x": vals[0], "y": vals[1], "z": vals[2],
        "rotation": tuple(vals[3:6]) if len(vals) == 6 else None,
        "index": index, "tool_id": "modify_placement",
    }


register_ir_kind(
    IRKindSpec(
        kind="modify_placement",
        tool_id="modify_placement",
        from_args=_modify_placement_from_args,
        from_record=_modify_placement_from_record,
    )
)

# ---------------------------------------------------------------------
# import_and_solidify_mesh — a single atomic operation, NOT a composite,
# despite genuinely needing several distinct FreeCAD state changes (Mesh ->
# raw Shape -> sewn Shell -> Solid -> Part::Feature). The distinction that
# actually decides atomic-vs-composite isn't "how many statements does the
# generated script need" — face_profile's own block is just as long — it's
# "does any intermediate state need its OWN document-visible identity that
# a LATER, separately-dispatched step must reference by name". Here, every
# intermediate (the raw Shape, the Shell, the Solid) is a bare in-memory
# geometry-kernel value, never added to the document at all except the
# transient ``Mesh::Feature`` (added and removed again within this SAME
# step, before any other step could ever observe it) — so one kind, one
# template block, no ``unroll_composite``/UUID-scope story needed at all.
# ---------------------------------------------------------------------

# Sew tolerance (mm) for Part.Shape.makeShapeFromMesh — the standard default
# used by FreeCAD's own "Mesh to Part" conversion recipe; loose enough to
# tolerate the small triangle-soup imprecision typical of an AI-reconstructed
# mesh without merging genuinely distinct nearby features. Moved here from
# ``mesh_ops.py`` (this kind's own default now, not a caller-supplied value).
_MESH_SEW_TOLERANCE_MM = 0.1


def _mesh_solidify_from_args(
    *, name: str, mesh_path: str, tolerance: float = _MESH_SEW_TOLERANCE_MM, var: str = "obj", index: int = 1,
) -> dict[str, Any]:
    return {
        "kind": "mesh_solidify", "var": var, "name": name,
        "mesh_path": str(mesh_path), "tolerance": float(tolerance),
        "index": index, "tool_id": "import_and_solidify_mesh",
    }


def _mesh_solidify_from_record(rec: Any, index: int) -> dict[str, Any]:
    name = str(rec.result.get("name", "Solid"))
    return {
        "kind": "mesh_solidify", "var": safe_var_name(name, index), "name": name,
        "mesh_path": str(rec.result.get("source_mesh") or rec.arguments.get("mesh_path", "")),
        "tolerance": _MESH_SEW_TOLERANCE_MM,
        "index": index, "tool_id": "import_and_solidify_mesh",
    }


register_ir_kind(
    IRKindSpec(
        kind="mesh_solidify",
        tool_id="import_and_solidify_mesh",
        from_args=_mesh_solidify_from_args,
        from_record=_mesh_solidify_from_record,
    )
)

# ---------------------------------------------------------------------
# batch_pattern_array — a single atomic kind ("pattern"). Unified onto the
# shared Session_Active.FCStd document (Document Lifecycle Unification): a
# PRIOR revision of this kind ran under ``doc_mode="standalone"``, opening
# its own source document by PATH and saving copies into a BRAND-NEW,
# separate output file — a real, observed bug, not a hypothetical one: any
# LATER tool in the same session (``perform_freecad_boolean``,
# ``export_freecad_model``, ...), which all resolve their own object
# arguments by NAME against ``Session_Active.FCStd``, could never find the
# array object at all, since it never lived in that document — an "object
# not located" failure every time a pattern's result fed into anything
# downstream. Every other creation kind above (and ``edge_operation``) has
# always built directly inside that ONE shared document; "pattern" now does
# too, via the ordinary ``doc_mode="session"`` path — ``source_object`` (an
# existing object already IN that document) is resolved by name, its copies
# are added to the SAME ``doc``, and the resulting ``Part::Compound`` is
# left there for anything later in the session to reference by name, same
# as a box or a boolean result.
#
# ``source_object`` is a required argument, resolved via the same
# Name/Label/case-insensitive ``resolve_object`` helper every other by-name
# lookup in this IR already uses — never the legacy "first object nothing
# references" heuristic (the "wrong object" bug class already fixed for six
# other engine.py tools).
# ---------------------------------------------------------------------

_PATTERN_TYPES = frozenset({"linear", "grid", "circular"})


def _pattern_offsets(
    pattern_type: str,
    *,
    count_x: int = 1,
    count_y: int = 1,
    spacing_x: float = 0.0,
    spacing_y: float = 0.0,
    count: int = 1,
    radius: float = 0.0,
) -> list[tuple[float, float, float, float]]:
    """Pure arithmetic: ``(dx, dy, dz, z_rotation_deg)`` offsets for every
    copy in a linear/grid/circular pattern — no FreeCAD needed.

    For ``"linear"``/``"grid"``, index 0 is always ``(0, 0, 0, 0)`` — the
    source object's own existing position — so ``count_x=8, count_y=8``
    produces 64 TOTAL placements in one call, not 64 additional ones.
    ``"circular"`` instead places all ``count`` copies on the circle (none
    necessarily coinciding with the source's original position).
    """
    pt = (pattern_type or "").strip().lower()
    if pt == "linear":
        n = max(1, int(count_x))
        return [(i * spacing_x, 0.0, 0.0, 0.0) for i in range(n)]
    if pt == "grid":
        nx, ny = max(1, int(count_x)), max(1, int(count_y))
        return [(i * spacing_x, j * spacing_y, 0.0, 0.0) for j in range(ny) for i in range(nx)]
    if pt == "circular":
        n = max(1, int(count))
        return [
            (
                radius * math.cos(2 * math.pi * i / n),
                radius * math.sin(2 * math.pi * i / n),
                0.0,
                360.0 * i / n,
            )
            for i in range(n)
        ]
    raise ValueError(f"unknown pattern_type: {pattern_type}")


def _pattern_from_args(
    *,
    name: str,
    source_object: str,
    pattern_type: str,
    count_x: int = 1,
    count_y: int = 1,
    spacing_x: float = 0.0,
    spacing_y: float = 0.0,
    count: int = 1,
    radius: float = 0.0,
    var: str = "obj",
    index: int = 1,
) -> dict[str, Any]:
    pt = (pattern_type or "").strip().lower()
    if pt not in _PATTERN_TYPES:
        raise ValueError(f"unknown pattern_type {pattern_type!r} — must be linear, grid, or circular")
    offsets = _pattern_offsets(
        pt, count_x=count_x, count_y=count_y, spacing_x=spacing_x, spacing_y=spacing_y, count=count, radius=radius
    )
    return {
        "kind": "pattern", "var": var, "name": name,
        "source_object": source_object, "offsets": offsets,
        "index": index, "tool_id": "batch_pattern_array",
    }


def _pattern_from_record(rec: Any, index: int) -> dict[str, Any]:
    name = str(rec.result.get("name", "Pattern"))
    return _pattern_from_args(
        name=name,
        source_object=str(rec.arguments.get("source_object", "")),
        pattern_type=str(rec.arguments.get("pattern_type", "linear")),
        count_x=int(rec.arguments.get("count_x", 1)),
        count_y=int(rec.arguments.get("count_y", 1)),
        spacing_x=float(rec.arguments.get("spacing_x") or 0.0),
        spacing_y=float(rec.arguments.get("spacing_y") or 0.0),
        count=int(rec.arguments.get("count", 1)),
        radius=float(rec.arguments.get("radius") or 0.0),
        var=safe_var_name(name, index),
        index=index,
    )


register_ir_kind(
    IRKindSpec(
        kind="pattern",
        tool_id="batch_pattern_array",
        from_args=_pattern_from_args,
        from_record=_pattern_from_record,
    )
)

# ---------------------------------------------------------------------
# create_freecad_feature_on_face — the Hierarchical/Composite IR's first
# real tool_id, and the exact "written in three places" example that
# motivated it. Unlike the five atomic kinds above, this LLM-facing call
# needs TWO real FreeCAD operations to realize: build a tool-solid anchored
# on the target's named face (a NEW atomic kind, "face_profile", ported
# verbatim from the geometry engine.py's own now-retired
# ``_FACE_FEATURE_SCRIPT`` computed — see that kind's template block in
# ``templates/universal_ir.py.jinja2``), then boolean it against the target
# — reusing the ALREADY-REGISTERED "boolean" kind's own ``from_args``
# rather than re-deriving Part::Cut/MultiFuse feature-type selection here a
# second time. ``_feature_on_face_composite`` is registered as a
# ``CompositeIRSpec``, not an ``IRKindSpec`` — it produces two atomic step
# dicts, not one, and never gets a template block of its own.
# ---------------------------------------------------------------------

# Local Coordinate System (LCS) resolution for create_freecad_feature_on_face:
# maps a semantic face label to (a) the WORLD-space unit normal/in-plane axes
# for that face — fixed regardless of the target's actual size, matching
# FreeCAD's own standard-view convention (top=+Z, front=-Y, right=+X) — and
# (b) the Python EXPRESSIONS (evaluated inside the generated script against
# `bb`, the target's real ``Shape.BoundBox`` at RUN time, not IR-build time)
# for that face's center point. Lives here (not in engine.py, where it
# originated) because ``_feature_on_face_composite`` below — this module's
# own composite resolver — needs it directly; engine.py now imports these
# back FROM here (see its own comment) rather than owning a second copy,
# the same upstream/downstream direction every other shared helper in this
# module already follows.
_FACE_NORMAL: dict[str, tuple[float, float, float]] = {
    "top": (0.0, 0.0, 1.0),
    "bottom": (0.0, 0.0, -1.0),
    "front": (0.0, -1.0, 0.0),
    "back": (0.0, 1.0, 0.0),
    "right": (1.0, 0.0, 0.0),
    "left": (-1.0, 0.0, 0.0),
}
_FACE_U_AXIS: dict[str, tuple[float, float, float]] = {
    "top": (1.0, 0.0, 0.0),
    "bottom": (1.0, 0.0, 0.0),
    "front": (1.0, 0.0, 0.0),
    "back": (-1.0, 0.0, 0.0),
    "right": (0.0, 1.0, 0.0),
    "left": (0.0, -1.0, 0.0),
}
_FACE_V_AXIS: dict[str, tuple[float, float, float]] = {
    "top": (0.0, 1.0, 0.0),
    "bottom": (0.0, -1.0, 0.0),
    "front": (0.0, 0.0, 1.0),
    "back": (0.0, 0.0, 1.0),
    "right": (0.0, 0.0, 1.0),
    "left": (0.0, 0.0, 1.0),
}
_FACE_ORIGIN_EXPR: dict[str, tuple[str, str, str]] = {
    "top": ("(bb.XMin + bb.XMax) / 2.0", "(bb.YMin + bb.YMax) / 2.0", "bb.ZMax"),
    "bottom": ("(bb.XMin + bb.XMax) / 2.0", "(bb.YMin + bb.YMax) / 2.0", "bb.ZMin"),
    "front": ("(bb.XMin + bb.XMax) / 2.0", "bb.YMin", "(bb.ZMin + bb.ZMax) / 2.0"),
    "back": ("(bb.XMin + bb.XMax) / 2.0", "bb.YMax", "(bb.ZMin + bb.ZMax) / 2.0"),
    "right": ("bb.XMax", "(bb.YMin + bb.YMax) / 2.0", "(bb.ZMin + bb.ZMax) / 2.0"),
    "left": ("bb.XMin", "(bb.YMin + bb.YMax) / 2.0", "(bb.ZMin + bb.ZMax) / 2.0"),
}

# Nudges the feature's tool-solid this far past the target face's surface
# (mm) before extruding — a "cut" starts this far OUTSIDE the surface and
# extrudes inward, a "add" starts this far INSIDE the surface and extrudes
# outward — so the boolean step always has clean overlapping material to
# work with instead of a razor-thin coincident-face touch.
_FACE_FEATURE_CLEARANCE_MM = 0.5

# operation ("cut"/"add", the LLM-facing vocabulary) -> the "boolean" kind's
# own operation vocabulary, and the +/-1 signs the face_profile step uses to
# decide which side of the face the tool-solid starts on and which way it
# extrudes.
_FACE_FEATURE_BOOLEAN_OP: dict[str, str] = {"cut": "cut", "add": "union"}
_FACE_FEATURE_PUSH_SIGN: dict[str, float] = {"cut": 1.0, "add": -1.0}
_FACE_FEATURE_EXTRUDE_SIGN: dict[str, float] = {"cut": -1.0, "add": 1.0}

# The "boolean" kind's own feature_type vocabulary (mirrors engine.py's
# ``_BOOLEAN_FEATURE_TYPE`` for exactly the two operations feature-on-face
# ever needs) — not imported from engine.py to avoid this module depending
# on one of its own downstream consumers (see this module's own docstring,
# point about ``safe_var_name``, for why that direction is deliberately
# never taken).
_FACE_FEATURE_TYPE: dict[str, str] = {"cut": "Part::Cut", "union": "Part::MultiFuse"}


def _face_axes(
    face: str,
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float], tuple[str, str, str]]:
    """``(normal, u_axis, v_axis, origin_expr)`` for a semantic face label —
    pure lookup, no FreeCAD needed. Raises ``ValueError`` for an unknown
    label."""
    key = (face or "").strip().lower()
    if key not in _FACE_NORMAL:
        raise ValueError(f"unknown face {face!r} — must be one of {sorted(_FACE_NORMAL)}")
    return _FACE_NORMAL[key], _FACE_U_AXIS[key], _FACE_V_AXIS[key], _FACE_ORIGIN_EXPR[key]


def _feature_on_face_composite(
    *,
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
) -> list[dict[str, Any]]:
    """``resolve_composite`` for ``create_freecad_feature_on_face``. Parameter
    mapping from the LLM's own top-level arguments to each child step's
    specific fields is just ordinary Python argument-passing here — no
    separate declarative mapping layer: ``extent``/``u``/``v``/``face``/
    ``shape``/radius`` etc. feed the new ``face_profile`` step directly,
    while ``name``/``operation``/``object_name`` feed the boolean step via
    the ALREADY-REGISTERED "boolean" kind's own ``from_args`` (so
    Part::Cut/MultiFuse selection is derived exactly once, not re-forked
    here).

    Raises ``ValueError`` for any invalid argument (unknown face/operation/
    shape, non-positive dimensions) — callers (engine.py's real-time
    execution) catch this the same way they already caught ``_face_axes``
    raising ``ValueError`` pre-migration, converting it to their own
    ``_error(...)`` payload shape.
    """
    op = (operation or "").strip().lower()
    if op not in _FACE_FEATURE_BOOLEAN_OP:
        raise ValueError(f"unknown operation {operation!r} — must be 'cut' or 'add'")
    shape_key = (shape or "").strip().lower()
    if shape_key not in ("circle", "rectangle"):
        raise ValueError(f"unknown shape {shape!r} — must be 'circle' or 'rectangle'")
    normal, u_axis, v_axis, origin_expr = _face_axes(face)
    extent_f = float(extent)
    if extent_f <= 0:
        raise ValueError("extent must be positive")
    if shape_key == "circle":
        if radius is None or float(radius) <= 0:
            raise ValueError("radius must be positive for shape='circle'")
    else:
        if width is None or length is None or float(width) <= 0 or float(length) <= 0:
            raise ValueError("width/length must be positive for shape='rectangle'")

    # Bare LOCAL label — unroll_composite namespaces it (and every reference
    # to it) through a fresh per-invocation scope before this ever reaches
    # FreeCAD; see CompositeIRSpec's own docstring.
    profile_step: dict[str, Any] = {
        "kind": "face_profile",
        "name": "profile",
        "object_name": object_name,
        "face": face,
        "shape": shape_key,
        "radius": float(radius) if radius is not None else None,
        "width": float(width) if width is not None else None,
        "length": float(length) if length is not None else None,
        "u": float(u),
        "v": float(v),
        "extent": extent_f,
        "normal": normal,
        "u_axis": u_axis,
        "v_axis": v_axis,
        "origin_expr": origin_expr,
        "push_sign": _FACE_FEATURE_PUSH_SIGN[op],
        "extrude_sign": _FACE_FEATURE_EXTRUDE_SIGN[op],
        "clearance": _FACE_FEATURE_CLEARANCE_MM,
    }

    bool_op = _FACE_FEATURE_BOOLEAN_OP[op]
    boolean_step = get_ir_kind("perform_freecad_boolean").from_args(
        name=name or "Feature",
        operation=bool_op,
        feature_type=_FACE_FEATURE_TYPE[bool_op],
        base_object=object_name,
        tool_object="profile",  # same bare local label as profile_step["name"] above
    )
    boolean_step.pop("var", None)
    boolean_step.pop("index", None)

    return [profile_step, boolean_step]


register_composite_ir(
    CompositeIRSpec(
        tool_id="create_freecad_feature_on_face",
        resolve_composite=_feature_on_face_composite,
    )
)


__all__ = (
    "CompositeIRSpec",
    "IRKindSpec",
    "get_composite_ir",
    "get_ir_kind",
    "is_ir_migrated",
    "migrated_tool_ids",
    "register_composite_ir",
    "register_ir_kind",
    "render_ir_script",
    "unroll_composite",
    "unroll_steps",
)
