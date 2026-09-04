"""Mesh-to-solid bridge — imports a triangle mesh (``.obj``/``.glb``, e.g.
``dana.tools.image_to_3d.generate_3d_from_image``'s own output) into the
shared session document as a genuine Boundary-Representation (B-Rep) solid,
so it can actually take part in ``perform_freecad_boolean``/
``perform_freecad_edge_operation``/assembly-mate operations the same way any
``create_freecad_*`` primitive can — those all require a real B-Rep solid,
not a raw triangle mesh, which is all a mesh-reconstruction model ever hands
back.

Reuses ``dana.plugins.freecad.engine``'s stateless FreeCADCmd script-runner
via the shared ``_execute_ir_tool`` pipeline (the "mesh_solidify" IRKindSpec
registered in ``dana.plugins.freecad.ir``) rather than going through the
``BaseCADEngine`` platform abstraction — same architectural role and same
reasoning as ``dana.plugins.freecad.standard_parts`` (see that module's own
docstring): this is a FreeCAD-plugin-specific operation, not a primitive
that needs a headless/non-Windows mock stand-in.
"""

from __future__ import annotations

from pathlib import Path

from dana.plugins.freecad.engine import (
    _auto_show,
    _dry_run_result,
    _error,
    _execute_ir_tool,
    _ok,
    _session_dir,
)
from dana.platform.factory import IS_HF_SPACE
from dana.security.dry_run import is_dry_run_enabled


def _resolve_mesh_path(mesh_path: str) -> Path | None:
    """Multi-Stage file resolution for a caller-supplied ``mesh_path`` — the
    same 3-candidate poll ``dana.tools.urdf_builder``'s own mesh-existence
    check uses (its own module docstring has the full reasoning): the path
    as given, then its basename under THIS session's own output directory
    (``_session_dir()`` — cwd-independent, where ``generate_3d_from_image``'s
    own output actually lands for this session), then its basename under
    the process's current working directory. Returns the first candidate
    that's an actual file, or ``None`` if none of the three are.
    """
    candidate = Path(mesh_path)
    if candidate.is_file():
        return candidate
    basename = candidate.name
    alt = _session_dir() / basename
    if alt.is_file():
        return alt
    alt2 = Path.cwd() / basename
    if alt2.is_file():
        return alt2
    return None


def import_and_solidify_mesh(mesh_path: str, object_name: str) -> str:
    """Import ``mesh_path`` (a ``.obj``/``.glb``/other Mesh-module-readable
    triangle mesh) into the shared ``Session_Active.FCStd`` document as a
    new B-Rep solid ``Part::Feature`` named ``object_name`` — the bridge a
    ``generate_3d_from_image`` mesh needs before any ``perform_freecad_*``
    operation (boolean, fillet/chamfer, assembly mate) can touch it, since
    every one of those requires a real solid, not a triangle mesh.

    ``mesh_path`` is resolved via the same 3-stage poll (as given, under
    ``freecad_output/``, under the current directory) other Dana tools use
    to catch a caller referencing a file that was never actually produced.

    Migrated to the Universal CAD IR's "mesh_solidify" atomic kind — the
    Mesh -> Shape -> Shell -> Solid -> Part::Feature pipeline is several
    FreeCAD state changes, but every intermediate is a bare in-memory value
    (or a transient document object removed again within this SAME step)
    that no LATER, separately-dispatched step ever needs to reference by
    name — so it's one atomic kind, not a composite (see ir.py's own
    comment on this kind for the general principle that distinguishes the
    two).
    """
    if IS_HF_SPACE:
        # Unlike every create_* op in engine.py's own get_cad_engine() path,
        # this never goes through the Mock/Real platform switch (see this
        # module's own docstring) — it always shells out to a real
        # FreeCADCmd subprocess. Gated here, at the shell-out itself, so
        # it's closed regardless of which caller reaches it.
        return _error("import_and_solidify_mesh is disabled in the hosted cloud demo — it requires the real FreeCAD engine.")

    name = (object_name or "").strip()
    if not name:
        return _error("import_and_solidify_mesh requires a non-empty object_name")

    resolved_path = _resolve_mesh_path(mesh_path)
    if resolved_path is None:
        return _error(
            f"import_and_solidify_mesh: mesh file '{mesh_path}' does not exist "
            "(checked as given, under freecad_output/, and under the current directory)"
        )

    if is_dry_run_enabled():
        return _dry_run_result(
            "import_and_solidify_mesh", name=name, type="Part::Feature", source_mesh=str(resolved_path)
        )

    result, steps, session_path = _execute_ir_tool(
        "import_and_solidify_mesh", name=name, mesh_path=str(resolved_path),
    )
    if not result["ok"]:
        return _error(f"import_and_solidify_mesh failed: {result['error']}")
    return _ok(
        name=result.get("resolved_name") or name,
        type="Part::Feature",
        source_mesh=str(resolved_path),
        bounding_box=result.get("bounding_box"),
        path=str(session_path),
        gui_shown=_auto_show(session_path),
    )


__all__ = ("import_and_solidify_mesh",)
