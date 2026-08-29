"""Mesh-to-solid bridge — imports a triangle mesh (``.obj``/``.glb``, e.g.
``dana.tools.image_to_3d.generate_3d_from_image``'s own output) into the
shared session document as a genuine Boundary-Representation (B-Rep) solid,
so it can actually take part in ``perform_freecad_boolean``/
``perform_freecad_edge_operation``/assembly-mate operations the same way any
``create_freecad_*`` primitive can — those all require a real B-Rep solid,
not a raw triangle mesh, which is all a mesh-reconstruction model ever hands
back.

Reuses ``dana.plugins.freecad.engine``'s stateless FreeCADCmd script-runner
directly (``_run_freecad_script``/``_session_document_path``/
``_SESSION_OPEN_SNIPPET``/``_SESSION_SAVE_SNIPPET``/``_SESSION_RESULT_PRINT``)
rather than going through the ``BaseCADEngine`` platform abstraction — same
architectural role and same reasoning as
``dana.plugins.freecad.standard_parts`` (see that module's own docstring):
this is a FreeCAD-plugin-specific operation, not a primitive that needs a
headless/non-Windows mock stand-in.
"""

from __future__ import annotations

from pathlib import Path

from dana.plugins.freecad.engine import (
    _OK_MARKER,
    _OUTPUT_DIR,
    _SESSION_DOCUMENT_NAME,
    _SESSION_OPEN_SNIPPET,
    _SESSION_RESULT_PRINT,
    _SESSION_SAVE_SNIPPET,
    _auto_show,
    _dry_run_result,
    _error,
    _ok,
    _run_freecad_script,
    _session_document_path,
)
from dana.platform.factory import IS_HF_SPACE
from dana.security.dry_run import is_dry_run_enabled

# Sew tolerance (mm) for Part.Shape.makeShapeFromMesh — the standard default
# used by FreeCAD's own "Mesh to Part" conversion recipe; loose enough to
# tolerate the small triangle-soup imprecision typical of an AI-reconstructed
# mesh without merging genuinely distinct nearby features.
_MESH_SEW_TOLERANCE = 0.1

# The standard FreeCAD mesh-to-solid pipeline: import via the Mesh module,
# build a Part.Shape from its triangle topology, sew/solidify it, then hand
# the resulting solid to a normal Part::Feature — exactly what File -> Import
# followed by Part -> "Convert to solid" does in the GUI, just scripted.
# `_mesh_obj` (a Mesh::Feature) is only a transient stepping stone to reach
# `_mesh_obj.Mesh.Topology` and is removed again once the solid exists;
# `Part.Shape()`/the sewn shape are bare geometry-kernel values from the
# `Part` module, never handed to `doc.addObject`, so there is no separate
# "shape document object" alongside it to clean up — only the Mesh::Feature
# is ever actually added to `doc.Objects`.
#
# `makeShapeFromMesh` hands back a bare Compound of independent Faces (0
# Shells) — confirmed live against FreeCAD 1.1.3 (`shape.ShapeType ==
# "Compound"`, `len(shape.Shells) == 0`) — so `Part.makeSolid()` must be
# called on an explicit `Part.Shell(shape.Faces)`, not on that Compound
# directly, or it raises "No shells or compsolids found in shape" outright.
_IMPORT_SOLIDIFY_SCRIPT = ("""\
import FreeCAD as App
import Mesh
import Part

""" + _SESSION_OPEN_SNIPPET + """\
_mesh_obj = doc.addObject("Mesh::Feature", "DanaTempMesh")
_mesh_obj.Mesh = Mesh.Mesh({mesh_path!r})

_shape = Part.Shape()
_shape.makeShapeFromMesh(_mesh_obj.Mesh.Topology, {tolerance}, False)
_shell = Part.Shell(_shape.Faces)
_solid = Part.makeSolid(_shell)
_solid = _solid.removeSplitter()
if _solid.isNull():
    raise RuntimeError(
        "mesh-to-solid conversion produced an empty/invalid solid — the "
        "source mesh may not be watertight"
    )

obj = doc.addObject("Part::Feature", {name!r})
obj.Shape = _solid
doc.removeObject(_mesh_obj.Name)
doc.recompute()
""" + _SESSION_SAVE_SNIPPET + _SESSION_RESULT_PRINT)


def _resolve_mesh_path(mesh_path: str) -> Path | None:
    """Multi-Stage file resolution for a caller-supplied ``mesh_path`` — the
    same 3-candidate poll ``dana.tools.urdf_builder``'s own mesh-existence
    check uses (its own module docstring has the full reasoning): the path
    as given, then its basename under the canonical ``freecad_output/``
    (``_OUTPUT_DIR`` — cwd-independent, where ``generate_3d_from_image``'s
    own output actually lands), then its basename under the process's
    current working directory. Returns the first candidate that's an
    actual file, or ``None`` if none of the three are.
    """
    candidate = Path(mesh_path)
    if candidate.is_file():
        return candidate
    basename = candidate.name
    alt = _OUTPUT_DIR / basename
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

    session_path = _session_document_path()
    script = _IMPORT_SOLIDIFY_SCRIPT.format(
        mesh_path=str(resolved_path),
        name=name,
        tolerance=_MESH_SEW_TOLERANCE,
        session_path=str(session_path),
        session_doc_name=_SESSION_DOCUMENT_NAME,
        marker=_OK_MARKER,
    )
    result = _run_freecad_script(script)
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
