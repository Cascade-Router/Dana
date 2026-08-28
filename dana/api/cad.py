"""REST API for the frontend CAD tab's toolbar — artifact listing/download
and an on-demand "open in the real FreeCAD GUI" action.

Every generated file lives under one of exactly two directories (see
``dana.plugins.freecad.engine``): ``freecad_output/`` (native per-object
``.FCStd``/``.stl`` from create_freecad_*/perform_freecad_boolean/...) and
``exports/`` (explicit ``export_freecad_model`` STEP/STL output). Both are
re-declared here as plain ``Path`` literals (same precedent as
``dana.plugins.freecad.py_export``/``techdraw_export``, which each declare
their own copy rather than importing engine.py's underscore-prefixed
module attribute across modules) rather than reaching into engine.py's
private globals.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from dana.api import artifacts_registry
from dana.paths import DANA_WORKSPACE

router = APIRouter(prefix="/api/cad", tags=["cad"])

_FREECAD_OUTPUT_DIR = DANA_WORKSPACE / "freecad_output"
_FREECAD_EXPORT_DIR = DANA_WORKSPACE / "exports"

# Only these two dirs, only these extensions — an artifact "download by
# filename" route is a path built from user-controlled input, so the
# directory allowlist below (never an arbitrary path) is what makes path
# traversal a structural impossibility rather than a string-sanitizing check.
_ARTIFACT_DIRS: tuple[Path, ...] = (_FREECAD_OUTPUT_DIR, _FREECAD_EXPORT_DIR)
_ARTIFACT_EXTENSIONS = frozenset({".step", ".stp", ".stl", ".fcstd", ".urdf", ".glb", ".obj"})

_MEDIA_TYPES = {
    ".step": "model/step",
    ".stp": "model/step",
    ".stl": "model/stl",
    ".fcstd": "application/octet-stream",
    ".urdf": "application/xml",
    # generate_3d_from_image's own output formats (dana.tools.image_to_3d).
    ".glb": "model/gltf-binary",
    ".obj": "model/obj",
}


def _list_artifacts() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for directory in _ARTIFACT_DIRS:
        if not directory.is_dir():
            continue
        for path in directory.iterdir():
            if not path.is_file() or path.suffix.lower() not in _ARTIFACT_EXTENSIONS:
                continue
            stat = path.stat()
            out.append(
                {
                    "filename": path.name,
                    "format": path.suffix.lstrip(".").lower(),
                    "size_bytes": stat.st_size,
                    "modified_at": stat.st_mtime,
                    "source": "generated" if directory == _FREECAD_OUTPUT_DIR else "exported",
                }
            )
    seen_filenames = {a["filename"] for a in out}
    # dana.api.artifacts_registry covers what this directory scan structurally
    # can't: MockFreeCADEngine (dana.platform.mock, used whenever
    # dana.platform.factory.IS_HF_SPACE) writes every generated mesh to an
    # arbitrary system-temp path via tempfile.mkstemp, never under either
    # directory above. Registry entries win no priority over a same-named
    # on-disk file — just fill in whatever the scan didn't already find.
    for entry in artifacts_registry.list_artifacts():
        if entry["filename"] in seen_filenames:
            continue
        seen_filenames.add(entry["filename"])
        out.append({k: v for k, v in entry.items() if k != "path"})
    out.sort(key=lambda a: a["modified_at"], reverse=True)
    return out


def _resolve_artifact(filename: str) -> Path:
    """Only a bare filename (no path separators) matching either a file that
    ACTUALLY exists directly inside one of ``_ARTIFACT_DIRS``, or one already
    recorded in ``artifacts_registry`` (the mock engine's arbitrary temp-path
    outputs — see ``_list_artifacts``), is ever returned — never resolved
    relative to an arbitrary/absolute caller path, so ``../../whatever`` or
    an absolute path never matches anything; the registry lookup is likewise
    a match against paths THIS process itself already generated, never the
    caller-supplied string used as a path directly.
    """
    name = Path(filename).name
    if name != filename or Path(name).suffix.lower() not in _ARTIFACT_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"invalid artifact filename: {filename!r}")
    for directory in _ARTIFACT_DIRS:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    for entry in artifacts_registry.list_artifacts():
        if entry["filename"] == name:
            return Path(entry["path"])
    raise HTTPException(status_code=404, detail=f"artifact not found: {filename!r}")


@router.get("/artifacts")
def list_artifacts() -> dict[str, Any]:
    return {"ok": True, "artifacts": _list_artifacts()}


@router.get("/artifacts/{filename}/download")
def download_artifact(filename: str) -> FileResponse:
    path = _resolve_artifact(filename)
    media_type = _MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")
    return FileResponse(path, media_type=media_type, filename=path.name)


def _newest_fcstd() -> Path | None:
    candidates = [p for p in _FREECAD_OUTPUT_DIR.glob("*.FCStd") if p.is_file()] if _FREECAD_OUTPUT_DIR.is_dir() else []
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


@router.post("/open-desktop")
def open_desktop() -> dict[str, Any]:
    """Opens the most recently modified ``.FCStd`` document (i.e. the
    active/last-touched model) in the real FreeCAD GUI, reusing
    ``dana.plugins.freecad.engine.show_in_freecad_gui`` — the exact same
    "never steal focus, never spawn a duplicate instance, push to the
    secondary monitor" logic already used automatically after every
    create_freecad_*/perform_freecad_boolean tool call. This is just an
    on-demand trigger for the SAME mechanism, not a new one.
    """
    path = _newest_fcstd()
    if path is None:
        raise HTTPException(status_code=404, detail="no FreeCAD document has been generated yet")
    from dana.plugins.freecad.engine import show_in_freecad_gui

    result = json.loads(show_in_freecad_gui(str(path)))
    if not result.get("ok"):
        raise HTTPException(status_code=500, detail=result.get("error") or "failed to open FreeCAD GUI")
    return result
