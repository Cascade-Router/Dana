"""In-process registry of every mesh/export artifact a CAD tool call has
produced this run — the one source both ``dana.api.cad``'s REST endpoint
and ``app.py``'s Gradio "artifacts" endpoint read from.

Needed because ``dana.api.cad``'s directory scan (``freecad_output/``/
``exports/`` under ``DANA_WORKSPACE``) only ever finds files a REAL FreeCAD
engine writes there. ``dana.platform.mock.MockFreeCADEngine`` — used
whenever ``dana.platform.factory.IS_HF_SPACE`` (i.e. on the actual deployed
Hugging Face Space) — writes every mesh to an arbitrary system-temp path via
``tempfile.mkstemp`` instead, which that directory scan never sees. Recording
each artifact HERE, at the moment ``dana.api.server``'s auto-export hook (or
an explicit ``export_freecad_model`` call) produces one, makes both surfaces
correct for both engines instead of only the real one.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

_artifacts: list[dict[str, Any]] = []


def register_artifact(path: str, *, format: str, source: str, session_id: str) -> dict[str, Any]:
    """Record one generated file. ``source`` is ``"generated"`` for the
    automatic per-tool-call viewer/STEP export, or ``"exported"`` for an
    explicit ``export_freecad_model`` call — same vocabulary
    ``dana.api.cad._list_artifacts`` already uses for its directory-scanned
    entries, so the two sources merge without a caller needing to know which
    one a given entry came from.

    ``session_id`` (required — every real call site already has
    ``session["session_id"]`` in scope) is what ``list_artifacts`` filters
    on, so one chat session's Scoped Mini-Explorer never lists a mesh
    another session generated — this registry is the ONLY per-artifact
    metadata store for the mock-engine path (arbitrary tempfile.mkstemp
    paths, never under a per-session directory a plain scan could isolate
    on its own).
    """
    p = Path(path)
    try:
        size_bytes = p.stat().st_size
    except OSError:
        size_bytes = 0
    entry = {
        "filename": p.name,
        "format": format,
        "size_bytes": size_bytes,
        "modified_at": time.time(),
        "source": source,
        "path": str(p),
        "session_id": session_id,
    }
    _artifacts.append(entry)
    return entry


def list_artifacts(session_id: str | None = None) -> list[dict[str, Any]]:
    """Every artifact recorded so far belonging to ``session_id``, newest
    first, skipping any whose file no longer exists (a stale temp path from
    an earlier, since-cleaned run). ``session_id=None`` returns every
    recorded artifact across every session — no real caller does this
    today (dana.api.cad's endpoint always has a session_id to filter on),
    kept only so a future cross-session admin view doesn't need a second
    function.
    """
    return sorted(
        (
            a
            for a in _artifacts
            if Path(a["path"]).is_file() and (session_id is None or a.get("session_id") == session_id)
        ),
        key=lambda a: a["modified_at"],
        reverse=True,
    )


__all__ = ("list_artifacts", "register_artifact")
