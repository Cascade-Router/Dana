"""Session state persistence across server restarts.

``capture_session_state`` builds the ``state`` payload ``dana.api.sessions``
stores next to a session's chat history; ``restore_session_state`` puts it
back when ``/ws/chat`` resumes that session in a fresh process. It covers the
ReAct engine's per-session registries (plan, object paths, topology DAG,
assembly state, gates), the task board plan, the CAD call log, the mock
driver's objects, and four per-connection fields (capability decay clock,
hidden tools, turn counter, LLM usage/cost totals).

Never persisted (security invariant, see ``_PERSISTED_SERVER_FIELDS``): the
HITL allowlist (``hitl_approved_tools``), ``auto_approve``, ``api_keys`` and
any suspended HITL/visual turn. A resumed session always starts with
approvals and secrets empty.

The session's FreeCAD document is paired with its state: every save copies
``Session_Active.FCStd`` to ``freecad_output/sessions/<id>.FCStd`` and records
its SHA-256. On restore, a live document that no longer matches is replaced
by that snapshot when the snapshot matches; otherwise the object registry and
DAG are resynced to whatever the live document really contains.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
from pathlib import Path
from typing import Any

from dana.api import artifacts_registry
from dana.core import react_dispatch
from dana.plugins.freecad import engine as freecad_engine
from dana.plugins.freecad.call_log import CadCallLog
from dana.plugins.planning import task_board

log = logging.getLogger(__name__)

STATE_VERSION = 1

# The only per-connection session-dict fields that are persisted. Anything
# not listed here (hitl_approved_tools, auto_approve, api_keys, react_state,
# visual_state, ...) never reaches disk.
_PERSISTED_SERVER_FIELDS = ("capability_unlocked_at_turn", "hidden_tool_ids", "turn_counter", "cost_tracking")


# --- codec: JSON with sets, frozensets, tuples and non-str dict keys --------


def encode(value: Any) -> Any:
    if isinstance(value, dict):
        if all(isinstance(k, str) for k in value):
            return {k: encode(v) for k, v in value.items()}
        return {"__dict_items__": [[encode(k), encode(v)] for k, v in value.items()]}
    if isinstance(value, (set, frozenset)):
        tag = "__frozenset__" if isinstance(value, frozenset) else "__set__"
        return {tag: sorted((encode(v) for v in value), key=repr)}
    if isinstance(value, tuple):
        return {"__tuple__": [encode(v) for v in value]}
    if isinstance(value, list):
        return [encode(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _freeze(value: Any) -> Any:
    """Set members and dict keys must be hashable."""
    return tuple(_freeze(v) for v in value) if isinstance(value, list) else value


def decode(value: Any) -> Any:
    if isinstance(value, list):
        return [decode(v) for v in value]
    if not isinstance(value, dict):
        return value
    if len(value) == 1:
        (tag, inner), = value.items()
        if tag == "__set__":
            return {_freeze(decode(v)) for v in inner}
        if tag == "__frozenset__":
            return frozenset(_freeze(decode(v)) for v in inner)
        if tag == "__tuple__":
            return tuple(decode(v) for v in inner)
        if tag == "__dict_items__":
            return {_freeze(decode(k)): decode(v) for k, v in inner}
    return {k: decode(v) for k, v in value.items()}


# --- document pairing -------------------------------------------------------


def _sha256(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def document_snapshot_path(session_id: str) -> Path:
    return freecad_engine._OUTPUT_DIR / "sessions" / f"{session_id}.FCStd"


def _snapshot_document(session_id: str) -> dict[str, Any] | None:
    """Copies the live session document to its per-session snapshot and
    fingerprints it. None for a session that never created a document."""
    live = freecad_engine.session_document_path_for(session_id)
    if not live.is_file() or live.stat().st_size == 0:
        return None
    snapshot = document_snapshot_path(session_id)
    try:
        shutil.copy2(live, snapshot)
    except OSError as exc:
        log.warning("session %s: could not snapshot %s: %s", session_id, live, exc)
        return None
    return {"sha256": _sha256(snapshot), "size": snapshot.stat().st_size}


def session_disk_paths(session_id: str) -> list[Path]:
    """Everything a session leaves on disk besides its ``<id>.json``: its
    document folder and snapshot, its exports (CAD files, macros, drawings,
    BOM CSVs) and its print-check STLs. Paths may not exist."""
    from dana.plugins.manufacturing import bom_exporter
    from dana.plugins.os import file_system

    return [
        freecad_engine._OUTPUT_DIR / "sessions" / session_id,
        document_snapshot_path(session_id),
        freecad_engine._EXPORT_DIR / "sessions" / session_id,
        bom_exporter._EXPORT_DIR / "sessions" / session_id,
        Path(file_system._SANDBOX_ROOT) / "prints" / "sessions" / session_id,
    ]


def forget_session(session_id: str) -> list[str]:
    """Removes what a deleted session left behind: its in-memory engine
    state (registries, task board plan, mock meshes' registry, artifact
    list) and the paths in ``session_disk_paths``. ``session_id`` must
    already be validated (``dana.api.sessions.is_valid_session_id``); each
    path is also checked to sit directly in its ``sessions/`` folder before
    anything is deleted, and a symlink is unlinked, never followed.
    Returns the paths removed."""
    from dana.platform.mock import replace_mock_session

    react_dispatch.forget_session_state(session_id)
    task_board.forget_plan(session_id)
    replace_mock_session(session_id, {})
    artifacts_registry.forget_session(session_id)

    removed = []
    for path in session_disk_paths(session_id):
        sessions_dir = path.parent
        if sessions_dir.name != "sessions" or path.name not in (session_id, f"{session_id}.FCStd"):
            continue
        try:
            if path.is_symlink():
                path.unlink()  # the link only, wherever it points
            elif path.resolve().parent != sessions_dir.resolve():
                continue
            elif path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)
            else:
                continue
        except OSError as exc:
            log.warning("session %s: could not remove %s: %s", session_id, path, exc)
            continue
        removed.append(str(path))
    return removed


# --- capture / restore --------------------------------------------------------


def _is_mock(engine: Any) -> bool:
    from dana.platform.mock import MockFreeCADEngine

    return isinstance(engine, MockFreeCADEngine)


def capture_session_state(session: dict[str, Any], engine: Any) -> dict[str, Any]:
    """The JSON-safe ``state`` payload for ``session`` (see module docstring)."""
    sid = session["session_id"]
    call_log = session.get("call_log")
    state: dict[str, Any] = {
        "version": STATE_VERSION,
        "react": react_dispatch.snapshot_session_state(sid),
        "task_board": task_board.snapshot_plan(sid),
        "call_log": call_log.to_payload() if isinstance(call_log, CadCallLog) else [],
        "server": {field: session[field] for field in _PERSISTED_SERVER_FIELDS if field in session},
    }
    if _is_mock(engine):
        from dana.platform.mock import snapshot_mock_session

        state["mock"] = snapshot_mock_session(sid)
    else:
        state["document"] = _snapshot_document(sid)
    return encode(state)


def _sync_document(sid: str, recorded: dict[str, Any] | None) -> str | None:
    """Real engine: make the live document agree with the restored state.
    Returns a user-facing note when something had to be repaired."""
    if not isinstance(recorded, dict) or not recorded.get("sha256"):
        return None  # this session never had a document
    live = freecad_engine.session_document_path_for(sid)
    if _sha256(live) == recorded["sha256"]:
        return None
    snapshot = document_snapshot_path(sid)
    if _sha256(snapshot) == recorded["sha256"]:
        shutil.copy2(snapshot, live)
        return None
    if not live.is_file():
        dropped = react_dispatch.prune_session_objects(sid, set(), None)
        return (
            "This session's FreeCAD document is missing, so its objects were cleared"
            + (f" ({', '.join(dropped)})." if dropped else ".")
        )
    names = freecad_engine.list_document_objects(live)
    if names is None:
        return (
            "This session's FreeCAD document changed outside Dana and could not be inspected; "
            "object names from before the restart may be stale."
        )
    dropped = react_dispatch.prune_session_objects(sid, names, str(live))
    return (
        "This session's FreeCAD document changed outside Dana; its object list was resynced from the document"
        + (f" (no longer present: {', '.join(dropped)})." if dropped else ".")
    )


def restore_session_state(session: dict[str, Any], raw_state: Any, engine: Any) -> str | None:
    """Hydrates ``session`` (a fresh /ws/chat session dict) from ``raw_state``.

    The call log and per-connection fields always come from disk (a new
    connection has none of its own). The process-wide registries are only
    restored when this process holds nothing for the session, so a same-process
    reconnect keeps its live memory. Missing or legacy state is a no-op.
    Returns a user-facing note when the CAD document had to be resynced.
    """
    if not isinstance(raw_state, dict) or raw_state.get("version") != STATE_VERSION:
        return None
    state = decode(raw_state)
    sid = session["session_id"]

    session["call_log"] = CadCallLog.from_payload(state.get("call_log"))
    for field, value in (state.get("server") or {}).items():
        if field in _PERSISTED_SERVER_FIELDS:
            session[field] = value

    if not react_dispatch.session_state_is_empty(sid):
        return None
    react_dispatch.restore_session_state(sid, state.get("react"))
    task_board.restore_plan(sid, state.get("task_board"))

    if _is_mock(engine):
        from dana.platform.mock import restore_mock_session

        surviving = restore_mock_session(sid, state.get("mock"))
        dropped = react_dispatch.prune_session_objects(sid, surviving, None)
        if dropped:
            return f"Some objects from before the restart are gone and were cleared: {', '.join(dropped)}."
        return None
    return _sync_document(sid, state.get("document"))


__all__ = (
    "forget_session",
    "session_disk_paths",
    "STATE_VERSION",
    "capture_session_state",
    "decode",
    "document_snapshot_path",
    "encode",
    "restore_session_state",
)
