"""Ambient per-request session_id — the propagation mechanism for CAD
workspace isolation (one ``Session_Active.FCStd``, one object-name
registry, one export directory PER chat session, not one shared globally).

Set once per turn, at the top of ``dana.api.server._execute_and_continue``
(the single production call site of ``dana.core.react_dispatch
.dispatch_tool_call``), this propagates through the entire synchronous
``dispatch_tool_call`` -> ``_tool_*`` handler -> ``dana.plugins.freecad
.engine.*`` call chain WITHOUT a new parameter threaded through every one
of those ~40 handler/~20 engine function signatures — every one of them
just calls ``get_session_id()`` at the point it needs to build a path or
key a registry, the same way they'd read any other module-level constant.

``contextvars.ContextVar`` (not a plain module global) is what makes this
safe under FastAPI's concurrency model: each WebSocket connection's message
handling runs as its own ``asyncio`` Task, and a Task gets its own
copy-on-write view of the context — one session's ``set_session_id`` call
can never bleed into a DIFFERENT session's concurrently-running turn the
way a bare global variable would.
"""

from __future__ import annotations

import contextvars
import re
from pathlib import Path

DEFAULT_SESSION_ID = "default"

# Same charset dana.api.sessions._VALID_SESSION_ID already enforces before a
# client-supplied session_id is ever accepted into a session dict — this is
# about to become a filesystem PATH COMPONENT (freecad_output/sessions/
# <session_id>/...), so re-validated here too rather than trusting it
# stayed unmodified across every module boundary in between. No "/", "..",
# or absolute-path characters are in this charset at all, so this closes
# off path traversal regardless of what called set_session_id().
_VALID_SESSION_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")

_session_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "dana_session_id", default=DEFAULT_SESSION_ID
)


def sanitize_session_id(session_id: str | None) -> str:
    """Validates ``session_id`` against the same safe charset
    ``dana.api.sessions._VALID_SESSION_ID`` already enforces before a
    client-supplied session_id is ever accepted into a session dict, and
    falls back to ``DEFAULT_SESSION_ID`` for anything else (``None``,
    empty, or containing a character that isn't alnum/``_``/``-``) —
    never raises, since a malformed/malicious value here must never crash
    a request or escape ``sessions/`` on disk. Shared by ``set_session_id``
    (the ambient-context path) and any direct caller that has an explicit
    session_id of its own to use instead of the ambient one (e.g.
    ``dana.api.cad``'s REST routes, which aren't part of the WebSocket
    turn's async call chain the contextvar propagates through).
    """
    candidate = (session_id or "").strip()
    return candidate if _VALID_SESSION_ID.fullmatch(candidate) else DEFAULT_SESSION_ID


def set_session_id(session_id: str) -> None:
    """Sets the ambient session_id for the rest of THIS asyncio Task's
    execution."""
    _session_id.set(sanitize_session_id(session_id))


def get_session_id() -> str:
    """The current turn's session_id, or ``DEFAULT_SESSION_ID`` for any
    caller outside ``dana.api.server``'s real WebSocket handling that never
    called ``set_session_id`` at all (ad hoc scripts, a bare REPL probe,
    most of the existing test suite) — kept for backward compatibility,
    not because "default" is a real shared session anything should rely on
    for actual isolation.
    """
    return _session_id.get()


def session_scoped_dir(base: Path, session_id: str | None = None) -> Path:
    """``base / "sessions" / <session_id>``, created if it doesn't exist
    yet. The one shared helper every per-session output directory
    (``dana.plugins.freecad.engine``'s ``freecad_output/``/``exports/``,
    ``dana.tools.urdf_builder``/``dana.tools.image_to_3d``'s own
    independent ``freecad_output/`` copies) is built from, so "which
    session owns this directory" has exactly one implementation instead of
    N call sites reimplementing the same ``base / "sessions" / id`` join.

    ``session_id``, when given, is sanitized and used directly — for a
    caller with its own explicit session_id (e.g. a REST route's query
    param) rather than the ambient one. Omit it (``None``, the default) to
    use the current ``get_session_id()`` instead, for every call site
    that's part of the WebSocket turn's contextvar-propagated call chain.
    """
    sid = sanitize_session_id(session_id) if session_id is not None else get_session_id()
    path = base / "sessions" / sid
    path.mkdir(parents=True, exist_ok=True)
    return path


__all__ = (
    "DEFAULT_SESSION_ID",
    "sanitize_session_id",
    "set_session_id",
    "get_session_id",
    "session_scoped_dir",
)
