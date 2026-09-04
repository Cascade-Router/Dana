"""Local Chat Session Persistence — plain JSON file storage under
``AGENT_WORKSPACE_DIR/data/sessions/<session_id>.json``, backing both the
frontend's ChatSidebar (browse/resume/delete past conversations via the
REST endpoints below) and ``dana.api.server``'s ``/ws/chat`` WebSocket
(hydrate-on-reconnect + auto-save after every completed ReAct turn).

Deliberately plain JSON, no SQLite/ORM — one small file per session,
consistent with ``dana.plugins.memory.core_memory``'s "no database"
precedent for on-disk agent state. Save frequency is low (once per
completed turn) and each file is tiny (a short chat transcript), so a full
read-modify-write per save is more than fast enough.

Persisted history is deliberately reduced to plain ``{"role", "content"}``
text pairs — NOT the full frontend ``ChatMessage`` shape (no attachments,
tool-activity feed, or HITL card state, and NOT the OpenAI-wire multimodal/
tool-call messages the ReAct loop itself works with turn-to-turn). This is
enough to "retain, browse, and resume past conversations" per the feature's
goal while keeping session files small and the storage format simple.

Also carries ``working_memory`` — dana.core.context_distiller's rolling,
per-session distilled summary (``{"summary": str, "turn": int}``) — so a
session resumed on a fresh ``/ws/chat`` connection (or a fresh server
process) picks its distilled context back up instead of starting cold, the
same way its message history already does.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from dana.paths import AGENT_WORKSPACE_DIR

SESSIONS_DIR: Path = AGENT_WORKSPACE_DIR / "data" / "sessions"

# A brief, human-glance sidebar title derived from the first user message —
# hard-truncated so one very long opening message can't blow up the
# sidebar's row width.
_TITLE_MAX_CHARS = 60

# dana.core.context_distiller's empty-state shape — same default a brand-new
# /ws/chat session's in-memory dict starts with, so a session saved before
# working_memory existed (or with a corrupt value) loads exactly as if it
# had never been distilled yet, rather than crashing or losing its history.
_EMPTY_WORKING_MEMORY: dict[str, Any] = {"summary": "", "turn": 0}

# Session-Specific Terminal History: mirrors dana/api/server.py's
# _log_terminal_event calls (user_message/assistant_message/tool_dispatch_
# start/tool_dispatch_end — the same "type"-tagged shape already streamed
# over /ws/chat as ServerEvent) so a session resumed on a fresh connection,
# or after a server restart, can re-populate the frontend's Terminal History
# panel instead of showing it empty until new activity happens. Capped the
# same way the frontend's own log buffer already is (useChatSocket.ts's
# MAX_LOG_LINES) so a long-lived session's on-disk file can't grow
# unbounded.
_TERMINAL_LOG_MAX_ENTRIES = 500


def _sanitize_working_memory(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict) or not isinstance(raw.get("summary"), str):
        return dict(_EMPTY_WORKING_MEMORY)
    turn = raw.get("turn", 0)
    return {"summary": raw["summary"], "turn": turn if isinstance(turn, int) and not isinstance(turn, bool) else 0}


def _sanitize_terminal_log(raw: Any) -> list[dict[str, Any]]:
    """Degrades to an empty log rather than raising — a session saved before
    this field existed, or a corrupt entry, must never break hydration the
    way a bad ``messages``/``working_memory`` value already can't (see
    ``load_session``'s own defensive parsing)."""
    if not isinstance(raw, list):
        return []
    return [entry for entry in raw if isinstance(entry, dict) and isinstance(entry.get("type"), str)][
        -_TERMINAL_LOG_MAX_ENTRIES:
    ]

# session_id is used to build a filename directly (see _session_path) — this
# allowlist makes a path-traversal payload (e.g. "../../etc/passwd") a
# structural impossibility rather than something needing a separate
# resolve()/relative_to() check: none of these characters can form ".." or
# "/". UUIDs (this module's own new_session_id) always satisfy it.
_VALID_SESSION_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def is_valid_session_id(session_id: str) -> bool:
    return bool(_VALID_SESSION_ID.fullmatch(session_id or ""))


def _session_path(session_id: str) -> Path:
    return SESSIONS_DIR / f"{session_id}.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_session_id() -> str:
    return str(uuid.uuid4())


def derive_title(user_text: str) -> str:
    """A brief sidebar title from the first user message: first line,
    whitespace-collapsed, hard-truncated. Never empty — an all-whitespace
    or blank first message falls back to "New chat" rather than leaving a
    session untitled in the sidebar.
    """
    stripped = (user_text or "").strip()
    first_line = stripped.splitlines()[0] if stripped else ""
    collapsed = re.sub(r"\s+", " ", first_line).strip()
    if not collapsed:
        return "New chat"
    if len(collapsed) > _TITLE_MAX_CHARS:
        return collapsed[: _TITLE_MAX_CHARS - 1].rstrip() + "…"
    return collapsed


def load_session(session_id: str) -> dict[str, Any] | None:
    """Returns the full on-disk session record, or ``None`` if it doesn't
    exist or is corrupt/foreign content — degrades to "nothing to hydrate"
    rather than raising, since this runs on every ``/ws/chat`` connect and
    must never crash a fresh session over a bad on-disk file.
    """
    if not is_valid_session_id(session_id):
        return None
    try:
        raw = _session_path(session_id).read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("messages"), list):
        return None

    messages = [
        {"role": m["role"], "content": m["content"]}
        for m in data["messages"]
        if isinstance(m, dict) and m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)
    ]
    return {
        "id": session_id,
        "title": str(data.get("title") or "New chat"),
        "created_at": str(data.get("created_at") or _now_iso()),
        "updated_at": str(data.get("updated_at") or _now_iso()),
        "messages": messages,
        "working_memory": _sanitize_working_memory(data.get("working_memory")),
        "terminal_log": _sanitize_terminal_log(data.get("terminal_log")),
    }


def save_session(
    session_id: str,
    *,
    title: str,
    created_at: str | None,
    messages: list[dict[str, str]],
    working_memory: dict[str, Any] | None = None,
    terminal_log: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Overwrites ``session_id``'s on-disk file with the given
    title/messages/working_memory/terminal_log — the ONE write path, used by
    both ``dana.api.server``'s auto-save hook and (indirectly, via
    ``load_session``+re-save) anything else that ever needs to persist a
    session, so there is exactly one place that knows this file's on-disk
    shape. ``created_at`` is stamped fresh (now) on a session's very first
    save; pass the PREVIOUS record's ``created_at`` on every save after
    that so it never drifts. ``working_memory``/``terminal_log`` default to
    the same empty-state shape a brand-new session starts with, for callers
    (e.g. the reset endpoint below) that intentionally save with no
    distilled context/terminal history of their own.
    """
    record = {
        "id": session_id,
        "title": title,
        "created_at": created_at or _now_iso(),
        "updated_at": _now_iso(),
        "messages": messages,
        "working_memory": working_memory if working_memory is not None else dict(_EMPTY_WORKING_MEMORY),
        "terminal_log": terminal_log if terminal_log is not None else [],
    }
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    _session_path(session_id).write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def delete_session(session_id: str) -> bool:
    """Removes ``session_id``'s file if present. Returns whether anything
    was actually deleted — never raises for an already-missing file, since
    delete is expected to be idempotent (a double-click in the sidebar, or
    deleting a session the frontend has stale metadata for, must not
    surface as an error).
    """
    if not is_valid_session_id(session_id):
        return False
    try:
        _session_path(session_id).unlink()
        return True
    except FileNotFoundError:
        return False


def list_sessions() -> list[dict[str, Any]]:
    """Metadata only (id/title/updated_at) for every stored session,
    most-recently-updated first — the frontend ChatSidebar's list. A
    corrupt/unreadable individual file is skipped rather than failing the
    whole listing (same "degrade gracefully, per-item" policy as
    ``load_session``).
    """
    if not SESSIONS_DIR.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for path in SESSIONS_DIR.glob("*.json"):
        session = load_session(path.stem)
        if session is None:
            continue
        out.append({"id": session["id"], "title": session["title"], "updated_at": session["updated_at"]})
    out.sort(key=lambda s: s["updated_at"], reverse=True)
    return out


router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.get("")
def get_sessions() -> dict[str, Any]:
    return {"ok": True, "sessions": list_sessions()}


@router.get("/{session_id}")
def get_session(session_id: str) -> dict[str, Any]:
    if not is_valid_session_id(session_id):
        raise HTTPException(status_code=400, detail=f"invalid session_id: {session_id!r}")
    session = load_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"session not found: {session_id!r}")
    return {"ok": True, "session": session}


@router.delete("/{session_id}")
def delete_session_endpoint(session_id: str) -> dict[str, Any]:
    if not is_valid_session_id(session_id):
        raise HTTPException(status_code=400, detail=f"invalid session_id: {session_id!r}")
    return {"ok": True, "deleted": delete_session(session_id)}


@router.delete("")
def clear_all_sessions_endpoint() -> dict[str, Any]:
    """The Settings modal's "Clear All Sessions" button — deletes every
    persisted session file in one call instead of the sidebar's one-at-a-
    time delete button. Registered on the router's bare prefix (``DELETE
    /api/sessions``), one level up from the single-session route above, so
    it never collides with a specific ``{session_id}``. Best-effort per
    file, same as ``delete_session``: a file that vanishes mid-sweep (e.g. a
    concurrent single-session delete) is simply not counted, not an error.
    """
    deleted = 0
    if SESSIONS_DIR.is_dir():
        for path in SESSIONS_DIR.glob("*.json"):
            try:
                path.unlink()
                deleted += 1
            except FileNotFoundError:
                pass
    return {"ok": True, "deleted": deleted}


@router.post("/{session_id}/reset")
def reset_session_endpoint(session_id: str) -> dict[str, Any]:
    """Clears an EXISTING session's persisted transcript in place (same id,
    title reset to "New chat", empty messages) — distinct from "+ New Chat"
    in the frontend, which already achieves a full reset (fresh tool
    accumulation, fresh capability/turn state, fresh token budget) by simply
    reconnecting the websocket with no ``session_id`` at all, letting the
    server mint a brand-new one; that in-memory turn state
    (``dana.api.server``'s per-websocket session dict — react_state,
    capability_unlocked_at_turn, turn_counter, ...) isn't reachable from a
    stateless REST call in the first place, since it lives on whichever
    live websocket connection owns it, not in this on-disk store. This
    endpoint exists for the separate, real case of clearing a SPECIFIC
    already-saved session's history without abandoning its id (e.g. a
    "clear this chat" action from the sidebar rather than starting a
    brand-new one).
    """
    if not is_valid_session_id(session_id):
        raise HTTPException(status_code=400, detail=f"invalid session_id: {session_id!r}")
    existing = load_session(session_id)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"session not found: {session_id!r}")
    record = save_session(session_id, title="New chat", created_at=existing["created_at"], messages=[])
    return {"ok": True, "session": record}


__all__ = (
    "SESSIONS_DIR",
    "delete_session",
    "derive_title",
    "is_valid_session_id",
    "list_sessions",
    "load_session",
    "new_session_id",
    "router",
    "save_session",
)
