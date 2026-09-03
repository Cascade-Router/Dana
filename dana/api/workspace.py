"""REST API for the frontend's Workspace Explorer plugin — exposes
AGENT_WORKSPACE_DIR's file tree/file contents (read-only), plus Dynamic
Workspace Mounting's registry of user-granted external directories.

Every path the tree/file endpoints touch goes through
``dana.plugins.os.file_system.resolve_sandboxed_path`` — the SAME
traversal-rejecting helper the os_tools ReAct tools (list_directory,
read_file, write_file, run_python_script) already use. This router adds no
separate path-validation logic of its own, so a traversal attempt is
rejected identically here as it would be for an LLM-driven tool call.

Nothing on the tree/file endpoints can mutate a file — those stay GET-only,
matching the frontend plugin's own read-only viewer. All actual file
mutation still goes through the agent's own os_tools ReAct tools.

Dynamic Workspace Mounting (``mount_workspace_directory``/
``load_mounted_directories`` below) is the one thing here that isn't
read-only in effect, even though it never touches a file's bytes: it's a
TRUST decision — which external absolute directories
``resolve_sandboxed_path``'s ``allowed_mounts`` param will subsequently
treat as safe for the os_tools trio. Persisted to
``AGENT_WORKSPACE_DIR/data/mounts.json`` rather than kept in any one
session's in-memory state, so a mount registered from one connected window
is immediately visible to every other one (dana.api.server reads this
fresh on every ReAct turn — see its ``_execute_and_continue``/
``_process_user_text``), and survives a server restart.
"""

from __future__ import annotations

import json
import mimetypes
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from dana.api.sessions import SESSIONS_DIR, is_valid_session_id, load_session
from dana.paths import AGENT_WORKSPACE_DIR
from dana.plugins.os.file_system import PathEscapeError, resolve_sandboxed_path

router = APIRouter(prefix="/api/workspace", tags=["workspace"])

# Lives under AGENT_WORKSPACE_DIR itself (not some separate config root) so
# it travels with the same sandbox tree everything else here already keys
# off, and survives a restart the same way dana.plugins.memory.core_memory's
# on-disk store does.
_MOUNTS_PATH = AGENT_WORKSPACE_DIR / "data" / "mounts.json"


def _format_display_datetime(dt: datetime) -> str:
    """"Mon D, HH:MM" in the server's local time — avoids %-d/%#d (not
    portable between Linux and Windows strftime) by formatting the
    zero-padded day separately."""
    local = dt.astimezone()
    return f"{local.strftime('%b')} {local.day}, {local.strftime('%H:%M')}"


def _session_display(path: Path) -> tuple[str, str]:
    """One ``load_session()`` call → (humanized label, sort key) for a
    ``data/sessions/<uuid>.json`` leaf, reusing the SAME title/created_at
    metadata ChatSidebar already renders (dana.api.sessions.load_session) —
    a session reads identically in both places instead of showing a raw
    UUID here. Falls back to the file's own mtime for both the label
    ("Session • <mtime>") and the sort key if the record fails to load
    (corrupt/foreign file), rather than a bare UUID.
    """
    session_id = path.stem
    session = load_session(session_id) if is_valid_session_id(session_id) else None
    if session is not None:
        try:
            created = datetime.fromisoformat(session["created_at"])
        except ValueError:
            created = datetime.now(timezone.utc)
        return f"{session['title']} • {_format_display_datetime(created)}", session["updated_at"]

    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        mtime = datetime.now(timezone.utc)
    return f"Session • {_format_display_datetime(mtime)}", mtime.isoformat()


def _build_tree(path: Path, rel_path: str, *, display_name: str | None = None) -> dict[str, Any]:
    """Recursive directory walk into the same JSON shape the frontend's
    WorkspacePlugin renders. A symlinked directory is deliberately NOT
    recursed into (treated as a leaf) — a cheap guard against a symlink
    loop turning this into infinite recursion; the file endpoint below
    would reject it as "not a file" anyway if something tried to open it.
    """
    name = display_name if display_name is not None else path.name
    if path.is_dir() and not path.is_symlink():
        entries = list(path.iterdir())
        # data/sessions/*.json humanization: one metadata dict, reused for
        # both the display label AND recency ordering (most-recent first,
        # matching ChatSidebar's own list_sessions()) — so the Workspace
        # tree's dates read top-to-bottom instead of shuffled by raw UUID.
        # Every other directory keeps the plain dirs-first/alphabetical sort.
        session_meta = {child: _session_display(child) for child in entries} if path == SESSIONS_DIR else {}
        if session_meta:
            entries.sort(key=lambda p: session_meta[p][1], reverse=True)
        else:
            entries.sort(key=lambda p: (0 if p.is_dir() else 1, p.name.lower()))
        return {
            "name": name,
            "path": rel_path,
            "type": "directory",
            "children": [
                _build_tree(
                    child,
                    f"{rel_path}/{child.name}" if rel_path else child.name,
                    display_name=session_meta[child][0] if child in session_meta else None,
                )
                for child in entries
            ],
        }
    try:
        size = path.stat().st_size
    except OSError:
        size = 0
    return {"name": name, "path": rel_path, "type": "file", "size": size}


@router.get("/tree")
def get_workspace_tree() -> dict[str, Any]:
    root = resolve_sandboxed_path(".")
    return {"ok": True, "tree": _build_tree(root, "", display_name="workspace")}


@router.get("/file/{file_path:path}")
def get_workspace_file(file_path: str) -> FileResponse:
    try:
        target = resolve_sandboxed_path(file_path)
    except PathEscapeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not target.exists():
        raise HTTPException(status_code=404, detail=f"file not found: {file_path!r}")
    if not target.is_file():
        raise HTTPException(status_code=400, detail=f"path is not a file: {file_path!r}")

    media_type, _ = mimetypes.guess_type(str(target))
    return FileResponse(target, media_type=media_type or "text/plain", filename=target.name)


def load_mounted_directories() -> list[str]:
    """Every currently-registered mount, read fresh off disk on each call —
    deliberately not cached in memory. dana.api.server reads this on every
    ReAct turn (dispatch_tool_call's allowed_mounts, build_system_prompt's
    mounted_directories), so there's no in-memory registry to keep in sync
    across sessions or to go stale after a mount registered elsewhere.
    """
    try:
        raw = json.loads(_MOUNTS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [p for p in raw if isinstance(p, str)] if isinstance(raw, list) else []


def _save_mounted_directories(paths: list[str]) -> None:
    _MOUNTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _MOUNTS_PATH.write_text(json.dumps(paths, indent=2), encoding="utf-8")


def mount_workspace_directory(raw_path: str) -> dict[str, Any]:
    """Validates and registers ``raw_path`` as a Dynamic Workspace Mounting
    trust root.

    Requires an ABSOLUTE, EXISTING directory: a relative path has no
    unambiguous meaning here (there's no "current directory" a native OS
    folder picker result is relative TO), and a path that doesn't exist
    can't be a directory the user actually just selected. Idempotent:
    mounting the same (resolved) directory twice is a no-op, never a
    duplicate registry entry.
    """
    candidate = (raw_path or "").strip()
    if not candidate:
        return {"ok": False, "error": "path must not be empty"}

    parsed = Path(candidate)
    if not parsed.is_absolute():
        return {"ok": False, "error": f"mount path must be absolute: {candidate!r}"}

    try:
        resolved = parsed.resolve(strict=True)
    except OSError as exc:
        return {"ok": False, "error": f"could not resolve path {candidate!r}: {exc}"}
    if not resolved.is_dir():
        return {"ok": False, "error": f"not a directory: {candidate!r}"}

    mounts = load_mounted_directories()
    resolved_str = str(resolved)
    if resolved_str not in mounts:
        mounts.append(resolved_str)
        _save_mounted_directories(mounts)
    return {"ok": True, "mounted_directories": mounts}


@router.get("/mounts")
def get_mounted_directories() -> dict[str, Any]:
    return {"ok": True, "mounted_directories": load_mounted_directories()}


@router.post("/mount")
def post_mount_directory(body: dict[str, str]) -> dict[str, Any]:
    result = mount_workspace_directory(body.get("path") or "")
    if not result["ok"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


__all__ = ("router", "load_mounted_directories", "mount_workspace_directory")
