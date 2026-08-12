"""Transactional file editor — stage writes, verify syntax, then commit.

Worker mutations never touch the live workspace until ``verify_and_commit``
approves the staging diff. Crashes / bad syntax call ``rollback_workspace``.
"""

from __future__ import annotations

import ast
import re
import threading
from pathlib import Path

from dana.exec.shadow_workspace import (
    ShadowWorkspace,
    bind_shadow_workspace,
    get_active_shadow,
)
from dana.paths import PROJECT_ROOT

_ROOT = Path(PROJECT_ROOT).resolve()
_PROTECTED_DIRS = ("dana", ".git", ".github")
_MAX_OUTPUT_CHARS = 2000

_PY_SUFFIXES = {".py", ".pyi"}
_CPP_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".inl"}

_lock = threading.RLock()
# session_id → ShadowWorkspace (durable until commit/rollback)
_SESSIONS: dict[str, ShadowWorkspace] = {}


def _truncate_file_body(text: str, limit: int = _MAX_OUTPUT_CHARS) -> str:
    raw = text if isinstance(text, str) else str(text or "")
    if len(raw) <= limit:
        return raw
    half = limit // 2
    return f"{raw[:half]}\n...[truncated]...\n{raw[-half:]}"


def _resolve_jailed(filepath: str) -> Path:
    from dana.tools.system_repl import _resolve_jailed as _jail

    return _jail(filepath)


def _is_protected(target: Path) -> str | None:
    for p_dir in _PROTECTED_DIRS:
        protected_path = (_ROOT / p_dir).resolve()
        if target.is_relative_to(protected_path):
            return p_dir
    return None


def begin_staging_session(
    session_id: str,
    *,
    base_dir: Path | str | None = None,
) -> ShadowWorkspace:
    """Open (or reuse) a staging session for transactional worker edits."""
    sid = str(session_id or "").strip() or "default"
    with _lock:
        existing = _SESSIONS.get(sid)
        if existing is not None and not existing._committed and not existing._rolled_back:
            existing.ensure()
            return existing
        ws = ShadowWorkspace(sid, base_dir=base_dir)
        ws.ensure()
        _SESSIONS[sid] = ws
        return ws


def get_staging_session(session_id: str) -> ShadowWorkspace | None:
    sid = str(session_id or "").strip()
    if not sid:
        return None
    with _lock:
        return _SESSIONS.get(sid)


def _drop_session(session_id: str) -> None:
    with _lock:
        _SESSIONS.pop(str(session_id or "").strip(), None)


def _brace_balanced(src: str) -> bool:
    depth = 0
    in_str: str | None = None
    line_c = False
    block_c = False
    i = 0
    while i < len(src):
        ch = src[i]
        nxt = src[i + 1] if i + 1 < len(src) else ""
        if line_c:
            if ch == "\n":
                line_c = False
            i += 1
            continue
        if block_c:
            if ch == "*" and nxt == "/":
                block_c = False
                i += 2
                continue
            i += 1
            continue
        if in_str:
            if ch == "\\" and in_str != "'":
                i += 2
                continue
            if ch == in_str:
                in_str = None
            i += 1
            continue
        if ch == "/" and nxt == "/":
            line_c = True
            i += 2
            continue
        if ch == "/" and nxt == "*":
            block_c = True
            i += 2
            continue
        if ch in {'"', "'"}:
            in_str = ch
            i += 1
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                return False
        i += 1
    return depth == 0


def verify_staged_content(path: Path, content: str) -> str | None:
    """Return an error string when staged content is malformed; else None."""
    suf = path.suffix.lower()
    body = content if isinstance(content, str) else str(content or "")
    if suf in _PY_SUFFIXES:
        try:
            ast.parse(body)
        except SyntaxError as exc:
            return f"Python SyntaxError in {path.name}: {exc}"
        return None
    if suf in _CPP_SUFFIXES:
        if not _brace_balanced(body):
            return f"C/C++ brace mismatch in {path.name}"
        # Reject truncated-looking dumps that end mid-token.
        if re.search(r"\b(class|struct|namespace)\s*$", body.rstrip()):
            return f"C/C++ truncated type declaration in {path.name}"
        return None
    # Non-code: always acceptable for commit.
    return None


def staging_diff_summary(session_id: str) -> str:
    """Human-readable list of staged destination paths."""
    ws = get_staging_session(session_id)
    if ws is None:
        return f"ERROR: unknown staging session {session_id!r}"
    rows = []
    for dest_key, staged in ws.staged_paths().items():
        try:
            rel = Path(dest_key).resolve().relative_to(_ROOT).as_posix()
        except Exception:  # noqa: BLE001
            rel = dest_key
        size = staged.stat().st_size if staged.is_file() else 0
        rows.append(f"- {rel} ({size} bytes staged)")
    if not rows:
        return f"OK: staging session {session_id!r} has no pending files"
    return f"OK: staging session {session_id!r}\n" + "\n".join(rows)


def verify_and_commit(session_id: str) -> str:
    """Approve staging diffs (syntax checks) then persist to the workspace.

    On verification failure the staging buffer is rolled back and live files
    remain untouched.
    """
    sid = str(session_id or "").strip()
    ws = get_staging_session(sid)
    if ws is None:
        return f"ERROR: unknown staging session {sid!r}"
    if ws._committed:
        _drop_session(sid)
        return f"OK: session {sid!r} already committed"
    if ws._rolled_back:
        _drop_session(sid)
        return f"ERROR: session {sid!r} already rolled back"

    errors: list[str] = []
    staged_items = list(ws.staged_paths().items())
    if not staged_items:
        ws.rollback()
        _drop_session(sid)
        return f"OK: session {sid!r} empty — nothing to commit"

    for dest_key, staged in staged_items:
        if not staged.is_file():
            continue
        try:
            text = staged.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"read staged failed for {dest_key}: {exc}")
            continue
        err = verify_staged_content(Path(dest_key), text)
        if err:
            errors.append(err)

    if errors:
        try:
            ws.rollback()
        finally:
            _drop_session(sid)
        joined = "; ".join(errors)
        return f"ERROR: verify failed — rolled back session {sid!r}: {joined}"

    try:
        written = ws.commit()
    except Exception as exc:  # noqa: BLE001
        try:
            if not ws._rolled_back:
                ws.rollback()
        except Exception:  # noqa: BLE001
            pass
        _drop_session(sid)
        return f"ERROR: commit failed for session {sid!r}: {exc}"

    _drop_session(sid)
    rels = []
    for p in written:
        try:
            rels.append(p.resolve().relative_to(_ROOT).as_posix())
        except Exception:  # noqa: BLE001
            rels.append(str(p))
    return f"OK: committed session {sid!r} files={rels}"


def rollback_workspace(session_id: str) -> str:
    """Clear the staging buffer; never touch live workspace files."""
    sid = str(session_id or "").strip()
    ws = get_staging_session(sid)
    if ws is None:
        # Also try discarding a leftover scratch dir by constructing a workspace.
        try:
            ghost = ShadowWorkspace(sid)
            if ghost.scratch_dir.exists():
                ghost.rollback()
                return f"OK: rolled back orphan scratch for {sid!r}"
        except Exception:  # noqa: BLE001
            pass
        return f"OK: no staging session {sid!r} (already clear)"
    try:
        if not ws._committed and not ws._rolled_back:
            ws.rollback()
        elif not ws._rolled_back and ws.scratch_dir.exists():
            ws._clear_scratch()
    except Exception as exc:  # noqa: BLE001
        _drop_session(sid)
        return f"ERROR: rollback failed for {sid!r}: {exc}"
    _drop_session(sid)
    return f"OK: rolled back staging session {sid!r}"


def file_editor(
    action: str,
    filepath: str,
    content: str | None = None,
    *,
    staging_session: str | None = None,
) -> str:
    """Read/write inside PROJECT_ROOT with optional transactional staging.

    When ``staging_session`` is set (or a shadow is already bound), write/append
    land only in the staging copy. Call ``verify_and_commit`` to persist.
    """
    act = (action or "").strip().lower()
    if act not in {"read", "write", "append"}:
        return "ERROR: action must be 'read', 'write', or 'append'"

    try:
        target = _resolve_jailed(filepath)
    except ValueError as exc:
        return f"ERROR: {exc}"

    if act in {"write", "append"}:
        blocked = _is_protected(target)
        if blocked:
            return (
                f"ERROR: Write access to {blocked} core system files is "
                "denied by safety protocols."
            )

    shadow = get_active_shadow()
    if staging_session and shadow is None:
        shadow = begin_staging_session(staging_session)
    # Prefer an explicit session workspace when ids match a registry entry.
    if staging_session:
        registered = get_staging_session(staging_session)
        if registered is not None:
            shadow = registered

    rel = (
        target.relative_to(_ROOT).as_posix()
        if target.is_relative_to(_ROOT)
        else target.as_posix()
    )

    if act == "read":
        try:
            if shadow is not None:
                staged = shadow.map_path(target)
                if staged.is_file():
                    body = staged.read_text(encoding="utf-8", errors="replace")
                    return (
                        f"OK: read {rel} ({len(body)} chars) [staged]\n"
                        f"{_truncate_file_body(body)}"
                    )
            if not target.is_file():
                return f"ERROR: file not found: {rel}"
            body = target.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001
            return f"ERROR: read failed: {exc}"
        return f"OK: read {rel} ({len(body)} chars)\n{_truncate_file_body(body)}"

    if content is None:
        return f"ERROR: content is required for {act}"

    # Domain clamp: refuse HTML/CSS/JS dumps into Python sources.
    body_text = str(content)
    if target.suffix.lower() in {".py", ".pyi"} and re.search(
        r"(?i)<(?:!DOCTYPE\s+html|html|script|style)\b|```(?:html|css|javascript)\b",
        body_text,
    ):
        return (
            "ERROR: refused HTML/CSS/JS content for Python path "
            f"{rel}; write valid Python only via file_editor"
        )

    # Auto-stage when a session id is provided even without bind_shadow.
    if shadow is None and staging_session:
        shadow = begin_staging_session(staging_session)

    try:
        if shadow is not None:
            if act == "append":
                prior = ""
                staged = shadow.map_path(target)
                if staged.is_file():
                    prior = staged.read_text(encoding="utf-8")
                elif target.is_file():
                    prior = target.read_text(encoding="utf-8", errors="replace")
                shadow.stage_write(target, prior + str(content))
            else:
                shadow.stage_write(target, str(content))
            return (
                f"OK: {act} {len(str(content))} chars to {rel} (shadow staged)"
            )

        # Legacy direct write (no staging context) — kept for non-worker callers.
        target.parent.mkdir(parents=True, exist_ok=True)
        if act == "append" and target.is_file():
            with target.open("a", encoding="utf-8") as fh:
                fh.write(str(content))
        else:
            target.write_text(str(content), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: {act} failed: {exc}"
    return f"OK: {act} {len(str(content))} chars to {rel}"


def transactional_file_tool(session_id: str):
    """Return a worker ``tool_fn(action, filepath, content)`` bound to staging."""

    sid = str(session_id or "").strip() or "worker"
    begin_staging_session(sid)

    def _tool(action: str, filepath: str, content: str | None = None) -> str:
        ws = begin_staging_session(sid)
        with bind_shadow_workspace(ws):
            return file_editor(
                action,
                filepath,
                content,
                staging_session=sid,
            )

    return _tool


__all__ = (
    "begin_staging_session",
    "file_editor",
    "get_staging_session",
    "rollback_workspace",
    "staging_diff_summary",
    "transactional_file_tool",
    "verify_and_commit",
    "verify_staged_content",
)
