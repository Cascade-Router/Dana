"""Sandboxed file-system tools backing Dana's real "os_tools" capability
domain (see dana.core.react_dispatch's _OS_TOOLS_TOOL_IDS/_CAPABILITY_TOOL_IDS)
— list_directory, read_file, write_file, edit_file, search_files.

Every path is confined to AGENT_WORKSPACE_DIR (dana.paths) — a dedicated
subdirectory of Dana's own workspace tree, deliberately NOT the whole
DANA_WORKSPACE/PROJECT_ROOT: that also contains .env (a real OpenAI API
key — see dana.core.model_provider's BYOK wiring), .git, and Dana's own
source. An LLM-driven read/write tool has no business reaching any of
that, so it gets its own narrower root instead.

Dynamic Workspace Mounting (dana.api.workspace) lets a user additionally
grant access to specific EXTERNAL absolute directories on top of that —
``resolve_sandboxed_path``'s ``allowed_mounts`` param is the single place
that trust decision is enforced; see its docstring for exactly how an
absolute path is (and is not) allowed through.

Only pathlib file I/O lives here — no subprocess/shell execution of any
kind, ever.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dana.paths import AGENT_WORKSPACE_DIR

# A plain module global, not a function-default value — tests monkeypatch
# this directly (see tests/plugins/os/test_file_system.py) to redirect
# every operation at a throwaway temp directory instead of the real one.
_SANDBOX_ROOT = AGENT_WORKSPACE_DIR.resolve()
_SANDBOX_ROOT.mkdir(parents=True, exist_ok=True)


class PathEscapeError(ValueError):
    """A requested path would resolve outside every trusted root (the
    sandbox, and any currently-mounted external directory)."""


def _resolve_mount_roots(allowed_mounts: list[str] | None) -> list[Path]:
    """Best-effort resolve of each dynamically-trusted mount path — an
    entry that's no longer a valid path on this machine (unmounted drive,
    deleted directory, malformed string) is silently skipped rather than
    raising, since a stale mounts.json entry must never crash an otherwise
    legitimate sandboxed-workspace file operation.
    """
    roots: list[Path] = []
    for mount in allowed_mounts or ():
        if not isinstance(mount, str) or not mount.strip():
            continue
        try:
            roots.append(Path(mount).resolve())
        except OSError:
            continue
    return roots


def resolve_sandboxed_path(raw_path: str, allowed_mounts: list[str] | None = None) -> Path:
    """Resolves ``raw_path`` against ``_SANDBOX_ROOT`` (Dana's own agent
    workspace) OR, if given, one of ``allowed_mounts`` — the absolute
    external directories a user has explicitly granted access to via
    Dynamic Workspace Mounting (``dana.api.workspace.mount_workspace_directory``,
    threaded down through ``dana.core.react_dispatch.dispatch_tool_call``'s
    own ``allowed_mounts`` param the same way BYOK ``api_keys`` already is).

    A relative ``raw_path`` is always interpreted relative to
    ``_SANDBOX_ROOT`` first, chroot-style, same as before this parameter
    existed. An absolute ``raw_path`` is no longer rejected outright the
    way it used to be: it's resolved (following symlinks, collapsing
    ``.``/``..``) and then checked against every trusted root —
    ``_SANDBOX_ROOT``, then each of ``allowed_mounts``, in order — allowed
    the moment ``relative_to()`` succeeds against any one of them. An
    absolute path outside every trusted root (no mounts registered, or one
    that resolves outside all of them) still raises ``PathEscapeError``,
    exactly like a relative ``..`` traversal past the sandbox root already
    did — path containment is checked AFTER ``Path.resolve()`` normalizes
    the candidate, not before, so a symlink pointing back out of a
    mounted directory is caught the same way a symlink escaping the
    sandbox itself already was. Raises ``PathEscapeError`` with a clear
    message on any violation; never silently clamps or truncates.
    """
    raw = (raw_path or "").strip()
    if not raw:
        raise PathEscapeError("path must not be empty")

    parsed = Path(raw)
    candidate = parsed.resolve() if parsed.is_absolute() else (_SANDBOX_ROOT / parsed).resolve()

    trusted_roots = [_SANDBOX_ROOT, *_resolve_mount_roots(allowed_mounts)]
    for root in trusted_roots:
        try:
            candidate.relative_to(root)
            return candidate
        except ValueError:
            continue
    raise PathEscapeError(
        f"path {raw!r} resolves outside the sandbox ({_SANDBOX_ROOT}) and every mounted directory"
    )


def list_directory(path: str, allowed_mounts: list[str] | None = None) -> dict[str, Any]:
    """Lists the immediate files/subdirectories under ``path``. Read-only."""
    try:
        target = resolve_sandboxed_path(path, allowed_mounts)
    except PathEscapeError as exc:
        return {"ok": False, "error": str(exc)}
    if not target.exists():
        return {"ok": False, "error": f"path does not exist: {path!r}"}
    if not target.is_dir():
        return {"ok": False, "error": f"path is not a directory: {path!r}"}
    entries = [
        {"name": child.name, "type": "directory" if child.is_dir() else "file"}
        for child in sorted(target.iterdir(), key=lambda p: p.name)
    ]
    return {"ok": True, "path": path, "entries": entries}


def read_file(path: str, allowed_mounts: list[str] | None = None) -> dict[str, Any]:
    """Returns the text content of a file under the sandbox root (or a
    mounted directory). Read-only. Decoding is best-effort: invalid UTF-8
    bytes are replaced with the standard replacement character instead of
    raising.
    """
    try:
        target = resolve_sandboxed_path(path, allowed_mounts)
    except PathEscapeError as exc:
        return {"ok": False, "error": str(exc)}
    if not target.exists():
        return {"ok": False, "error": f"file does not exist: {path!r}"}
    if not target.is_file():
        return {"ok": False, "error": f"path is not a file: {path!r}"}
    try:
        raw_bytes = target.read_bytes()
    except OSError as exc:
        return {"ok": False, "error": f"could not read file: {exc}"}
    return {"ok": True, "path": path, "content": raw_bytes.decode("utf-8", errors="replace")}


def write_file(path: str, content: str, allowed_mounts: list[str] | None = None) -> dict[str, Any]:
    """Writes ``content`` (UTF-8 text) to a file under the sandbox root (or
    a mounted directory), creating parent directories as needed. MUTATING —
    tools.json declares no "read_only": true for this tool, so
    dana.core.react_dispatch.is_mutating_tool's fail-closed schema check
    gates it, and the ReAct loop suspends for explicit user approval
    before this ever actually runs.
    """
    try:
        target = resolve_sandboxed_path(path, allowed_mounts)
    except PathEscapeError as exc:
        return {"ok": False, "error": str(exc)}
    if target.is_dir():
        return {"ok": False, "error": f"path is a directory, not a file: {path!r}"}
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    except OSError as exc:
        return {"ok": False, "error": f"could not write file: {exc}"}
    return {"ok": True, "path": path, "bytes_written": len(content.encode("utf-8"))}


def edit_file(
    path: str, search_block: str, replace_block: str, allowed_mounts: list[str] | None = None
) -> dict[str, Any]:
    """Surgical search-and-replace edit on an EXISTING file, in place of
    regenerating and rewriting its entire content via ``write_file`` — the
    intended fix for a 1000-line file needing a one-line change: no reason
    to burn tokens (and risk a truncated/"..." LLM output silently
    destroying the rest of the file) reproducing the other 999 lines
    verbatim just to change one of them.

    Deliberately a strict, literal ``str.count()``/``str.replace()`` — NOT
    a diff/patch library or an AST-aware transform. A plain string match is
    something the calling LLM can reason about and self-correct on failure
    (it already has to reproduce ``search_block`` byte-for-byte to succeed,
    so it can just as easily read back why zero or 2+ matches happened);
    a heavier diffing/AST layer would hide exactly how the match failed
    behind machinery this tool has no business depending on.

    MUTATING — gated by the same fail-closed schema check as write_file
    (dana.core.react_dispatch.is_mutating_tool), so the ReAct loop suspends
    for explicit user approval before this ever actually runs.

    Reads and writes raw bytes (not ``Path.write_text``'s platform-default
    text mode) precisely so the file's EXISTING, un-edited bytes round-trip
    unchanged: ``write_text`` re-translates every bare ``\\n`` in the string
    to the platform newline on write, which would double up any line
    already using ``\\r\\n`` that got preserved as literal characters by
    the plain ``decode()`` below — silently corrupting every OTHER line in
    the file on Windows, not just the one actually being edited.

    Validation is strict on purpose: ``search_block`` must match the
    file's current content EXACTLY ONCE. Zero matches almost always means
    the LLM's copy of the "original" text has already drifted from what's
    actually on disk (stale context, whitespace difference, ...); 2+
    matches means the requested change is ambiguous — which occurrence was
    meant is not something this tool guesses at. Either way this returns a
    normal digested failure (never touching the file) so the model can
    read the error, re-read the file if needed, and retry with a more
    unique ``search_block`` — never silently editing the wrong location.
    """
    try:
        target = resolve_sandboxed_path(path, allowed_mounts)
    except PathEscapeError as exc:
        return {"ok": False, "error": str(exc)}
    if not target.exists():
        return {"ok": False, "error": f"file does not exist: {path!r}"}
    if not target.is_file():
        return {"ok": False, "error": f"path is not a file: {path!r}"}
    try:
        raw_bytes = target.read_bytes()
    except OSError as exc:
        return {"ok": False, "error": f"could not read file: {exc}"}

    original = raw_bytes.decode("utf-8", errors="replace")
    if original.count(search_block) != 1:
        return {
            "ok": False,
            "error": "Search block not found or multiple matches found. Provide a more unique search block.",
        }

    updated = original.replace(search_block, replace_block, 1)
    try:
        target.write_bytes(updated.encode("utf-8"))
    except OSError as exc:
        return {"ok": False, "error": f"could not write file: {exc}"}
    return {"ok": True, "path": path, "bytes_written": len(updated.encode("utf-8"))}


# Directory NAMES (not full paths — matched at every depth) never worth
# descending into for a text search: dependency/build trees that are
# typically huge, mostly-binary or generated, and never what "find this
# variable/function" is actually asking about. Any directory whose name
# starts with "." (.git, .venv, .idea, ...) is skipped by a separate,
# simpler rule in _iter_searchable_files below — kept there rather than
# folded into this set since "starts with a dot" already covers .venv
# without needing to enumerate every dotfile convention by name.
_SKIP_DIR_NAMES = frozenset({"node_modules", "__pycache__", "venv"})

# Hard cap on returned matches (Payload Limits) — a query that's too
# generic (e.g. a common short identifier) across a large mounted codebase
# could otherwise return thousands of hits, which would blow up the LLM's
# context window for no benefit: past the first ~50, the model should
# narrow the query rather than read more of the same signal.
_MAX_SEARCH_MATCHES = 50

# Per-match "content" snippet is hard-truncated at this length — a single
# absurdly long line (a minified bundle, a generated data file) must not
# blow up one match's payload even though the overall match COUNT is
# already capped above.
_MAX_MATCH_SNIPPET_CHARS = 200

# Files larger than this are skipped outright, before ever being read into
# memory line-by-line — a search tool has no business choking on a
# multi-hundred-MB log/bundle file that happened to live outside a
# _SKIP_DIR_NAMES directory (e.g. directly inside a mounted repo).
_MAX_SEARCHABLE_FILE_BYTES = 2 * 1024 * 1024


def _iter_searchable_files(root: Path):
    """Depth-first walk of ``root``, yielding every regular file — pruning
    hidden directories (name starts with ``.``, e.g. ``.git``/``.venv``)
    and ``_SKIP_DIR_NAMES`` (``node_modules``/``__pycache__``/``venv``)
    BEFORE recursing into them, not merely filtering them out of the
    results afterward — the whole point being to never even walk into a
    multi-hundred-thousand-file ``node_modules`` tree in the first place.
    A directory pathlib can't list (permission error, broken symlink) is
    silently skipped rather than aborting the whole search.
    """
    try:
        entries = sorted(root.iterdir(), key=lambda p: p.name)
    except OSError:
        return
    for entry in entries:
        if entry.is_dir():
            if entry.name.startswith(".") or entry.name in _SKIP_DIR_NAMES:
                continue
            yield from _iter_searchable_files(entry)
        elif entry.is_file():
            yield entry


def search_files(
    directory_path: str, query: str, allowed_mounts: list[str] | None = None
) -> dict[str, Any]:
    """Recursively greps ``directory_path`` for ``query`` (case-insensitive
    substring match) — a fast way to LOCATE where something lives in a
    large (possibly mounted-external) codebase before spending a whole
    ReAct turn guessing directory structure with list_directory, or wasting
    an edit_file call on a search_block that turns out not to exist where
    the model assumed it would.

    Pure Python (pathlib + string matching) — deliberately never shells out
    to a real ``grep``/``ripgrep`` binary, so this behaves identically
    regardless of what's actually installed on the host OS.

    Read-only reconnaissance: unlike write_file/edit_file, its tools.json
    entry declares "read_only": true, so dana.core.react_dispatch.
    is_mutating_tool's fail-closed schema check lets it dispatch
    immediately with no HITL approval.

    Graceful degradation: each candidate file is opened and decoded as
    STRICT UTF-8 (unlike read_file/edit_file's lenient
    ``errors="replace"``) — a binary file (image, compiled artifact, ...)
    will almost always fail that decode, and the intent here is to
    actually SKIP it, not search a garbled replacement-character rendering
    of its bytes for false-positive matches. A file over
    ``_MAX_SEARCHABLE_FILE_BYTES`` is skipped the same way, before ever
    being read into memory. Every skip (decode failure, read error,
    oversized file, unreadable directory) is silent, never aborting the
    rest of the search.

    Returns at most ``_MAX_SEARCH_MATCHES`` matches, each
    ``{"file": ..., "line": ..., "content": ...}``; ``"truncated": True``
    on the result if the cap was hit, so the model knows to narrow its
    query rather than assume it just saw every occurrence. ``"file"`` is
    the SANDBOX-RELATIVE path when the match is under ``_SANDBOX_ROOT``, or
    the absolute path when it's under a mount — in both cases, exactly the
    string read_file/write_file/edit_file's own ``path`` argument expects,
    so the model can act on a match with no path arithmetic of its own.
    """
    try:
        target = resolve_sandboxed_path(directory_path, allowed_mounts)
    except PathEscapeError as exc:
        return {"ok": False, "error": str(exc)}
    if not target.exists():
        return {"ok": False, "error": f"path does not exist: {directory_path!r}"}
    if not target.is_dir():
        return {"ok": False, "error": f"path is not a directory: {directory_path!r}"}

    needle = (query or "").strip().lower()
    if not needle:
        return {"ok": False, "error": "query must not be empty"}

    matches: list[dict[str, Any]] = []
    truncated = False
    for file_path in _iter_searchable_files(target):
        try:
            if file_path.stat().st_size > _MAX_SEARCHABLE_FILE_BYTES:
                continue
            text = file_path.read_bytes().decode("utf-8")
        except (OSError, UnicodeDecodeError):
            continue  # unreadable or not a text file — skip, don't abort the search

        try:
            file_id = file_path.relative_to(_SANDBOX_ROOT).as_posix()
        except ValueError:
            file_id = str(file_path)

        for line_number, line in enumerate(text.splitlines(), start=1):
            if needle not in line.lower():
                continue
            snippet = line.strip()
            if len(snippet) > _MAX_MATCH_SNIPPET_CHARS:
                snippet = snippet[: _MAX_MATCH_SNIPPET_CHARS - 1] + "…"
            matches.append({"file": file_id, "line": line_number, "content": snippet})
            if len(matches) >= _MAX_SEARCH_MATCHES:
                truncated = True
                break
        if truncated:
            break

    return {"ok": True, "path": directory_path, "matches": matches, "truncated": truncated}


__all__ = (
    "PathEscapeError",
    "resolve_sandboxed_path",
    "list_directory",
    "read_file",
    "write_file",
    "edit_file",
    "search_files",
)
