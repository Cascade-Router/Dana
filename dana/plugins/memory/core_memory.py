"""Persistent "Core Memory" — a small on-disk key/value store the agent can
autonomously update via the ``update_core_memory`` tool (see
``dana.core.react_dispatch``'s ``_tool_update_core_memory``/``_CORE_TOOL_IDS``)
and that gets read back into every session's system prompt
(``build_system_prompt``). This is the fix for Dana's "session amnesia": a
user preference, an active project's constraints, or a learned workflow,
once written here, survives a server restart — unlike ``messages``, which
lives only for one ``/ws/chat`` connection's lifetime.

Deliberately separate from ``dana.plugins.os.file_system``'s sandboxed
``list_directory``/``read_file``/``write_file`` surface (arbitrary-path,
``os_tools``-gated, HITL-approved text I/O): core memory is one fixed file
the agent's own memory tool manages directly, with no path argument, no
capability gate, and no approval step — it's always available, the same as
``system_state``/``check_plugin_registry`` (see ``_CORE_TOOL_IDS``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dana.paths import AGENT_WORKSPACE_DIR

# A plain module global, not a function-default value — tests monkeypatch
# this directly (same pattern as dana.plugins.os.file_system's
# _SANDBOX_ROOT) to redirect every read/write at a throwaway temp file
# instead of the real one.
CORE_MEMORY_PATH: Path = AGENT_WORKSPACE_DIR / "data" / "core_memory.json"


def read_core_memory() -> dict[str, str]:
    """Returns the current core-memory dict, or ``{}`` if the file doesn't
    exist yet, or contains anything that isn't a flat string->string JSON
    object. Corrupt/foreign content degrades to "no memory yet" rather than
    raising — this is read on EVERY turn's system-prompt build (see
    ``build_system_prompt``) and must never crash a chat turn over a bad
    on-disk file.
    """
    try:
        raw = CORE_MEMORY_PATH.read_text(encoding="utf-8")
    except OSError:
        return {}
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items() if isinstance(k, str) and isinstance(v, str)}


def _write_memory_file(memory: dict[str, str]) -> str | None:
    """Serializes ``memory`` to ``CORE_MEMORY_PATH``, creating the file/
    parent directories on first use. Returns an error string on failure,
    ``None`` on success — the one place either write path below actually
    touches disk, so ``write_core_memory`` (single-section
    read-modify-write) and ``replace_core_memory`` (full overwrite) can
    never drift apart on file path/serialization/error-handling details.
    """
    try:
        CORE_MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        CORE_MEMORY_PATH.write_text(json.dumps(memory, indent=2, sort_keys=True), encoding="utf-8")
    except OSError as exc:
        return f"could not write core memory: {exc}"
    return None


def write_core_memory(section: str, content: str) -> dict[str, Any]:
    """Sets ``section`` to ``content`` in the on-disk core-memory file.

    Read-modify-write over the WHOLE file, no per-section locking — core
    memory is a small, single-agent, low-write-frequency store, not a
    concurrent database, so a lost-update race between two writes in the
    same turn isn't a real risk worth guarding against here. This is the
    agent's OWN write path (the ``update_core_memory`` ReAct tool) — it only
    ever touches one section, never drops any other section already on
    disk. See ``replace_core_memory`` for the frontend MemoryPlugin's full-
    overwrite path.
    """
    section = (section or "").strip()
    if not section:
        return {"ok": False, "error": "section must not be empty"}
    memory = read_core_memory()
    memory[section] = str(content or "")
    error = _write_memory_file(memory)
    if error:
        return {"ok": False, "error": error}
    return {"ok": True, "section": section, "content": memory[section], "memory": memory}


def replace_core_memory(memory: dict[str, Any]) -> dict[str, Any]:
    """Overwrites the ENTIRE on-disk core-memory file with ``memory`` —
    unlike ``write_core_memory``'s single-section read-modify-write, any
    section currently on disk but absent from ``memory`` is dropped. This
    is dana.api.memory's ``POST /api/memory`` write path: the frontend's
    MemoryPlugin always edits a full local copy of the dict (add/edit/
    delete a section) and saves the whole thing back at once, so a full
    replace is the correct semantics there — never used by the agent's own
    ``update_core_memory`` tool, which must never accidentally erase a
    section it wasn't asked to touch.

    Keys/values are coerced to ``str`` the same way ``read_core_memory``
    already tolerates on read (non-string keys dropped, values stringified)
    so a round-trip through this function can never write a file
    ``read_core_memory`` itself would then reject.
    """
    clean = {str(k): str(v) for k, v in (memory or {}).items() if isinstance(k, str)}
    error = _write_memory_file(clean)
    if error:
        return {"ok": False, "error": error}
    return {"ok": True, "memory": clean}


def format_core_memory_for_prompt(memory: dict[str, str] | None = None) -> str:
    """Renders ``memory`` (defaults to a fresh ``read_core_memory()``) as the
    "## Persistent Core Memory" block ``build_system_prompt`` appends to the
    bottom of every turn's system prompt. Returns ``""`` when there's
    nothing to show — callers should skip appending an empty section header
    entirely rather than show the model a heading with no content under it.
    """
    memory = read_core_memory() if memory is None else memory
    if not memory:
        return ""
    lines = ["## Persistent Core Memory"]
    lines.extend(f"- {section}: {content}" for section, content in sorted(memory.items()))
    return "\n".join(lines)


__all__ = (
    "CORE_MEMORY_PATH",
    "format_core_memory_for_prompt",
    "read_core_memory",
    "replace_core_memory",
    "write_core_memory",
)
