"""REST API for the frontend's MemoryPlugin — direct user visibility into,
and control over, the agent's persistent Core Memory
(``dana.plugins.memory.core_memory``): the "black box" transparency goal.

This is the SAME on-disk file the agent's own ``update_core_memory`` ReAct
tool writes into (see ``dana.core.react_dispatch``) and the same content
every future session's system prompt reads back
(``build_system_prompt``/``format_core_memory_for_prompt``) — this router
adds no separate storage, path, or serialization logic of its own; both
endpoints go straight through ``core_memory``'s own read/write helpers so
the two write paths (this API, and the agent's own tool call) can never
drift apart on what a valid on-disk file looks like.

``GET`` returns the current dict as-is. ``POST`` is a full overwrite — the
frontend's MemoryPlugin always edits a full local copy (add/edit/delete a
section) and saves the whole thing back at once — unlike the agent's own
``update_core_memory`` tool, which only ever read-modify-writes one
section. See ``dana.plugins.memory.core_memory.replace_core_memory`` for
why that distinction matters (a full overwrite here must never be the same
code path a single autonomous tool call uses, or one errant PUT could wipe
out every other section the agent had saved).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from dana.plugins.memory.core_memory import read_core_memory, replace_core_memory

router = APIRouter(prefix="/api/memory", tags=["memory"])


@router.get("")
def get_memory() -> dict[str, Any]:
    return {"ok": True, "memory": read_core_memory()}


@router.post("")
def post_memory(memory: dict[str, str]) -> dict[str, Any]:
    return replace_core_memory(memory)


__all__ = ("router",)
