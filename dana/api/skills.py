"""REST API for the frontend's SkillsPlugin — Autonomous Skill Acquisition
transparency: lets the user see (and delete) every skill the agent has
taught itself (dana.core.skill_loader / dana.core.react_dispatch's
"user_skills" capability domain).

``GET`` reflects EXACTLY what's currently registered as a dispatchable tool
(``dana.core.react_dispatch.list_user_skills``) — not an independent
re-scan of disk — so what the user sees here always matches what the agent
can actually call right now. ``DELETE`` removes the skill's ``.py`` file
(``dana.core.skill_loader.delete_skill``, through the SAME
``resolve_sandboxed_path`` traversal check every other skill/os_tools
operation uses) and hot-reloads the registry
(``dana.core.react_dispatch.refresh_user_skills``) — the exact same two-step
sequence the agent's own ``delete_skill`` tool runs, so the two write paths
can never drift apart on what "deleted" actually means.

``PUT`` is the frontend SkillsPlugin's manual-edit path: it overwrites the
skill's file VERBATIM with whatever the user typed
(``dana.core.skill_loader.write_skill_source`` — no ``TOOL_SCHEMA``
reshaping the way the agent's own ``save_new_skill`` tool does), then
hot-reloads the registry regardless of outcome (so a skill just broken by
a bad edit is correctly unloaded, not left silently stale), and returns
``400`` with the SPECIFIC parse/validation error
(``dana.core.skill_loader.validate_skill_file``) if the edited file no
longer loads cleanly.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from dana.core import react_dispatch
from dana.core.skill_loader import (
    delete_skill,
    is_valid_skill_name,
    read_skill_source,
    validate_skill_file,
    write_skill_source,
)

router = APIRouter(prefix="/api/skills", tags=["skills"])


def _skill_summary(tool_id: str, schema: dict[str, Any]) -> dict[str, Any]:
    fn = schema.get("function") if isinstance(schema, dict) else None
    fn = fn if isinstance(fn, dict) else {}
    return {
        "name": tool_id,
        "description": str(fn.get("description") or ""),
        "code": read_skill_source(tool_id) or "",
    }


@router.get("")
def get_skills() -> dict[str, Any]:
    skills = [_skill_summary(tool_id, schema) for tool_id, schema in sorted(react_dispatch.list_user_skills().items())]
    return {"ok": True, "skills": skills}


@router.delete("/{skill_name}")
def delete_skill_endpoint(skill_name: str) -> dict[str, Any]:
    if not is_valid_skill_name(skill_name):
        raise HTTPException(status_code=400, detail=f"invalid skill_name: {skill_name!r}")
    result = delete_skill(skill_name)
    if not result.get("ok"):
        raise HTTPException(status_code=500, detail=result.get("error") or "could not delete skill")
    react_dispatch.refresh_user_skills()
    return {"ok": True, "deleted": result.get("deleted", False)}


@router.put("/{skill_name}")
def put_skill_endpoint(skill_name: str, body: dict[str, str]) -> dict[str, Any]:
    if not is_valid_skill_name(skill_name):
        raise HTTPException(status_code=400, detail=f"invalid skill_name: {skill_name!r}")
    code = body.get("code")
    if not isinstance(code, str) or not code.strip():
        raise HTTPException(status_code=400, detail="request body must include a non-empty 'code' string")

    write_result = write_skill_source(skill_name, code)
    if not write_result.get("ok"):
        raise HTTPException(status_code=400, detail=write_result.get("error") or "could not write skill file")

    # Always re-check + hot-reload, even on failure below — a skill just
    # broken by this edit must be dropped from the registry, not left
    # silently dispatchable with its PREVIOUS (now overwritten) code.
    validation_error = validate_skill_file(skill_name)
    react_dispatch.refresh_user_skills()
    if validation_error:
        raise HTTPException(status_code=400, detail=validation_error)
    return {"ok": True, "skill_name": skill_name}


__all__ = ("router",)
