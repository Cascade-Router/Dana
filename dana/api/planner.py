"""Read-only REST API for the frontend's Planner plugin — direct user
visibility into the agent's Task Planner / Executive Function scratchpad
(``dana.plugins.planning.task_board``).

Adds NO independent state or copy of its own: ``GET /api/planner`` reads
straight through ``task_board.get_active_plan()`` — the EXACT SAME
accessor ``dana.core.react_dispatch.build_system_prompt`` calls to render
the "## Current Active Plan" block the LLM itself sees every turn — so
this plugin's UI and the model's own system-prompt view of the plan can
never drift apart on what "the current plan" is. There is no write
endpoint here on purpose: the plan is mutated ONLY via the agent's own
``create_plan``/``mark_task_completed`` ReAct tools, never directly from
this UI — unlike Background Process Management's Services plugin (which
CAN kill a service the user didn't ask the agent to), letting a human
silently rewrite the agent's own executive-function state out from under
it has no equivalent safe use case.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from dana.plugins.planning.task_board import get_active_plan

router = APIRouter(prefix="/api/planner", tags=["planner"])


@router.get("")
def get_planner_state() -> dict[str, Any]:
    return {"ok": True, "plan": get_active_plan()}


__all__ = ("router",)
