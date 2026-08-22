"""Task Planner / Executive Function — a structured scratchpad the agent
manages FOR ITSELF across a long-horizon, multi-turn goal ("build a full
web app from scratch"), backing ``create_plan``/``mark_task_completed``
(see ``dana.core.react_dispatch``'s ``_CORE_TOOL_IDS``).

The problem this exists to fix: Dana's ReAct loop re-sends the system
prompt plus the running conversation on every turn, but a sufficiently
long project's raw tool-call history is a poor substitute for "what step
am I actually on" — a model re-deriving its place from a wall of past
tool_result messages is exactly how it starts hallucinating progress,
repeating a step it already did, or wandering off the original objective.
A plan this module tracks gets rendered straight into EVERY turn's system
prompt (``dana.core.react_dispatch.build_system_prompt``'s
"## Current Active Plan" block) as an explicit anchor, instead.

Deliberately a single GLOBAL plan, not one keyed per chat session: this is
the agent's own executive-function scratchpad for whatever long-horizon
objective it is CURRENTLY working, the same way there is only one
Persistent Core Memory store (``dana.plugins.memory.core_memory``) —
multiple concurrent independent plans is a real feature, but a larger one
than "give the agent a checklist so it doesn't lose its place," which is
what was asked for here. Also deliberately IN-MEMORY only, not persisted
to disk (same choice ``dana.plugins.os.background_services`` made for its
own ``_ACTIVE_PROCESSES``): a server restart clearing the current plan is
an acceptable trade for the simplicity of not needing a second on-disk
format to keep in sync with this module's own in-memory shape.

Tool schemas are deliberately minimal (a list of plain strings for
``create_plan``, two integers for ``mark_task_completed``) — the LLM never
has to construct or edit a nested JSON task object itself; every ``id``/
``status`` field is assigned and mutated ENTIRELY by this module.
"""

from __future__ import annotations

from typing import Any, Literal

TaskStatus = Literal["pending", "active", "completed"]

# A plain module global, not a function-default value — tests read/mutate
# this directly (see tests/api/test_planner_api.py) the same way
# dana.plugins.os.background_services's _ACTIVE_PROCESSES already is.
# Shape: {"objective": str, "tasks": [{"id": int, "description": str,
# "status": "pending"|"active"|"completed"}, ...], "current_task_id": int | None}.
_ACTIVE_PLAN: dict[str, Any] = {"objective": "", "tasks": [], "current_task_id": None}


def _snapshot_plan() -> dict[str, Any]:
    """A shallow-but-safe COPY of ``_ACTIVE_PLAN`` — every caller (the
    REST API, the system-prompt renderer, a tool's own return payload)
    gets its own list/dict instances, so nothing downstream can mutate
    this module's actual state by holding onto (and editing) a returned
    reference.
    """
    return {
        "objective": _ACTIVE_PLAN["objective"],
        "tasks": [dict(task) for task in _ACTIVE_PLAN["tasks"]],
        "current_task_id": _ACTIVE_PLAN["current_task_id"],
    }


def get_active_plan() -> dict[str, Any]:
    """Read-only accessor — the ONE place both ``dana.api.planner``'s
    ``GET /api/planner`` and ``dana.core.react_dispatch.build_system_prompt``
    read the current plan from, so the REST API and the LLM's own system
    prompt can never drift apart on what "the current plan" means.
    """
    return _snapshot_plan()


def create_plan(objective: str, tasks: list[str]) -> dict[str, Any]:
    """Overwrites the active plan with a fresh ``objective`` and ordered
    ``tasks`` list — starting a NEW long-horizon goal always replaces
    whatever plan (if any) was active before, rather than merging with it;
    an agent that wants to preserve unfinished work from a previous plan
    should finish or explicitly note it (e.g. via ``update_core_memory``)
    before replacing it.

    Each task string becomes ``{"id": <1-based position>, "description":
    <the string>, "status": "pending"}`` — task ids are assigned here,
    purely by list position; the LLM never invents or tracks its own ids.
    The FIRST task is immediately promoted to ``"active"`` (and
    ``current_task_id`` set to its id) — a freshly created plan always
    starts already pointed at its own first step, needing no separate
    call to begin.
    """
    clean_objective = (objective or "").strip()
    if not clean_objective:
        return {"ok": False, "error": "objective must not be empty"}

    clean_tasks = [t.strip() for t in (tasks or []) if isinstance(t, str) and t.strip()]
    if not clean_tasks:
        return {"ok": False, "error": "tasks must be a non-empty list of non-empty strings"}

    task_objs: list[dict[str, Any]] = [
        {"id": position, "description": description, "status": "pending"}
        for position, description in enumerate(clean_tasks, start=1)
    ]
    task_objs[0]["status"] = "active"

    _ACTIVE_PLAN["objective"] = clean_objective
    _ACTIVE_PLAN["tasks"] = task_objs
    _ACTIVE_PLAN["current_task_id"] = task_objs[0]["id"]

    return {"ok": True, "plan": _snapshot_plan()}


def mark_task_completed(task_id: int, next_task_id: int | None = None) -> dict[str, Any]:
    """Marks ``task_id`` ``"completed"``, and — only if ``next_task_id`` is
    given — promotes THAT task to ``"active"`` and updates
    ``current_task_id`` to point at it. ``next_task_id`` is optional
    exactly because finishing a task doesn't always mean the next one is
    obvious yet (the agent may need to re-check the plan, or the objective
    is fully done); leaving it out simply clears ``current_task_id`` (if it
    was pointing at the task just completed) rather than guessing which
    task comes next.

    Both ids are validated to actually exist in the CURRENT plan's tasks
    BEFORE anything is mutated — an unknown ``task_id`` OR an unknown
    ``next_task_id`` fails the whole call with no partial state change,
    so a typo'd id can never leave the plan in an inconsistent state (e.g.
    a task marked completed with no new active task actually promoted
    because the id it meant to promote didn't exist).
    """
    tasks = _ACTIVE_PLAN.get("tasks") or []
    if not tasks:
        return {"ok": False, "error": "no active plan — call create_plan first"}

    task = next((t for t in tasks if t["id"] == task_id), None)
    if task is None:
        return {"ok": False, "error": f"no task with id {task_id} in the active plan"}

    next_task: dict[str, Any] | None = None
    if next_task_id is not None:
        if next_task_id == task_id:
            return {"ok": False, "error": "next_task_id must be different from task_id"}
        next_task = next((t for t in tasks if t["id"] == next_task_id), None)
        if next_task is None:
            return {"ok": False, "error": f"no task with id {next_task_id} in the active plan"}

    task["status"] = "completed"
    if next_task is not None:
        next_task["status"] = "active"
        _ACTIVE_PLAN["current_task_id"] = next_task_id
    elif _ACTIVE_PLAN.get("current_task_id") == task_id:
        _ACTIVE_PLAN["current_task_id"] = None

    return {"ok": True, "plan": _snapshot_plan()}


__all__ = ("get_active_plan", "create_plan", "mark_task_completed")
