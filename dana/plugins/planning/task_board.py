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

Keyed by ``session_id`` (Concurrency Bug fix): this used to be a single
GLOBAL plan, on the reasoning that it's the agent's own executive-function
scratchpad for whatever long-horizon objective it is CURRENTLY working,
the same way there is only one Persistent Core Memory store
(``dana.plugins.memory.core_memory``). That reasoning silently assumed
one plan is ever "current" at a time process-wide — false the moment two
chat sessions are open concurrently: confirmed live (dana_runtime.log,
2026-09-13) that a SECOND, unrelated session's ``create_plan`` call was
flatly rejected ("Cannot overwrite an active plan") solely because a
FIRST session's plan — from an earlier, already-abandoned/failed turn —
was still sitting in this one global slot. Each ``session_id`` now gets
its own independent scratchpad (``_PLANS_BY_SESSION``, keyed exactly like
``dana.plugins.freecad.engine``'s own per-session ``Session_Active.FCStd``
already is), created lazily on first touch and never seen by any other
session. Still deliberately IN-MEMORY only, not persisted to disk (same
choice ``dana.plugins.os.background_services`` made for its own
``_ACTIVE_PROCESSES``): a server restart clearing every session's current
plan is an acceptable trade for the simplicity of not needing a second
on-disk format to keep in sync with this module's own in-memory shape.

Tool schemas are deliberately minimal (a list of plain strings for
``create_plan``, two integers for ``mark_task_completed``) — the LLM never
has to construct or edit a nested JSON task object itself; every ``id``/
``status`` field is assigned and mutated ENTIRELY by this module.
"""

from __future__ import annotations

from typing import Any, Literal

from dana.session_context import get_session_id

TaskStatus = Literal["pending", "active", "completed", "cancelled"]


def _new_plan() -> dict[str, Any]:
    """A fresh, empty plan slot — same shape every session's own entry in
    ``_PLANS_BY_SESSION`` starts as.

    Shape: {"objective": str, "tasks": [{"id": int, "description": str,
    "status": "pending"|"active"|"completed"|"cancelled"}, ...],
    "current_task_id": int | None, "auto_seeded": bool}.
    """
    return {
        "objective": "",
        "tasks": [],
        "current_task_id": None,
        # Auto-Seed Handoff: True only when THIS plan was planted by dana.api.
        # server._process_user_text's _looks_multi_step heuristic before the
        # model ever got a turn -- a placeholder anchor, never a plan the
        # model itself committed to. Lets _tool_create_plan's Plan Immutability
        # guard (dana.core.react_dispatch) tell "the model's own in-progress
        # plan" (protected) apart from "a heuristic guess sitting in a
        # genuinely idle slot" (safe to replace) -- see that guard's own
        # comment for why they look identical without this flag.
        "auto_seeded": False,
        # Monotonic Task IDs (Dynamic FSM Replanning): the next id
        # insert_task will hand out. create_plan resets this to len(tasks)+1
        # every time it (re)creates a plan from scratch — a fresh plan's
        # ids still start at 1 and run consecutively exactly as before,
        # this only starts diverging from "id == list position" once
        # insert_task allocates one beyond the original set. Deliberately
        # NOT named anything containing "next_task_id" — mark_task_completed
        # already has a parameter by that name meaning something else
        # entirely (which task to activate next), and this is an id
        # ALLOCATOR, not a task reference.
        "_id_seq": 1,
    }


# Every session's own plan, keyed by session_id — tests read/mutate this
# directly (see tests/conftest.py's _reset_task_board_plan, which now
# clears the whole dict rather than one plan's fields) the same way
# dana.plugins.os.background_services's _ACTIVE_PROCESSES already is.
_PLANS_BY_SESSION: dict[str, dict[str, Any]] = {}


def _plan_for(session_id: str | None) -> dict[str, Any]:
    """This session's own plan slot, created empty on first touch.
    ``session_id=None`` (every call site inside the real dispatch_tool_call
    chain — see dana.session_context's own module docstring for exactly
    which call sites that covers) resolves the AMBIENT current session via
    ``get_session_id()``, same convention
    ``dana.plugins.freecad.engine._session_dir()`` already uses for its own
    per-session state. A caller running OUTSIDE that chain (dana.api.
    server._process_user_text, before this turn's first tool dispatch has
    called ``set_session_id`` for it) must pass its own real ``session_id``
    explicitly instead of relying on this default — exactly the same
    "explicit off the WebSocket call chain, ambient on it" split
    ``dana.api.cad``'s REST routes already draw.
    """
    sid = session_id if session_id is not None else get_session_id()
    return _PLANS_BY_SESSION.setdefault(sid, _new_plan())


def _snapshot_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """A shallow-but-safe COPY of one session's plan — every caller (the
    REST API, the system-prompt renderer, a tool's own return payload)
    gets its own list/dict instances, so nothing downstream can mutate
    this module's actual state by holding onto (and editing) a returned
    reference.
    """
    return {
        "objective": plan["objective"],
        "tasks": [dict(task) for task in plan["tasks"]],
        "current_task_id": plan["current_task_id"],
        "auto_seeded": plan["auto_seeded"],
    }


def get_active_plan(session_id: str | None = None) -> dict[str, Any]:
    """Read-only accessor for THIS session's own active plan — the ONE
    place both ``dana.api.planner``'s ``GET /api/planner`` and
    ``dana.core.react_dispatch.build_system_prompt`` read the current plan
    from, so the REST API and the LLM's own system prompt can never drift
    apart on what "the current plan" means for a given session.
    """
    return _snapshot_plan(_plan_for(session_id))


def create_plan(
    objective: str, tasks: list[str], *, auto_seeded: bool = False, session_id: str | None = None
) -> dict[str, Any]:
    """Overwrites THIS session's own active plan with a fresh ``objective``
    and ordered ``tasks`` list — starting a NEW long-horizon goal always
    replaces whatever plan (if any) was active before FOR THIS SESSION,
    rather than merging with it, and never touches any OTHER session's own
    plan; an agent that wants to preserve unfinished work from a previous
    plan should finish or explicitly note it (e.g. via
    ``update_core_memory``) before replacing it.

    Each task string becomes ``{"id": <1-based position>, "description":
    <the string>, "status": "pending"}`` — task ids are assigned here,
    purely by list position; the LLM never invents or tracks its own ids.
    The FIRST task is immediately promoted to ``"active"`` (and
    ``current_task_id`` set to its id) — a freshly created plan always
    starts already pointed at its own first step, needing no separate
    call to begin.

    ``auto_seeded`` (default ``False``, the LLM-driven path via
    ``dana.core.react_dispatch._tool_create_plan``) is ``True`` ONLY for
    ``dana.api.server``'s own heuristic pre-seed — see ``_new_plan``'s own
    comment on the flag this sets. A call that replaces an auto-seeded
    plan with a real one (the common case right after) correctly flips
    this back to whatever THIS call passes, so the model's own plan is
    never mistaken for a placeholder afterward.

    ``session_id``: see ``_plan_for``'s own docstring — omit it for any
    caller inside the real dispatch_tool_call chain (the ambient current
    session is correct there), pass it explicitly for a caller running
    before this turn's first tool dispatch (``_process_user_text``'s own
    auto-seed heuristic).
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

    plan = _plan_for(session_id)
    plan["objective"] = clean_objective
    plan["tasks"] = task_objs
    plan["current_task_id"] = task_objs[0]["id"]
    plan["auto_seeded"] = auto_seeded
    # Monotonic Task IDs: a brand-new plan still numbers its own tasks
    # 1..len(tasks) exactly as before (id == list position for the
    # initial set) — the allocator only starts handing out ids beyond
    # that once insert_task is used against THIS plan.
    plan["_id_seq"] = len(task_objs) + 1

    return {"ok": True, "plan": _snapshot_plan(plan)}


def insert_task(
    description: str,
    insert_after_task_id: int | None = None,
    *,
    insert_before_task_id: int | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Splices one new task into THIS session's own active plan — Dynamic
    FSM Replanning's core primitive: the one legal way a plan already in
    progress grows, instead of the model smuggling the missing work into
    an unrelated task's own execution window (confirmed live,
    dana_runtime.log session 41d1967d).

    Exactly one of ``insert_after_task_id``/``insert_before_task_id`` must
    be given.

    The new task's ``id`` comes from this plan's own monotonic
    ``_id_seq`` counter, NEVER from list position — a later task's id
    stays stable even after an earlier insertion shifts its position in
    the list, so an id the model already saw earlier this conversation
    never silently starts meaning a different task.

    ``insert_after_task_id`` must name an existing, NOT-``"completed"``
    task — inserting after completed history would let the model splice
    "past" work back into what still needs doing, which is a rewrite of
    the record, not a legitimate plan fix. (Cancelled tasks ARE a valid
    insertion point — inserting after one is just inserting into the
    remaining, still-open part of the plan.) The new task is always
    ``"pending"`` on this path.

    ``insert_before_task_id`` — FSM Recovery / Prerequisite Insertion — is
    the one way to splice a task BEFORE one that's already ``"active"``,
    covering exactly the case ``insert_after_task_id`` structurally cannot
    reach: a prerequisite discovered only after the FSM already advanced
    past the point where it belonged (confirmed live: an ``insert_task``
    aimed at the task before an already-``"active"`` one was rejected
    outright, then ``cancel_pending_task`` on the active one was ALSO
    rejected — "insert around it" and "cancel it" were both structurally
    impossible once a task went active, a genuine deadlock). Targeting the
    currently ``"active"`` task demotes it straight back to ``"pending"``
    and promotes the newly inserted task to ``"active"`` in the same call
    (``current_task_id`` updated to match) — there is always exactly one
    active task, so "insert before active" and "swap what's active" are
    the same operation, not two. Targeting a merely ``"pending"`` task
    instead just splices a new pending task ahead of it, no promotion
    involved. Rejects a ``"completed"``/``"cancelled"`` anchor either way
    — there is no reasonable position "before" finished or dropped work.
    """
    plan = _plan_for(session_id)
    tasks = plan.get("tasks") or []
    if not tasks:
        return {"ok": False, "error": "no active plan — call create_plan first"}

    if (insert_after_task_id is None) == (insert_before_task_id is None):
        return {
            "ok": False,
            "error": "insert_task requires EXACTLY ONE of insert_after_task_id or insert_before_task_id",
        }

    clean_description = (description or "").strip()
    if not clean_description:
        return {"ok": False, "error": "description must not be empty"}

    if insert_after_task_id is not None:
        anchor = next((t for t in tasks if t["id"] == insert_after_task_id), None)
        if anchor is None:
            return {"ok": False, "error": f"no task with id {insert_after_task_id} in the active plan"}
        if anchor["status"] == "completed":
            return {
                "ok": False,
                "error": (
                    f"cannot insert a task after task {insert_after_task_id} — it is already completed. "
                    "Insert after the current active task, or another still-open one, instead."
                ),
            }

        new_id = plan["_id_seq"]
        plan["_id_seq"] = new_id + 1
        new_task = {"id": new_id, "description": clean_description, "status": "pending"}
        position = tasks.index(anchor) + 1
        tasks.insert(position, new_task)
        return {"ok": True, "plan": _snapshot_plan(plan), "inserted_task_id": new_id}

    # insert_before_task_id path — FSM Recovery / Prerequisite Insertion.
    anchor = next((t for t in tasks if t["id"] == insert_before_task_id), None)
    if anchor is None:
        return {"ok": False, "error": f"no task with id {insert_before_task_id} in the active plan"}
    if anchor["status"] in ("completed", "cancelled"):
        return {
            "ok": False,
            "error": (
                f"cannot insert a task before task {insert_before_task_id} — it is "
                f"{anchor['status']!r}. Insert before the current active task, or another still-"
                "pending one, instead."
            ),
        }

    new_id = plan["_id_seq"]
    plan["_id_seq"] = new_id + 1
    activates = anchor["status"] == "active"
    new_task = {"id": new_id, "description": clean_description, "status": "active" if activates else "pending"}
    position = tasks.index(anchor)
    tasks.insert(position, new_task)

    result: dict[str, Any] = {"ok": True, "inserted_task_id": new_id}
    if activates:
        # State Shift: the anchor was the one and only "active" task —
        # demoting it to "pending" (never "completed", this function has no
        # evidence its work is actually done) and pointing current_task_id
        # at the new task keeps the plan's "exactly one active task"
        # invariant intact across the swap, the same invariant
        # mark_task_completed's own "demote any OTHER active task" step
        # protects on its side of the FSM.
        anchor["status"] = "pending"
        plan["current_task_id"] = new_id
        result["demoted_task_id"] = anchor["id"]
    result["plan"] = _snapshot_plan(plan)
    return result


def cancel_pending_task(
    task_id_to_cancel: int, reason: str, *, session_id: str | None = None
) -> dict[str, Any]:
    """Marks a still-``"pending"`` task ``"cancelled"`` in THIS session's
    own active plan — Dynamic FSM Replanning's other half: the legal way
    to drop a task the model now realizes it doesn't actually need,
    instead of leaving it sitting there forever un-completable (or
    completing it dishonestly just to clear it).

    Deliberately narrower than an ``edit``/``modify`` primitive: this can
    only ever REMOVE a not-yet-started task from what's left to do, never
    rewrite an existing task's own ``description``/declared tools — there
    is no way to use this to retroactively launder a task's own record to
    match whatever was already (possibly wrongly) dispatched, the way an
    editable ``description``/``expected_tools`` field could be.

    Hard-rejects a ``"completed"`` task (can't cancel history) and an
    ``"active"`` one (can't cancel the task currently being worked —
    complete it, or insert around it, instead); only ``"pending"``
    survives both checks.
    """
    plan = _plan_for(session_id)
    tasks = plan.get("tasks") or []
    if not tasks:
        return {"ok": False, "error": "no active plan — call create_plan first"}

    task = next((t for t in tasks if t["id"] == task_id_to_cancel), None)
    if task is None:
        return {"ok": False, "error": f"no task with id {task_id_to_cancel} in the active plan"}
    if task["status"] != "pending":
        return {
            "ok": False,
            "error": (
                f"cannot cancel task {task_id_to_cancel} — it is {task['status']!r}. Only a pending "
                "(not yet started) task can be cancelled."
            ),
        }

    clean_reason = (reason or "").strip()
    if not clean_reason:
        return {"ok": False, "error": "reason must not be empty"}

    task["status"] = "cancelled"
    task["cancel_reason"] = clean_reason

    return {"ok": True, "plan": _snapshot_plan(plan)}


def cancel_active_task(reason: str, *, session_id: str | None = None) -> dict[str, Any]:
    """Cancels THIS session's own currently-``"active"`` task outright —
    FSM Recovery's other half alongside ``insert_task``'s
    ``insert_before_task_id`` path. Use THIS when the active task itself
    turns out to be wrong/redundant/impossible; use ``insert_task(...,
    insert_before_task_id=<active id>)`` instead when the active task is
    still correct and only a prerequisite is missing before it (that path
    keeps the active task, just delays it — this one drops it for good).

    ``cancel_pending_task`` deliberately still refuses an ``"active"`` task
    (see its own docstring) — this is a SEPARATE primitive rather than a
    loosened check on that one, so a plain ``"pending"``-only cancel stays
    available with its existing, narrower contract intact.

    On cancel, the FSM falls back exactly like ``mark_task_completed``'s
    own Robust Task Auto-Advancement: the lowest-id still-``"pending"``
    task (if any) is promoted to ``"active"`` and ``current_task_id``
    updated to point at it; with nothing pending left, ``current_task_id``
    becomes ``None`` and the agent is free to ``insert_task``/
    ``create_plan`` to rebuild the queue from here.
    """
    plan = _plan_for(session_id)
    tasks = plan.get("tasks") or []
    if not tasks:
        return {"ok": False, "error": "no active plan — call create_plan first"}

    task = next((t for t in tasks if t["status"] == "active"), None)
    if task is None:
        return {"ok": False, "error": "no task is currently active — nothing to cancel"}

    clean_reason = (reason or "").strip()
    if not clean_reason:
        return {"ok": False, "error": "reason must not be empty"}

    cancelled_id = task["id"]
    task["status"] = "cancelled"
    task["cancel_reason"] = clean_reason

    auto_next = min((t for t in tasks if t["status"] == "pending"), key=lambda t: t["id"], default=None)
    if auto_next is not None:
        auto_next["status"] = "active"
        plan["current_task_id"] = auto_next["id"]
        message = (
            f"Task {cancelled_id} cancelled. Task {auto_next['id']} "
            f"({auto_next['description']!r}) — the lowest-id pending task — was auto-promoted to active."
        )
    else:
        plan["current_task_id"] = None
        message = f"Task {cancelled_id} cancelled. No pending task remains — insert_task/create_plan to continue."

    return {"ok": True, "plan": _snapshot_plan(plan), "cancelled_task_id": cancelled_id, "message": message}


def mark_task_completed(
    task_id: int, next_task_id: int | None = None, *, session_id: str | None = None
) -> dict[str, Any]:
    """Marks ``task_id`` ``"completed"`` in THIS session's own plan, and —
    only if ``next_task_id`` is given — promotes THAT task to ``"active"``
    and updates ``current_task_id`` to point at it. ``next_task_id`` is
    optional exactly because finishing a task doesn't always mean the next
    one is obvious yet (the agent may need to re-check the plan, or the
    objective is fully done); leaving it out simply clears
    ``current_task_id`` (if it was pointing at the task just completed)
    rather than guessing which task comes next.

    Both ids are validated to actually exist in THIS session's CURRENT
    plan's tasks BEFORE anything is mutated — an unknown ``task_id`` OR an
    unknown ``next_task_id`` fails the whole call with no partial state
    change, so a typo'd id can never leave the plan in an inconsistent
    state (e.g. a task marked completed with no new active task actually
    promoted because the id it meant to promote didn't exist).

    Robust Task Auto-Advancement: an OMITTED ``next_task_id`` (the caller
    passed ``None``, not a wrong id — a wrong id still fails outright
    above, this never silently papers over an actual typo'd argument) no
    longer just clears ``current_task_id`` to ``None`` while real work
    remains. Confirmed live: a local model completing a task without ever
    supplying ``next_task_id`` left ``current_task_id`` null with pending
    tasks still in the plan — ``dana.core.react_dispatch``'s own FSM reads
    "no active task" as "nothing to hard-restrict the schema to," which
    starved the model of its own geometry tools and produced an empty
    completion. The lowest-id still-``"pending"`` task is auto-promoted
    instead; ``current_task_id`` only ever becomes ``None`` when nothing
    ``"pending"`` is left (every remaining task is ``"completed"`` or
    ``"cancelled"``) — the plan is genuinely finished, not just under-specified.

    ``session_id``: see ``_plan_for``'s own docstring — same omit-inside/
    pass-explicitly-outside split ``create_plan`` above already documents.
    """
    plan = _plan_for(session_id)
    tasks = plan.get("tasks") or []
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
    # Strict FSM State Enforcement: unconditionally demote any OTHER task
    # still marked "active" before deciding what (if anything) gets
    # promoted below. Without this, calling mark_task_completed for a task
    # that ISN'T the one dana.core.react_dispatch's automatic
    # _advance_fsm_on_dispatch (or a previous out-of-order manual call —
    # see this function's own docstring on why an arbitrary task_id is a
    # supported "ultimate override") already promoted leaves THAT task
    # untouched while the fallback below promotes a SECOND one — two
    # simultaneously "active" tasks, with current_task_id only ever
    # pointing at one of them. Demoted to "pending", never "completed":
    # this function has no evidence a demoted task's work is actually
    # done, only that it's no longer the one being tracked as current.
    for other in tasks:
        if other is not task and other["status"] == "active":
            other["status"] = "pending"
    message: str | None = None
    if next_task is not None:
        next_task["status"] = "active"
        plan["current_task_id"] = next_task_id
    else:
        auto_next = min(
            (t for t in tasks if t["status"] == "pending"), key=lambda t: t["id"], default=None
        )
        if auto_next is not None:
            auto_next["status"] = "active"
            plan["current_task_id"] = auto_next["id"]
            message = (
                f"next_task_id was not given, so task {auto_next['id']} "
                f"({auto_next['description']!r}) — the lowest-id pending task — was auto-promoted to active."
            )
        elif plan.get("current_task_id") == task_id:
            plan["current_task_id"] = None

    result: dict[str, Any] = {"ok": True, "plan": _snapshot_plan(plan)}
    if message:
        result["message"] = message
    return result


__all__ = (
    "get_active_plan",
    "create_plan",
    "mark_task_completed",
    "insert_task",
    "cancel_pending_task",
    "cancel_active_task",
)
