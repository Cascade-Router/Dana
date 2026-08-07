"""LangGraph state schemas for the Supervisor ↔ Worker DAG swarm."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

try:
    from typing import NotRequired
except ImportError:  # Python < 3.11
    from typing_extensions import NotRequired


TaskStatus = Literal["pending", "ready", "running", "completed", "failed", "blocked"]


class DagTask(TypedDict):
    """One node in the supervisor's micro-task DAG."""

    task_id: int
    action: str
    dependencies: list[int]
    tool_name: NotRequired[str]
    status: NotRequired[TaskStatus]
    summary: NotRequired[str]
    error: NotRequired[str]
    attempts: NotRequired[int]


class WorkerState(TypedDict):
    """Isolated worker view — no global conversation history.

    Each worker receives only its task instructions, a fresh context window,
    and localized tool-call outputs so context cannot drift across the DAG.
    """

    task_id: int
    instructions: str
    context_window: list[dict[str, str]]
    tool_outputs: list[dict[str, Any]]
    summary: str
    status: TaskStatus
    staging_session_id: NotRequired[str]
    error: NotRequired[str]


class SupervisorState(TypedDict):
    """Global supervisor state: DAG, pending queue, completed summaries."""

    user_prompt: str
    dag: list[DagTask]
    pending_tasks: list[int]
    completed_summaries: list[dict[str, Any]]
    active_task_ids: list[int]
    worker_results: list[dict[str, Any]]
    final_response: str
    status: Literal[
        "planning",
        "dispatching",
        "awaiting_workers",
        "evaluating",
        "completed",
        "failed",
        "ABORTED",
    ]
    supervisor_cycles: int
    max_supervisor_cycles: int
    max_task_attempts: int
    last_dispatch_key: str
    stall_count: int
    # Intentionally empty / unused by workers — kept only so callers cannot
    # accidentally thread ReAct history into the swarm.
    global_conversation_history: list[dict[str, Any]]
    # Open transactional staging sessions awaiting verify_and_commit.
    open_staging_sessions: list[str]
    checkpoint_log: list[str]
    error: NotRequired[str]


def empty_supervisor_state(
    user_prompt: str,
    *,
    max_supervisor_cycles: int = 12,
    max_task_attempts: int = 2,
) -> SupervisorState:
    """Fresh supervisor state for a single complex prompt."""
    return {
        "user_prompt": user_prompt,
        "dag": [],
        "pending_tasks": [],
        "completed_summaries": [],
        "active_task_ids": [],
        "worker_results": [],
        "final_response": "",
        "status": "planning",
        "supervisor_cycles": 0,
        "max_supervisor_cycles": int(max_supervisor_cycles),
        "max_task_attempts": int(max_task_attempts),
        "last_dispatch_key": "",
        "stall_count": 0,
        "global_conversation_history": [],
        "open_staging_sessions": [],
        "checkpoint_log": [],
    }


def empty_worker_state(task_id: int, instructions: str) -> WorkerState:
    """Fresh isolated worker state (empty context window)."""
    return {
        "task_id": int(task_id),
        "instructions": instructions,
        "context_window": [],
        "tool_outputs": [],
        "summary": "",
        "status": "ready",
    }


class Epic(TypedDict):
    """One high-level sub-goal managed by the Meta-Broker."""

    epic_id: int
    title: str
    goal: str
    status: Literal["pending", "active", "repairing", "completed", "failed", "ABORTED"]
    repair_attempts: NotRequired[int]
    validation_command: NotRequired[str]
    workspace_path: NotRequired[str]


class RuntimeFeedback(TypedDict):
    """Structured result from ``run_validation_harness``."""

    success: bool
    exit_code: int
    stdout: str
    stderr: str
    command: NotRequired[str]
    workspace_path: NotRequired[str]
    epic_id: NotRequired[int]


class BrokerState(SupervisorState):
    """Meta-Broker state: macro intent → sequential epics → closed-loop feedback.

    Embeds ``SupervisorState`` fields so each active epic can run an isolated
    Supervisor ↔ Worker DAG with a fresh context window (no cross-epic history).
    """

    macro_intent: str
    epics: list[Epic]
    active_epic_index: int
    runtime_feedback: RuntimeFeedback | dict[str, Any]
    broker_phase: Literal[
        "plan",
        "dispatch_epic",
        "await_supervisor",
        "staging",
        "validate",
        "feedback",
        "repair",
        "advance",
        "done",
    ]
    max_repair_attempts: int
    workspace_path: NotRequired[str]
    validation_command: NotRequired[str]
    epic_log: NotRequired[list[str]]
    completed_epic_artifacts: NotRequired[list[dict[str, Any]]]


def empty_broker_state(
    macro_intent: str,
    *,
    max_supervisor_cycles: int = 12,
    max_task_attempts: int = 2,
    max_repair_attempts: int = 0,
    workspace_path: str | None = None,
    validation_command: str | None = None,
) -> BrokerState:
    """Fresh Meta-Broker state for a multi-epic macro prompt."""
    base = empty_supervisor_state(
        "",
        max_supervisor_cycles=max_supervisor_cycles,
        max_task_attempts=max_task_attempts,
    )
    state: dict[str, Any] = {
        **base,
        "macro_intent": str(macro_intent or ""),
        "epics": [],
        "active_epic_index": 0,
        "runtime_feedback": {},
        "broker_phase": "plan",
        "max_repair_attempts": int(max_repair_attempts),
        "epic_log": [],
        "completed_epic_artifacts": [],
        "status": "planning",
    }
    if workspace_path:
        state["workspace_path"] = str(workspace_path)
    if validation_command:
        state["validation_command"] = str(validation_command)
    return state  # type: ignore[return-value]


__all__ = (
    "BrokerState",
    "DagTask",
    "Epic",
    "RuntimeFeedback",
    "SupervisorState",
    "TaskStatus",
    "WorkerState",
    "empty_broker_state",
    "empty_supervisor_state",
    "empty_worker_state",
)
