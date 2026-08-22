"""Strict Pydantic schemas for supervisor DAG + worker tool-call outputs."""

from __future__ import annotations

from collections import deque
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _validate_dag_solvability(tasks: list[Any]) -> None:
    """Raise ``ValueError`` if the DAG cannot execute (deadlock / bad topology).

    Inlined from the now-removed ``dana.graph.dag_topology`` (the legacy
    LangGraph supervisor stack) — this is the only remaining caller of it,
    and the check itself (Kahn topological sort, stdlib-only) has no other
    dependency worth keeping a whole extra module around for.

    Checks
    ------
    1. At least one task has ``dependencies: []`` (entry point).
    2. Every dependency id exists in the plan.
    3. No circular dependencies (Kahn topological sort must cover all nodes).
    """
    rows: list[tuple[int, list[int]]] = [
        (int(node.task_id), [int(d) for d in node.dependencies]) for node in tasks
    ]
    if not rows:
        raise ValueError("Topological Error: empty task list; no executable DAG.")

    ids = {tid for tid, _ in rows}
    for tid, deps in rows:
        for dep in deps:
            if dep not in ids:
                raise ValueError(
                    f"Topological Error: task {tid} depends on missing "
                    f"prerequisite id {dep} (not in plan)."
                )
            if dep == tid:
                raise ValueError(
                    f"Topological Error: task {tid} cannot depend on itself."
                )

    roots = [tid for tid, deps in rows if not deps]
    if not roots:
        raise ValueError(
            "Topological Error: No starting tasks found. At least one task must "
            "have dependencies: []"
        )

    children: dict[int, list[int]] = {tid: [] for tid, _ in rows}
    indeg: dict[int, int] = {tid: 0 for tid, _ in rows}
    for tid, deps in rows:
        for dep in deps:
            children[dep].append(tid)
            indeg[tid] += 1

    queue: deque[int] = deque(roots)
    seen = 0
    while queue:
        node = queue.popleft()
        seen += 1
        for nxt in children[node]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                queue.append(nxt)

    if seen != len(rows):
        cyclic = sorted(tid for tid, deg in indeg.items() if deg > 0)
        raise ValueError(
            "Topological Error: circular dependency detected among task_ids "
            f"{cyclic} (A -> B -> A deadlock)."
        )


# Exact registered worker tool ids — shared by DAG TaskNode + WorkerToolCall.
WorkerToolName = Literal[
    "file_editor",
    "get_file_outline",
    "get_symbol_definition",
    "read_local_file",
]

_WORKER_TOOL_NAMES: tuple[str, ...] = (
    "file_editor",
    "get_file_outline",
    "get_symbol_definition",
    "read_local_file",
)


class TaskNode(BaseModel):
    """One micro-task in the supervisor DAG."""

    model_config = ConfigDict(extra="forbid")

    task_id: int = Field(..., ge=1, description="Stable integer id for this node")
    action: str = Field(
        ...,
        min_length=1,
        description=(
            "Concrete worker instruction (what to do / which file). "
            "Never invent tool names here — use tool_name instead."
        ),
    )
    tool_name: WorkerToolName = Field(
        ...,
        description=(
            "Registered worker tool id. Must be one of: "
            + ", ".join(_WORKER_TOOL_NAMES)
        ),
    )
    dependencies: list[int] = Field(
        default_factory=list,
        description="task_id values that must complete before this node runs",
    )

    @field_validator("action")
    @classmethod
    def _strip_action(cls, value: str) -> str:
        text = (value or "").strip()
        if not text:
            raise ValueError("action must be non-empty")
        # Bare hallucinated tool tokens are not valid instructions.
        lowered = text.lower().replace("-", "_")
        banned = {
            "create",
            "create_file",
            "write_file",
            "read_file",
            "edit_file",
            "pytest",
            "run_tests",
            "bash",
            "shell",
            "execute",
        }
        if lowered in banned or lowered in {t.lower() for t in _WORKER_TOOL_NAMES}:
            raise ValueError(
                f"action={text!r} looks like a tool name; put tools in "
                f"tool_name and describe the work in action "
                f"(e.g. 'Write TokenBucket tests to path/test_x.py')"
            )
        return text

    @field_validator("dependencies")
    @classmethod
    def _deps_nonneg(cls, value: list[int]) -> list[int]:
        out: list[int] = []
        for item in value or []:
            n = int(item)
            if n < 1:
                raise ValueError("dependency ids must be >= 1")
            if n not in out:
                out.append(n)
        return out


class EpicNode(BaseModel):
    """One high-level epic for Meta-Broker decomposition."""

    model_config = ConfigDict(extra="forbid")

    epic_id: int = Field(..., ge=1)
    title: str = Field(..., min_length=1)
    goal: str = Field(..., min_length=1)
    validation_command: str = Field(
        default="",
        description=(
            "Targeted shell command that validates ONLY this epic's work. "
            "Examples: 'python -m pytest tests/test_rate_limiter.py -q', "
            "'python -m py_compile popup_animation.py'. Never use a bare "
            "global 'pytest' / 'python -m pytest -q' over the whole repo."
        ),
    )

    @field_validator("title", "goal")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        text = (value or "").strip()
        if not text:
            raise ValueError("must be non-empty")
        return text

    @field_validator("validation_command")
    @classmethod
    def _strip_validation_command(cls, value: str) -> str:
        return (value or "").strip()


class EpicPlan(BaseModel):
    """Broker structured output — sequential epics for closed-loop repair."""

    model_config = ConfigDict(extra="forbid")

    epics: list[EpicNode] = Field(..., min_length=1)

    def to_epics(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for node in sorted(self.epics, key=lambda n: n.epic_id):
            row: dict[str, Any] = {
                "epic_id": int(node.epic_id),
                "title": str(node.title),
                "goal": str(node.goal),
                "status": "pending",
                "repair_attempts": 0,
            }
            cmd = str(node.validation_command or "").strip()
            if cmd:
                row["validation_command"] = cmd
            rows.append(row)
        return rows


class DAGPlan(BaseModel):
    """Supervisor structured output — ordered DAG of TaskNode entries."""

    model_config = ConfigDict(extra="forbid")

    tasks: list[TaskNode] = Field(..., min_length=1)

    @field_validator("tasks")
    @classmethod
    def _unique_ids(cls, value: list[TaskNode]) -> list[TaskNode]:
        seen: set[int] = set()
        for node in value:
            if node.task_id in seen:
                raise ValueError(f"duplicate task_id: {node.task_id}")
            seen.add(node.task_id)
        # Dependencies must reference known ids (or be omitted).
        for node in value:
            for dep in node.dependencies:
                if dep == node.task_id:
                    raise ValueError(f"task {node.task_id} cannot depend on itself")
                if dep not in seen:
                    raise ValueError(
                        f"task {node.task_id} depends on unknown id {dep}"
                    )
        return value

    @model_validator(mode="after")
    def _require_solvable_topology(self) -> DAGPlan:
        """Reject deadlocked / entry-less DAGs so json_schema_retry can re-prompt."""
        _validate_dag_solvability(self.tasks)
        return self

    def to_dag_tasks(self) -> list[dict[str, Any]]:
        """Convert to supervisor ``DagTask`` dicts."""
        rows: list[dict[str, Any]] = []
        for node in sorted(self.tasks, key=lambda n: n.task_id):
            rows.append(
                {
                    "task_id": int(node.task_id),
                    "action": str(node.action),
                    "tool_name": str(node.tool_name),
                    "dependencies": [int(d) for d in node.dependencies],
                    "status": "pending",
                    "summary": "",
                    "error": "",
                    "attempts": 0,
                }
            )
        return rows


# Alias requested by planning docs / UI copy.
SupervisorPlan = DAGPlan


class WorkerToolCall(BaseModel):
    """Single localized worker tool invocation (no free-form prose)."""

    model_config = ConfigDict(extra="forbid")

    tool: WorkerToolName = Field(..., description="Registered worker tool id")
    filepath: str | None = Field(default=None, description="Target path when applicable")
    symbol: str | None = Field(
        default=None, description="Symbol name for get_symbol_definition"
    )
    action: Literal["read", "write", "append"] | None = Field(
        default=None, description="file_editor action"
    )
    content: str | None = Field(default=None, description="Write/append payload")
    arguments: dict[str, Any] = Field(default_factory=dict)

    @field_validator("filepath", "symbol", "content")
    @classmethod
    def _strip_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None


class WorkerToolPlan(BaseModel):
    """Worker structured output — tool calls + dense summary for the supervisor."""

    model_config = ConfigDict(extra="forbid")

    tool_calls: list[WorkerToolCall] = Field(default_factory=list)
    summary: str = Field(default="", description="Dense outcome for the supervisor")
    status: Literal["completed", "failed"] = "completed"
    error: str = ""

    @field_validator("summary", "error")
    @classmethod
    def _strip_text(cls, value: str) -> str:
        return (value or "").strip()


def schema_for_model(model: type[BaseModel]) -> dict[str, Any]:
    """JSON Schema object suitable for Ollama ``format``."""
    return model.model_json_schema()


__all__ = (
    "DAGPlan",
    "EpicNode",
    "EpicPlan",
    "SupervisorPlan",
    "TaskNode",
    "WorkerToolCall",
    "WorkerToolName",
    "WorkerToolPlan",
    "schema_for_model",
)
