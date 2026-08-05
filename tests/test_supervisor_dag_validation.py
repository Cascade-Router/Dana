"""Topological DAG solvability — reject cycles / missing entry points + retry."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from dana.graph.dag_topology import validate_dag_solvability
from dana.graph.monitor_bus import write_broker_crash_dump
from dana.graph.nodes.supervisor import plan_dag_with_llm, validate_dag_solvability as exported
from dana.llm_schemas import DAGPlan, SupervisorPlan, TaskNode
from dana.middleware.json_schema_retry import StructuredOutputError, parse_with_schema_retry


def _task(
    task_id: int,
    *,
    deps: list[int] | None = None,
    action: str | None = None,
) -> dict:
    return {
        "task_id": task_id,
        "action": action or f"Work on step {task_id} in x/step_{task_id}.py",
        "tool_name": "file_editor",
        "dependencies": list(deps or []),
    }


def test_validate_dag_solvability_rejects_no_entry_point() -> None:
    tasks = [
        TaskNode.model_validate(_task(1, deps=[2])),
        TaskNode.model_validate(_task(2, deps=[1])),
    ]
    with pytest.raises(ValueError, match="No starting tasks found"):
        validate_dag_solvability(tasks)


def test_validate_dag_solvability_rejects_missing_prerequisite() -> None:
    tasks = [
        TaskNode.model_validate(_task(1, deps=[])),
        TaskNode.model_validate(_task(2, deps=[9])),
    ]
    with pytest.raises(ValueError, match="missing prerequisite id 9"):
        validate_dag_solvability(tasks)


def test_validate_dag_solvability_rejects_cycle_with_entry() -> None:
    # Task 1 is a root; 2 <-> 3 is a disconnected cycle.
    tasks = [
        TaskNode.model_validate(_task(1, deps=[])),
        TaskNode.model_validate(_task(2, deps=[3])),
        TaskNode.model_validate(_task(3, deps=[2])),
    ]
    with pytest.raises(ValueError, match="circular dependency"):
        validate_dag_solvability(tasks)


def test_validate_dag_solvability_accepts_linear_chain() -> None:
    tasks = [
        TaskNode.model_validate(_task(1, deps=[])),
        TaskNode.model_validate(_task(2, deps=[1])),
        TaskNode.model_validate(_task(3, deps=[2])),
    ]
    validate_dag_solvability(tasks)  # no raise
    assert exported is validate_dag_solvability


def test_supervisor_plan_model_rejects_circular_deps() -> None:
    with pytest.raises(ValidationError, match="Topological Error"):
        SupervisorPlan.model_validate(
            {
                "tasks": [
                    _task(1, deps=[2]),
                    _task(2, deps=[1]),
                ]
            }
        )


def test_parse_with_schema_retry_recovers_from_circular_llm_output() -> None:
    """Mock LLM returns a cycle first; retry middleware feeds ValueError back."""
    calls = {"n": 0}
    observations: list[str] = []

    def invoke(messages: list[dict[str, str]]) -> str:
        calls["n"] += 1
        if len(messages) > 2:
            observations.append(str(messages[-1].get("content") or ""))
        if calls["n"] == 1:
            return json.dumps(
                {
                    "tasks": [
                        _task(1, deps=[2]),
                        _task(2, deps=[1]),
                    ]
                }
            )
        return json.dumps(
            {
                "tasks": [
                    _task(1, deps=[], action="Write test to x/test_a.py"),
                    _task(2, deps=[1], action="Implement module in x/a.py"),
                ]
            }
        )

    plan = parse_with_schema_retry(
        [{"role": "user", "content": "TDD plan"}],
        DAGPlan,
        invoke=invoke,
        max_retries=3,
    )
    assert calls["n"] == 2
    assert plan.tasks[0].dependencies == []
    assert any("Topological Error" in obs for obs in observations)


def test_plan_dag_with_llm_dumps_raw_tasks_on_exhausted_retries(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dump_path = tmp_path / "broker_crash_dump.txt"
    monkeypatch.setattr(
        "dana.graph.monitor_bus.broker_crash_dump_path",
        lambda: dump_path,
    )

    def always_cycle(messages, model, **_k):  # noqa: ANN001
        raise StructuredOutputError(
            "structured output failed after 4 attempt(s): "
            "ValidationError: Topological Error: circular dependency",
            attempts=4,
            last_raw=json.dumps(
                {
                    "tasks": [
                        _task(1, deps=[2]),
                        _task(2, deps=[1]),
                    ]
                }
            ),
        )

    import dana.llm_client as llm_client

    monkeypatch.setattr(llm_client, "ask_planner_structured", always_cycle)

    # Falls back to heuristic after dump — must not raise to the caller.
    planned = plan_dag_with_llm("do something simple", use_structured=True)
    assert planned  # heuristic fallback
    assert dump_path.is_file()
    body = dump_path.read_text(encoding="utf-8")
    assert "generated tasks JSON" in body
    assert '"task_id": 1' in body or '"task_id":1' in body


def test_write_broker_crash_dump_includes_tasks_json(tmp_path, monkeypatch) -> None:
    dump_path = tmp_path / "broker_crash_dump.txt"
    monkeypatch.setattr(
        "dana.graph.monitor_bus.broker_crash_dump_path",
        lambda: dump_path,
    )
    write_broker_crash_dump(
        ValueError("Topological Error: circular dependency"),
        context="unit-test",
        tasks_json=[{"task_id": 1, "dependencies": [2]}],
        raw_llm_output='{"tasks":[]}',
    )
    text = dump_path.read_text(encoding="utf-8")
    assert "generated tasks JSON" in text
    assert "raw LLM output" in text
    assert '"task_id": 1' in text or '"task_id":1' in text
