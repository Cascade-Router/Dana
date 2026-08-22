"""Strict Pydantic schemas + JSON retry middleware for small-model outputs."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from dana.llm_schemas import DAGPlan, TaskNode, WorkerToolCall, WorkerToolPlan
from dana.middleware.json_schema_retry import (
    StructuredOutputError,
    extract_json_payload,
    parse_model,
    parse_with_schema_retry,
)


def test_dag_plan_schema_rejects_bad_deps() -> None:
    with pytest.raises(ValidationError):
        DAGPlan.model_validate(
            {
                "tasks": [
                    {
                        "task_id": 1,
                        "action": "read a.py",
                        "tool_name": "read_local_file",
                        "dependencies": [],
                    },
                    {
                        "task_id": 2,
                        "action": "edit b.py",
                        "tool_name": "file_editor",
                        "dependencies": [9],
                    },
                ]
            }
        )
    plan = DAGPlan.model_validate(
        {
            "tasks": [
                {
                    "task_id": 1,
                    "action": "read a.py",
                    "tool_name": "read_local_file",
                    "dependencies": [],
                },
                {
                    "task_id": 2,
                    "action": "edit b.py",
                    "tool_name": "file_editor",
                    "dependencies": [1],
                },
            ]
        }
    )
    rows = plan.to_dag_tasks()
    assert [r["task_id"] for r in rows] == [1, 2]
    assert rows[1]["dependencies"] == [1]
    assert rows[1]["tool_name"] == "file_editor"


def test_task_node_rejects_hallucinated_tool_names() -> None:
    with pytest.raises(ValidationError):
        TaskNode.model_validate(
            {
                "task_id": 1,
                "action": "Write rate limiter",
                "tool_name": "create_file",
                "dependencies": [],
            }
        )
    with pytest.raises(ValidationError):
        TaskNode.model_validate(
            {
                "task_id": 1,
                "action": "create_file",
                "tool_name": "file_editor",
                "dependencies": [],
            }
        )
    node = TaskNode.model_validate(
        {
            "task_id": 1,
            "action": "Write TokenBucket tests to x/test_rate_limiter.py",
            "tool_name": "file_editor",
            "dependencies": [],
        }
    )
    assert node.tool_name == "file_editor"


def test_worker_tool_plan_schema() -> None:
    plan = WorkerToolPlan.model_validate(
        {
            "tool_calls": [
                {
                    "tool": "get_file_outline",
                    "filepath": "dana/graph/builder.py",
                },
                {
                    "tool": "file_editor",
                    "filepath": "logs/x.py",
                    "action": "write",
                    "content": "x = 1\n",
                },
            ],
            "summary": "outlined builder and wrote x.py",
            "status": "completed",
        }
    )
    assert plan.tool_calls[0].tool == "get_file_outline"
    with pytest.raises(ValidationError):
        WorkerToolCall.model_validate({"tool": "not_a_tool"})


def test_parse_model_accepts_bare_task_list() -> None:
    raw = json.dumps(
        [
            {
                "task_id": 1,
                "action": "read a.py",
                "tool_name": "read_local_file",
                "dependencies": [],
            },
            {
                "task_id": 2,
                "action": "edit b.py",
                "tool_name": "file_editor",
                "dependencies": [1],
            },
        ]
    )
    plan = parse_model(raw, DAGPlan)
    assert len(plan.tasks) == 2


def test_extract_json_from_fence() -> None:
    blob = (
        "Here you go:\n```json\n"
        '{"tasks":[{"task_id":1,"action":"outline x.py",'
        '"tool_name":"get_file_outline","dependencies":[]}]}\n```\n'
    )
    payload = extract_json_payload(blob)
    assert payload.startswith("{")
    assert parse_model(blob, DAGPlan).tasks[0].task_id == 1


def test_retry_parser_recovers_after_malformed_json() -> None:
    calls = {"n": 0}

    def invoke(messages: list[dict[str, str]]) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            return "NOT JSON at all {{{"
        if calls["n"] == 2:
            # Invalid schema (missing tasks)
            return '{"nope": true}'
        return json.dumps(
            {
                "tasks": [
                    {
                        "task_id": 1,
                        "action": "read a.py",
                        "tool_name": "read_local_file",
                        "dependencies": [],
                    },
                ]
            }
        )

    plan = parse_with_schema_retry(
        [{"role": "user", "content": "plan it"}],
        DAGPlan,
        invoke=invoke,
        max_retries=3,
    )
    assert plan.tasks[0].action == "read a.py"
    assert plan.tasks[0].tool_name == "read_local_file"
    assert calls["n"] == 3
    # Temporary context must include error observation on retries.
    # invoke sees growing message lists; verify via a capturing wrapper.


def test_retry_parser_rejects_create_file_then_accepts_file_editor() -> None:
    """Hallucinated tool names must fail closed and retry into registered tools."""
    calls = {"n": 0}

    def invoke(messages: list[dict[str, str]]) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            return json.dumps(
                {
                    "tasks": [
                        {
                            "task_id": 1,
                            "action": "Write test",
                            "tool_name": "create_file",
                            "dependencies": [],
                        }
                    ]
                }
            )
        return json.dumps(
            {
                "tasks": [
                    {
                        "task_id": 1,
                        "action": "Write failing pytest for TokenBucket to "
                        "execution_jail/x/test_rate_limiter.py",
                        "tool_name": "file_editor",
                        "dependencies": [],
                    },
                    {
                        "task_id": 2,
                        "action": "Implement TokenBucket in "
                        "execution_jail/x/rate_limiter.py",
                        "tool_name": "file_editor",
                        "dependencies": [1],
                    },
                ]
            }
        )

    plan = parse_with_schema_retry(
        [{"role": "user", "content": "TDD rate limiter"}],
        DAGPlan,
        invoke=invoke,
        max_retries=3,
    )
    assert calls["n"] == 2
    assert all(t.tool_name == "file_editor" for t in plan.tasks)
    assert "create_file" not in {t.tool_name for t in plan.tasks}
    assert "test" in plan.tasks[0].action.lower()


def test_retry_parser_fails_after_max_retries() -> None:
    def invoke(_messages: list[dict[str, str]]) -> str:
        return "still-not-json"

    with pytest.raises(StructuredOutputError) as excinfo:
        parse_with_schema_retry(
            [{"role": "user", "content": "plan"}],
            DAGPlan,
            invoke=invoke,
            max_retries=3,
        )
    assert excinfo.value.attempts == 4  # initial + 3 retries
