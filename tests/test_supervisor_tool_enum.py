"""Supervisor DAG planning must emit registered tool_name values only."""

from __future__ import annotations

import json

from dana.graph.nodes.supervisor import plan_dag_with_llm
from dana.llm_schemas import DAGPlan


def test_plan_dag_tdd_prompt_uses_file_editor_not_create_file(
    monkeypatch: object,
) -> None:
    """TDD-style prompt → structured plan with file_editor only."""
    captured: dict = {}

    def fake_structured(messages, model, max_retries=3, **_k):  # noqa: ANN001
        captured["system"] = messages[0]["content"]
        captured["user"] = messages[1]["content"]
        assert model is DAGPlan
        assert "file_editor" in messages[0]["content"]
        assert "create_file" in messages[0]["content"]  # banned example text
        assert "EXAMPLE VALID DAG PLAN" in messages[0]["content"]
        return DAGPlan.model_validate(
            {
                "tasks": [
                    {
                        "task_id": 1,
                        "action": (
                            "Write failing pytest for TokenBucket to "
                            "execution_jail/broker_diag/test_rate_limiter.py"
                        ),
                        "tool_name": "file_editor",
                        "dependencies": [],
                    },
                    {
                        "task_id": 2,
                        "action": (
                            "Implement TokenBucket in "
                            "execution_jail/broker_diag/rate_limiter.py"
                        ),
                        "tool_name": "file_editor",
                        "dependencies": [1],
                    },
                ]
            }
        )

    import dana.llm_client as llm_client

    monkeypatch.setattr(llm_client, "ask_planner_structured", fake_structured)

    prompt = (
        "Use TDD: first write tests for a TokenBucket rate limiter at "
        "execution_jail/broker_diag/test_rate_limiter.py, then implement "
        "execution_jail/broker_diag/rate_limiter.py."
    )
    planned = plan_dag_with_llm(prompt, use_structured=True)
    assert len(planned) == 2
    assert all(t.get("tool_name") == "file_editor" for t in planned)
    assert "create_file" not in json.dumps(planned)
    assert "test" in planned[0]["action"].lower()


def test_legacy_llm_invoke_few_shot_mentions_file_editor() -> None:
    def fake_invoke(instruction: str) -> str:
        assert "file_editor" in instruction
        assert "create_file" in instruction
        return json.dumps(
            [
                {
                    "task_id": 1,
                    "action": "Write test to x/test_rate_limiter.py",
                    "tool_name": "file_editor",
                    "dependencies": [],
                },
                {
                    "task_id": 2,
                    "action": "Write implementation to x/rate_limiter.py",
                    "tool_name": "file_editor",
                    "dependencies": [1],
                },
            ]
        )

    planned = plan_dag_with_llm(
        "TDD rate limiter",
        llm_invoke=fake_invoke,
        use_structured=False,
    )
    assert [t["tool_name"] for t in planned] == ["file_editor", "file_editor"]
