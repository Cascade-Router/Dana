"""Unit tests for closed-loop verifier_node (offline / injectable)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END

from dana.agentic_react_graph import (
    compile_donna_react_graph,
    route_after_execution,
    route_after_verifier,
)
from dana.graph.nodes.verifier import (
    MAX_VERIFICATION_ATTEMPTS,
    default_physical_evidence_check,
    make_verifier_node,
)
from dana.graph.task_tracker import TaskStatus, TaskTracker
from dana.graph.workflow import remap_execution_end_to_verifier
from dana.schema import ReactGraphState
from langchain_core.messages import AIMessage, HumanMessage


def test_schema_has_verification_result() -> None:
    assert "verification_result" in ReactGraphState.__annotations__


def test_route_after_execution_goes_to_verifier_on_halt() -> None:
    assert route_after_execution({"execution_error": None, "halt": True}) == "verifier"
    assert remap_execution_end_to_verifier(END) == "verifier"


def test_default_check_json_artifact(tmp_path: Path) -> None:
    out = tmp_path / "audit_summary.json"
    out.write_text(
        json.dumps(
            {
                "total_warnings": 2,
                "total_errors": 1,
                "summary": "ok",
            }
        ),
        encoding="utf-8",
    )
    state: dict[str, Any] = {
        "halt": True,
        "final_raw": "done",
        "verification_targets": {
            "files": [str(out)],
            "json_schema": {
                "path": str(out),
                "required_keys": ["total_warnings", "total_errors", "summary"],
            },
        },
    }
    result = default_physical_evidence_check(state)
    assert result["verified"] is True


def test_default_check_uia_bounds() -> None:
    state = {
        "halt": True,
        "final_raw": "inspected",
        "uia_nodes": [
            {"name": "Notepad", "title": "Notepad", "bbox": [10, 20, 300, 400]},
        ],
        "verification_targets": {"uia_nodes": True},
    }
    result = default_physical_evidence_check(state)
    assert result["verified"] is True
    assert result["evidence"]["uia_nodes"]


def test_verifier_retries_then_fail_closed(tmp_path: Path) -> None:
    tracker = TaskTracker(
        dropped_log_path=tmp_path / "dropped.log",
        ledger_path=tmp_path / "ledger.md",
    )
    tracker.start_task("v-1", "verify me")

    def always_fail(_state: ReactGraphState) -> dict[str, Any]:
        return {"verified": False, "evidence": {"reason": "missing artifact"}}

    node = make_verifier_node(always_fail, tracker=tracker, max_attempts=3)

    state: ReactGraphState = {
        "session_id": "v-1",
        "task_id": "v-1",
        "halt": True,
        "final_raw": "claimed done",
        "verification_result": {},
    }
    for i in range(1, 4):
        patch = node(state)
        state = {**state, **patch}
        assert state["verification_result"]["verified"] is False
        assert state["verification_result"]["attempts"] == i
        route = route_after_verifier(state)
        if i < 3:
            assert route == "agent"
        else:
            assert route == "fail_closed"

    rec = tracker.get_task("v-1")
    assert rec is not None
    assert rec.status == TaskStatus.FAILED


def test_verifier_marks_completed_on_success(tmp_path: Path) -> None:
    tracker = TaskTracker(
        dropped_log_path=tmp_path / "dropped.log",
        ledger_path=tmp_path / "ledger.md",
    )
    tracker.start_task("v-ok", "ok task")
    node = make_verifier_node(
        lambda _s: {"verified": True, "evidence": {"ok": True}},
        tracker=tracker,
    )
    patch = node(
        {
            "session_id": "v-ok",
            "task_id": "v-ok",
            "halt": True,
            "final_raw": "done",
        }
    )
    assert patch["verification_result"]["verified"] is True
    assert route_after_verifier(patch) == END
    assert tracker.get_task("v-ok").status == TaskStatus.COMPLETED


def test_graph_tools_halt_hits_verifier() -> None:
    path: list[str] = []

    def planner(state: ReactGraphState) -> dict[str, Any]:
        return {"always_include": ["python_repl"]}

    def executor(state: ReactGraphState) -> dict[str, Any]:
        return {}

    def agent(state: ReactGraphState) -> dict[str, Any]:
        path.append("agent")
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "python_repl",
                            "args": {"code": "print(1)"},
                            "id": "c1",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        }

    def tools(state: ReactGraphState) -> dict[str, Any]:
        path.append("tools")
        from langchain_core.messages import ToolMessage

        msgs = state.get("messages") or []
        last = msgs[-1] if msgs else None
        tcs = getattr(last, "tool_calls", None) or []
        tool_msgs = [
            ToolMessage(
                content="exit_code=0\n1",
                tool_call_id=str(
                    (tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", ""))
                    or "c1"
                ),
            )
            for tc in tcs
        ]
        return {
            "messages": tool_msgs,
            "halt": True,
            "final_raw": "ok",
            "last_obs": "exit_code=0\n1",
        }

    def verifier(state: ReactGraphState) -> dict[str, Any]:
        path.append("verifier")
        return {
            "verification_result": {
                "verified": True,
                "evidence": {"mode": "test"},
                "attempts": 1,
            },
            "halt": True,
            "pending_synthesis": False,
        }

    graph = compile_donna_react_graph(
        agent,
        tools,
        planner_node_fn=planner,
        executor_node_fn=executor,
        verifier_node_fn=verifier,
        checkpointer=MemorySaver(),
    )
    cfg = {"configurable": {"thread_id": "verifier-wire"}}
    list(
        graph.stream(
            {
                "messages": [HumanMessage(content="run repl")],
                "halt": False,
                "always_include": ["python_repl"],
                "session_id": "verifier-wire",
            },
            cfg,
            stream_mode="values",
        )
    )
    assert path == ["agent", "tools", "verifier"]
    final = graph.get_state(cfg).values
    assert final.get("verification_result", {}).get("verified") is True


def test_max_attempts_constant() -> None:
    assert MAX_VERIFICATION_ATTEMPTS == 3
