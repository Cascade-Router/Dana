"""State integrity evals — schema preservation + HITL pending-approval halt."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from donna.agentic_planning import executor_node, planner_node
from donna.agentic_react_graph import (
    _route_after_agent,
    _route_after_jason_review,
    _route_after_ticket_approval,
    _route_after_ticket_validate,
    compile_donna_react_graph,
)
from donna.middleware import hitl_ticket as hitl
from donna.schema import ReactGraphState


_REQUIRED_CONTROL_PLANE = frozenset(
    {"session_id", "current_agent", "active_intent"}
)

_VALID_CTX = (
    "Target files: donna/agentic_react_graph.py\n"
    "Root cause: HITL must freeze graph until operator approves draft ticket.\n"
    "Step-by-step changes: 1) validate 2) jason 3) interrupt 4) tools.\n"
    "Acceptance criteria: interrupt pending; Approve resumes tools; Deny halts."
)


@pytest.fixture(autouse=True)
def _hitl_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DONNA_HITL_TICKET", "1")
    monkeypatch.delenv("DONNA_HITL_AUTO_APPROVE", raising=False)
    monkeypatch.delenv("DONNA_HITL_AUTO_DENY", raising=False)
    monkeypatch.delenv("DONNA_HITL_REQUIRE_GUI", raising=False)
    hitl.clear_pending()
    yield
    hitl.clear_pending()


def _ai_with_draft(
    objective: str = "Log desktop summary ticket for HITL eval",
    context: str = _VALID_CTX,
) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "draft_cursor_prompt",
                "args": {"objective": objective, "context": context},
                "id": "call-draft-eval-1",
                "type": "tool_call",
            }
        ],
    )


def _assert_control_plane(state: dict[str, Any]) -> None:
    missing = [k for k in _REQUIRED_CONTROL_PLANE if not str(state.get(k) or "").strip()]
    assert not missing, f"missing control-plane fields: {missing} in {state.keys()}"


def test_planner_executor_preserve_required_schema_fields() -> None:
    """Multi-node hydrate must keep bureaucratic triad + always_include list."""
    seed: ReactGraphState = {
        "session_id": "eval-state-1",
        "current_agent": "ReAct_Agent",
        "active_intent": "Summarize active window text",
        "messages": [HumanMessage(content="Summarize active window text")],
        "always_include": [],
        "halt": False,
        "iterations": 0,
        "ticket_validated": False,
        "ticket_validation_retries": 0,
    }
    planned = planner_node(seed)
    merged: ReactGraphState = {**seed, **planned}
    # add_messages-style: keep human + planner system card
    merged["messages"] = list(seed["messages"]) + list(planned.get("messages") or [])
    _assert_control_plane(merged)
    assert isinstance(merged.get("execution_plan"), dict)
    assert isinstance(merged.get("env_context"), dict)
    assert isinstance(merged.get("always_include"), list)

    executed = executor_node(merged)
    after: ReactGraphState = {**merged, **executed}
    _assert_control_plane(after)
    assert after["session_id"] == "eval-state-1"
    assert after["execution_plan"]["status"] == "executing"
    assert after.get("halt") is False


def test_draft_ticket_routes_into_hitl_corridor() -> None:
    """Agent draft_cursor_prompt → ticket_validate → jason → ticket_approval."""
    state: ReactGraphState = {
        "messages": [_ai_with_draft()],
        "session_id": "eval-hitl-route",
        "current_agent": "ReAct_Agent",
        "active_intent": "draft_cursor_prompt",
        "halt": False,
        "ticket_validated": False,
    }
    assert _route_after_agent(state) == "ticket_validate"
    assert _route_after_ticket_validate({"ticket_validated": True}) == "jason_ticket_review"
    assert _route_after_jason_review({"halt": False}) == "ticket_approval"
    # Pending approval: not halted yet — approval node will interrupt.
    assert _route_after_ticket_approval({"halt": False}) == "tools"
    assert _route_after_ticket_approval({"halt": True}) != "tools"


def test_hitl_ticket_node_halts_pending_approval(monkeypatch: pytest.MonkeyPatch) -> None:
    """ticket_approval interrupt freezes execution until resume (pending state)."""
    path: list[str] = []
    tools_ran = {"n": 0}

    def planner(state: ReactGraphState) -> dict[str, Any]:
        path.append("planner")
        return {
            "session_id": state.get("session_id") or "eval-hitl",
            "current_agent": "Planner",
            "active_intent": state.get("active_intent") or "draft",
            "always_include": ["draft_cursor_prompt"],
        }

    def executor(state: ReactGraphState) -> dict[str, Any]:
        path.append("executor")
        return {"current_agent": "Executor"}

    def agent(state: ReactGraphState) -> dict[str, Any]:
        path.append("agent")
        return {
            "messages": [_ai_with_draft()],
            "session_id": state.get("session_id") or "eval-hitl",
            "current_agent": "ReAct_Agent",
            "active_intent": "draft_cursor_prompt",
        }

    def tools(state: ReactGraphState) -> dict[str, Any]:
        path.append("tools")
        tools_ran["n"] += 1
        return {
            "halt": True,
            "final_raw": "queued",
            "last_obs": "Action queued successfully.",
            "session_id": state.get("session_id") or "eval-hitl",
            "current_agent": "Tools",
            "active_intent": state.get("active_intent") or "draft",
        }

    monkeypatch.setattr(
        "donna.agentic_react_graph.generate_jason_ticket_critique",
        lambda *_a, **_k: "Critique ok — HITL pending approval eval.",
    )
    monkeypatch.setattr("donna.core_agent.enqueue_speech", lambda *_a, **_k: None)

    graph = compile_donna_react_graph(
        agent,
        tools,
        planner_node_fn=planner,
        executor_node_fn=executor,
        checkpointer=MemorySaver(),
    )
    cfg = {"configurable": {"thread_id": "eval-hitl-pending"}}
    list(
        graph.stream(
            {
                "messages": [HumanMessage(content="create a desktop log ticket")],
                "halt": False,
                "always_include": [],
                "session_id": "eval-hitl",
                "current_agent": "ReAct_Agent",
                "active_intent": "create a desktop log ticket",
            },
            cfg,
            stream_mode="values",
        )
    )

    snap = graph.get_state(cfg)
    assert snap.interrupts, "HITL ticket_approval must interrupt (pending approval)"
    assert tools_ran["n"] == 0, "tools must not run while HITL is pending"
    values = snap.values
    _assert_control_plane(values)
    # Validated ticket payload is present while waiting on operator.
    assert values.get("ticket_validated") is True
    drafted = values.get("drafted_ticket") or {}
    assert str(drafted.get("objective") or "").strip()
    assert hitl.is_pending() or snap.interrupts

    # Resume approve → tools execute; control plane still intact.
    list(
        graph.stream(
            Command(resume={"approved": True, "action": "approve"}),
            cfg,
            stream_mode="values",
        )
    )
    final = graph.get_state(cfg).values
    assert tools_ran["n"] == 1
    assert final.get("final_raw") == "queued"
    _assert_control_plane(final)
    assert not graph.get_state(cfg).interrupts


def test_multi_node_stream_preserves_session_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """session_id survives planner → executor → agent → tools mutations."""
    sid = "eval-session-preserve"

    def planner(state: ReactGraphState) -> dict[str, Any]:
        return {
            "session_id": state.get("session_id") or sid,
            "current_agent": "Planner",
            "active_intent": state.get("active_intent") or "ping",
        }

    def executor(state: ReactGraphState) -> dict[str, Any]:
        return {
            "session_id": state.get("session_id") or sid,
            "current_agent": "Executor",
            "active_intent": state.get("active_intent") or "ping",
        }

    def agent(state: ReactGraphState) -> dict[str, Any]:
        return {
            "messages": [AIMessage(content="FINAL: ok")],
            "halt": True,
            "final_raw": "ok",
            "session_id": state.get("session_id") or sid,
            "current_agent": "ReAct_Agent",
            "active_intent": state.get("active_intent") or "ping",
        }

    def tools(state: ReactGraphState) -> dict[str, Any]:
        raise AssertionError("tools must not run on text-only FINAL")

    graph = compile_donna_react_graph(
        agent,
        tools,
        planner_node_fn=planner,
        executor_node_fn=executor,
        checkpointer=MemorySaver(),
    )
    cfg = {"configurable": {"thread_id": sid}}
    list(
        graph.stream(
            {
                "messages": [HumanMessage(content="hello")],
                "halt": False,
                "session_id": sid,
                "current_agent": "ReAct_Agent",
                "active_intent": "hello",
                "always_include": [],
            },
            cfg,
            stream_mode="values",
        )
    )
    final = graph.get_state(cfg).values
    assert final.get("session_id") == sid
    _assert_control_plane(final)
