"""Pre-answer Plan-Then-Execute corridor (planner → executor → agent)."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from donna.agentic import requires_tool_graph
from donna.agentic_planning import (
    build_structured_plan,
    desktop_plan_intent,
    executor_node,
    planner_node,
)
from donna.agentic_react_graph import compile_donna_react_graph
from donna.schema import ReactGraphState


_DESKTOP_UTTERANCE = (
    "Summarize active window and create a desktop log ticket"
)
_VALID_CTX = (
    "Target files: donna/agentic_planning.py\n"
    "Root cause: user asked for an active-window summary plus a desktop log ticket.\n"
    "Step-by-step changes: 1) vision summarize 2) draft_cursor_prompt 3) HITL gate.\n"
    "Acceptance criteria: Planner arms vision+ticket; Approve runs tools after HITL."
)


def test_desktop_plan_intent_escalates_from_chat() -> None:
    assert desktop_plan_intent(_DESKTOP_UTTERANCE) is True
    assert requires_tool_graph(_DESKTOP_UTTERANCE) is True
    assert requires_tool_graph("Hello Donna, how are you?") is False


def test_build_structured_plan_selects_vision_and_ticket() -> None:
    plan = build_structured_plan(
        _DESKTOP_UTTERANCE,
        tool_ids=[
            "analyze_visual_context",
            "ocr_with_region",
            "draft_cursor_prompt",
            "file_editor",
        ],
        window_meta={"title": "Notepad", "available": True, "excluded_self": False},
    )
    assert "analyze_visual_context" in plan["required_tools"]
    assert "draft_cursor_prompt" in plan["required_tools"]
    assert plan["status"] == "planned"
    assert "Notepad" in plan["environment_assessment"]


def test_planner_then_executor_hydrate_state() -> None:
    state: ReactGraphState = {
        "messages": [HumanMessage(content=_DESKTOP_UTTERANCE)],
        "session_id": "plan-1",
        "current_agent": "ReAct_Agent",
        "active_intent": _DESKTOP_UTTERANCE,
        "always_include": [],
        "halt": False,
    }
    planned = planner_node(state)
    assert planned.get("execution_plan")
    assert "env_context" in planned
    assert planned.get("current_agent") == "Planner"
    merged: ReactGraphState = {**state, **planned}
    merged["messages"] = list(state["messages"]) + list(planned.get("messages") or [])
    executed = executor_node(merged)
    assert executed["execution_plan"]["status"] == "executing"
    always = executed.get("always_include") or []
    assert "analyze_visual_context" in always or "ocr_with_region" in always
    assert "draft_cursor_prompt" in always


def test_compile_graph_planner_executor_before_hitl(monkeypatch) -> None:
    path: list[str] = []

    monkeypatch.setenv("DONNA_HITL_TICKET", "1")
    monkeypatch.delenv("DONNA_HITL_AUTO_APPROVE", raising=False)
    monkeypatch.delenv("DONNA_HITL_AUTO_DENY", raising=False)
    monkeypatch.delenv("DONNA_HITL_REQUIRE_GUI", raising=False)

    def planner(state: ReactGraphState) -> dict[str, Any]:
        path.append("planner")
        return {
            "execution_plan": {
                "intended_goal": _DESKTOP_UTTERANCE,
                "required_tools": [
                    "analyze_visual_context",
                    "draft_cursor_prompt",
                ],
                "execution_steps": ["vision", "ticket"],
                "status": "planned",
            },
            "always_include": [
                "analyze_visual_context",
                "draft_cursor_prompt",
            ],
            "current_agent": "Planner",
        }

    def executor(state: ReactGraphState) -> dict[str, Any]:
        path.append("executor")
        plan = dict(state.get("execution_plan") or {})
        plan["status"] = "executing"
        return {
            "execution_plan": plan,
            "always_include": list(state.get("always_include") or []),
            "current_agent": "Executor",
        }

    def agent(state: ReactGraphState) -> dict[str, Any]:
        path.append("agent")
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "draft_cursor_prompt",
                            "args": {
                                "objective": "Log desktop summary ticket for HITL path",
                                "context": _VALID_CTX,
                            },
                            "id": "tc-plan-1",
                        }
                    ],
                )
            ]
        }

    tools_ran = {"n": 0}

    def tools(state: ReactGraphState) -> dict[str, Any]:
        path.append("tools")
        tools_ran["n"] += 1
        return {"halt": True, "final_raw": "queued", "last_obs": "ok"}

    monkeypatch.setattr(
        "donna.agentic_react_graph.generate_jason_ticket_critique",
        lambda *_a, **_k: "Critique ok for planning corridor.",
    )
    monkeypatch.setattr("donna.core_agent.enqueue_speech", lambda *_a, **_k: None)

    graph = compile_donna_react_graph(
        agent,
        tools,
        planner_node_fn=planner,
        executor_node_fn=executor,
        checkpointer=MemorySaver(),
    )
    node_ids = set(graph.get_graph().nodes)
    assert {"planner", "executor", "agent", "ticket_validate", "tools"} <= node_ids

    cfg = {"configurable": {"thread_id": "plan-corridor"}}
    list(
        graph.stream(
            {
                "messages": [HumanMessage(content=_DESKTOP_UTTERANCE)],
                "halt": False,
                "always_include": [],
            },
            cfg,
            stream_mode="values",
        )
    )
    assert path[:3] == ["planner", "executor", "agent"]
    assert graph.get_state(cfg).interrupts

    list(
        graph.stream(
            Command(resume={"approved": True, "action": "approve"}),
            cfg,
            stream_mode="values",
        )
    )
    assert tools_ran["n"] == 1
    assert "tools" in path
