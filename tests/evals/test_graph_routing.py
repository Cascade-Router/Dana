"""Deterministic LangGraph routing evals (mocked supervisor / no live LLM)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END

from dana.agentic import requires_tool_graph, set_donna_mode
from dana.agentic_planning import build_structured_plan, desktop_plan_intent
from dana.agentic_react_graph import (
    _route_after_agent,
    compile_donna_react_graph,
)
from dana.schema import ReactGraphState
from dana.tools.broker import merge_bound_tool_ids


# Catalog subset used by planner / binder evals (no disk registry dependency).
_KNOWN_TOOLS = (
    "analyze_visual_context",
    "ocr_with_region",
    "python_repl",
    "shell_execute",
    "file_editor",
    "draft_cursor_prompt",
)


def _mock_supervisor_llm(script: list[AIMessage]) -> MagicMock:
    """Scripted ChatOllama stand-in: bind_tools → invoke yield AIMessages."""
    bound = MagicMock()
    queue = list(script)

    def _next_msg(*_a: Any, **_k: Any) -> AIMessage:
        if not queue:
            return AIMessage(content="FINAL: done")
        return queue.pop(0)

    bound.invoke.side_effect = _next_msg
    llm = MagicMock()
    llm.bind_tools.return_value = bound
    return llm


def _tool_call_msg(name: str, args: dict[str, Any] | None = None) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": name,
                "args": dict(args or {}),
                "id": f"call-{name}",
                "type": "tool_call",
            }
        ],
    )


@pytest.fixture(autouse=True)
def _chat_mode() -> None:
    set_donna_mode("chat")
    yield
    set_donna_mode("chat")


def test_summarize_text_routes_to_vision_plan() -> None:
    """Supervisor-style foresight: summarize → vision / OCR required tools."""
    q = "Summarize active window text and highlight errors"
    assert desktop_plan_intent(q) is True
    assert requires_tool_graph(q) is True
    plan = build_structured_plan(
        q,
        tool_ids=list(_KNOWN_TOOLS),
        window_meta={"title": "Notepad", "available": True, "excluded_self": False},
    )
    required = plan["required_tools"]
    assert any(
        t in required for t in ("analyze_visual_context", "ocr_with_region")
    ), required


def test_code_patch_routes_to_repl_suite() -> None:
    """Code / patch intents bind the local REPL suite (python_repl)."""
    q = "Please run a python_repl code patch to fix core_agent.py"
    assert requires_tool_graph(q) is True
    bound = merge_bound_tool_ids(
        user_text=q,
        forced_tool_id=None,
        mode="developer",
        known_ids=_KNOWN_TOOLS,
    )
    assert "python_repl" in bound


def test_mocked_supervisor_routes_vision_then_tools() -> None:
    """Compile graph with a mocked agent that emits vision tool_calls → tools."""
    path: list[str] = []
    llm = _mock_supervisor_llm(
        [_tool_call_msg("analyze_visual_context", {"source": "screen"})]
    )

    def planner(state: ReactGraphState) -> dict[str, Any]:
        path.append("planner")
        return {
            "execution_plan": {
                "required_tools": ["analyze_visual_context"],
                "status": "planned",
            },
            "always_include": ["analyze_visual_context"],
            "current_agent": "Planner",
        }

    def executor(state: ReactGraphState) -> dict[str, Any]:
        path.append("executor")
        return {"current_agent": "Executor", "execution_plan": {"status": "executing"}}

    def agent(state: ReactGraphState) -> dict[str, Any]:
        path.append("agent")
        bound = llm.bind_tools([])
        msg = bound.invoke(state.get("messages") or [])
        return {"messages": [msg], "halt": False}

    def tools(state: ReactGraphState) -> dict[str, Any]:
        path.append("tools")
        return {"halt": True, "final_raw": "vision_ok", "last_obs": "OK: screen"}

    graph = compile_donna_react_graph(
        agent,
        tools,
        planner_node_fn=planner,
        executor_node_fn=executor,
        checkpointer=MemorySaver(),
    )
    cfg = {"configurable": {"thread_id": "eval-vision-route"}}
    list(
        graph.stream(
            {
                "messages": [HumanMessage(content="summarize the screen")],
                "halt": False,
                "always_include": ["analyze_visual_context"],
                "session_id": "eval-vision",
                "active_intent": "summarize the screen",
            },
            cfg,
            stream_mode="values",
        )
    )
    assert path[:3] == ["planner", "executor", "agent"]
    assert "tools" in path
    final = graph.get_state(cfg).values
    assert final.get("final_raw") == "vision_ok"
    assert final.get("halt") is True


def test_mocked_supervisor_routes_repl_tool() -> None:
    path: list[str] = []

    def planner(state: ReactGraphState) -> dict[str, Any]:
        path.append("planner")
        return {"always_include": ["python_repl"], "current_agent": "Planner"}

    def executor(state: ReactGraphState) -> dict[str, Any]:
        path.append("executor")
        return {"current_agent": "Executor"}

    def agent(state: ReactGraphState) -> dict[str, Any]:
        path.append("agent")
        return {
            "messages": [
                _tool_call_msg(
                    "python_repl",
                    {"code": "print(1+1)"},
                )
            ]
        }

    def tools(state: ReactGraphState) -> dict[str, Any]:
        path.append("tools")
        msgs = state.get("messages") or []
        last = msgs[-1] if msgs else None
        names = [tc.get("name") for tc in (getattr(last, "tool_calls", None) or [])]
        assert "python_repl" in names
        return {"halt": True, "final_raw": "repl_ok", "last_obs": "OK: 2"}

    graph = compile_donna_react_graph(
        agent,
        tools,
        planner_node_fn=planner,
        executor_node_fn=executor,
        checkpointer=MemorySaver(),
    )
    cfg = {"configurable": {"thread_id": "eval-repl-route"}}
    list(
        graph.stream(
            {
                "messages": [
                    HumanMessage(content="run a python_repl code patch")
                ],
                "halt": False,
                "always_include": ["python_repl"],
            },
            cfg,
            stream_mode="values",
        )
    )
    assert path == ["planner", "executor", "agent", "tools"]


def test_invalid_empty_input_routes_to_end_without_crash() -> None:
    """Empty / nonsense inputs must not raise; graph ends or stays recoverable."""
    assert requires_tool_graph("") is False
    assert requires_tool_graph("   ") is False
    plan = build_structured_plan(
        "",
        tool_ids=list(_KNOWN_TOOLS),
        window_meta={"title": "", "available": False},
    )
    assert plan["intended_goal"] == "(empty)"
    assert isinstance(plan["required_tools"], list)

    state: ReactGraphState = {
        "messages": [AIMessage(content="I cannot help with that.")],
        "halt": True,
        "always_include": [],
        "session_id": "eval-empty",
        "active_intent": "",
    }
    assert _route_after_agent(state) == END


def test_invalid_unknown_tool_call_still_routes_to_tools_safely() -> None:
    """Unknown tool ids still take the tools edge; tools node handles recovery."""
    msg = _tool_call_msg("not_a_real_tool", {})
    state: ReactGraphState = {
        "messages": [msg],
        "halt": False,
        "always_include": [],
    }
    assert _route_after_agent(state) == "tools"
