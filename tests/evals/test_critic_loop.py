"""Self-healing python_repl critic loop evals (mocked critic / execution; no network)."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from donna.agentic import set_donna_mode
from donna.agentic_react_graph import compile_donna_react_graph, route_after_execution
from donna.graph.nodes.critic import (
    fail_closed_node,
    heuristic_critique,
    is_python_repl_failure,
    make_critic_node,
    python_repl_state_patch,
)
from donna.schema import ReactGraphState


@pytest.fixture(autouse=True)
def _chat_mode() -> None:
    set_donna_mode("chat")
    yield
    set_donna_mode("chat")


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


def test_is_python_repl_failure_detects_traceback_and_exit() -> None:
    assert is_python_repl_failure(
        "exit_code=1\nstdout:\n(empty)\nstderr:\nZeroDivisionError: division by zero"
    )
    assert is_python_repl_failure("ERROR: python_repl failed: boom")
    assert not is_python_repl_failure("exit_code=0\nstdout:\n2")


def test_python_repl_state_patch_only_on_failure() -> None:
    fail = python_repl_state_patch(
        code="print(1/0)",
        observation="exit_code=1\nstderr:\nZeroDivisionError",
    )
    assert fail["execution_error"]
    assert fail["last_code_snippet"] == "print(1/0)"
    ok = python_repl_state_patch(code="print(1)", observation="exit_code=0\nstdout:\n1")
    assert ok["execution_error"] is None


def test_route_after_execution_critic_then_fail_closed() -> None:
    assert (
        route_after_execution(
            {"execution_error": "ZeroDivisionError", "retry_count": 0, "max_retries": 3}
        )
        == "critic"
    )
    assert (
        route_after_execution(
            {"execution_error": "ZeroDivisionError", "retry_count": 3, "max_retries": 3}
        )
        == "fail_closed"
    )
    assert route_after_execution({"execution_error": None, "halt": True}) == "__end__"


def test_critic_loop_self_heals_within_two_retries() -> None:
    """Failing 1/0 → critic proposes 1/1 → tools succeeds; ≤2 critic visits."""
    path: list[str] = []
    exec_codes: list[str] = []

    def planner(state: ReactGraphState) -> dict[str, Any]:
        path.append("planner")
        return {
            "always_include": ["python_repl"],
            "current_agent": "Planner",
            "retry_count": 0,
            "max_retries": 3,
            "critique_history": [],
            "execution_error": None,
        }

    def executor(state: ReactGraphState) -> dict[str, Any]:
        path.append("executor")
        return {"current_agent": "Executor"}

    def agent(state: ReactGraphState) -> dict[str, Any]:
        path.append("agent")
        return {
            "messages": [_tool_call_msg("python_repl", {"code": "print(1/0)"})],
            "halt": False,
        }

    def tools(state: ReactGraphState) -> dict[str, Any]:
        path.append("tools")
        code = str(state.get("last_code_snippet") or "").strip()
        if not code:
            msgs = state.get("messages") or []
            last = msgs[-1] if msgs else None
            tcs = getattr(last, "tool_calls", None) or []
            if tcs:
                args = tcs[0].get("args") if isinstance(tcs[0], dict) else {}
                code = str((args or {}).get("code") or "")
        exec_codes.append(code)
        if "1/0" in code:
            obs = (
                "exit_code=1\nstdout:\n(empty)\nstderr:\n"
                "ZeroDivisionError: division by zero"
            )
            return {
                **python_repl_state_patch(code=code, observation=obs),
                "last_obs": obs,
                "halt": False,
                "final_raw": "",
            }
        obs = "exit_code=0\nstdout:\n1"
        return {
            **python_repl_state_patch(code=code, observation=obs),
            "last_obs": obs,
            "halt": True,
            "final_raw": "repl_healed",
        }

    def critic_llm(error: str, code: str) -> str:
        fixed = code.replace("1/0", "1/1")
        return (
            "ZeroDivisionError: change divisor to non-zero.\n"
            f"FIXED_CODE:\n```python\n{fixed}\n```"
        )

    critic = make_critic_node(critic_llm)

    def _critic(state: ReactGraphState) -> dict[str, Any]:
        path.append("critic")
        return critic(state)

    graph = compile_donna_react_graph(
        agent,
        tools,
        planner_node_fn=planner,
        executor_node_fn=executor,
        critic_node_fn=_critic,
        checkpointer=MemorySaver(),
    )
    cfg = {"configurable": {"thread_id": "eval-critic-heal"}}
    list(
        graph.stream(
            {
                "messages": [HumanMessage(content="run print(1/0)")],
                "halt": False,
                "always_include": ["python_repl"],
                "session_id": "eval-critic",
                "active_intent": "run print(1/0)",
                "retry_count": 0,
                "max_retries": 3,
                "critique_history": [],
                "execution_error": None,
                "last_code_snippet": "",
            },
            cfg,
            stream_mode="values",
        )
    )
    final = graph.get_state(cfg).values
    assert path[:3] == ["planner", "executor", "agent"]
    assert path.count("critic") == 1
    assert path.count("tools") == 2
    assert path.count("critic") <= 2
    assert any("1/0" in c for c in exec_codes)
    assert any("1/1" in c for c in exec_codes)
    assert final.get("execution_error") in (None, "")
    assert final.get("final_raw") == "repl_healed"
    assert final.get("halt") is True
    assert int(final.get("retry_count") or 0) <= 2
    assert final.get("critique_history")


def test_fail_closed_after_max_retries() -> None:
    path: list[str] = []

    def planner(state: ReactGraphState) -> dict[str, Any]:
        return {
            "always_include": ["python_repl"],
            "retry_count": 0,
            "max_retries": 2,
            "critique_history": [],
        }

    def executor(state: ReactGraphState) -> dict[str, Any]:
        return {}

    def agent(state: ReactGraphState) -> dict[str, Any]:
        path.append("agent")
        return {
            "messages": [_tool_call_msg("python_repl", {"code": "print(1/0)"})],
        }

    def tools(state: ReactGraphState) -> dict[str, Any]:
        path.append("tools")
        code = str(state.get("last_code_snippet") or "print(1/0)")
        msgs = state.get("messages") or []
        last = msgs[-1] if msgs else None
        tcs = getattr(last, "tool_calls", None) or []
        if tcs and not str(state.get("last_code_snippet") or "").strip():
            args = tcs[0].get("args") if isinstance(tcs[0], dict) else {}
            code = str((args or {}).get("code") or code)
        # Prefer last_code_snippet when critic re-injected a tool call.
        if str(state.get("last_code_snippet") or "").strip():
            code = str(state.get("last_code_snippet"))
        obs = "exit_code=1\nstderr:\nZeroDivisionError: division by zero"
        return {
            **python_repl_state_patch(code=code, observation=obs),
            "last_obs": obs,
            "halt": False,
        }

    def critic(state: ReactGraphState) -> dict[str, Any]:
        path.append("critic")
        hist = list(state.get("critique_history") or [])
        hist.append("no fix available")
        retry = int(state.get("retry_count") or 0) + 1
        return {
            "critique_history": hist,
            "retry_count": retry,
            "last_code_snippet": "print(1/0)",
            "execution_error": None,
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "python_repl",
                            "args": {"code": "print(1/0)"},
                            "id": f"critic-retry-{retry}",
                            "type": "tool_call",
                        }
                    ],
                )
            ],
        }

    graph = compile_donna_react_graph(
        agent,
        tools,
        planner_node_fn=planner,
        executor_node_fn=executor,
        critic_node_fn=critic,
        fail_closed_node_fn=fail_closed_node,
        checkpointer=MemorySaver(),
    )
    cfg: dict[str, Any] = {
        "configurable": {"thread_id": "eval-critic-fail-closed"},
        "recursion_limit": 40,
    }
    list(
        graph.stream(
            {
                "messages": [HumanMessage(content="always fail")],
                "halt": False,
                "session_id": "eval-fail-closed",
                "active_intent": "always fail",
                "retry_count": 0,
                "max_retries": 2,
                "critique_history": [],
                "execution_error": None,
            },
            cfg,
            stream_mode="values",
        )
    )
    final = graph.get_state(cfg).values
    assert str(final.get("final_raw") or "").startswith("FAIL_CLOSED")
    assert int(final.get("retry_count") or 0) >= 2
    assert path.count("critic") >= 2


def test_heuristic_critique_offline() -> None:
    out = heuristic_critique("ZeroDivisionError: division by zero", "print(1/0)")
    assert "FIXED_CODE" in out
    assert "1/1" in out or "/ 1" in out
