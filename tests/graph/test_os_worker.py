"""Focused unit tests for OS execution worker + supervisor routing."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from dana.agentic_react_graph import compile_dana_react_graph
from dana.graph.workers.os_worker import (
    OS_WORKER_NODE,
    OS_WORKER_SYSTEM_PROMPT,
    make_os_worker_node,
    route_after_executor,
    should_route_to_os_worker,
)
from dana.schema import ReactGraphState


def test_should_route_powershell_hint() -> None:
    state: ReactGraphState = {
        "messages": [
            HumanMessage(
                content="Use PowerShell to check my current active network adapters"
            )
        ],
        "active_intent": "network adapters",
    }
    assert should_route_to_os_worker(state) is True
    assert route_after_executor(state) == OS_WORKER_NODE


def test_should_not_route_vision_or_chat() -> None:
    vision: ReactGraphState = {
        "messages": [HumanMessage(content="What do you see on my screen?")],
    }
    chat: ReactGraphState = {
        "messages": [HumanMessage(content="Hello, how are you today?")],
    }
    assert should_route_to_os_worker(vision) is False
    assert should_route_to_os_worker(chat) is False
    assert route_after_executor(chat) == "agent"


def test_os_worker_offline_fallback_calls_execute_powershell() -> None:
    calls: list[str] = []

    def fake_ps(command: str) -> str:
        calls.append(command)
        return "returncode=0\nstdout:\nActuator Online\nstderr:\n(empty)"

    node = make_os_worker_node(llm=None, execute_powershell_fn=fake_ps)
    out = node(
        {
            "messages": [
                HumanMessage(content="Use PowerShell: Write-Output 'Actuator Online'")
            ],
            "halt": False,
        }
    )
    assert calls, "offline fallback must invoke execute_powershell"
    assert "Write-Output" in calls[0] or "Actuator Online" in calls[0]
    assert out["halt"] is True
    assert out["current_agent"] == "OS_Worker"
    assert "Actuator Online" in str(out.get("final_raw") or "")
    assert OS_WORKER_SYSTEM_PROMPT.startswith("You are the OS Execution Worker")


def test_os_worker_binds_only_execute_powershell_via_llm() -> None:
    bound = MagicMock()
    bound.invoke.return_value = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "execute_powershell",
                "args": {"command": "Get-Process | Select-Object -First 1"},
                "id": "call-ps-1",
                "type": "tool_call",
            }
        ],
    )
    llm = MagicMock()
    llm.bind_tools.return_value = bound
    seen: list[str] = []

    def fake_ps(command: str) -> str:
        seen.append(command)
        return "returncode=0\nstdout:\nok\nstderr:\n(empty)"

    node = make_os_worker_node(llm=llm, execute_powershell_fn=fake_ps)
    out = node({"messages": [HumanMessage(content="run powershell Get-Process")]})

    assert llm.bind_tools.called
    tools_arg = llm.bind_tools.call_args[0][0]
    names = [getattr(t, "name", None) for t in tools_arg]
    assert names == ["execute_powershell"]
    assert seen == ["Get-Process | Select-Object -First 1"]
    assert out["final_raw"].startswith("returncode=0")


def test_compile_routes_os_intent_to_worker_then_verifier() -> None:
    path: list[str] = []

    def planner(state: ReactGraphState) -> dict[str, Any]:
        path.append("planner")
        return {
            "always_include": ["execute_powershell"],
            "execution_plan": {
                "required_tools": ["execute_powershell"],
                "status": "planned",
            },
            "current_agent": "Planner",
        }

    def executor(state: ReactGraphState) -> dict[str, Any]:
        path.append("executor")
        return {"current_agent": "Executor"}

    def agent(state: ReactGraphState) -> dict[str, Any]:
        path.append("agent")
        return {"halt": True, "final_raw": "should_not_reach_agent"}

    def tools(state: ReactGraphState) -> dict[str, Any]:
        path.append("tools")
        return {"halt": True, "final_raw": "should_not_reach_tools"}

    def os_worker(state: ReactGraphState) -> dict[str, Any]:
        path.append("os_worker")
        return {
            "halt": True,
            "final_raw": "returncode=0\nstdout:\nworker_ok\nstderr:\n(empty)",
            "last_obs": "worker_ok",
            "current_agent": "OS_Worker",
        }

    def verifier(state: ReactGraphState) -> dict[str, Any]:
        path.append("verifier")
        return {
            "verification_result": {"verified": True, "attempts": 1, "evidence": {}},
            "halt": True,
            "current_agent": "Verifier",
        }

    def hydrate(state: ReactGraphState) -> dict[str, Any]:
        path.append("hydrate_memory")
        return {}

    def consolidate(state: ReactGraphState) -> dict[str, Any]:
        path.append("consolidate_memory")
        return {}

    graph = compile_dana_react_graph(
        agent,
        tools,
        planner_node_fn=planner,
        executor_node_fn=executor,
        os_worker_node_fn=os_worker,
        verifier_node_fn=verifier,
        hydrate_memory_node_fn=hydrate,
        consolidate_memory_node_fn=consolidate,
        checkpointer=MemorySaver(),
    )
    cfg = {"configurable": {"thread_id": "os-worker-route"}}
    list(
        graph.stream(
            {
                "messages": [
                    HumanMessage(
                        content="Use PowerShell to list network adapters"
                    )
                ],
                "halt": False,
                "always_include": ["execute_powershell"],
                "active_intent": "Use PowerShell to list network adapters",
            },
            cfg,
            stream_mode="values",
        )
    )
    assert "os_worker" in path
    assert "agent" not in path
    assert path.index("os_worker") > path.index("executor")
    assert "verifier" in path
    final = graph.get_state(cfg).values
    assert "worker_ok" in str(final.get("final_raw") or "")
