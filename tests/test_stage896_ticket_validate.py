"""Stage 8.9.6 — Pydantic validation loop before HITL + rich ticket display."""

from __future__ import annotations

from typing import Any

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from dana.agentic_react_graph import (
    _MAX_TICKET_VALIDATION_RETRIES,
    _route_after_agent,
    _route_after_ticket_validate,
    compile_donna_react_graph,
    jason_ticket_review_node,
    ticket_approval_node,
    ticket_validate_node,
)
from dana.middleware import hitl_ticket as hitl
from dana.schema import ReactGraphState

_VALID_CONTEXT = (
    "Target files: dana/agentic_react_graph.py, dana/tools/guards.py\n"
    "Root cause: Intent-echo contexts were reaching HITL before Pydantic ran.\n"
    "Step-by-step changes: 1) validate payload 2) bounce MoA up to 3 times "
    "3) only then Jason/HITL.\n"
    "Acceptance criteria: invalid tickets never interrupt; max retries yields "
    "cleanly to the operator."
)


@pytest.fixture(autouse=True)
def _hitl_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DONNA_HITL_TICKET", "1")
    monkeypatch.delenv("DONNA_HITL_AUTO_APPROVE", raising=False)
    monkeypatch.delenv("DONNA_HITL_AUTO_DENY", raising=False)
    monkeypatch.setenv("DONNA_HITL_REQUIRE_GUI", "1")
    hitl.clear_pending()
    yield
    hitl.clear_pending()


def _ai_draft(objective: str, context: str, *, call_id: str = "call-d1") -> Any:
    from langchain_core.messages import AIMessage

    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "draft_cursor_prompt",
                "args": {"objective": objective, "context": context},
                "id": call_id,
                "type": "tool_call",
            }
        ],
    )


def test_route_agent_goes_to_ticket_validate() -> None:
    state: ReactGraphState = {
        "messages": [_ai_draft("Valid objective text", _VALID_CONTEXT)],
        "session_id": "s",
    }
    assert _route_after_agent(state) == "ticket_validate"


def test_validate_success_sets_drafted_ticket() -> None:
    state: ReactGraphState = {
        "messages": [
            _ai_draft(
                "Implement ticket validation gate before HITL",
                _VALID_CONTEXT,
            )
        ],
        "session_id": "ok",
        "ticket_validation_retries": 0,
    }
    out = ticket_validate_node(state)
    assert out.get("ticket_validated") is True
    assert out.get("halt") is not True
    drafted = out.get("drafted_ticket") or {}
    assert "Implement ticket validation" in drafted.get("objective", "")
    assert "Root cause" in drafted.get("context", "")
    assert _route_after_ticket_validate(out) == "jason_ticket_review"


def test_validate_failure_routes_back_to_agent() -> None:
    state: ReactGraphState = {
        "messages": [
            _ai_draft(
                "thin obj",
                "**Technical intent:** thin obj\n**Target Files:** dana/x.py",
            )
        ],
        "session_id": "bad",
        "ticket_validation_retries": 0,
    }
    out = ticket_validate_node(state)
    assert out.get("ticket_validated") is False
    assert out.get("halt") is not True
    assert int(out.get("ticket_validation_retries") or 0) == 1
    assert "Ticket validation failed" in str(out.get("last_obs") or "")
    assert _route_after_ticket_validate(out) == "agent"
    # Must close the tool_call with a ToolMessage (no HITL).
    msgs = out.get("messages") or []
    assert msgs
    assert getattr(msgs[0], "tool_call_id", None)


def test_validate_max_retries_yields_to_user() -> None:
    state: ReactGraphState = {
        "messages": [
            _ai_draft(
                "still thin",
                "**Technical intent:** still thin\n**Target Files:** dana/x.py",
            )
        ],
        "session_id": "max",
        "ticket_validation_retries": _MAX_TICKET_VALIDATION_RETRIES - 1,
    }
    out = ticket_validate_node(state)
    assert out.get("halt") is True
    assert "Max retries reached" in str(out.get("final_raw") or "")
    assert _route_after_ticket_validate(out) == END


def test_invalid_never_hits_hitl_interrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bad payload bounces; Approve breakpoint must not fire."""

    attempts = {"n": 0}

    def agent(state: ReactGraphState) -> dict[str, Any]:
        attempts["n"] += 1
        # Always emit the same invalid intent-echo payload.
        return {
            "messages": [
                _ai_draft(
                    "enhance cursor rendering palette",
                    "**Technical intent:** enhance cursor rendering palette\n"
                    "**Target Files:** dana/agentic.py",
                    call_id=f"call-{attempts['n']}",
                )
            ]
        }

    tools_ran = {"n": 0}

    def tools(state: ReactGraphState) -> dict[str, Any]:
        tools_ran["n"] += 1
        return {"halt": True, "final_raw": "should-not-run"}

    monkeypatch.setattr(
        "dana.core_agent.enqueue_speech",
        lambda *_a, **_k: None,
    )

    g = StateGraph(ReactGraphState)
    g.add_node("agent", agent)
    g.add_node("ticket_validate", ticket_validate_node)
    g.add_node("jason_ticket_review", jason_ticket_review_node)
    g.add_node("ticket_approval", ticket_approval_node)
    g.add_node("tools", tools)
    g.add_edge(START, "agent")
    g.add_conditional_edges(
        "agent",
        _route_after_agent,
        {
            "ticket_validate": "ticket_validate",
            "tools": "tools",
            "agent": "agent",
            END: END,
        },
    )
    g.add_conditional_edges(
        "ticket_validate",
        _route_after_ticket_validate,
        {
            "jason_ticket_review": "jason_ticket_review",
            "agent": "agent",
            END: END,
        },
    )
    g.add_conditional_edges(
        "jason_ticket_review",
        lambda s: END if s.get("halt") else "ticket_approval",
        {"ticket_approval": "ticket_approval", END: END},
    )
    g.add_conditional_edges(
        "ticket_approval",
        lambda s: END if s.get("halt") else "tools",
        {"tools": "tools", END: END},
    )
    g.add_edge("tools", END)
    graph = g.compile(checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "no-hitl-invalid"}}

    list(graph.stream({"messages": [], "halt": False}, cfg, stream_mode="values"))
    snap = graph.get_state(cfg)
    assert not snap.interrupts
    assert tools_ran["n"] == 0
    final = snap.values
    assert final.get("halt") is True
    assert "Max retries" in str(final.get("final_raw") or "")
    assert attempts["n"] == _MAX_TICKET_VALIDATION_RETRIES


def test_valid_hits_hitl_with_full_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    def agent(state: ReactGraphState) -> dict[str, Any]:
        return {
            "messages": [
                _ai_draft(
                    "Implement pydantic ticket validation before HITL",
                    _VALID_CONTEXT,
                )
            ]
        }

    tools_ran = {"n": 0}

    def tools(state: ReactGraphState) -> dict[str, Any]:
        tools_ran["n"] += 1
        return {"halt": True, "final_raw": "queued"}

    monkeypatch.setattr(
        "dana.agentic_react_graph.generate_jason_ticket_critique",
        lambda *_a, **_k: "Looks solid — approve if you agree.",
    )
    monkeypatch.setattr(
        "dana.core_agent.enqueue_speech",
        lambda *_a, **_k: None,
    )

    graph = compile_donna_react_graph(agent, tools, checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "valid-hitl"}}
    list(graph.stream({"messages": [], "halt": False}, cfg, stream_mode="values"))
    snap = graph.get_state(cfg)
    assert snap.interrupts
    value = snap.interrupts[0].value
    assert "Implement pydantic" in str(value.get("objective") or "")
    assert "Root cause" in str(value.get("context") or "")
    assert value.get("jason_critique")
    formatted = hitl.format_ticket_payload(value)
    assert "Objective:" in formatted
    assert "Context:" in formatted
    assert "Files:" in formatted

    list(
        graph.stream(
            Command(resume={"approved": True, "action": "approve"}),
            cfg,
            stream_mode="values",
        )
    )
    assert tools_ran["n"] == 1


def test_format_and_orb_ticket_widgets() -> None:
    import customtkinter as ctk

    from dana.ui.assistive_orb import AssistiveTouchOrb

    text = hitl.format_ticket_payload(
        {
            "objective": "Fix validation loop",
            "context": _VALID_CONTEXT,
            "jason_critique": "Approve.",
            "files": "dana/tools/guards.py",
        }
    )
    assert "Objective:\nFix validation loop" in text
    assert "Files:" in text
    assert "dana/tools/guards.py" in text or "dana/agentic_react_graph.py" in text

    root = ctk.CTk()
    root.withdraw()
    orb = AssistiveTouchOrb(root, dictation_getter=lambda: False)
    assert hasattr(orb, "_ticket_lbl")
    hitl.publish_pending(
        {
            "objective": "Fix validation loop",
            "context": _VALID_CONTEXT,
            "jason_critique": "Approve.",
        },
        thread_id="orb",
    )
    orb.refresh_controls()
    root.update_idletasks()
    assert orb._hitl_pending is True
    shown = str(orb._ticket_lbl.cget("text"))
    assert "Objective:" in shown
    assert "Context:" in shown
    assert "Files:" in shown
    orb.destroy()
    root.destroy()


def test_compile_includes_ticket_validate() -> None:
    def agent(state: ReactGraphState) -> dict[str, Any]:
        return {"halt": True}

    def tools(state: ReactGraphState) -> dict[str, Any]:
        return {"halt": True}

    graph = compile_donna_react_graph(agent, tools, checkpointer=MemorySaver())
    nodes = set(graph.get_graph().nodes)
    assert "ticket_validate" in nodes
    assert "jason_ticket_review" in nodes
    assert "ticket_approval" in nodes
