"""Stage 8.6 — HITL ticket approval (LangGraph interrupt + GUI bridge)."""

from __future__ import annotations

import os
from typing import Any

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END
from langgraph.types import Command

from dana.agentic_react_graph import (
    _route_after_agent,
    _route_after_jason_review,
    _route_after_ticket_approval,
    _route_after_ticket_validate,
    compile_donna_react_graph,
    extract_draft_cursor_payload,
    message_has_draft_cursor_prompt,
)
from dana.middleware import hitl_ticket as hitl
from dana.schema import ReactGraphState

_VALID_CTX = (
    "Target files: dana/middleware/hitl_ticket.py\n"
    "Root cause: HITL must pause only on validated draft_cursor_prompt payloads.\n"
    "Step-by-step changes: 1) interrupt 2) GUI approve/deny 3) resume tools.\n"
    "Acceptance criteria: Approve runs tools; Deny halts without tool execution."
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
    objective: str = "Fix HITL ticket approval path",
    context: str = _VALID_CTX,
) -> Any:
    from langchain_core.messages import AIMessage

    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "draft_cursor_prompt",
                "args": {"objective": objective, "context": context},
                "id": "call-draft-1",
                "type": "tool_call",
            }
        ],
    )


def test_extract_draft_payload_and_route() -> None:
    msg = _ai_with_draft("Obj long enough", _VALID_CTX)
    assert message_has_draft_cursor_prompt(msg)
    state: ReactGraphState = {
        "messages": [msg],
        "session_id": "s1",
        "active_intent": "draft_cursor_prompt",
    }
    payload = extract_draft_cursor_payload(state)
    assert payload["objective"] == "Obj long enough"
    assert "Root cause" in payload["context"]
    assert payload["tool"] == "draft_cursor_prompt"
    assert _route_after_agent(state) == "ticket_validate"
    assert _route_after_ticket_validate({"ticket_validated": True}) == "jason_ticket_review"
    assert _route_after_jason_review({"halt": False}) == "ticket_approval"
    assert _route_after_ticket_approval({"halt": False}) == "tools"
    assert _route_after_ticket_approval({"halt": True}) == END


def test_hitl_bridge_approve_deny() -> None:
    os.environ["DONNA_HITL_REQUIRE_GUI"] = "1"
    hitl.publish_pending(
        {"tool": "draft_cursor_prompt", "objective": "O", "context": "C"},
        thread_id="t",
    )
    assert hitl.is_pending()
    formatted = hitl.format_ticket_payload(hitl.get_pending())
    assert "HITL TICKET APPROVAL" in formatted
    assert "JASON REVIEW" in formatted

    assert hitl.submit_decision(True, action="approve")
    decision = hitl.wait_for_decision(timeout_s=1.0)
    assert hitl.decision_is_approved(decision)

    hitl.clear_pending()
    hitl.publish_pending({"objective": "X", "context": "Y"}, thread_id="t2")
    assert hitl.submit_decision(False, action="deny")
    decision = hitl.wait_for_decision(timeout_s=1.0)
    assert not hitl.decision_is_approved(decision)


def test_langgraph_interrupt_approve_runs_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    """Production path: ticket_validate → jason → ticket_approval → tools."""

    def agent(state: ReactGraphState) -> dict[str, Any]:
        return {
            "messages": [
                _ai_with_draft("Approve path for HITL resume flow", _VALID_CTX)
            ]
        }

    tools_ran = {"n": 0}

    def tools(state: ReactGraphState) -> dict[str, Any]:
        tools_ran["n"] += 1
        return {
            "halt": True,
            "final_raw": "queued",
            "last_obs": "Action queued successfully.",
        }

    monkeypatch.setattr(
        "dana.agentic_react_graph.generate_jason_ticket_critique",
        lambda *_a, **_k: "This ticket accurately captures the API constraints.",
    )
    monkeypatch.setattr(
        "dana.core_agent.enqueue_speech",
        lambda *_a, **_k: None,
    )

    graph = compile_donna_react_graph(agent, tools, checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "hitl-approve"}}

    list(graph.stream({"messages": [], "halt": False}, cfg, stream_mode="values"))
    snap = graph.get_state(cfg)
    assert snap.interrupts
    assert "Approve path" in str(snap.interrupts[0].value["objective"])
    assert "accurately" in str(snap.interrupts[0].value.get("jason_critique") or "")

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
    assert not graph.get_state(cfg).interrupts


def test_langgraph_interrupt_deny_skips_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    tools_ran = {"n": 0}

    def agent(state: ReactGraphState) -> dict[str, Any]:
        return {
            "messages": [_ai_with_draft("Deny path for HITL resume flow", _VALID_CTX)]
        }

    def tools(state: ReactGraphState) -> dict[str, Any]:
        tools_ran["n"] += 1
        return {"halt": True, "final_raw": "should-not-run"}

    monkeypatch.setattr(
        "dana.agentic_react_graph.generate_jason_ticket_critique",
        lambda *_a, **_k: "Missing visual bounds — deny.",
    )
    monkeypatch.setattr(
        "dana.core_agent.enqueue_speech",
        lambda *_a, **_k: None,
    )

    graph = compile_donna_react_graph(agent, tools, checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "hitl-deny"}}

    list(graph.stream({"messages": [], "halt": False}, cfg, stream_mode="values"))
    list(
        graph.stream(
            Command(resume={"approved": False, "action": "deny"}),
            cfg,
            stream_mode="values",
        )
    )
    final = graph.get_state(cfg).values
    assert tools_ran["n"] == 0
    assert final.get("halt") is True
    assert "cancelled" in str(final.get("final_raw") or "").lower()


def test_compile_includes_ticket_approval_node() -> None:
    def agent(state: ReactGraphState) -> dict[str, Any]:
        return {"halt": True, "final_raw": "ok"}

    def tools(state: ReactGraphState) -> dict[str, Any]:
        return {"halt": True}

    graph = compile_donna_react_graph(agent, tools, checkpointer=MemorySaver())
    nodes = set(graph.get_graph().nodes)
    assert "ticket_validate" in nodes
    assert "jason_ticket_review" in nodes
    assert "ticket_approval" in nodes
    assert "tools" in nodes
    assert "agent" in nodes


def test_gui_hitl_buttons_toggle() -> None:
    os.environ.setdefault("DONNA_OS_DRY_RUN", "1")
    import customtkinter as ctk

    from dana.ui.trace_window import LiveTracePanel
    from dana.schema import TraceEvent

    root = ctk.CTk()
    root.withdraw()
    panel = LiveTracePanel(root)
    panel.update_idletasks()
    assert panel._hitl_visible is False
    assert not panel._hitl_bar.winfo_ismapped()

    panel._handle_event(
        TraceEvent(
            event_type="status",
            node="ticket_approval",
            message="HITL_PENDING_APPROVAL",
            payload="=== HITL TICKET APPROVAL REQUIRED ===\nOBJECTIVE:\nTest",
            tool="draft_cursor_prompt",
        )
    )
    panel.update_idletasks()
    assert panel._hitl_visible is True
    # Withdrawn roots may report not-mapped; pack_info confirms the bar is shown.
    assert panel._hitl_bar.pack_info()

    hitl.publish_pending({"objective": "GUI", "context": "btn"}, thread_id="g")
    panel._on_hitl_approve()
    panel.update_idletasks()
    assert panel._hitl_visible is False
    with pytest.raises(Exception):
        panel._hitl_bar.pack_info()

    hitl.clear_pending()
    hitl.publish_pending({"objective": "GUI2", "context": "btn"}, thread_id="g2")
    panel._handle_event(
        TraceEvent(
            event_type="status",
            node="ticket_approval",
            message="HITL_PENDING_APPROVAL",
            payload="pending",
        )
    )
    assert panel._hitl_visible is True
    panel._on_hitl_deny()
    panel.update_idletasks()
    assert panel._hitl_visible is False

    root.destroy()
