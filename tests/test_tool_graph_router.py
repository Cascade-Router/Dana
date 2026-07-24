"""Tests for chat→tool-graph semantic escalation."""

from __future__ import annotations

from donna.agentic import requires_tool_graph, set_donna_mode
from donna.cascade_router import decide_route
from donna.tools.broker import get_broker


def test_greeting_stays_chat_path() -> None:
    assert requires_tool_graph("Hello Donna, how are you?") is False
    assert requires_tool_graph("What time is it?") is False


def test_system_repl_read_forces_tool_graph() -> None:
    q = "Do you see your input? Use your tools to read the system_repl.py file."
    assert requires_tool_graph(q) is True


def test_keywords_and_extensions_force_tool() -> None:
    assert requires_tool_graph("please run this in the terminal") is True
    assert requires_tool_graph("execute a script") is True
    assert requires_tool_graph("fix the bug in core_agent.py") is True
    assert requires_tool_graph("update settings.json") is True


def test_tool_graph_escalates_research_store_and_ollama() -> None:
    assert requires_tool_graph("Don't you have access to your research store?") is True
    assert requires_tool_graph("Is my Olama local server online?") is True
    assert requires_tool_graph("Is Ollama online?") is True
    assert requires_tool_graph("Hello Donna, how are you?") is False


def test_cascade_chat_mode_escalates_on_tool_intent() -> None:
    set_donna_mode("chat")
    d = decide_route(
        "Use your tools to read the system_repl.py file.",
        forced_tool=None,
    )
    # Must not take the lightweight chat early-return reason.
    assert "tools/MoA bypassed" not in (d.reason or "")


def test_broker_foresight_file_editor_for_system_repl() -> None:
    set_donna_mode("chat")
    broker = get_broker()
    call = broker.parse_utterance(
        "Use your tools to read the system_repl.py file."
    )
    assert call is not None
    assert call.tool_id == "file_editor"
    assert call.arguments.get("action") == "read"
    path = str(call.arguments.get("filepath") or "")
    assert "system_repl.py" in path.replace("\\", "/")
