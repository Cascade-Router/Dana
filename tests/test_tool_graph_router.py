"""Tests for chat→tool-graph semantic escalation."""

from __future__ import annotations

from dana.agentic import (
    clear_chat_memory,
    requires_tool_graph,
    run_lightweight_chat,
    set_donna_mode,
)
from dana.agentic_planning import desktop_plan_intent
from dana.cascade_router import decide_route
from dana.graph.completion_gate import (
    is_filler_response,
    is_verbal_tool_graph_escalation,
    should_reject_chat_final,
)
from dana.tools.broker import get_broker


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


def test_tool_graph_escalates_desktop_window_ticket() -> None:
    assert (
        requires_tool_graph(
            "Summarize active window and create a desktop log ticket"
        )
        is True
    )


def test_tool_graph_escalates_window_app_and_log_prompts() -> None:
    """Audit prompts that previously soft-dropped on lightweight chat."""
    prompts = [
        "How many windows do I have open",
        "how many windows tabs do I have open?",
        "Please close Discord app",
        "Can you open your latest logs?",
        "open logs",
        "Please retrack your latest logs...",
    ]
    for prompt in prompts:
        assert requires_tool_graph(prompt) is True, prompt
        assert desktop_plan_intent(prompt) is True, prompt


def test_readonly_agent_verification_prompts_require_tool_graph() -> None:
    """Read-only audit prompts must escalate (never filler-only chat)."""
    prompts = [
        "How many windows do I have open?",
        "Can you open your latest logs and summarize the last boot session?",
        "Describe what you see on my screen right now.",
        "Search episodic memory for user preferences.",
    ]
    for prompt in prompts:
        assert requires_tool_graph(prompt) is True, prompt
    # Optional: cascade must not take lightweight chat early-return.
    set_donna_mode("chat")
    for prompt in prompts:
        d = decide_route(prompt, forced_tool=None)
        assert "tools/MoA bypassed" not in (d.reason or ""), prompt


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


def test_filler_and_verbal_escalate_reject_chat_final() -> None:
    assert is_filler_response("Let me see...")
    assert is_filler_response("Let me check again...")
    assert is_verbal_tool_graph_escalation(
        "To help with your request, I need to escalate this to the tool-graph route."
    )
    assert should_reject_chat_final("Let me see...") is True
    assert should_reject_chat_final(
        "I need to escalate this to the tool-graph route."
    ) is True
    assert should_reject_chat_final("You have three windows open.") is False


def test_lightweight_chat_skips_memory_on_filler() -> None:
    clear_chat_memory()

    def _ask(_messages, model="llama3.2"):  # noqa: ARG001
        return "Let me see..."

    result = run_lightweight_chat(
        user_text="How many windows do I have open",
        ask_fn=_ask,
        use_chat_memory=True,
    )
    assert "see" in (result.final_text or "").lower()
    from dana.agentic import chat_memory_size

    assert chat_memory_size() == 0
    clear_chat_memory()


def test_lightweight_chat_skips_memory_on_verbal_escalate() -> None:
    clear_chat_memory()

    def _ask(_messages, model="llama3.2"):  # noqa: ARG001
        return (
            "To help with your request, I need to escalate this to the "
            "tool-graph route."
        )

    result = run_lightweight_chat(
        user_text="how many windows tabs do I have open?",
        ask_fn=_ask,
        use_chat_memory=True,
    )
    assert should_reject_chat_final(result.final_text) is True
    from dana.agentic import chat_memory_size

    assert chat_memory_size() == 0
    clear_chat_memory()
