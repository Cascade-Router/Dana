"""Stage 2 regression: chat memory roles, fuzzy mode switch, MoA ledger gate."""

from __future__ import annotations

from donna.agentic import (
    append_chat_memory_turn,
    build_lightweight_chat_system_prompt,
    chat_memory_as_role_messages,
    chat_memory_size,
    clear_chat_memory,
    enrich_draft_cursor_args,
    parse_mode_switch,
    requires_tool_graph,
    run_lightweight_chat,
)
from donna.cascade_router import match_state_toggle
from donna.moa_tool_shim import parse_plan_fields, plan_has_structured_context
from donna.tools.broker import parse_draft_cursor_prompt_args
from donna.tools.general.draft_cursor_prompt import (
    _context_is_sufficiently_specific,
    draft_cursor_prompt,
)


def test_chat_memory_role_separated_across_turns() -> None:
    clear_chat_memory()
    captured: list[list[dict[str, str]]] = []

    def _ask(messages, model="llama3.2"):  # noqa: ANN001, ARG001
        captured.append(list(messages))
        # Echo last user content so retention is observable.
        last_user = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"),
            "",
        )
        return f"ack:{last_user}"

    r1 = run_lightweight_chat(
        user_text="My favorite color is teal.",
        ask_fn=_ask,
        use_chat_memory=True,
    )
    assert r1.final_text.startswith("ack:")
    assert chat_memory_size() == 1

    r2 = run_lightweight_chat(
        user_text="What color did I just mention?",
        ask_fn=_ask,
        use_chat_memory=True,
    )
    assert chat_memory_size() == 2
    assert len(captured) == 2
    second = captured[1]
    assert second[0]["role"] == "system"
    assert "Capability card" in second[0]["content"]
    # Role-separated history precedes the new user turn.
    roles = [m["role"] for m in second]
    assert roles == ["system", "user", "assistant", "user"]
    assert "teal" in second[1]["content"]
    assert second[-1]["content"] == "What color did I just mention?"
    # History must NOT be flattened into the system prompt.
    assert "Recent Conversation History:" not in second[0]["content"]
    assert r2.final_text.startswith("ack:")
    clear_chat_memory()


def test_chat_memory_as_role_messages_helper() -> None:
    clear_chat_memory()
    append_chat_memory_turn("hello", "hi there")
    append_chat_memory_turn("second", "reply two")
    msgs = chat_memory_as_role_messages()
    assert msgs == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "reply two"},
    ]
    clear_chat_memory()


def test_capability_card_in_system_prompt() -> None:
    prompt = build_lightweight_chat_system_prompt(reply_lang="en")
    assert "Capability card" in prompt
    assert "switch to vision" in prompt.lower()
    assert "ollama" in prompt.lower()


def test_fuzzy_mode_switch_without_literal_mode() -> None:
    assert parse_mode_switch("Switch to vision, Mo.") == "vision"
    assert parse_mode_switch("switch to vision") == "vision"
    assert match_state_toggle("switch to vision") == "vision"
    assert parse_mode_switch("go to developer") == "developer"
    assert parse_mode_switch("switch to research") == "research"
    # Bare single words must not flip mode.
    assert parse_mode_switch("vision") is None
    assert parse_mode_switch("research") is None


def test_tool_graph_escalates_research_store_and_ollama() -> None:
    assert requires_tool_graph("Don't you have access to your research store?") is True
    assert requires_tool_graph("Is my Olama local server online?") is True
    assert requires_tool_graph("Is Ollama online?") is True
    assert requires_tool_graph("Hello Donna, how are you?") is False


def test_broker_objective_not_midword_truncated() -> None:
    long = (
        "Donna, use the draft_cursor_prompt tool to log a self-improvement ticket "
        "to improve cursor handling in the deepseek pipeline by adding error logging "
        "for failed cursor handoffs and enhancing cursor prompt validation to reduce "
        "false positives."
    )
    args = parse_draft_cursor_prompt_args(long)
    obj = str(args.get("objective") or "")
    assert obj
    assert not obj.endswith("...")
    assert "pro" != obj[-3:]  # not the old mid-word "…cursor pro"
    assert "false positives" in obj or "validation" in obj or len(obj) > 120


def test_enrich_does_not_hard_cap_objective() -> None:
    obj = (
        "Harden the MoA draft_cursor_prompt pipeline so reasoner CONTEXT always "
        "includes target files, root cause, step-by-step changes, and acceptance "
        "criteria before ledger write."
    )
    enriched = enrich_draft_cursor_args(
        raw_text=obj,
        objective=obj,
        context="",
    )
    assert enriched["objective"] == obj or enriched["objective"].startswith(
        "Harden the MoA"
    )
    assert "..." not in enriched["objective"]


def test_parse_plan_fields_markdown_headings() -> None:
    plan = (
        "**INTENT**: draft_cursor_prompt\n\n"
        "**OBJECTIVE**: Harden MoA ticket formatting for Cursor.\n\n"
        "**CONTEXT**:\n"
        "Target files: donna/moa_tool_shim.py\n"
        "Root cause: markdown headings broke parsers\n"
        "Step-by-step changes: 1. normalize headings 2. reject thin plans\n"
        "Acceptance criteria: parse_plan_fields reads INTENT from **INTENT**:\n"
        "ARGS:\nNOTES: none\n"
    )
    fields = parse_plan_fields(plan)
    assert fields["intent"] == "draft_cursor_prompt"
    assert "Harden MoA" in fields["objective"]
    assert "moa_tool_shim.py" in fields["context"]
    assert plan_has_structured_context(plan) is True


def test_unstructured_plan_rejected() -> None:
    thin = (
        "INTENT: draft_cursor_prompt\n"
        "OBJECTIVE: improve cursor handling somehow\n"
        "CONTEXT: just make it better please\n"
    )
    assert plan_has_structured_context(thin) is False


def test_ledger_rejects_intent_echo_only() -> None:
    echo = (
        "**Technical intent:** improve cursor handling in the deepseek pipeline\n"
        "**Target Files:** donna/agentic.py"
    )
    assert _context_is_sufficiently_specific(echo) is False
    result = draft_cursor_prompt(
        objective="improve cursor handling in the deepseek pipeline",
        context=echo,
    )
    assert result.startswith("Error: Draft rejected") or (
        result.startswith("ERROR:") and "Validation Error" in result
    )


def test_ledger_accepts_structured_context(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    ledger = tmp_path / "patch_ledger.md"
    ledger.write_text("# Donna Patch Ledger\n\n", encoding="utf-8")

    def _fake_ledger_path():
        return ledger

    monkeypatch.setattr(
        "donna.tools.general.draft_cursor_prompt._ledger_path",
        _fake_ledger_path,
    )
    # Bypass enrich auto-map so we test the gate + write path directly.
    monkeypatch.setattr(
        "donna.agentic.enrich_draft_cursor_args",
        lambda **kwargs: {
            "objective": kwargs.get("objective") or "",
            "context": kwargs.get("context") or "",
            "target_files": "donna/moa_tool_shim.py",
        },
    )
    ctx = (
        "**Technical intent:** Harden MoA ticket CONTEXT validation.\n"
        "**Target Files:** donna/moa_tool_shim.py, donna/tools/general/draft_cursor_prompt.py\n\n"
        "Root cause: Thin reasoner plans wrote intent-echo tickets Cursor rejected.\n"
        "Step-by-step changes:\n"
        "1. Require Target files / Root cause / Steps / Acceptance in CONTEXT.\n"
        "2. Reject unstructured plans before stage-2 formatting.\n"
        "Acceptance criteria: Malformed fixtures return Draft rejected; structured "
        "fixtures append PENDING tickets with full objectives.\n"
    )
    assert _context_is_sufficiently_specific(ctx) is True
    result = draft_cursor_prompt(
        objective="Harden MoA ticket CONTEXT validation before ledger writes.",
        context=ctx,
    )
    assert "Ticket added to patch_ledger.md" in result
    body = ledger.read_text(encoding="utf-8")
    assert "Harden MoA ticket CONTEXT validation" in body
    assert "Root cause:" in body
