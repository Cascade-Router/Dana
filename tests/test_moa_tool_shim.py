"""Two-stage MoA tool-binding shim unit checks."""

from __future__ import annotations

from donna.moa_tool_shim import (
    defer_forced_tool_for_moa,
    enrich_forced_tool_from_plan,
    formatter_system_injection,
    parse_plan_fields,
    should_use_moa_tool_shim,
)
from donna.tools.schema import ToolCall


def test_parse_plan_fields() -> None:
    plan = (
        "INTENT: draft_cursor_prompt\n"
        "OBJECTIVE: improve deepseek tool binding with a two-stage MoA shim\n"
        "CONTEXT: Target Files: donna/agentic_react_graph.py, donna/moa_tool_shim.py\n"
        "ARGS:\nNOTES: none\n"
    )
    fields = parse_plan_fields(plan)
    assert fields["intent"] == "draft_cursor_prompt"
    assert "two-stage MoA" in fields["objective"]
    assert "agentic_react_graph.py" in fields["context"]
    print("[PASS] parse_plan_fields")


def test_parse_plan_fields_markdown_bold() -> None:
    plan = (
        "**INTENT**: draft_cursor_prompt\n"
        "**OBJECTIVE**: improve deepseek tool binding\n"
        "**CONTEXT**: Target files: donna/moa_tool_shim.py\n"
    )
    fields = parse_plan_fields(plan)
    assert fields["intent"] == "draft_cursor_prompt"
    assert "improve deepseek" in fields["objective"]
    print("[PASS] parse_plan_fields_markdown_bold")


def test_enrich_draft_cursor_prefers_longer_reasoner_text() -> None:
    thin = ToolCall(
        tool_id="draft_cursor_prompt",
        arguments={"objective": "improve deepseek", "context": ""},
        raw_text="x",
    )
    plan = (
        "INTENT: draft_cursor_prompt\n"
        "OBJECTIVE: improve deepseek's MoA tool-binding shim so R1 plans and "
        "Llama formats draft_cursor_prompt without truncating ticket context\n"
        "CONTEXT: Target Files: donna/moa_tool_shim.py and donna/agentic_react_graph.py. "
        "Do not touch ToolForge gates.\n"
    )
    enriched = enrich_forced_tool_from_plan(thin, plan)
    assert "MoA tool-binding" in str(enriched.arguments.get("objective") or "")
    assert "moa_tool_shim.py" in str(enriched.arguments.get("context") or "")
    print("[PASS] enrich_forced_tool_from_plan")


def test_formatter_injection_contains_plan() -> None:
    inj = formatter_system_injection("INTENT: draft_cursor_prompt\nOBJECTIVE: demo")
    assert "MOA REASONER PLAN" in inj
    assert "draft_cursor_prompt" in inj
    print("[PASS] formatter_system_injection")


def test_defer_and_route_helpers() -> None:
    assert defer_forced_tool_for_moa("draft_cursor_prompt")
    assert not defer_forced_tool_for_moa("web_search")
    # draft_cursor_prompt keyword forces high / MoA when cascade enabled
    assert should_use_moa_tool_shim(
        "Please use draft_cursor_prompt to log a ticket about MoA shim",
        forced_tool_id="draft_cursor_prompt",
    )
    print("[PASS] defer/route helpers")


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
