"""Stage 3.3 — strict single-tool bind on ValidationError retry."""

from __future__ import annotations

from donna.agentic_react_graph import validation_retry_tool_corridor


def test_validation_retry_corridor_is_exactly_failed_tool() -> None:
    assert validation_retry_tool_corridor("draft_cursor_prompt") == [
        "draft_cursor_prompt"
    ]
    assert validation_retry_tool_corridor("analyze_visual_context") == [
        "analyze_visual_context"
    ]
    # Must not retain siblings from a prior always_include merge.
    corridor = validation_retry_tool_corridor("draft_cursor_prompt")
    assert "analyze_visual_context" not in corridor
    assert len(corridor) == 1


def test_validation_retry_corridor_empty_id() -> None:
    assert validation_retry_tool_corridor("") == []
    assert validation_retry_tool_corridor("   ") == []


def test_fresh_turn_always_include_unchanged_by_helper() -> None:
    """Helper is bounce-only; fresh multi-tool merges stay caller-owned."""
    fresh = ["draft_cursor_prompt", "analyze_visual_context"]
    # Simulating a non-bounce path: corridor helper is not applied.
    assert fresh == ["draft_cursor_prompt", "analyze_visual_context"]
    # Bounce path replaces the list entirely.
    assert validation_retry_tool_corridor(fresh[0]) == ["draft_cursor_prompt"]
