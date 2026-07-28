"""Mode foresight must not starve explicitly named tools in bind lists."""

from __future__ import annotations

from dana.tools.broker import explicit_tool_ids_in_text, merge_bound_tool_ids

_KNOWN = (
    "analyze_visual_context",
    "draft_cursor_prompt",
    "read_local_file",
    "web_search",
)


def test_explicit_keyword_sweep_finds_named_tools() -> None:
    text = (
        "Donna, check my screen but also use the draft_cursor_prompt "
        "tool to write a test."
    )
    found = explicit_tool_ids_in_text(text, _KNOWN)
    assert "draft_cursor_prompt" in found
    assert found.count("draft_cursor_prompt") == 1


def test_vision_mode_merges_explicit_draft_cursor() -> None:
    text = (
        "Donna, check my screen but also use the draft_cursor_prompt "
        "tool to write a test."
    )
    merged = merge_bound_tool_ids(
        user_text=text,
        forced_tool_id="analyze_visual_context",
        mode="vision",
        known_ids=_KNOWN,
    )
    assert "analyze_visual_context" in merged
    assert "draft_cursor_prompt" in merged
    # Deduped even if mode + forced both add vision tool.
    assert merged.count("analyze_visual_context") == 1
    assert merged.count("draft_cursor_prompt") == 1
    print(f"[PASS] tools={merged}")


def test_dedupe_when_mode_and_text_overlap() -> None:
    text = "please call analyze_visual_context on the webcam"
    merged = merge_bound_tool_ids(
        user_text=text,
        forced_tool_id="analyze_visual_context",
        mode="vision",
        known_ids=_KNOWN,
    )
    assert merged == ["analyze_visual_context"]
