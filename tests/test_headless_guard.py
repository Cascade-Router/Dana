"""DANA_HEADLESS prunes screen/canvas-capture tools from the LLM's tools=
and short-circuits take_canvas_screenshot to a "skipped" result."""

from __future__ import annotations

import pytest

import dana.core.react_dispatch as rd


def _tool_names(schemas: list[dict]) -> set[str]:
    return {entry["function"]["name"] for entry in schemas}


@pytest.mark.parametrize(
    "active_plugins",
    [None, frozenset(), frozenset({"vision_tools", "os_tools"})],
)
def test_headless_prunes_capture_tools_from_schema(monkeypatch: pytest.MonkeyPatch, active_plugins) -> None:
    monkeypatch.setenv("DANA_HEADLESS", "true")
    names = _tool_names(rd._llm_tools_schema(active_plugins))
    assert not names & rd.HEADLESS_PRUNED_TOOL_IDS
    assert "create_plan" in names  # the rest of the core set is untouched


def test_headless_prunes_even_forced_and_hard_restricted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DANA_HEADLESS", "true")
    forced = rd._llm_tools_schema(frozenset(), force_include=frozenset({"analyze_desktop_screen"}))
    restricted = rd._llm_tools_schema(hard_restrict_to=frozenset({"take_canvas_screenshot", "create_plan"}))
    assert not _tool_names(forced) & rd.HEADLESS_PRUNED_TOOL_IDS
    assert _tool_names(restricted) == {"create_plan"}


def test_interactive_mode_keeps_screenshot_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DANA_HEADLESS", "false")
    assert "take_canvas_screenshot" in _tool_names(rd._llm_tools_schema(frozenset()))
    assert rd.is_visual_inspection_tool("take_canvas_screenshot")


def test_headless_screenshot_fallback_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DANA_HEADLESS", "true")
    # Never routed to the server's frontend-suspend path...
    assert not rd.is_visual_inspection_tool("take_canvas_screenshot")
    # ...and the handler answers immediately instead.
    result = rd.TOOL_HANDLERS["take_canvas_screenshot"]({}, None, None)
    assert result["status"] == "skipped"
    assert result["message"] == "Headless mode active; screenshot bypassed."
