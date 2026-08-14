"""Unit tests for dana.core.react_dispatch's LLM-driven ReAct parsing step
and its (unchanged) tool execution / HITL-gating layer.

``parse_utterance`` now calls a real LLM (via ``ModelProvider.
complete_with_tool_calls``) instead of matching regex, so every test here
mocks that one call site (``rd.ModelProvider``) rather than starting a real
Ollama daemon — these tests are about the ReAct wiring (system prompt
construction, tool-call translation, HITL/DAG-safe fallbacks), not about
LLM quality.

No async test functions/plugin needed: each test drives the coroutine with
a plain ``asyncio.run(...)`` call from an ordinary sync ``def test_...()``.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from dana.core import react_dispatch as rd
from dana.tools.schema import ToolCall


class _FakeProvider:
    """Stands in for ``dana.core.model_provider.ModelProvider``."""

    def __init__(
        self,
        tool_calls: list[ToolCall] | None = None,
        content: str = "",
        raises: Exception | None = None,
    ) -> None:
        self._tool_calls = tool_calls or []
        self._content = content
        self._raises = raises
        self.calls: list[dict[str, Any]] = []

    def complete_with_tool_calls(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        provider: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.calls.append({"messages": messages, "tools": tools, "provider": provider})
        if self._raises is not None:
            raise self._raises
        return {"content": self._content, "tool_calls": self._tool_calls, "provider": "test"}


def _mock_llm(
    monkeypatch: pytest.MonkeyPatch,
    *,
    tool_calls: list[ToolCall] | None = None,
    content: str = "",
    raises: Exception | None = None,
) -> _FakeProvider:
    fake = _FakeProvider(tool_calls=tool_calls, content=content, raises=raises)
    monkeypatch.setattr(rd, "ModelProvider", lambda: fake)
    return fake


def _parse(text: str, active_selection: dict[str, Any] | None = None) -> ToolCall | None:
    return asyncio.run(rd.parse_utterance(text, active_selection))


# --------------------------------------------------------------------------
# build_system_prompt
# --------------------------------------------------------------------------


def test_build_system_prompt_without_selection_omits_selection_text() -> None:
    prompt = rd.build_system_prompt(None)
    assert "canvas selection" not in prompt.lower()


def test_build_system_prompt_includes_active_selection() -> None:
    selection = {"centroid": [1.0, 2.0, 3.0], "normal": [0.0, 1.0, 0.0]}
    prompt = rd.build_system_prompt(selection)
    assert "[1.0, 2.0, 3.0]" in prompt
    assert "[0.0, 1.0, 0.0]" in prompt


# --------------------------------------------------------------------------
# parse_utterance — control flow
# --------------------------------------------------------------------------


def test_parse_utterance_empty_text_short_circuits_without_calling_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom() -> None:
        raise AssertionError("LLM should not be called for empty text")

    monkeypatch.setattr(rd, "ModelProvider", _boom)
    assert _parse("   ") is None


def test_parse_utterance_no_tool_calls_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_llm(monkeypatch, tool_calls=[], content="Sure, happy to chat!")
    assert _parse("thanks!") is None


def test_parse_utterance_llm_exception_returns_none_gracefully(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_llm(monkeypatch, raises=RuntimeError("ollama unreachable"))
    assert _parse("build a box") is None


def test_parse_utterance_unknown_tool_id_from_llm_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_llm(monkeypatch, tool_calls=[ToolCall(tool_id="not_a_real_tool", arguments={})])
    assert _parse("do something weird") is None


def test_parse_utterance_returns_first_proposed_tool_call(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _mock_llm(
        monkeypatch,
        tool_calls=[ToolCall(tool_id="create_freecad_box", arguments={"length": 60, "width": 40, "height": 20})],
    )
    call = _parse("Create a parametric 60x40x20mm box")
    assert call is not None
    assert call.tool_id == "create_freecad_box"
    assert call.arguments == {"length": 60, "width": 40, "height": 20}
    assert call.raw_text == "Create a parametric 60x40x20mm box"
    # The tools handed to the LLM are exactly the wired subset, not the
    # full tools.json registry (which also serves the legacy regex broker).
    tool_names = {t["function"]["name"] for t in fake.calls[0]["tools"]}
    assert tool_names == rd._LLM_TOOL_IDS
    assert fake.calls[0]["provider"] == "ollama"


# --------------------------------------------------------------------------
# parse_utterance — camera preset resolution
# --------------------------------------------------------------------------


def test_parse_utterance_camera_preset_resolves_to_position_target(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_llm(monkeypatch, tool_calls=[ToolCall(tool_id="manipulate_camera", arguments={"preset": "iso"})])
    call = _parse("Show me the isometric view")
    assert call is not None
    assert call.tool_id == "manipulate_camera"
    assert call.arguments["position"] == list(rd._CAMERA_PRESETS["iso"])
    assert call.arguments["target"] == [0.0, 0.0, 0.0]


def test_parse_utterance_camera_preset_uses_active_selection_as_target(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_llm(monkeypatch, tool_calls=[ToolCall(tool_id="manipulate_camera", arguments={"preset": "top"})])
    selection = {"centroid": [5.0, 6.0, 7.0], "normal": [0.0, 0.0, 1.0]}
    call = _parse("orbit to the top view", selection)
    assert call is not None
    assert call.arguments["position"] == list(rd._CAMERA_PRESETS["top"])
    assert call.arguments["target"] == [5.0, 6.0, 7.0]


def test_parse_utterance_camera_preset_invalid_value_defaults_to_iso(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_llm(monkeypatch, tool_calls=[ToolCall(tool_id="manipulate_camera", arguments={"preset": "bottom"})])
    call = _parse("show me the bottom")
    assert call is not None
    assert call.arguments["position"] == list(rd._CAMERA_PRESETS["iso"])


# --------------------------------------------------------------------------
# parse_utterance — selection-injection fallback
# --------------------------------------------------------------------------


def test_parse_utterance_injects_selection_when_llm_omits_it_but_user_said_here(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_llm(monkeypatch, tool_calls=[ToolCall(tool_id="create_freecad_box", arguments={})])
    selection = {"centroid": [1.0, 2.0, 3.0], "normal": [0.0, 1.0, 0.0]}
    call = _parse("add a box here", selection)
    assert call is not None
    assert call.arguments["target_position"] == [1.0, 2.0, 3.0]
    assert call.arguments["target_normal"] == [0.0, 1.0, 0.0]


def test_parse_utterance_no_fallback_injection_without_reference_word(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_llm(monkeypatch, tool_calls=[ToolCall(tool_id="create_freecad_box", arguments={})])
    selection = {"centroid": [1.0, 2.0, 3.0], "normal": [0.0, 1.0, 0.0]}
    call = _parse("build a cube", selection)
    assert call is not None
    assert "target_position" not in call.arguments


def test_parse_utterance_respects_llm_provided_target_over_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_llm(
        monkeypatch,
        tool_calls=[
            ToolCall(
                tool_id="create_freecad_extrusion",
                arguments={"height": 25, "target_position": [9.0, 9.0, 9.0], "target_normal": [0.0, 0.0, 1.0]},
            )
        ],
    )
    selection = {"centroid": [1.0, 2.0, 3.0], "normal": [0.0, 1.0, 0.0]}
    call = _parse("extrude this by 25mm", selection)
    assert call is not None
    # The LLM's own values win — the fallback never overwrites a real answer.
    assert call.arguments["target_position"] == [9.0, 9.0, 9.0]


# --------------------------------------------------------------------------
# next_react_turn / ReactTurn — the multi-step loop's single-iteration
# building block. parse_utterance (tested above) is now a thin single-turn
# wrapper around this; dana.api.server drives it directly, iteration after
# iteration, with an evolving `messages` history.
# --------------------------------------------------------------------------


def test_next_react_turn_returns_final_when_no_tool_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_llm(monkeypatch, content="All done here.")
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    turn = asyncio.run(rd.next_react_turn(messages))
    assert turn.kind == "final"
    assert turn.content == "All done here."


def test_next_react_turn_returns_tool_call_and_finalizes_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_llm(monkeypatch, tool_calls=[ToolCall(tool_id="create_freecad_box", arguments={})])
    selection = {"centroid": [1.0, 2.0, 3.0], "normal": [0.0, 1.0, 0.0]}
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "add a box here"}]
    turn = asyncio.run(rd.next_react_turn(messages, selection, raw_text="add a box here"))
    assert turn.kind == "tool_call"
    assert turn.call.tool_id == "create_freecad_box"
    # _finalize_call_arguments still runs — same selection-injection fallback
    # parse_utterance relies on, now reachable at any loop iteration.
    assert turn.call.arguments["target_position"] == [1.0, 2.0, 3.0]


def test_next_react_turn_uses_explicit_raw_text_not_latest_message(monkeypatch: pytest.MonkeyPatch) -> None:
    """raw_text must be the ORIGINAL user utterance, not whatever the last
    message in a multi-turn history happens to be — a later loop iteration's
    last message is a tool result, which has no bearing on whether the user
    said "here" several tool calls ago."""
    _mock_llm(monkeypatch, tool_calls=[ToolCall(tool_id="create_freecad_box", arguments={})])
    selection = {"centroid": [4.0, 5.0, 6.0], "normal": [0.0, 0.0, 1.0]}
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "add a box here"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "system_state", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": '{"ok": true}'},
    ]
    turn = asyncio.run(rd.next_react_turn(messages, selection, raw_text="add a box here"))
    assert turn.call.arguments["target_position"] == [4.0, 5.0, 6.0]


def test_next_react_turn_unknown_tool_id_yields_final(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_llm(monkeypatch, tool_calls=[ToolCall(tool_id="not_a_real_tool", arguments={})], content="")
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "do something weird"}]
    turn = asyncio.run(rd.next_react_turn(messages))
    assert turn.kind == "final"


def test_next_react_turn_llm_exception_yields_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_llm(monkeypatch, raises=RuntimeError("ollama unreachable"))
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "build a box"}]
    turn = asyncio.run(rd.next_react_turn(messages))
    assert turn.kind == "error"


def test_build_assistant_tool_call_message_shape() -> None:
    call = ToolCall(tool_id="create_freecad_box", arguments={"length": 30, "width": 30, "height": 30})
    message, call_id = rd.build_assistant_tool_call_message(call)
    assert message["role"] == "assistant"
    assert message["tool_calls"][0]["id"] == call_id
    assert message["tool_calls"][0]["function"]["name"] == "create_freecad_box"
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {"length": 30, "width": 30, "height": 30}


def test_build_assistant_tool_call_message_ids_are_unique() -> None:
    call = ToolCall(tool_id="system_state", arguments={})
    _msg1, id1 = rd.build_assistant_tool_call_message(call)
    _msg2, id2 = rd.build_assistant_tool_call_message(call)
    assert id1 != id2


def test_build_tool_result_message_success() -> None:
    result = rd.ToolResult("create_freecad_box", True, {"ok": True, "name": "Box", "path": "x.FCStd"}, "ok", 5)
    message = rd.build_tool_result_message("call_abc", result)
    assert message == {
        "role": "tool",
        "tool_call_id": "call_abc",
        "content": json.dumps({"ok": True, "name": "Box", "path": "x.FCStd"}),
    }


def test_build_tool_result_message_failure_reports_error() -> None:
    result = rd.ToolResult("create_freecad_box", False, {}, "boom", 5)
    message = rd.build_tool_result_message("call_abc", result)
    assert json.loads(message["content"]) == {"ok": False, "error": "boom"}


def test_multi_iteration_message_history_round_trips_through_next_react_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A small end-to-end check of the actual pattern dana.api.server drives:
    build the assistant+tool messages for one tool call, append them, then
    ask next_react_turn again — proving the shapes these helpers produce
    are exactly what next_react_turn's own LLM call site can consume."""
    fake = _mock_llm(
        monkeypatch,
        tool_calls=[ToolCall(tool_id="get_freecad_bounding_box", arguments={"target_object": "Box"})],
    )
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "check the box then stop"}]
    turn1 = asyncio.run(rd.next_react_turn(messages, raw_text="check the box then stop"))
    assert turn1.kind == "tool_call"

    assistant_message, call_id = rd.build_assistant_tool_call_message(turn1.call)
    messages.append(assistant_message)
    result = rd.ToolResult("get_freecad_bounding_box", True, {"ok": True, "z_max": 30.0}, "ok", 5)
    messages.append(rd.build_tool_result_message(call_id, result))

    fake._tool_calls = []
    fake._content = "The box is 30mm tall."
    turn2 = asyncio.run(rd.next_react_turn(messages, raw_text="check the box then stop"))
    assert turn2.kind == "final"
    assert turn2.content == "The box is 30mm tall."
    # The second LLM call actually received the full history, tool result included.
    assert fake.calls[-1]["messages"] == messages


# --------------------------------------------------------------------------
# Tool execution / HITL layer — unchanged by the LLM swap, so ToolCall is
# constructed directly here rather than routed through parse_utterance.
# --------------------------------------------------------------------------


def test_is_mutating_tool_classification() -> None:
    assert rd.is_mutating_tool("create_freecad_box") is True
    assert rd.is_mutating_tool("create_freecad_cylinder") is True
    assert rd.is_mutating_tool("create_freecad_extrusion") is True
    assert rd.is_mutating_tool("create_freecad_pyramid") is True
    assert rd.is_mutating_tool("create_freecad_star_prism") is True
    assert rd.is_mutating_tool("perform_freecad_boolean") is True
    assert rd.is_mutating_tool("perform_freecad_edge_operation") is True
    assert rd.is_mutating_tool("modify_freecad_parameter") is True
    assert rd.is_mutating_tool("create_freecad_pipe") is True
    assert rd.is_mutating_tool("align_freecad_objects") is True
    assert rd.is_mutating_tool("resync_workspace") is True
    assert rd.is_mutating_tool("system_state") is False
    assert rd.is_mutating_tool("execute_vision_analysis") is False
    assert rd.is_mutating_tool("manipulate_camera") is False
    # CRITICAL: read-only/file-IO, must never require HITL approval.
    assert rd.is_mutating_tool("get_freecad_bounding_box") is False
    assert rd.is_mutating_tool("export_freecad_model") is False


def test_describe_tool_call_box() -> None:
    call = ToolCall(tool_id="create_freecad_box", arguments={"length": 60, "width": 40, "height": 20})
    description = rd.describe_tool_call(call)
    assert "60" in description and "40" in description and "20" in description


def test_manipulate_camera_tool_handler_requires_vectors() -> None:
    payload = rd._tool_manipulate_camera({"position": [1, 2, 3]}, None, None)
    assert payload["ok"] is False

    payload = rd._tool_manipulate_camera({"position": [1, 2, 3], "target": [0, 0, 0]}, None, None)
    assert payload == {"ok": True, "position": [1.0, 2.0, 3.0], "target": [0.0, 0.0, 0.0]}


def test_dispatch_manipulate_camera_via_registry() -> None:
    call = ToolCall(tool_id="manipulate_camera", arguments={"position": [200, 0, 0], "target": [0, 0, 0]})
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None)
    assert result.ok is True
    assert result.payload["position"] == [200.0, 0.0, 0.0]


def test_dispatch_extrusion_without_profile_or_selection_fails_cleanly() -> None:
    call = ToolCall(tool_id="create_freecad_extrusion", arguments={"height": 25})
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None)
    assert result.ok is False
    assert "profile points" in result.message or "selected face" in result.message


def test_dispatch_extrusion_rejects_non_z_normal() -> None:
    call = ToolCall(
        tool_id="create_freecad_extrusion",
        arguments={"height": 25, "target_position": [0, 0, 50], "target_normal": [1, 0, 0]},
    )
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None)
    assert result.ok is False
    assert "Z axis" in result.message


def test_dispatch_extrusion_with_selection_builds_default_footprint() -> None:
    from dana.platform.mock import MockControlPlane, MockFreeCADEngine

    call = ToolCall(
        tool_id="create_freecad_extrusion",
        arguments={"height": 25, "target_position": [0, 0, 50], "target_normal": [0, 0, 1]},
    )
    result = rd.dispatch_tool_call(call, MockFreeCADEngine(), MockControlPlane())
    assert result.ok is True
    assert result.payload["dimensions"]["height"] == 25.0
    assert result.payload["dimensions"]["profile_points"] == 4


# --------------------------------------------------------------------------
# Sharp-edged primitives: pyramid + star prism
# --------------------------------------------------------------------------


def test_describe_tool_call_pyramid() -> None:
    call = ToolCall(tool_id="create_freecad_pyramid", arguments={"length": 50, "width": 50, "height": 75})
    description = rd.describe_tool_call(call)
    assert "50" in description and "75" in description


def test_describe_tool_call_star_prism() -> None:
    call = ToolCall(
        tool_id="create_freecad_star_prism",
        arguments={"points": 8, "outer_radius": 60, "inner_radius": 20, "height": 5},
    )
    description = rd.describe_tool_call(call)
    assert "8" in description and "60" in description and "20" in description and "5" in description


def test_dispatch_pyramid_via_mock_engine() -> None:
    from dana.platform.mock import MockControlPlane, MockFreeCADEngine

    call = ToolCall(tool_id="create_freecad_pyramid", arguments={"length": 50, "width": 50, "height": 75})
    result = rd.dispatch_tool_call(call, MockFreeCADEngine(), MockControlPlane())
    assert result.ok is True
    assert result.payload["dimensions"] == {"length": 50.0, "width": 50.0, "height": 75.0}
    assert result.payload["bounding_box"] == [-25.0, -25.0, 0.0, 25.0, 25.0, 75.0]


def test_dispatch_star_prism_via_mock_engine() -> None:
    from dana.platform.mock import MockControlPlane, MockFreeCADEngine

    call = ToolCall(
        tool_id="create_freecad_star_prism",
        arguments={"points": 8, "outer_radius": 60, "inner_radius": 20, "height": 5},
    )
    result = rd.dispatch_tool_call(call, MockFreeCADEngine(), MockControlPlane())
    assert result.ok is True
    assert result.payload["dimensions"] == {
        "points": 8,
        "outer_radius": 60.0,
        "inner_radius": 20.0,
        "height": 5.0,
    }
    assert result.payload["bounding_box"] == [-60.0, -60.0, 0.0, 60.0, 60.0, 5.0]


def test_dispatch_star_prism_rejects_too_few_points() -> None:
    call = ToolCall(tool_id="create_freecad_star_prism", arguments={"points": 2, "outer_radius": 60, "inner_radius": 20})
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None)
    assert result.ok is False
    assert "at least 3 points" in result.message


def test_extract_placement_defaults_to_origin() -> None:
    assert rd._extract_placement({}) == (0.0, 0.0, 0.0)


def test_extract_placement_reads_xyz_args() -> None:
    assert rd._extract_placement({"placement_x": 1, "placement_y": 2, "placement_z": 3}) == (1.0, 2.0, 3.0)


def test_dispatch_box_with_placement_passes_through_to_engine() -> None:
    from dana.platform.mock import MockControlPlane, MockFreeCADEngine

    call = ToolCall(
        tool_id="create_freecad_box",
        arguments={
            "length": 20,
            "width": 20,
            "height": 20,
            "placement_x": 0,
            "placement_y": 0,
            "placement_z": 25,
        },
    )
    result = rd.dispatch_tool_call(call, MockFreeCADEngine(), MockControlPlane())
    assert result.ok is True
    assert result.payload["placement"] == [0.0, 0.0, 25.0]


def test_dispatch_cylinder_without_placement_defaults_to_origin() -> None:
    from dana.platform.mock import MockControlPlane, MockFreeCADEngine

    call = ToolCall(tool_id="create_freecad_cylinder", arguments={"radius": 10, "height": 30})
    result = rd.dispatch_tool_call(call, MockFreeCADEngine(), MockControlPlane())
    assert result.ok is True
    assert result.payload["placement"] == [0.0, 0.0, 0.0]


def test_dispatch_pyramid_and_star_prism_with_placement() -> None:
    from dana.platform.mock import MockControlPlane, MockFreeCADEngine

    pyramid_call = ToolCall(
        tool_id="create_freecad_pyramid",
        arguments={"length": 50, "width": 50, "height": 75, "placement_x": 10, "placement_y": -5},
    )
    result = rd.dispatch_tool_call(pyramid_call, MockFreeCADEngine(), MockControlPlane())
    assert result.ok is True
    assert result.payload["placement"] == [10.0, -5.0, 0.0]

    star_call = ToolCall(
        tool_id="create_freecad_star_prism",
        arguments={"points": 8, "outer_radius": 60, "inner_radius": 20, "height": 5, "placement_z": 12},
    )
    result = rd.dispatch_tool_call(star_call, MockFreeCADEngine(), MockControlPlane())
    assert result.ok is True
    assert result.payload["placement"] == [0.0, 0.0, 12.0]


def test_parse_utterance_pyramid_and_star_prism_pass_through(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_llm(
        monkeypatch,
        tool_calls=[
            ToolCall(tool_id="create_freecad_pyramid", arguments={"length": 50, "width": 50, "height": 75})
        ],
    )
    call = _parse("Build me a pyramid with a 50x50 base and a height of 75.")
    assert call is not None
    assert call.tool_id == "create_freecad_pyramid"

    _mock_llm(
        monkeypatch,
        tool_calls=[
            ToolCall(
                tool_id="create_freecad_star_prism",
                arguments={"points": 8, "outer_radius": 60, "inner_radius": 20, "height": 5},
            )
        ],
    )
    call = _parse("Create a sharp-edged ninja star with 8 points, outer radius 60mm, inner radius 20mm, thickness 5mm.")
    assert call is not None
    assert call.tool_id == "create_freecad_star_prism"


# --------------------------------------------------------------------------
# Boolean CSG operations: perform_freecad_boolean
# --------------------------------------------------------------------------


def test_describe_tool_call_boolean_cut() -> None:
    call = ToolCall(
        tool_id="perform_freecad_boolean",
        arguments={"operation": "cut", "base_object": "Box", "tool_object": "Cylinder"},
    )
    description = rd.describe_tool_call(call)
    assert "Cylinder" in description and "Box" in description


def test_dispatch_boolean_rejects_unknown_operation() -> None:
    call = ToolCall(
        tool_id="perform_freecad_boolean",
        arguments={"operation": "bogus", "base_object": "Box", "tool_object": "Cylinder"},
    )
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None)
    assert result.ok is False
    assert "cut, union, intersect" in result.message


def test_dispatch_boolean_requires_base_and_tool_object() -> None:
    call = ToolCall(tool_id="perform_freecad_boolean", arguments={"operation": "cut"})
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None)
    assert result.ok is False
    assert "base_object" in result.message


def test_dispatch_boolean_rejects_unknown_object_names() -> None:
    call = ToolCall(
        tool_id="perform_freecad_boolean",
        arguments={"operation": "cut", "base_object": "NeverCreated1", "tool_object": "NeverCreated2"},
    )
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None)
    assert result.ok is False
    assert "NeverCreated1" in result.message


def test_dispatch_boolean_end_to_end_via_object_name_registry() -> None:
    """create_freecad_box/cylinder register their (name -> path) in
    rd._OBJECT_PATH_REGISTRY as a side effect of dispatch_tool_call; a later
    perform_freecad_boolean call resolves base_object/tool_object against
    that registry instead of needing a persistent FreeCAD ActiveDocument."""
    from dana.platform.mock import MockControlPlane, MockFreeCADEngine

    engine = MockFreeCADEngine()
    control_plane = MockControlPlane()

    box_result = rd.dispatch_tool_call(
        ToolCall(tool_id="create_freecad_box", arguments={"length": 50, "width": 50, "height": 50, "name": "CsgBox"}),
        engine,
        control_plane,
    )
    assert box_result.ok is True

    cyl_result = rd.dispatch_tool_call(
        ToolCall(
            tool_id="create_freecad_cylinder",
            arguments={
                "radius": 15,
                "height": 50,
                "name": "CsgCylinder",
                "placement_x": 25,
                "placement_y": 25,
            },
        ),
        engine,
        control_plane,
    )
    assert cyl_result.ok is True

    cut_result = rd.dispatch_tool_call(
        ToolCall(
            tool_id="perform_freecad_boolean",
            arguments={"operation": "cut", "base_object": "CsgBox", "tool_object": "CsgCylinder"},
        ),
        engine,
        control_plane,
    )
    assert cut_result.ok is True
    assert cut_result.payload["type"] == "Part::Cut"
    assert cut_result.payload["name"] == "Cut"

    # The boolean result itself registers too, so it can chain into a
    # further boolean op as someone else's base_object/tool_object.
    assert rd._OBJECT_PATH_REGISTRY["Cut"] == cut_result.payload["path"]


def test_dispatch_boolean_union_and_intersect_use_default_names() -> None:
    from dana.platform.mock import MockControlPlane, MockFreeCADEngine

    engine = MockFreeCADEngine()
    control_plane = MockControlPlane()
    rd.dispatch_tool_call(
        ToolCall(tool_id="create_freecad_box", arguments={"length": 20, "width": 20, "height": 20, "name": "UBoxA"}),
        engine,
        control_plane,
    )
    rd.dispatch_tool_call(
        ToolCall(tool_id="create_freecad_cylinder", arguments={"radius": 5, "height": 20, "name": "UCylA"}),
        engine,
        control_plane,
    )

    union_result = rd.dispatch_tool_call(
        ToolCall(
            tool_id="perform_freecad_boolean",
            arguments={"operation": "union", "base_object": "UBoxA", "tool_object": "UCylA"},
        ),
        engine,
        control_plane,
    )
    assert union_result.ok is True
    assert union_result.payload["name"] == "Fusion"
    assert union_result.payload["type"] == "Part::MultiFuse"

    intersect_result = rd.dispatch_tool_call(
        ToolCall(
            tool_id="perform_freecad_boolean",
            arguments={"operation": "intersect", "base_object": "UBoxA", "tool_object": "UCylA"},
        ),
        engine,
        control_plane,
    )
    assert intersect_result.ok is True
    assert intersect_result.payload["name"] == "Common"
    assert intersect_result.payload["type"] == "Part::MultiCommon"


def test_parse_utterance_boolean_pass_through(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_llm(
        monkeypatch,
        tool_calls=[
            ToolCall(
                tool_id="perform_freecad_boolean",
                arguments={"operation": "cut", "base_object": "Box", "tool_object": "Cylinder"},
            )
        ],
    )
    call = _parse("Drill a hole through the box using the cylinder.")
    assert call is not None
    assert call.tool_id == "perform_freecad_boolean"
    assert call.arguments["operation"] == "cut"


# --------------------------------------------------------------------------
# Edge manipulation: perform_freecad_edge_operation
# --------------------------------------------------------------------------


def test_describe_tool_call_edge_operation_whole_object() -> None:
    call = ToolCall(
        tool_id="perform_freecad_edge_operation",
        arguments={"operation": "fillet", "target_object": "Box", "value": 5},
    )
    description = rd.describe_tool_call(call)
    assert "Fillet" in description and "Box" in description and "every edge" in description


def test_describe_tool_call_edge_operation_face_targeted() -> None:
    call = ToolCall(
        tool_id="perform_freecad_edge_operation",
        arguments={"operation": "chamfer", "target_object": "Box", "value": 3, "face_centroid": [25, 25, 50]},
    )
    description = rd.describe_tool_call(call)
    assert "Chamfer" in description and "selected face" in description


def test_dispatch_edge_operation_rejects_unknown_operation() -> None:
    call = ToolCall(
        tool_id="perform_freecad_edge_operation",
        arguments={"operation": "bogus", "target_object": "Box", "value": 5},
    )
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None)
    assert result.ok is False
    assert "fillet, chamfer" in result.message


def test_dispatch_edge_operation_requires_target_object() -> None:
    call = ToolCall(tool_id="perform_freecad_edge_operation", arguments={"operation": "fillet", "value": 5})
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None)
    assert result.ok is False
    assert "target_object" in result.message


def test_dispatch_edge_operation_requires_numeric_value() -> None:
    call = ToolCall(
        tool_id="perform_freecad_edge_operation",
        arguments={"operation": "fillet", "target_object": "Box", "value": "not-a-number"},
    )
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None)
    assert result.ok is False
    assert "numeric value" in result.message


def test_dispatch_edge_operation_rejects_unknown_object_name() -> None:
    call = ToolCall(
        tool_id="perform_freecad_edge_operation",
        arguments={"operation": "fillet", "target_object": "NeverCreated", "value": 5},
    )
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None)
    assert result.ok is False
    assert "NeverCreated" in result.message


def test_dispatch_edge_operation_whole_object_via_registry() -> None:
    from dana.platform.mock import MockControlPlane, MockFreeCADEngine

    engine = MockFreeCADEngine()
    control_plane = MockControlPlane()
    rd.dispatch_tool_call(
        ToolCall(tool_id="create_freecad_box", arguments={"length": 20, "width": 20, "height": 20, "name": "EdgeBoxA"}),
        engine,
        control_plane,
    )

    result = rd.dispatch_tool_call(
        ToolCall(
            tool_id="perform_freecad_edge_operation",
            arguments={"operation": "fillet", "target_object": "EdgeBoxA", "value": 3},
        ),
        engine,
        control_plane,
    )
    assert result.ok is True
    assert result.payload["type"] == "Part::Fillet"
    assert result.payload["face_targeted"] is False
    # The edge-op result itself registers too, so it can chain further.
    assert rd._OBJECT_PATH_REGISTRY["Fillet"] == result.payload["path"]


def test_dispatch_edge_operation_face_targeted_via_explicit_centroid() -> None:
    from dana.platform.mock import MockControlPlane, MockFreeCADEngine

    engine = MockFreeCADEngine()
    control_plane = MockControlPlane()
    rd.dispatch_tool_call(
        ToolCall(tool_id="create_freecad_box", arguments={"length": 20, "width": 20, "height": 20, "name": "EdgeBoxB"}),
        engine,
        control_plane,
    )

    result = rd.dispatch_tool_call(
        ToolCall(
            tool_id="perform_freecad_edge_operation",
            arguments={
                "operation": "chamfer",
                "target_object": "EdgeBoxB",
                "value": 2,
                "face_centroid": [10, 10, 20],
            },
        ),
        engine,
        control_plane,
    )
    assert result.ok is True
    assert result.payload["type"] == "Part::Chamfer"
    assert result.payload["face_targeted"] is True


def test_finalize_call_arguments_injects_face_centroid_for_edge_operation() -> None:
    call = ToolCall(
        tool_id="perform_freecad_edge_operation",
        arguments={"operation": "fillet", "target_object": "Box", "value": 5},
        raw_text="Fillet the edges of this face by 5mm.",
    )
    selection = {"centroid": [25.0, 25.0, 50.0], "normal": [0.0, 0.0, 1.0]}
    rd._finalize_call_arguments(call, selection)
    assert call.arguments["face_centroid"] == [25.0, 25.0, 50.0]


def test_finalize_call_arguments_edge_operation_no_selection_leaves_whole_object() -> None:
    call = ToolCall(
        tool_id="perform_freecad_edge_operation",
        arguments={"operation": "fillet", "target_object": "Box", "value": 5},
    )
    rd._finalize_call_arguments(call, None)
    assert "face_centroid" not in call.arguments


def test_finalize_call_arguments_edge_operation_respects_llm_provided_centroid() -> None:
    call = ToolCall(
        tool_id="perform_freecad_edge_operation",
        arguments={
            "operation": "fillet",
            "target_object": "Box",
            "value": 5,
            "face_centroid": [1.0, 2.0, 3.0],
        },
    )
    selection = {"centroid": [9.0, 9.0, 9.0], "normal": [0.0, 0.0, 1.0]}
    rd._finalize_call_arguments(call, selection)
    assert call.arguments["face_centroid"] == [1.0, 2.0, 3.0]


def test_parse_utterance_edge_operation_pass_through_with_active_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_llm(
        monkeypatch,
        tool_calls=[
            ToolCall(
                tool_id="perform_freecad_edge_operation",
                arguments={"operation": "fillet", "target_object": "Box", "value": 5},
            )
        ],
    )
    selection = {"centroid": [25.0, 25.0, 50.0], "normal": [0.0, 0.0, 1.0]}
    call = _parse("Fillet the edges of this face by 5mm.", selection)
    assert call is not None
    assert call.tool_id == "perform_freecad_edge_operation"
    assert call.arguments["face_centroid"] == [25.0, 25.0, 50.0]


# --------------------------------------------------------------------------
# Parametric modification: modify_freecad_parameter
# --------------------------------------------------------------------------


def test_describe_tool_call_modify_parameter() -> None:
    call = ToolCall(
        tool_id="modify_freecad_parameter",
        arguments={"target_object": "Box", "parameter_name": "Height", "new_value": 100},
    )
    description = rd.describe_tool_call(call)
    assert "Box" in description and "Height" in description and "100" in description


def test_dispatch_modify_parameter_requires_target_object() -> None:
    call = ToolCall(
        tool_id="modify_freecad_parameter",
        arguments={"parameter_name": "Height", "new_value": 100},
    )
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None)
    assert result.ok is False
    assert "target_object" in result.message


def test_dispatch_modify_parameter_rejects_unknown_object_name() -> None:
    call = ToolCall(
        tool_id="modify_freecad_parameter",
        arguments={"target_object": "NeverCreated", "parameter_name": "Height", "new_value": 100},
    )
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None)
    assert result.ok is False
    assert "NeverCreated" in result.message


def test_dispatch_modify_parameter_requires_numeric_new_value() -> None:
    call = ToolCall(
        tool_id="modify_freecad_parameter",
        arguments={"target_object": "Box", "parameter_name": "Height", "new_value": "not-a-number"},
    )
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None)
    assert result.ok is False
    assert "numeric new_value" in result.message


def test_dispatch_modify_parameter_end_to_end_via_registry() -> None:
    from dana.platform.mock import MockControlPlane, MockFreeCADEngine

    engine = MockFreeCADEngine()
    control_plane = MockControlPlane()
    rd.dispatch_tool_call(
        ToolCall(tool_id="create_freecad_box", arguments={"length": 20, "width": 20, "height": 20, "name": "ModBoxA"}),
        engine,
        control_plane,
    )

    result = rd.dispatch_tool_call(
        ToolCall(
            tool_id="modify_freecad_parameter",
            arguments={"target_object": "ModBoxA", "parameter_name": "Height", "new_value": 99},
        ),
        engine,
        control_plane,
    )
    assert result.ok is True
    assert result.payload["parameter_name"] == "Height"
    assert result.payload["new_value"] == 99.0


# --------------------------------------------------------------------------
# Non-mutating spatial query: get_freecad_bounding_box
# --------------------------------------------------------------------------


def test_describe_tool_call_get_bounding_box() -> None:
    call = ToolCall(tool_id="get_freecad_bounding_box", arguments={"target_object": "Box"})
    description = rd.describe_tool_call(call)
    assert "Box" in description


def test_dispatch_get_bounding_box_requires_target_object() -> None:
    call = ToolCall(tool_id="get_freecad_bounding_box", arguments={})
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None)
    assert result.ok is False
    assert "target_object" in result.message


def test_dispatch_get_bounding_box_rejects_unknown_object_name() -> None:
    call = ToolCall(tool_id="get_freecad_bounding_box", arguments={"target_object": "NeverCreated"})
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None)
    assert result.ok is False
    assert "NeverCreated" in result.message


def test_dispatch_get_bounding_box_end_to_end_via_registry() -> None:
    from dana.platform.mock import MockControlPlane, MockFreeCADEngine

    engine = MockFreeCADEngine()
    control_plane = MockControlPlane()
    rd.dispatch_tool_call(
        ToolCall(tool_id="create_freecad_box", arguments={"length": 30, "width": 30, "height": 30, "name": "BBoxBoxA"}),
        engine,
        control_plane,
    )

    result = rd.dispatch_tool_call(
        ToolCall(tool_id="get_freecad_bounding_box", arguments={"target_object": "BBoxBoxA"}),
        engine,
        control_plane,
    )
    assert result.ok is True
    for key in ("x_min", "y_min", "z_min", "x_max", "y_max", "z_max"):
        assert key in result.payload


def test_get_bounding_box_never_registers_a_new_object() -> None:
    """A read shouldn't mutate the object registry — get_bounding_box's
    payload has no "name" key at all, so dispatch_tool_call's generic
    post-success registration (keyed on payload["name"]) is a no-op here."""
    from dana.platform.mock import MockControlPlane, MockFreeCADEngine

    engine = MockFreeCADEngine()
    control_plane = MockControlPlane()
    rd.dispatch_tool_call(
        ToolCall(tool_id="create_freecad_box", arguments={"length": 10, "width": 10, "height": 10, "name": "BBoxBoxB"}),
        engine,
        control_plane,
    )
    before = dict(rd._OBJECT_PATH_REGISTRY)
    rd.dispatch_tool_call(
        ToolCall(tool_id="get_freecad_bounding_box", arguments={"target_object": "BBoxBoxB"}),
        engine,
        control_plane,
    )
    assert rd._OBJECT_PATH_REGISTRY == before


# --------------------------------------------------------------------------
# 2D-to-3D sweeps: create_freecad_pipe
# --------------------------------------------------------------------------


def test_describe_tool_call_pipe_straight() -> None:
    call = ToolCall(
        tool_id="create_freecad_pipe",
        arguments={"pipe_radius": 8, "path_type": "straight", "length_or_angle": 60},
    )
    description = rd.describe_tool_call(call)
    assert "straight" in description.lower() and "8" in description and "60" in description


def test_describe_tool_call_pipe_arc() -> None:
    call = ToolCall(
        tool_id="create_freecad_pipe",
        arguments={"pipe_radius": 10, "path_type": "arc", "length_or_angle": 90},
    )
    description = rd.describe_tool_call(call)
    assert "curved" in description.lower() and "90" in description


def test_dispatch_pipe_rejects_unknown_path_type() -> None:
    call = ToolCall(
        tool_id="create_freecad_pipe",
        arguments={"pipe_radius": 10, "path_type": "bogus", "length_or_angle": 90},
    )
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None)
    assert result.ok is False
    assert "straight, arc" in result.message


def test_dispatch_pipe_requires_numeric_fields() -> None:
    call = ToolCall(
        tool_id="create_freecad_pipe",
        arguments={"pipe_radius": "nope", "path_type": "straight", "length_or_angle": 60},
    )
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None)
    assert result.ok is False
    assert "numeric" in result.message


def test_dispatch_pipe_straight_via_mock_engine() -> None:
    from dana.platform.mock import MockControlPlane, MockFreeCADEngine

    call = ToolCall(
        tool_id="create_freecad_pipe",
        arguments={"pipe_radius": 5, "path_type": "straight", "length_or_angle": 40, "name": "PipeA"},
    )
    result = rd.dispatch_tool_call(call, MockFreeCADEngine(), MockControlPlane())
    assert result.ok is True
    assert result.payload["type"] == "Part::Sweep"
    assert result.payload["dimensions"]["path_type"] == "straight"


def test_dispatch_pipe_arc_with_placement_via_mock_engine() -> None:
    from dana.platform.mock import MockControlPlane, MockFreeCADEngine

    call = ToolCall(
        tool_id="create_freecad_pipe",
        arguments={
            "pipe_radius": 10,
            "path_type": "arc",
            "length_or_angle": 90,
            "name": "PipeB",
            "placement_x": 5,
        },
    )
    result = rd.dispatch_tool_call(call, MockFreeCADEngine(), MockControlPlane())
    assert result.ok is True
    assert result.payload["placement"] == [5.0, 0.0, 0.0]
    assert result.payload["dimensions"]["path_type"] == "arc"


def test_parse_utterance_pipe_pass_through(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_llm(
        monkeypatch,
        tool_calls=[
            ToolCall(
                tool_id="create_freecad_pipe",
                arguments={"pipe_radius": 10, "path_type": "arc", "length_or_angle": 90},
            )
        ],
    )
    call = _parse("Create a curved pipe with a 10mm radius that bends at a 90-degree angle.")
    assert call is not None
    assert call.tool_id == "create_freecad_pipe"
    assert call.arguments["path_type"] == "arc"
    assert call.arguments["length_or_angle"] == 90


# --------------------------------------------------------------------------
# Assembly alignment: align_freecad_objects
# --------------------------------------------------------------------------


def test_describe_tool_call_align() -> None:
    call = ToolCall(
        tool_id="align_freecad_objects",
        arguments={"source_object": "Cylinder", "target_object": "Box", "alignment_type": "top_center"},
    )
    description = rd.describe_tool_call(call)
    assert "Cylinder" in description and "Box" in description and "top_center" in description


def test_dispatch_align_rejects_unknown_alignment_type() -> None:
    call = ToolCall(
        tool_id="align_freecad_objects",
        arguments={"source_object": "Cylinder", "target_object": "Box", "alignment_type": "sideways"},
    )
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None)
    assert result.ok is False
    assert "top_center" in result.message


def test_dispatch_align_requires_source_and_target_object() -> None:
    call = ToolCall(tool_id="align_freecad_objects", arguments={"alignment_type": "top_center"})
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None)
    assert result.ok is False
    assert "source_object" in result.message


def test_dispatch_align_rejects_unknown_object_names() -> None:
    call = ToolCall(
        tool_id="align_freecad_objects",
        arguments={"source_object": "NeverCreated1", "target_object": "NeverCreated2", "alignment_type": "top_center"},
    )
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None)
    assert result.ok is False
    assert "NeverCreated1" in result.message


def test_dispatch_align_end_to_end_via_registry() -> None:
    from dana.platform.mock import MockControlPlane, MockFreeCADEngine

    engine = MockFreeCADEngine()
    control_plane = MockControlPlane()
    rd.dispatch_tool_call(
        ToolCall(tool_id="create_freecad_box", arguments={"length": 60, "width": 60, "height": 5, "name": "AlignBaseA"}),
        engine,
        control_plane,
    )
    rd.dispatch_tool_call(
        ToolCall(tool_id="create_freecad_cylinder", arguments={"radius": 20, "height": 40, "name": "AlignCylA"}),
        engine,
        control_plane,
    )

    result = rd.dispatch_tool_call(
        ToolCall(
            tool_id="align_freecad_objects",
            arguments={"source_object": "AlignCylA", "target_object": "AlignBaseA", "alignment_type": "top_center"},
        ),
        engine,
        control_plane,
    )
    assert result.ok is True
    assert result.payload["alignment_type"] == "top_center"
    assert len(result.payload["placement"]) == 3
    # The aligned object re-registers under the same name -> same path
    # (it moved in place, it didn't get a new identity or file).
    assert rd._OBJECT_PATH_REGISTRY["AlignCylA"] == result.payload["path"]


# --------------------------------------------------------------------------
# Export pipelines: export_freecad_model
# --------------------------------------------------------------------------


def test_describe_tool_call_export() -> None:
    call = ToolCall(
        tool_id="export_freecad_model",
        arguments={"format": "stl", "target_objects": ["Box", "Cylinder"], "filename": "assembly"},
    )
    description = rd.describe_tool_call(call)
    assert "Box" in description and "Cylinder" in description and "STL" in description and "assembly" in description


def test_dispatch_export_rejects_unknown_format() -> None:
    call = ToolCall(
        tool_id="export_freecad_model",
        arguments={"format": "obj", "target_objects": ["Box"], "filename": "x"},
    )
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None)
    assert result.ok is False
    assert "stl, step" in result.message


def test_dispatch_export_requires_non_empty_target_objects() -> None:
    call = ToolCall(tool_id="export_freecad_model", arguments={"format": "stl", "target_objects": [], "filename": "x"})
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None)
    assert result.ok is False
    assert "target_objects" in result.message


def test_dispatch_export_requires_filename() -> None:
    call = ToolCall(tool_id="export_freecad_model", arguments={"format": "stl", "target_objects": ["Box"]})
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None)
    assert result.ok is False
    assert "filename" in result.message


def test_dispatch_export_rejects_unknown_object_name() -> None:
    call = ToolCall(
        tool_id="export_freecad_model",
        arguments={"format": "stl", "target_objects": ["NeverCreated"], "filename": "x"},
    )
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None)
    assert result.ok is False
    assert "NeverCreated" in result.message


def test_dispatch_export_end_to_end_via_registry() -> None:
    from dana.platform.mock import MockControlPlane, MockFreeCADEngine

    engine = MockFreeCADEngine()
    control_plane = MockControlPlane()
    rd.dispatch_tool_call(
        ToolCall(tool_id="create_freecad_box", arguments={"length": 60, "width": 60, "height": 5, "name": "ExportBaseA"}),
        engine,
        control_plane,
    )
    rd.dispatch_tool_call(
        ToolCall(tool_id="create_freecad_cylinder", arguments={"radius": 20, "height": 40, "name": "ExportCylA"}),
        engine,
        control_plane,
    )

    result = rd.dispatch_tool_call(
        ToolCall(
            tool_id="export_freecad_model",
            arguments={"format": "stl", "target_objects": ["ExportBaseA", "ExportCylA"], "filename": "motor_mount_assembly"},
        ),
        engine,
        control_plane,
    )
    assert result.ok is True
    assert result.payload["target_count"] == 2
    # An export result has no "name" of its own — it must NOT register as
    # a fresh object in the name->path registry.
    assert "motor_mount_assembly" not in rd._OBJECT_PATH_REGISTRY


def test_dispatch_export_step_reports_mock_limitation_not_a_crash() -> None:
    """The mock engine can't produce real STEP (B-rep) data via trimesh —
    this must surface as a clean ok:False, not an exception."""
    from dana.platform.mock import MockControlPlane, MockFreeCADEngine

    engine = MockFreeCADEngine()
    control_plane = MockControlPlane()
    rd.dispatch_tool_call(
        ToolCall(tool_id="create_freecad_box", arguments={"length": 10, "width": 10, "height": 10, "name": "StepBoxA"}),
        engine,
        control_plane,
    )
    result = rd.dispatch_tool_call(
        ToolCall(
            tool_id="export_freecad_model",
            arguments={"format": "step", "target_objects": ["StepBoxA"], "filename": "x"},
        ),
        engine,
        control_plane,
    )
    assert result.ok is False
    assert "STEP" in result.message


def test_parse_utterance_align_and_export_pass_through(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_llm(
        monkeypatch,
        tool_calls=[
            ToolCall(
                tool_id="align_freecad_objects",
                arguments={"source_object": "Cylinder", "target_object": "Box", "alignment_type": "top_center"},
            )
        ],
    )
    call = _parse("Snap the cylinder to the top center of the box.")
    assert call is not None
    assert call.tool_id == "align_freecad_objects"

    _mock_llm(
        monkeypatch,
        tool_calls=[
            ToolCall(
                tool_id="export_freecad_model",
                arguments={"format": "stl", "target_objects": ["Box", "Cylinder"], "filename": "motor_mount_assembly"},
            )
        ],
    )
    call = _parse("Export both objects as an STL file named motor_mount_assembly.")
    assert call is not None
    assert call.tool_id == "export_freecad_model"


# --------------------------------------------------------------------------
# Phase A: schema registry unification — the LLM tool subset must actually
# resolve against tools.json (dana/tools/tools.json), and must line up
# exactly with the dispatch-side TOOL_HANDLERS/MUTATING_TOOLS sets.
# --------------------------------------------------------------------------


def test_llm_tool_ids_all_have_handlers() -> None:
    for tool_id in rd._LLM_TOOL_IDS:
        assert tool_id in rd.TOOL_HANDLERS, tool_id


def test_llm_tools_schema_resolves_against_tools_json() -> None:
    schema = rd._llm_tools_schema()
    names = {t["function"]["name"] for t in schema}
    assert names == rd._LLM_TOOL_IDS


def test_llm_tools_schema_has_no_duplicate_or_missing_parameter_names() -> None:
    for entry in rd._llm_tools_schema():
        fn = entry["function"]
        properties = fn["parameters"]["properties"]
        for required_name in fn["parameters"]["required"]:
            assert required_name in properties, f"{fn['name']}: required {required_name!r} not in properties"
