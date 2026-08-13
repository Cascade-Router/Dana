"""Unit tests for dana.core.react_dispatch's intent parsing and tool gate."""

from __future__ import annotations

from dana.core import react_dispatch as rd


def test_parse_utterance_box_with_dimensions() -> None:
    call = rd.parse_utterance("build a box 60x40x20")
    assert call is not None
    assert call.tool_id == "create_freecad_box"
    assert call.arguments == {"length": "60", "width": "40", "height": "20"}


def test_parse_utterance_injects_selection_when_referenced() -> None:
    selection = {"centroid": [1.0, 2.0, 3.0], "normal": [0.0, 1.0, 0.0]}
    call = rd.parse_utterance("extrude a box here", selection)
    assert call is not None
    assert call.tool_id == "create_freecad_box"
    assert call.arguments["target_position"] == [1.0, 2.0, 3.0]
    assert call.arguments["target_normal"] == [0.0, 1.0, 0.0]


def test_parse_utterance_no_selection_injection_without_reference_word() -> None:
    selection = {"centroid": [1.0, 2.0, 3.0], "normal": [0.0, 1.0, 0.0]}
    call = rd.parse_utterance("build a cube", selection)
    assert call is not None
    assert "target_position" not in call.arguments


def test_parse_utterance_camera_preset_uses_selection_centroid() -> None:
    selection = {"centroid": [5.0, 6.0, 7.0]}
    call = rd.parse_utterance("look at it from the top", selection)
    assert call is not None
    assert call.tool_id == "manipulate_camera"
    assert call.arguments["target"] == [5.0, 6.0, 7.0]
    assert call.arguments["position"] == list(rd._CAMERA_PRESETS["top"])


def test_parse_utterance_camera_preset_defaults_target_without_selection() -> None:
    call = rd.parse_utterance("orbit to the front view")
    assert call is not None
    assert call.tool_id == "manipulate_camera"
    assert call.arguments["target"] == [0.0, 0.0, 0.0]


def test_parse_utterance_unmatched_returns_none() -> None:
    assert rd.parse_utterance("tell me a joke") is None


def test_is_mutating_tool_classification() -> None:
    assert rd.is_mutating_tool("create_freecad_box") is True
    assert rd.is_mutating_tool("resync_workspace") is True
    assert rd.is_mutating_tool("system_state") is False
    assert rd.is_mutating_tool("execute_vision_analysis") is False
    assert rd.is_mutating_tool("manipulate_camera") is False


def test_describe_tool_call_box() -> None:
    call = rd.parse_utterance("build a box 60x40x20")
    assert call is not None
    description = rd.describe_tool_call(call)
    assert "60" in description and "40" in description and "20" in description


def test_manipulate_camera_tool_handler_requires_vectors() -> None:
    payload = rd._tool_manipulate_camera({"position": [1, 2, 3]}, None, None)
    assert payload["ok"] is False

    payload = rd._tool_manipulate_camera({"position": [1, 2, 3], "target": [0, 0, 0]}, None, None)
    assert payload == {"ok": True, "position": [1.0, 2.0, 3.0], "target": [0.0, 0.0, 0.0]}


def test_dispatch_manipulate_camera_via_registry() -> None:
    call = rd.parse_utterance("orbit to the side view")
    assert call is not None
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None)
    assert result.ok is True
    assert result.payload["position"] == list(rd._CAMERA_PRESETS["side"])
