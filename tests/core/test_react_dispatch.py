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


def test_parse_utterance_box_dims_prefix_order_with_units() -> None:
    call = rd.parse_utterance("Create a parametric 50x50x20mm box")
    assert call is not None
    assert call.tool_id == "create_freecad_box"
    assert float(call.arguments["length"]) == 50.0
    assert float(call.arguments["width"]) == 50.0
    assert float(call.arguments["height"]) == 20.0


def test_parse_utterance_box_dims_suffix_order_unchanged() -> None:
    call = rd.parse_utterance("Build a box 40x30x10")
    assert call is not None
    assert call.tool_id == "create_freecad_box"
    assert float(call.arguments["length"]) == 40.0
    assert float(call.arguments["width"]) == 30.0
    assert float(call.arguments["height"]) == 10.0


def test_parse_utterance_box_dims_all_word_order_and_unit_variants() -> None:
    variants = [
        "50x50x20 box",
        "50 x 50 x 20 mm box",
        "box 50x50x20",
        "box 50x50x20mm",
        "box 50 x 50 x 20 mm",
    ]
    for text in variants:
        call = rd.parse_utterance(text)
        assert call is not None, text
        assert call.tool_id == "create_freecad_box", text
        assert float(call.arguments["length"]) == 50.0, text
        assert float(call.arguments["width"]) == 50.0, text
        assert float(call.arguments["height"]) == 20.0, text


def test_parse_utterance_cylinder_bare_dims_either_word_order() -> None:
    call = rd.parse_utterance("Create a 10x30 cylinder")
    assert call is not None
    assert call.tool_id == "create_freecad_cylinder"
    assert float(call.arguments["radius"]) == 10.0
    assert float(call.arguments["height"]) == 30.0

    call = rd.parse_utterance("cylinder 10x30mm")
    assert call is not None
    assert call.tool_id == "create_freecad_cylinder"
    assert float(call.arguments["radius"]) == 10.0
    assert float(call.arguments["height"]) == 30.0


def test_parse_utterance_cylinder_radius_height_keywords_unchanged() -> None:
    call = rd.parse_utterance("Create a cylinder radius 10 height 30")
    assert call is not None
    assert call.tool_id == "create_freecad_cylinder"
    assert float(call.arguments["radius"]) == 10.0
    assert float(call.arguments["height"]) == 30.0


def test_parse_utterance_prefix_dims_no_longer_falls_back_to_defaults() -> None:
    """Regression test for the exact bug found live-testing the WS dispatch
    endpoint: "<dims> box" word order silently fell through to the
    dims-less generic box pattern, creating a 40x25x15mm default box
    instead of the requested 50x50x20mm one."""
    call = rd.parse_utterance(
        "Create a parametric 50x50x20mm box in FreeCAD, then add a 10mm radius cylinder directly on top of it."
    )
    assert call is not None
    assert call.tool_id == "create_freecad_box"
    assert float(call.arguments["length"]) == 50.0
    assert float(call.arguments["width"]) == 50.0
    assert float(call.arguments["height"]) == 20.0


def test_parse_utterance_extrude_routes_to_extrusion_tool() -> None:
    call = rd.parse_utterance("Extrude this by 25mm")
    assert call is not None
    assert call.tool_id == "create_freecad_extrusion"
    assert float(call.arguments["height"]) == 25.0


def test_parse_utterance_extrude_injects_selection_as_anchor() -> None:
    selection = {"centroid": [0, 0, 50], "normal": [0, 0, 1]}
    call = rd.parse_utterance("Extrude this by 25mm", selection)
    assert call is not None
    assert call.tool_id == "create_freecad_extrusion"
    assert call.arguments["target_position"] == [0, 0, 50]
    assert call.arguments["target_normal"] == [0, 0, 1]


def test_is_mutating_tool_includes_extrusion() -> None:
    assert rd.is_mutating_tool("create_freecad_extrusion") is True


def test_dispatch_extrusion_without_profile_or_selection_fails_cleanly() -> None:
    call = rd.parse_utterance("Extrude this by 25mm")
    assert call is not None
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None)
    assert result.ok is False
    assert "profile points" in result.message or "selected face" in result.message


def test_dispatch_extrusion_rejects_non_z_normal() -> None:
    selection = {"centroid": [0, 0, 50], "normal": [1, 0, 0]}
    call = rd.parse_utterance("Extrude this by 25mm", selection)
    assert call is not None
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None)
    assert result.ok is False
    assert "Z axis" in result.message


def test_dispatch_extrusion_with_selection_builds_default_footprint() -> None:
    from dana.platform.mock import MockControlPlane, MockFreeCADEngine

    selection = {"centroid": [0, 0, 50], "normal": [0, 0, 1]}
    call = rd.parse_utterance("Extrude this by 25mm", selection)
    assert call is not None
    result = rd.dispatch_tool_call(call, MockFreeCADEngine(), MockControlPlane())
    assert result.ok is True
    assert result.payload["dimensions"]["height"] == 25.0
    assert result.payload["dimensions"]["profile_points"] == 4


def test_parse_utterance_camera_preset_first_word_order() -> None:
    """Regression test for the word-order bug found live-testing the WS
    dispatch endpoint: "Show me the isometric view" (preset word before
    the trigger word "view") didn't match any INTENT_PATTERNS entry."""
    call = rd.parse_utterance("Show me the isometric view")
    assert call is not None
    assert call.tool_id == "manipulate_camera"
    assert call.arguments["position"] == list(rd._CAMERA_PRESETS["iso"])

    call = rd.parse_utterance("top view")
    assert call is not None
    assert call.tool_id == "manipulate_camera"
    assert call.arguments["position"] == list(rd._CAMERA_PRESETS["top"])


def test_parse_utterance_camera_preset_trigger_first_word_order_unchanged() -> None:
    call = rd.parse_utterance("orbit to the iso view")
    assert call is not None
    assert call.tool_id == "manipulate_camera"
    assert call.arguments["position"] == list(rd._CAMERA_PRESETS["iso"])
