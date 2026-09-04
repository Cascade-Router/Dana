"""Targeted tests for the Standard Parametric Part Generator sprint:
dana.plugins.freecad.engineering_standards's new programmatic accessors
(parse_screw_spec/get_bearing_geometry/get_nema17_dimensions) and
dana.plugins.freecad.standard_parts.insert_standard_part, plus one
end-to-end dispatch_tool_call integration check. Not a full test suite by
design — pure-logic + dry-run coverage only, no live FreeCADCmd required.
"""

from __future__ import annotations

import json

import pytest

from dana.plugins.freecad.engineering_standards import (
    get_bearing_geometry,
    get_nema17_dimensions,
    parse_screw_spec,
)
from dana.plugins.freecad.standard_parts import insert_standard_part


@pytest.fixture(autouse=True)
def _plan_gate_open() -> None:
    """This module's dispatch_tool_call integration tests exercise
    insert_standard_part directly — dana.core.react_dispatch's
    Plan-and-Execute Gatekeeper now requires create_plan before any
    geometry-mutating tool (insert_standard_part included) dispatches; see
    tests/core/test_react_dispatch.py's own dedicated gatekeeper tests for
    that feature itself. Safe to leave open with no explicit teardown here
    — tests/conftest.py's own ``_reset_plan_gate_state`` clears the
    underlying registry after every test in the whole suite.
    """
    import dana.core.react_dispatch as react_dispatch

    react_dispatch._set_has_plan(True, "test-harness plan")


# --------------------------------------------------------------------------
# engineering_standards.py's programmatic (non-fuzzy) accessors
# --------------------------------------------------------------------------


def test_parse_screw_spec_extracts_diameter_length_and_head_geometry():
    geo = parse_screw_spec("M3x12")
    assert geo == {
        "nominal_diameter_mm": 3.0,
        "length_mm": 12.0,
        "head_diameter_mm": 5.5,
        "head_height_mm": 3.0,
    }


def test_parse_screw_spec_accepts_uppercase_x_and_decimal_length():
    geo = parse_screw_spec("M4X16.5")
    assert geo["nominal_diameter_mm"] == 4.0
    assert geo["length_mm"] == 16.5


def test_parse_screw_spec_rejects_malformed_spec():
    with pytest.raises(ValueError, match="invalid screw spec"):
        parse_screw_spec("not-a-spec")


def test_parse_screw_spec_rejects_unsupported_diameter():
    with pytest.raises(ValueError, match="no screw geometry"):
        parse_screw_spec("M8x20")


def test_get_bearing_geometry_returns_exact_608_dimensions():
    assert get_bearing_geometry("608") == {"bore_diameter_mm": 8.0, "outer_diameter_mm": 22.0, "width_mm": 7.0}


def test_get_bearing_geometry_rejects_unknown_designation():
    with pytest.raises(ValueError, match="no bearing geometry"):
        get_bearing_geometry("999")


def test_get_nema17_dimensions_includes_shaft_length_for_geometry_generation():
    dims = get_nema17_dimensions()
    assert dims["mounting_hole_spacing_mm"] == 31.0
    assert dims["default_shaft_length_mm"] == 24.0


# --------------------------------------------------------------------------
# insert_standard_part — dry-run mode (no FreeCADCmd needed)
# --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _dry_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DANA_OS_DRY_RUN", "1")


def test_insert_nema17_motor_dry_run_reports_correct_dimensions():
    result = json.loads(insert_standard_part("nema17_motor"))
    assert result["ok"] is True
    assert result["part_type"] == "nema17_motor"
    assert result["dimensions"]["mounting_hole_spacing_mm"] == 31.0


def test_insert_socket_head_screw_dry_run_resolves_spec():
    result = json.loads(insert_standard_part("socket_head_screw", specification="M3x12"))
    assert result["ok"] is True
    assert result["name"] == "Screw_M3X12"
    assert result["dimensions"]["head_diameter_mm"] == 5.5


def test_insert_ball_bearing_dry_run_resolves_designation():
    result = json.loads(insert_standard_part("ball_bearing", specification="608"))
    assert result["ok"] is True
    assert result["dimensions"]["outer_diameter_mm"] == 22.0


def test_insert_standard_part_rejects_unknown_part_type():
    result = json.loads(insert_standard_part("gearbox"))
    assert result["ok"] is False
    assert "unknown part_type" in result["error"]


def test_insert_standard_part_rejects_malformed_screw_spec():
    result = json.loads(insert_standard_part("socket_head_screw", specification="bogus"))
    assert result["ok"] is False
    assert "invalid screw spec" in result["error"]


def test_insert_standard_part_custom_name_and_placement_are_honored():
    result = json.loads(insert_standard_part("ball_bearing", specification="608", name="MyBearing", placement=(1.0, 2.0, 3.0)))
    assert result["name"] == "MyBearing"
    assert result["placement"] == [1.0, 2.0, 3.0]


# --------------------------------------------------------------------------
# insert_standard_part(part_type="fastener") — fastener_type sanitization
#
# FreeCAD's Fasteners workbench keys its type registry on an exact,
# space-free, uppercase token: passing "iso4017" through unchanged used to
# reach FastenersCmd.FSScrewObject and crash with FreeCAD's own
# "'iso4017' is not part of the enumeration" error (confirmed live against
# a real FreeCADCmd), and "ISO 4017" failed Dana's own token regex before
# ever reaching FreeCAD. Both must now resolve to "ISO4017".
# --------------------------------------------------------------------------


@pytest.mark.parametrize("raw", ["ISO4017", "iso4017", "ISO 4017", "iso 4017", "  iso4017  "])
def test_insert_fastener_dry_run_normalizes_type_case_and_spaces(raw: str):
    result = json.loads(insert_standard_part("fastener", fastener_type=raw, size="M8", length=30))
    assert result["ok"] is True, result
    assert result["dimensions"]["fastener_type"] == "ISO4017"
    assert result["name"] == "Fastener_ISO4017_M8"


def test_insert_fastener_dry_run_rejects_empty_type():
    result = json.loads(insert_standard_part("fastener", fastener_type="   ", size="M8", length=30))
    assert result["ok"] is False
    assert "fastener_type" in result["error"]


# --------------------------------------------------------------------------
# End-to-end dispatch_tool_call integration
# --------------------------------------------------------------------------


def test_dispatch_tool_call_insert_standard_part_end_to_end():
    """insert_standard_part bypasses the engine/control_plane driver
    abstraction by design (see standard_parts.py's module docstring), so
    engine=None/control_plane=None here proves dispatch never needs them
    for this tool_id — dry-run mode keeps this CI-safe without FreeCADCmd."""
    from dana.core import react_dispatch as rd
    from dana.tools.schema import ToolCall

    result = rd.dispatch_tool_call(
        ToolCall(tool_id="insert_standard_part", arguments={"part_type": "nema17_motor"}),
        engine=None,
        control_plane=None,
    )
    assert result.ok is True
    assert result.payload["part_type"] == "nema17_motor"
    assert rd.is_mutating_tool("insert_standard_part") is True


def test_dispatch_tool_call_insert_standard_part_missing_part_type():
    from dana.core import react_dispatch as rd
    from dana.tools.schema import ToolCall

    result = rd.dispatch_tool_call(
        ToolCall(tool_id="insert_standard_part", arguments={}), engine=None, control_plane=None
    )
    assert result.ok is False
    assert "part_type" in result.message
