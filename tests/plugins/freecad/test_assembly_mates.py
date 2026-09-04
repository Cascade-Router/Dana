"""Targeted tests for Kinematic Assembly Mates: dana.plugins.freecad.engine
's pure ``_mate_delta`` math (mirrors the existing ``_alignment_delta``
tests in test_engine.py) plus one end-to-end dispatch_tool_call integration
check through the mock CAD engine driver — no live FreeCADCmd required, and
no unrelated legacy test files exercised. Not a full test suite by design.
"""

from __future__ import annotations

import pytest

from dana.plugins.freecad import engine


@pytest.fixture(autouse=True)
def _plan_gate_open() -> None:
    """This module's dispatch_tool_call integration test builds its own
    Base/Shaft primitives as setup — dana.core.react_dispatch's
    Plan-and-Execute Gatekeeper now requires create_plan before any
    geometry-mutating tool (including those two primitives) dispatches; see
    tests/core/test_react_dispatch.py's own dedicated gatekeeper tests for
    that feature itself. Safe to leave open with no explicit teardown here
    — tests/conftest.py's own ``_reset_plan_gate_state`` clears the
    underlying registry after every test in the whole suite.
    """
    import dana.core.react_dispatch as react_dispatch

    react_dispatch._set_has_plan(True, "test-harness plan")


def _bbox(x_min, y_min, z_min, x_max, y_max, z_max):
    return {"x_min": x_min, "y_min": y_min, "z_min": z_min, "x_max": x_max, "y_max": y_max, "z_max": z_max}


# --------------------------------------------------------------------------
# _mate_delta — pure arithmetic, no FreeCAD needed
# --------------------------------------------------------------------------


def test_concentric_mate_centers_xy_and_applies_raw_z_offset():
    fixed = _bbox(0, 0, 0, 20, 20, 10)  # center (10, 10)
    moving = _bbox(-5, -5, 0, 5, 5, 30)  # center (0, 0)
    dx, dy, dz = engine._mate_delta("concentric", {"z_offset": 5.0}, fixed, moving)
    assert (dx, dy, dz) == (10.0, 10.0, 5.0)


def test_concentric_mate_defaults_z_offset_to_zero():
    fixed = _bbox(0, 0, 0, 20, 20, 10)
    moving = _bbox(0, 0, 0, 10, 10, 30)
    dx, dy, dz = engine._mate_delta("concentric", {}, fixed, moving)
    assert dz == 0.0


def test_coincident_planar_mate_stacks_moving_flush_on_fixed_top():
    fixed = _bbox(0, 0, 0, 20, 20, 10)  # top at z=10, center (10, 10)
    moving = _bbox(0, 0, 0, 30, 30, 5)  # bottom at z=0, center (15, 15)
    dx, dy, dz = engine._mate_delta("coincident_planar", {"offset_x": 2.0}, fixed, moving)
    assert dx == -3.0  # (10 - 15) + 2
    assert dy == -5.0  # (10 - 15) + 0
    assert dz == 10.0  # fixed.z_max(10) - moving.z_min(0)


def test_offset_axial_mate_stands_moving_off_above_fixed_top_by_distance():
    fixed = _bbox(0, 0, 0, 20, 20, 10)  # top at z=10, center (10, 10)
    moving = _bbox(-5, -5, 0, 5, 5, 30)  # bottom at z=0, center (0, 0)
    dx, dy, dz = engine._mate_delta("offset_axial", {"distance": 3.0}, fixed, moving)
    assert (dx, dy) == (10.0, 10.0)
    assert dz == 13.0  # (fixed.z_max(10) + 3) - moving.z_min(0)


def test_offset_axial_mate_from_bottom_face_hangs_moving_below_fixed():
    fixed = _bbox(0, 0, 0, 20, 20, 10)  # bottom at z=0
    moving = _bbox(0, 0, 0, 10, 10, 30)  # top at z=30
    dx, dy, dz = engine._mate_delta("offset_axial", {"distance": 3.0, "from_face": "bottom"}, fixed, moving)
    assert dz == -33.0  # (fixed.z_min(0) - 3) - moving.z_max(30)


def test_mate_delta_rejects_unknown_mate_type():
    with pytest.raises(ValueError, match="unknown mate_type"):
        engine._mate_delta("welded", {}, _bbox(0, 0, 0, 1, 1, 1), _bbox(0, 0, 0, 1, 1, 1))


# --------------------------------------------------------------------------
# End-to-end dispatch_tool_call integration, through the mock CAD engine
# (no live FreeCADCmd needed — MockFreeCADEngine.create_assembly_mate
# reuses this same _mate_delta function directly).
# --------------------------------------------------------------------------


def test_dispatch_tool_call_create_assembly_mate_end_to_end():
    from dana.core import react_dispatch as rd
    from dana.platform.mock import MockControlPlane, MockFreeCADEngine
    from dana.tools.schema import ToolCall

    engine_driver = MockFreeCADEngine()
    control_plane = MockControlPlane()

    base = rd.dispatch_tool_call(
        ToolCall(tool_id="create_freecad_box", arguments={"length": 20, "width": 20, "height": 10, "name": "Base"}),
        engine_driver,
        control_plane,
    )
    shaft = rd.dispatch_tool_call(
        ToolCall(tool_id="create_freecad_cylinder", arguments={"radius": 5, "height": 30, "name": "Shaft"}),
        engine_driver,
        control_plane,
    )
    assert base.ok and shaft.ok

    mate = rd.dispatch_tool_call(
        ToolCall(
            tool_id="create_assembly_mate",
            arguments={
                "fixed_obj": "Base",
                "moving_obj": "Shaft",
                "mate_type": "offset_axial",
                "mate_params": {"distance": 3.0},
            },
        ),
        engine_driver,
        control_plane,
    )
    assert mate.ok is True
    assert mate.payload["mate_type"] == "offset_axial"
    assert rd.is_mutating_tool("create_assembly_mate") is True


def test_dispatch_tool_call_create_assembly_mate_rejects_unknown_mate_type():
    from dana.core import react_dispatch as rd
    from dana.tools.schema import ToolCall

    result = rd.dispatch_tool_call(
        ToolCall(
            tool_id="create_assembly_mate",
            arguments={"fixed_obj": "Base", "moving_obj": "Shaft", "mate_type": "welded"},
        ),
        engine=None,
        control_plane=None,
    )
    assert result.ok is False
    assert "mate_type" in result.message


def test_dispatch_tool_call_create_assembly_mate_rejects_unknown_object_name():
    from dana.core import react_dispatch as rd
    from dana.tools.schema import ToolCall

    result = rd.dispatch_tool_call(
        ToolCall(
            tool_id="create_assembly_mate",
            arguments={"fixed_obj": "Ghost", "moving_obj": "Shaft", "mate_type": "concentric"},
        ),
        engine=None,
        control_plane=None,
    )
    assert result.ok is False
    assert "fixed_obj" in result.message
