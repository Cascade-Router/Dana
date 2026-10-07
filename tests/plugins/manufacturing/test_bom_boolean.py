"""generate_assembly_bom on an assembly holding a boolean result: only the
finished solid is billed, never the block/tool it was cut from.

Runs on both CAD drivers. The mock driver's boolean needs trimesh's
manifold3d backend (requirements-dev.txt) and its cylinder is a 32-sided
prism, so its volumes match only to ~0.2%; the real FreeCAD engine is exact
and is skipped when FreeCADCmd isn't installed (it isn't in CI).
"""

from __future__ import annotations

import csv
import math
import uuid
from pathlib import Path
from typing import Any

import pytest

from dana.core import react_dispatch as rd
from dana.plugins.freecad import engine as fc_engine
from dana.plugins.manufacturing import bom_exporter
from dana.session_context import DEFAULT_SESSION_ID, set_session_id
from dana.tools.schema import ToolCall

_REAL_FREECAD = pytest.param(
    "freecad",
    marks=[
        pytest.mark.e2e,
        pytest.mark.skipif(
            fc_engine.detect_freecadcmd() is None,
            reason="real FreeCADCmd not found (PATH or DANA_FREECADCMD_PATH) — this case drives the real engine",
        ),
    ],
)
# Relative volume tolerance per driver.
_TOLERANCE = {"mock": 5e-3, "freecad": 1e-6}
# Cylinder placement that drills through the middle of the block. FreeCAD
# places a box by its corner and a cylinder by its base centre; the mock
# centres every primitive on its placement, so the same hole needs different
# coordinates there.
_DRILL_PLACEMENT = {
    "freecad": {"placement_x": 10, "placement_y": 10, "placement_z": -5},
    "mock": {},
}

BLOCK_MM3 = 20 * 20 * 10
HOLE_MM3 = math.pi * 5**2 * 10  # r=5 cylinder through the full 10 mm height
PLA_DENSITY = bom_exporter.load_materials()["materials"]["PLA"]["density_g_cm3"]


@pytest.fixture(params=["mock", _REAL_FREECAD])
def driver(request) -> str:
    return request.param


@pytest.fixture
def call(driver: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from dana.platform.mock import MockControlPlane, MockFreeCADEngine
    from dana.platform.win32 import RealFreeCADEngine

    monkeypatch.setattr(fc_engine, "_OUTPUT_DIR", tmp_path / "freecad_output")
    monkeypatch.setattr(fc_engine, "_EXPORT_DIR", tmp_path / "freecad_exports")
    monkeypatch.setattr(bom_exporter, "_EXPORT_DIR", tmp_path / "exports")
    monkeypatch.setenv("DANA_HEADLESS", "true")
    monkeypatch.delenv("DANA_OS_DRY_RUN", raising=False)
    set_session_id(f"bom-bool-{uuid.uuid4().hex[:8]}")
    rd._set_has_plan(True, "BOM boolean test")
    cad = RealFreeCADEngine() if driver == "freecad" else MockFreeCADEngine()
    control_plane = MockControlPlane()

    def _call(tool_id: str, **arguments: Any) -> dict[str, Any]:
        result = rd.dispatch_tool_call(ToolCall(tool_id=tool_id, arguments=arguments), cad, control_plane)
        assert result.ok, (tool_id, result.message)
        return result.payload

    yield _call
    set_session_id(DEFAULT_SESSION_ID)


def _drilled_block(call, driver: str) -> None:
    call("create_freecad_box", name="Block", length=20, width=20, height=10)
    # Centred on the block and overhanging it top and bottom, so the hole
    # goes all the way through: removed volume is exactly pi * r^2 * 10.
    call("create_freecad_cylinder", name="Drill", radius=5, height=20, **_DRILL_PLACEMENT[driver])
    call("perform_freecad_boolean", operation="cut", base_object="Block", tool_object="Drill", name="DrilledBlock")
    call("create_freecad_assembly", name="Kit")


@pytest.mark.parametrize(
    "members",
    [
        pytest.param(["DrilledBlock"], id="cut-only"),
        # Operands added alongside the result: the case that double-counts if
        # boolean inputs aren't filtered out.
        pytest.param(["Block", "Drill", "DrilledBlock"], id="cut-plus-operands"),
    ],
)
def test_bom_bills_only_the_boolean_result(call, driver: str, members: list[str]) -> None:
    _drilled_block(call, driver)
    call("add_parts_to_assembly", assembly_name="Kit", part_names=members)

    bom = call("generate_assembly_bom", assembly_name="Kit", material="PLA")

    expected_cm3 = (BLOCK_MM3 - HOLE_MM3) / 1000.0
    assert [p["name"] for p in bom["parts"]] == ["DrilledBlock"]
    rel = _TOLERANCE[driver]
    assert bom["parts"][0]["volume_cm3"] == pytest.approx(expected_cm3, rel=rel, abs=1e-4)
    assert bom["total_mass_g"] == pytest.approx(expected_cm3 * PLA_DENSITY, rel=rel, abs=1e-4)

    with open(bom["path"], newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))[1:]
    assert [r[0] for r in rows] == ["DrilledBlock"]


def test_mock_refuses_a_boolean_that_never_ran(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Without manifold3d the mock's cut silently returns the uncut base; the
    BOM must refuse it rather than bill the whole block."""
    import trimesh

    from dana.platform.mock import MockControlPlane, MockFreeCADEngine

    def _no_backend(*_args: Any, **_kwargs: Any):
        raise ImportError("manifold3d")

    monkeypatch.setattr(trimesh.Trimesh, "difference", _no_backend)
    monkeypatch.setattr(bom_exporter, "_EXPORT_DIR", tmp_path / "exports")
    set_session_id(f"bom-bool-{uuid.uuid4().hex[:8]}")
    rd._set_has_plan(True, "BOM boolean fallback test")
    cad, control_plane = MockFreeCADEngine(), MockControlPlane()
    try:
        for tool_id, args in [
            ("create_freecad_box", {"name": "Block", "length": 20, "width": 20, "height": 10}),
            ("create_freecad_cylinder", {"name": "Drill", "radius": 5, "height": 20}),
            ("perform_freecad_boolean", {"operation": "cut", "base_object": "Block", "tool_object": "Drill"}),
            ("create_freecad_assembly", {"name": "Kit"}),
            ("add_parts_to_assembly", {"assembly_name": "Kit", "part_names": ["Cut"]}),
        ]:
            assert rd.dispatch_tool_call(ToolCall(tool_id=tool_id, arguments=args), cad, control_plane).ok, tool_id

        result = rd.dispatch_tool_call(
            ToolCall(tool_id="generate_assembly_bom", arguments={"assembly_name": "Kit"}), cad, control_plane
        )
        assert not result.ok
        assert "manifold3d" in result.message
        assert not (tmp_path / "exports").exists()
    finally:
        set_session_id(DEFAULT_SESSION_ID)
