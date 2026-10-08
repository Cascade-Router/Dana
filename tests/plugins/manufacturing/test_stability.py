"""Mass-weighted assembly center of mass and the static tip-over check.

The reference assembly is a 20 mm cube base with a 60 x 20 x 10 mm arm
cantilevered off its +X side, 10 mm above the ground. All PLA, the center of
mass lands at x = 34 mm, 14 mm past the base's 20 mm footprint: it tips. A
stainless base (8.0 g/cm^3 against PLA's 1.24) pulls it to x = 17.55 mm,
2.45 mm inside: it stands.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import pytest

from dana.core import react_dispatch as rd
from dana.plugins.freecad import engine as fc_engine
from dana.plugins.manufacturing import bom_exporter
from dana.plugins.manufacturing.stability import convex_hull, signed_margin, stability_report
from dana.session_context import DEFAULT_SESSION_ID, set_session_id
from dana.tools.schema import ToolCall

SQUARE = [[0, 0], [20, 0], [20, 20], [0, 20]]
BASE_VOLUME, ARM_VOLUME = 20 * 20 * 20, 60 * 20 * 10
STEEL_X = (BASE_VOLUME * 8.0 * 10 + ARM_VOLUME * 1.24 * 50) / (BASE_VOLUME * 8.0 + ARM_VOLUME * 1.24)


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(bom_exporter, "_EXPORT_DIR", tmp_path / "bom")
    set_session_id(f"stability-{uuid.uuid4().hex[:8]}")
    rd._set_has_plan(True, "stability test")
    yield
    set_session_id(DEFAULT_SESSION_ID)


# -- geometry ---------------------------------------------------------------------------


def test_hull_drops_interior_and_collinear_points() -> None:
    hull = convex_hull(SQUARE + [[10, 10], [10, 0], [5, 5]])
    assert sorted(hull) == [(0.0, 0.0), (0.0, 20.0), (20.0, 0.0), (20.0, 20.0)]


@pytest.mark.parametrize(
    ("point", "margin"),
    [((10, 10), 10.0), ((18, 10), 2.0), ((25, 10), -5.0), ((23, 24), -5.0), ((20, 10), 0.0)],
)
def test_signed_margin_is_positive_inside_negative_outside(point: tuple[float, float], margin: float) -> None:
    assert signed_margin(point, convex_hull(SQUARE)) == pytest.approx(margin)


def test_balancing_on_an_edge_is_never_stable() -> None:
    report = stability_report([10, 0, 5], [[0, 0], [20, 0]])  # a knife edge
    assert report["statically_stable"] is False and report["stability_margin_mm"] == pytest.approx(0.0)


def test_no_ground_contact_cannot_be_judged() -> None:
    report = stability_report([0, 0, 0], [])
    assert report["statically_stable"] is None and "No ground contact" in report["stability_warning"]


# -- BOM ---------------------------------------------------------------------------------


def _cantilever_parts() -> list[dict[str, Any]]:
    return [
        {"name": "Base", "volume_mm3": BASE_VOLUME, "center_of_mass_mm": [10, 10, 10]},
        {"name": "Arm", "volume_mm3": ARM_VOLUME, "center_of_mass_mm": [50, 10, 15]},
    ]


def test_uniform_pla_cantilever_tips() -> None:
    bom = bom_exporter.build_bom("Rig", _cantilever_parts(), "PLA", footprint_xy=SQUARE)
    assert bom["assembly_center_of_mass_mm"] == [34.0, 10.0, 13.0]
    assert bom["statically_stable"] is False and bom["stability_margin_mm"] == pytest.approx(-14.0)
    assert "14.0 mm outside the base footprint" in bom["stability_warning"]
    assert bom["parts"][1]["center_of_mass_mm"] == [50.0, 10.0, 15.0]


def test_steel_base_rescues_it() -> None:
    bom = bom_exporter.build_bom(
        "Rig", _cantilever_parts(), "PLA", part_materials={"Base": "Stainless Steel 304"}, footprint_xy=SQUARE
    )
    assert bom["assembly_center_of_mass_mm"][0] == pytest.approx(STEEL_X, abs=1e-3)
    assert bom["statically_stable"] is True
    assert bom["stability_margin_mm"] == pytest.approx(20 - STEEL_X, abs=1e-3)  # ~2.45 mm
    assert bom["stability_warning"] is None  # more than 2 mm to spare


def test_aluminium_base_is_not_dense_enough() -> None:
    bom = bom_exporter.build_bom(
        "Rig", _cantilever_parts(), "PLA", part_materials={"Base": "Aluminum 6061"}, footprint_xy=SQUARE
    )
    assert bom["statically_stable"] is False


def test_thin_margin_stands_but_warns() -> None:
    parts = [{"name": "Post", "volume_mm3": 1000, "center_of_mass_mm": [19, 10, 5]}]
    bom = bom_exporter.build_bom("Post", parts, footprint_xy=SQUARE)
    assert bom["statically_stable"] is True and bom["stability_margin_mm"] == pytest.approx(1.0)
    assert "only 1.0 mm inside" in bom["stability_warning"]


def test_single_centered_part_has_the_symmetry_margin() -> None:
    parts = [{"name": "Block", "volume_mm3": 8000, "center_of_mass_mm": [10, 10, 10]}]
    bom = bom_exporter.build_bom("Block", parts, footprint_xy=SQUARE)
    assert (bom["statically_stable"], bom["stability_margin_mm"], bom["stability_warning"]) == (True, 10.0, None)


def test_without_centers_there_is_no_stability_section() -> None:
    bom = bom_exporter.build_bom("Rig", [{"name": "A", "volume_mm3": 1000}], footprint_xy=SQUARE)
    assert "assembly_center_of_mass_mm" not in bom and "statically_stable" not in bom


# -- through the tool --------------------------------------------------------------------


def _dispatcher(cad: Any):
    from dana.platform.mock import MockControlPlane

    control_plane = MockControlPlane()

    def call(tool_id: str, **arguments: Any) -> dict[str, Any]:
        result = rd.dispatch_tool_call(ToolCall(tool_id=tool_id, arguments=arguments), cad, control_plane)
        assert result.ok, (tool_id, result.message)
        return result.payload

    return call


def _build_cantilever(call, *, corner_placement: bool) -> None:
    """Base occupies x 0..20, y 0..20, z 0..20; the arm x 20..80, z 10..20.
    FreeCAD places a box by its corner, the mock by its centre."""
    if corner_placement:
        base, arm = {}, {"placement_x": 20, "placement_z": 10}
    else:
        base = {"placement_x": 10, "placement_y": 10, "placement_z": 10}
        arm = {"placement_x": 50, "placement_y": 10, "placement_z": 15}
    call("create_freecad_box", name="Base", length=20, width=20, height=20, **base)
    call("create_freecad_box", name="Arm", length=60, width=20, height=10, **arm)
    call("create_freecad_assembly", name="Rig")
    call("add_parts_to_assembly", assembly_name="Rig", part_names=["Base", "Arm"])


def _check_rescue(call) -> None:
    pla = call("generate_assembly_bom", assembly_name="Rig")
    assert pla["statically_stable"] is False and pla["stability_margin_mm"] == pytest.approx(-14.0, abs=1e-3)
    steel = call("generate_assembly_bom", assembly_name="Rig", part_materials={"Base": "Stainless Steel 304"})
    assert steel["statically_stable"] is True
    assert steel["stability_margin_mm"] == pytest.approx(20 - STEEL_X, abs=1e-3)
    assert steel["support_polygon_xy"] and all(0 <= x <= 20 and 0 <= y <= 20 for x, y in steel["support_polygon_xy"])


def test_mock_driver_cantilever_rescued_by_material() -> None:
    from dana.platform.mock import MockFreeCADEngine

    call = _dispatcher(MockFreeCADEngine())
    _build_cantilever(call, corner_placement=False)
    _check_rescue(call)


@pytest.mark.e2e
@pytest.mark.skipif(
    fc_engine.detect_freecadcmd() is None,
    reason="real FreeCADCmd not found (PATH or DANA_FREECADCMD_PATH) — this test drives the real engine",
)
def test_live_cantilever_rescued_by_material(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from dana.platform.win32 import RealFreeCADEngine

    monkeypatch.setattr(fc_engine, "_OUTPUT_DIR", tmp_path / "freecad_output")
    monkeypatch.setattr(fc_engine, "_EXPORT_DIR", tmp_path / "exports")
    monkeypatch.setenv("DANA_HEADLESS", "true")
    monkeypatch.delenv("DANA_OS_DRY_RUN", raising=False)
    call = _dispatcher(RealFreeCADEngine())
    _build_cantilever(call, corner_placement=True)
    _check_rescue(call)
    centers = {p["name"]: p["center_of_mass_mm"] for p in call("generate_assembly_bom", assembly_name="Rig")["parts"]}
    assert centers == {"Base": [10.0, 10.0, 10.0], "Arm": [50.0, 10.0, 15.0]}
