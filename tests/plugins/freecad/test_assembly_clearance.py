"""validate_assembly_collisions' clearance report: the tightest gap between
parts that aren't directly mated or jointed, and warnings for non-mated
gaps under min_safe_clearance_mm, alongside the existing collision keys.

The dispatch tests build the assembly through the mock driver (so mates and
joints are recorded the real way) and fake only the engine's per-pair
distances, since the mock driver can't intersect solids. The live tests
measure real gaps in FreeCAD.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import pytest

from dana.core import react_dispatch as rd
from dana.plugins.freecad import engine as fc_engine
from dana.session_context import DEFAULT_SESSION_ID, set_session_id
from dana.tools.schema import ToolCall

_LIVE = pytest.mark.skipif(
    fc_engine.detect_freecadcmd() is None,
    reason="real FreeCADCmd not found (PATH or DANA_FREECADCMD_PATH) — this test drives the real engine",
)


@pytest.fixture
def session():
    set_session_id(f"clearance-{uuid.uuid4().hex[:8]}")
    rd._set_has_plan(True, "clearance test")
    yield
    set_session_id(DEFAULT_SESSION_ID)


def _dispatcher(cad: Any):
    from dana.platform.mock import MockControlPlane

    control_plane = MockControlPlane()

    def call(tool_id: str, **arguments: Any):
        return rd.dispatch_tool_call(ToolCall(tool_id=tool_id, arguments=arguments), cad, control_plane)

    return call


def _mock_rig(monkeypatch: pytest.MonkeyPatch, distances: dict[tuple[str, str], float]):
    """Base + Lid (mated) + Pin (free), with the engine reporting ``distances``."""
    from dana.platform.mock import MockFreeCADEngine

    cad = MockFreeCADEngine()
    call = _dispatcher(cad)
    for name in ("Base", "Lid", "Pin"):
        assert call("create_freecad_box", name=name, length=10, width=10, height=10).ok
    assert call("create_freecad_assembly", name="Rig").ok
    assert call("add_parts_to_assembly", assembly_name="Rig", part_names=["Base", "Lid", "Pin"]).ok
    mate = call("apply_assembly_constraint", assembly_name="Rig", part1_name="Base", part1_element="Face6",
                part2_name="Lid", part2_element="Face5", constraint_type="Coincident")
    assert mate.ok, mate.message

    clearances = [{"part_a": a, "part_b": b, "distance_mm": d} for (a, b), d in distances.items()]
    monkeypatch.setattr(
        cad,
        "validate_assembly_collisions",
        lambda name: {"ok": True, "name": name, "collisions": [], "has_collisions": False,
                      "checked_members": 3, "clearances": clearances},
    )
    return call


@pytest.mark.usefixtures("session")
def test_tight_non_mated_gap_warns_and_mated_contact_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    call = _mock_rig(monkeypatch, {("Base", "Lid"): 0.0, ("Base", "Pin"): 0.1, ("Lid", "Pin"): 3.0})

    result = call("validate_assembly_collisions", assembly_name="Rig")

    assert result.ok, result.message
    report = result.payload
    assert report["collision_free"] is True and report["collisions"] == []  # existing keys kept
    # Base/Lid touch (0.0) but are mated, so they don't count.
    assert report["min_clearance_mm"] == pytest.approx(0.1)
    assert report["closest_non_mated_pair"] == {"part_a": "Base", "part_b": "Pin", "clearance_mm": 0.1}
    assert report["clearance_warnings"] == [{"part_a": "Base", "part_b": "Pin", "clearance_mm": 0.1}]
    summary = rd.summarize_result(ToolCall(tool_id="validate_assembly_collisions", arguments={}), result)
    assert "closer than 0.2 mm" in summary and "`Base`/`Pin` (0.1 mm)" in summary


@pytest.mark.usefixtures("session")
def test_comfortable_gap_has_no_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    call = _mock_rig(monkeypatch, {("Base", "Lid"): 0.0, ("Base", "Pin"): 2.0, ("Lid", "Pin"): 5.0})

    report = call("validate_assembly_collisions", assembly_name="Rig").payload

    assert report["min_clearance_mm"] == pytest.approx(2.0)
    assert report["clearance_warnings"] == []


@pytest.mark.usefixtures("session")
def test_touching_non_mated_parts_report_zero_but_no_tolerance_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    call = _mock_rig(monkeypatch, {("Base", "Lid"): 0.0, ("Base", "Pin"): 0.0, ("Lid", "Pin"): 4.0})

    report = call("validate_assembly_collisions", assembly_name="Rig").payload

    assert report["min_clearance_mm"] == 0.0
    assert report["clearance_warnings"] == []  # touching is not "too tight", it is contact


@pytest.mark.usefixtures("session")
def test_threshold_is_configurable_and_validated(monkeypatch: pytest.MonkeyPatch) -> None:
    call = _mock_rig(monkeypatch, {("Base", "Lid"): 0.0, ("Base", "Pin"): 0.5, ("Lid", "Pin"): 3.0})

    loose = call("validate_assembly_collisions", assembly_name="Rig").payload
    strict = call("validate_assembly_collisions", assembly_name="Rig", min_safe_clearance_mm=1.0).payload
    bad = call("validate_assembly_collisions", assembly_name="Rig", min_safe_clearance_mm=-1)

    assert loose["clearance_warnings"] == []
    assert [w["part_b"] for w in strict["clearance_warnings"]] == ["Pin"]
    assert not bad.ok and "min_safe_clearance_mm" in bad.message


@pytest.mark.usefixtures("session")
def test_no_non_mated_pairs_means_no_minimum(monkeypatch: pytest.MonkeyPatch) -> None:
    from dana.platform.mock import MockFreeCADEngine

    cad = MockFreeCADEngine()
    call = _dispatcher(cad)
    for name in ("Base", "Lid"):
        assert call("create_freecad_box", name=name, length=10, width=10, height=10).ok
    assert call("create_freecad_assembly", name="Pair").ok
    assert call("add_parts_to_assembly", assembly_name="Pair", part_names=["Base", "Lid"]).ok
    assert call("define_kinematic_joint", assembly_name="Pair", child_link="Lid", parent_link="Base",
                joint_type="revolute").ok
    monkeypatch.setattr(
        cad,
        "validate_assembly_collisions",
        lambda name: {"ok": True, "name": name, "collisions": [], "has_collisions": False, "checked_members": 2,
                      "clearances": [{"part_a": "Base", "part_b": "Lid", "distance_mm": 0.05}]},
    )

    report = call("validate_assembly_collisions", assembly_name="Pair").payload

    # Jointed parts are meant to be close; nothing else to compare.
    assert (report["min_clearance_mm"], report["closest_non_mated_pair"], report["clearance_warnings"]) == (None, None, [])


# -- live: real FreeCAD -------------------------------------------------------------------


@pytest.fixture
def live_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, session):
    from dana.platform.win32 import RealFreeCADEngine

    monkeypatch.setattr(fc_engine, "_OUTPUT_DIR", tmp_path / "freecad_output")
    monkeypatch.setattr(fc_engine, "_EXPORT_DIR", tmp_path / "exports")
    monkeypatch.setenv("DANA_HEADLESS", "true")
    monkeypatch.delenv("DANA_OS_DRY_RUN", raising=False)
    call = _dispatcher(RealFreeCADEngine())

    def checked(tool_id: str, **arguments: Any) -> dict[str, Any]:
        result = call(tool_id, **arguments)
        assert result.ok, (tool_id, result.message)
        return result.payload

    return checked


@pytest.mark.e2e
@_LIVE
@pytest.mark.parametrize(("gap", "warns"), [(0.1, True), (2.0, False)])
def test_live_gap_between_unmated_blocks(live_call, gap: float, warns: bool) -> None:
    live_call("create_freecad_box", name="Left", length=10, width=10, height=10)
    live_call("create_freecad_box", name="Right", length=10, width=10, height=10, placement_x=10 + gap)
    live_call("create_freecad_assembly", name="Rig")
    live_call("add_parts_to_assembly", assembly_name="Rig", part_names=["Left", "Right"])

    report = live_call("validate_assembly_collisions", assembly_name="Rig")

    assert report["collision_free"] is True
    assert report["min_clearance_mm"] == pytest.approx(gap, abs=1e-4)
    assert report["closest_non_mated_pair"]["clearance_mm"] == pytest.approx(gap, abs=1e-4)
    assert bool(report["clearance_warnings"]) is warns


@pytest.mark.e2e
@_LIVE
def test_live_mated_contact_is_excluded_from_clearance(live_call) -> None:
    live_call("create_freecad_box", name="Base", length=20, width=20, height=10)
    live_call("create_freecad_box", name="Lid", length=20, width=20, height=5, placement_x=100)
    live_call("create_freecad_box", name="Pin", length=5, width=5, height=5, placement_x=20.15)
    live_call("create_freecad_assembly", name="Rig")
    live_call("add_parts_to_assembly", assembly_name="Rig", part_names=["Base", "Lid", "Pin"])
    # Lid's bottom onto Base's top: touching, but mated.
    live_call("apply_assembly_constraint", assembly_name="Rig", part1_name="Base", part1_element="[0, 0, 1]",
              part2_name="Lid", part2_element="[0, 0, -1]", constraint_type="Coincident")

    report = live_call("validate_assembly_collisions", assembly_name="Rig")

    assert report["collision_free"] is True
    pairs = {frozenset((c["part_a"], c["part_b"])): c["distance_mm"] for c in report["clearances"]}
    assert pairs[frozenset(("Base", "Lid"))] == pytest.approx(0.0, abs=1e-6)  # really touching
    assert report["closest_non_mated_pair"]["clearance_mm"] == pytest.approx(0.15, abs=1e-4)
    assert {report["closest_non_mated_pair"]["part_a"], report["closest_non_mated_pair"]["part_b"]} == {"Base", "Pin"}
    assert len(report["clearance_warnings"]) == 1
