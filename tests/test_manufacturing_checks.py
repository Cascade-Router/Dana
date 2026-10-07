"""Pre-print sanity check (dana.plugins.freecad.printability) and its
check_printability tool.

The report logic is pure and runs everywhere on trimesh meshes; the live
tests at the bottom build real parts in FreeCADCmd and check them through the
real engine (skipped without FreeCAD).
"""

from __future__ import annotations

import json
import math
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import trimesh

import dana.core.react_dispatch as rd
from dana.platform.mock import MockFreeCADEngine
from dana.plugins.freecad import engine, printability
from dana.plugins.freecad.printability import (
    DEFAULT_PRINTER,
    Facet,
    PrinterProfile,
    SolidFacet,
    build_report,
    evaluate_orientations,
)
from dana.plugins.os import file_system
from dana.session_context import DEFAULT_SESSION_ID, set_session_id

_CLEAN = {"shape_valid": True, "closed_solid": True, "mesh_solid": True, "non_manifold": False,
          "self_intersections": False, "open_edges": 0, "components": 1}


def _box(lo: tuple[float, float, float], hi: tuple[float, float, float]) -> trimesh.Trimesh:
    lo_a, hi_a = np.array(lo, float), np.array(hi, float)
    box = trimesh.creation.box(extents=hi_a - lo_a)
    box.apply_translation((lo_a + hi_a) / 2)
    return box


def _report(mesh: trimesh.Trimesh, **kw: Any) -> dict[str, Any]:
    """build_report on a trimesh mesh, with mesh checks computed the same way
    the FreeCAD side does (watertight, open edges = edges used by one facet)."""
    edges = np.sort(mesh.edges, axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    z = mesh.vertices[mesh.faces][:, :, 2]
    checks = {**_CLEAN, "mesh_solid": bool(mesh.is_watertight), "closed_solid": bool(mesh.is_watertight),
              "open_edges": int((counts == 1).sum())}
    facets = [Facet(float(n), float(a), float(lo), float(hi))
              for n, a, lo, hi in zip(mesh.face_normals[:, 2], mesh.area_faces, z.min(axis=1), z.max(axis=1))]
    return build_report(target="T", checks=checks, bbox=list(mesh.bounds.flatten()), facets=facets, **kw)


# -- printable ---------------------------------------------------------------------------


def test_clean_box_is_printable_without_supports() -> None:
    report = _report(_box((0, 0, 0), (20, 10, 15)))
    assert report["printable"] is True
    assert report["requires_supports"] is False
    assert report["warnings"] == []
    assert report["dimensions_mm"] == [20.0, 10.0, 15.0]


def test_open_shell_is_not_printable() -> None:
    box = _box((0, 0, 0), (10, 10, 10))
    open_shell = trimesh.Trimesh(box.vertices, box.faces[:-2], process=False)  # drop one side
    report = _report(open_shell)
    assert report["printable"] is False
    assert report["checks"]["open_edges"] == 4
    assert any("not watertight" in w for w in report["warnings"])


@pytest.mark.parametrize(
    ("flag", "value", "message"),
    [
        ("non_manifold", True, "non-manifold"),
        ("self_intersections", True, "intersects itself"),
        ("shape_valid", False, "shape is invalid"),
        ("closed_solid", False, "not a closed solid"),
    ],
)
def test_each_mesh_defect_blocks_printing(flag: str, value: bool, message: str) -> None:
    report = build_report(target="T", checks={**_CLEAN, flag: value}, bbox=[0, 0, 0, 10, 10, 10], facets=[])
    assert report["printable"] is False
    assert any(message in w for w in report["warnings"])


def test_separate_islands_warn_but_still_print() -> None:
    report = build_report(target="T", checks={**_CLEAN, "components": 2}, bbox=[0, 0, 0, 10, 10, 10], facets=[])
    assert report["printable"] is True
    assert "2 separate bodies" in report["warnings"][0]


# -- build volume ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("size", "fit", "printable"),
    [
        ((250, 210, 220), "fits", True),  # exactly the MK4 volume
        ((210, 250, 100), "fits_rotated", True),
        ((300, 10, 5), "too_large", False),
        ((100, 100, 221), "too_large", False),
    ],
)
def test_build_volume(size: tuple[float, float, float], fit: str, printable: bool) -> None:
    report = _report(_box((0, 0, 0), size))
    assert report["build_volume"]["fit"] == fit
    assert report["printable"] is printable


def test_custom_build_volume() -> None:
    report = _report(_box((0, 0, 0), (300, 10, 5)), printer=PrinterProfile("big", 350, 350, 350))
    assert report["printable"] is True
    assert printability.printer_profile(None) == DEFAULT_PRINTER
    with pytest.raises(ValueError):
        printability.printer_profile([0, 10, 10])


# -- overhangs ---------------------------------------------------------------------------


def test_ninety_degree_cantilever_requires_supports() -> None:
    column = _box((0, 0, 0), (10, 10, 40))
    arm = _box((10, 0, 30), (40, 10, 40))  # underside at Z=30, flat (90° from vertical)
    report = _report(trimesh.util.concatenate([column, arm]))
    assert report["requires_supports"] is True
    assert report["overhangs"]["area_mm2"] == pytest.approx(300.0)
    assert report["overhangs"]["z_range_mm"] == [30.0, 30.0]
    assert report["overhangs"]["worst_angle_from_vertical_deg"] == 90.0


@pytest.mark.parametrize(("angle_from_vertical", "overhangs"), [(30.0, False), (44.0, False), (46.0, True), (60.0, True)])
def test_overhang_threshold_is_45_degrees_from_vertical(angle_from_vertical: float, overhangs: bool) -> None:
    facet = Facet(normal_z=-math.sin(math.radians(angle_from_vertical)), area=50.0, z_min=5.0, z_max=10.0)
    report = build_report(target="T", checks=_CLEAN, bbox=[0, 0, 0, 10, 10, 10], facets=[facet])
    assert report["requires_supports"] is overhangs


def test_bottom_resting_on_the_bed_is_not_an_overhang() -> None:
    on_bed = Facet(normal_z=-1.0, area=500.0, z_min=0.0, z_max=0.0)
    assert build_report(target="T", checks=_CLEAN, bbox=[0, 0, 0, 10, 10, 10], facets=[on_bed])[
        "requires_supports"] is False


def test_sliver_overhang_below_the_area_floor_is_ignored() -> None:
    sliver = Facet(normal_z=-1.0, area=0.2, z_min=5.0, z_max=5.0)
    report = build_report(target="T", checks=_CLEAN, bbox=[0, 0, 0, 10, 10, 10], facets=[sliver])
    assert report["requires_supports"] is False
    assert report["overhangs"]["facet_count"] == 1  # still reported, just not decisive


def test_custom_overhang_limit() -> None:
    facet = Facet(normal_z=-math.sin(math.radians(40)), area=50.0, z_min=5.0, z_max=10.0)
    report = build_report(target="T", checks=_CLEAN, bbox=[0, 0, 0, 10, 10, 10], facets=[facet], max_overhang_deg=35)
    assert report["requires_supports"] is True


# -- the agent-facing tool ---------------------------------------------------------------


def test_tool_is_registered_read_only_and_reachable_with_the_freecad_domain() -> None:
    from dana.tools.schema import load_tool_registry

    spec = load_tool_registry()["check_printability"]
    assert spec.read_only is True
    assert "check_printability" in rd.TOOL_HANDLERS
    assert "check_printability" in rd._CAPABILITY_TOOL_IDS["freecad_full"]
    assert rd.is_mutating_tool("check_printability") is False


@pytest.mark.parametrize(
    "args",
    [{"build_volume_mm": [250, 210]}, {"build_volume_mm": [250, 0, 220]}, {"max_overhang_deg": 90},
     {"max_overhang_deg": "45"}],
)
def test_tool_rejects_bad_arguments(args: dict[str, Any]) -> None:
    result = rd._tool_check_printability(args, MockFreeCADEngine(), None)
    assert result["ok"] is False and "check_printability" in result["error"]


def test_mock_engine_says_it_cannot_check_rather_than_guessing() -> None:
    result = rd._tool_check_printability({}, MockFreeCADEngine(), None)
    assert result["ok"] is False and "real FreeCAD engine is required" in result["error"]


# -- orientation analysis ---------------------------------------------------------------


def _solid(mesh: trimesh.Trimesh) -> list[SolidFacet]:
    tri = mesh.vertices[mesh.faces]
    return [
        SolidFacet(tuple(map(float, n)), float(a), tuple(map(float, lo)), tuple(map(float, hi)))
        for n, a, lo, hi in zip(mesh.face_normals, mesh.area_faces, tri.min(axis=1), tri.max(axis=1))
    ]


def _union(*meshes: trimesh.Trimesh) -> trimesh.Trimesh:
    pytest.importorskip("manifold3d")  # trimesh's boolean backend (requirements-dev.txt)
    return trimesh.boolean.union(list(meshes))


def _upright_tee() -> trimesh.Trimesh:
    """A T in the XZ plane, 10 mm thick: a 40 mm stem under a 50 mm bar, whose
    400 mm² underside overhangs as modelled."""
    return _union(_box((0, 0, 0), (10, 10, 40)), _box((-20, 0, 40), (30, 10, 50)))


def _jack() -> trimesh.Trimesh:
    """Three 60 mm bars crossing at the origin: whichever bar stands up, the
    other two stick out sideways with overhanging undersides."""
    return _union(_box((-30, -5, -5), (30, 5, 5)), _box((-5, -30, -5), (5, 30, 5)), _box((-5, -5, -30), (5, 5, 30)))


def _rotated(mesh: trimesh.Trimesh, rotation: dict[str, Any]) -> trimesh.Trimesh:
    axis = {"X": [1, 0, 0], "Y": [0, 1, 0], "Z": [0, 0, 1]}[rotation["axis"]]
    turned = mesh.copy()
    turned.apply_transform(trimesh.transformations.rotation_matrix(math.radians(rotation["degrees"]), axis))
    return turned


def test_upright_tee_is_recommended_flat_and_the_rotation_really_removes_the_overhang() -> None:
    tee = _upright_tee()
    result = evaluate_orientations(_solid(tee))

    assert result["current_orientation_printable"] is False
    assert result["orientations"][0]["overhang_area_mm2"] == pytest.approx(400.0)
    best = result["recommended_orientation"]
    # Lying flat on a 10 mm face: support-free, biggest bed contact (the T's
    # whole 900 mm² profile), 10 mm tall.
    assert (best["up_axis"], best["rotation"], best["requires_supports"]) == ("+Y", {"axis": "X", "degrees": 90.0}, False)
    assert best["bed_contact_mm2"] == pytest.approx(900.0) and best["height_mm"] == pytest.approx(10.0)
    assert "rotated +90° about the part's X axis" in result["remediation_hint"]
    assert "no supports needed" in result["remediation_hint"]
    # Independent check of the rotation's sign: actually turning the mesh puts
    # it in a pose the plain as-modelled check calls support-free.
    turned = _report(_rotated(tee, best["rotation"]))
    assert turned["requires_supports"] is False
    assert turned["dimensions_mm"][2] == pytest.approx(10.0)


@pytest.mark.parametrize("pose", range(6))
def test_every_listed_rotation_turns_its_up_axis_to_plus_z(pose: int) -> None:
    entry = evaluate_orientations(_solid(_box((0, 0, 0), (10, 20, 30))))["orientations"][pose]
    up = np.zeros(3)
    up["XYZ".index(entry["up_axis"][1])] = 1.0 if entry["up_axis"][0] == "+" else -1.0
    if entry["rotation"] is not None:
        axis = np.eye(3)["XYZ".index(entry["rotation"]["axis"])]
        up = trimesh.transformations.rotation_matrix(math.radians(entry["rotation"]["degrees"]), axis)[:3, :3] @ up
    assert up == pytest.approx([0.0, 0.0, 1.0], abs=1e-9)


def test_jack_needs_supports_in_every_orientation() -> None:
    result = evaluate_orientations(_solid(_jack()))

    assert result["current_orientation_printable"] is False
    assert all(pose["requires_supports"] for pose in result["orientations"])
    # All six poses are equally bad, so the current one is kept rather than
    # suggesting a pointless rotation.
    assert result["recommended_orientation"]["is_current"] is True
    assert result["remediation_hint"].startswith("Needs supports in every principal orientation")
    assert "Enable supports" in result["remediation_hint"]


def test_part_that_already_prints_is_never_told_to_rotate() -> None:
    # Lying on its 20x30 side would give more bed contact, but the part is
    # already fine as modelled.
    result = evaluate_orientations(_solid(_box((0, 0, 0), (10, 20, 30))))
    assert result["current_orientation_printable"] is True
    assert result["recommended_orientation"]["is_current"] is True
    assert result["remediation_hint"] is None


def test_too_tall_as_modelled_but_fits_lying_down() -> None:
    result = evaluate_orientations(_solid(_box((0, 0, 0), (20, 20, 230))))  # MK4 is 220 mm tall
    assert result["orientations"][0]["fit"] == "too_large"
    best = result["recommended_orientation"]
    # 230 mm only fits along the bed's 250 mm side, hence fits_rotated.
    assert best["fit"] == "fits_rotated" and best["height_mm"] == pytest.approx(20.0)
    assert "and it fits the bed if turned 90° on it" in result["remediation_hint"]


def test_too_large_every_way() -> None:
    result = evaluate_orientations(_solid(_box((0, 0, 0), (300, 300, 300))))
    assert all(pose["fit"] == "too_large" for pose in result["orientations"])
    assert result["remediation_hint"].startswith("Too large for the Prusa MK4")


def test_mesh_defects_are_not_blamed_on_orientation() -> None:
    box = _box((0, 0, 0), (10, 10, 10))
    open_shell = trimesh.Trimesh(box.vertices, box.faces[:-2], process=False)
    z = open_shell.vertices[open_shell.faces][:, :, 2]
    report = build_report(
        target="T",
        checks={**_CLEAN, "mesh_solid": False, "closed_solid": False, "open_edges": 4},
        bbox=list(open_shell.bounds.flatten()),
        facets=[Facet(float(n), float(a), float(lo), float(hi))
                for n, a, lo, hi in zip(open_shell.face_normals[:, 2], open_shell.area_faces, z.min(1), z.max(1))],
        solid_facets=_solid(open_shell),
    )
    assert report["printable"] is False and report["current_orientation_printable"] is False
    assert report["remediation_hint"].startswith("Repair the mesh first")


def test_report_carries_orientation_fields_only_with_solid_facets() -> None:
    tee = _upright_tee()
    plain = _report(tee)
    assert "recommended_orientation" not in plain
    z = tee.vertices[tee.faces][:, :, 2]
    full = build_report(
        target="T",
        checks=_CLEAN,
        bbox=list(tee.bounds.flatten()),
        facets=[Facet(float(n), float(a), float(lo), float(hi))
                for n, a, lo, hi in zip(tee.face_normals[:, 2], tee.area_faces, z.min(1), z.max(1))],
        solid_facets=_solid(tee),
    )
    # The as-modelled pose agrees with the original single-pose check.
    assert full["requires_supports"] is True and full["orientations"][0]["requires_supports"] is True
    assert full["overhangs"]["area_mm2"] == full["orientations"][0]["overhang_area_mm2"]


# -- live: real FreeCAD -------------------------------------------------------------------

_live = pytest.mark.skipif(
    engine.detect_freecadcmd() is None,
    reason="real FreeCADCmd not found (PATH or DANA_FREECADCMD_PATH) — this test drives the real engine",
)


@pytest.fixture
def live_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(engine, "_OUTPUT_DIR", tmp_path / "freecad_output")
    monkeypatch.setattr(engine, "_EXPORT_DIR", tmp_path / "exports")
    monkeypatch.setattr(file_system, "_SANDBOX_ROOT", (tmp_path / "agent_workspace").resolve())
    monkeypatch.setenv("DANA_HEADLESS", "true")
    monkeypatch.delenv("DANA_OS_DRY_RUN", raising=False)
    set_session_id(f"print-{uuid.uuid4().hex[:8]}")
    yield
    set_session_id(DEFAULT_SESSION_ID)


def _poly(pts: list[tuple[float, float]]) -> list[dict[str, Any]]:
    return [{"type": "line", "start": list(pts[i]), "end": list(pts[(i + 1) % len(pts)])} for i in range(len(pts))]


def _padded(plane: str, profile: list[tuple[float, float]], length: float) -> None:
    assert json.loads(engine.create_sketch("Profile", plane, _poly(profile)))["ok"]
    assert json.loads(engine.create_pad("Profile", length))["ok"]


def _check(**kw: Any) -> dict[str, Any]:
    from dana.platform.win32 import RealFreeCADEngine

    result = rd._tool_check_printability(kw, RealFreeCADEngine(), None)
    assert result["ok"] is True, result
    return result


@pytest.mark.e2e
@_live
@pytest.mark.usefixtures("live_session")
def test_live_padded_block_is_printable() -> None:
    _padded("XY", [(0, 0), (20, 0), (20, 10), (0, 10)], 15)
    report = _check()
    assert (report["printable"], report["requires_supports"], report["target"]) == (True, False, "Body")
    assert report["checks"]["mesh_solid"] is True and report["checks"]["open_edges"] == 0
    stl = Path(report["stl_path"])
    assert stl.read_bytes()[:5] != b"solid"  # binary STL, not ASCII
    # In the sandbox, named after the part, so slice_stl_to_gcode can take it.
    assert stl.name == "Body.stl"
    assert file_system.resolve_sandboxed_path(str(stl)) == stl.resolve()
    assert [p.name for p in stl.parent.iterdir()] == ["Body.stl"]  # no temp file left


@pytest.mark.e2e
@_live
@pytest.mark.usefixtures("live_session")
def test_live_ninety_degree_cantilever_requires_supports() -> None:
    _padded("XZ", [(0, 0), (10, 0), (10, 30), (40, 30), (40, 40), (0, 40)], 10)
    report = _check()
    assert report["printable"] is True
    assert report["requires_supports"] is True
    assert report["overhangs"]["area_mm2"] == pytest.approx(300.0)


@pytest.mark.e2e
@_live
@pytest.mark.usefixtures("live_session")
def test_live_thirty_degree_slope_needs_no_supports() -> None:
    run = 40 * math.tan(math.radians(30))
    _padded("XZ", [(0, 0), (10, 0), (10 + run, 40), (0, 40)], 10)
    assert _check()["requires_supports"] is False


@pytest.mark.e2e
@_live
@pytest.mark.usefixtures("live_session")
def test_live_oversized_part_is_not_printable() -> None:
    _padded("XY", [(0, 0), (300, 0), (300, 10), (0, 10)], 5)
    report = _check()
    assert report["printable"] is False and report["build_volume"]["fit"] == "too_large"
    assert _check(build_volume_mm=[350, 350, 350])["printable"] is True


@pytest.mark.e2e
@_live
@pytest.mark.usefixtures("live_session")
def test_live_open_shell_is_not_printable() -> None:
    session = engine._session_document_path()
    session.parent.mkdir(parents=True, exist_ok=True)
    result = engine._run_freecad_script(
        "import FreeCAD as App, Part\n"
        "doc = App.newDocument('Session_Active')\n"
        "o = doc.addObject('Part::Feature', 'OpenShell')\n"
        "o.Shape = Part.Shell(Part.makeBox(10, 10, 10).Faces[:-1])\n"
        f"doc.recompute()\ndoc.saveAs({str(session)!r})\nprint({engine._OK_MARKER!r})\n"
    )
    assert result["ok"], result
    report = _check(object_name="OpenShell")
    assert report["printable"] is False
    assert report["checks"]["closed_solid"] is False
    assert report["checks"]["mesh_solid"] is False
    assert report["checks"]["open_edges"] > 0


@pytest.mark.e2e
@_live
@pytest.mark.usefixtures("live_session")
def test_live_unknown_object_is_a_clean_error() -> None:
    _padded("XY", [(0, 0), (10, 0), (10, 10), (0, 10)], 5)
    from dana.platform.win32 import RealFreeCADEngine

    result = rd._tool_check_printability({"object_name": "Nope"}, RealFreeCADEngine(), None)
    assert result["ok"] is False and "Object not found: Nope" in result["error"]


@pytest.mark.e2e
@_live
@pytest.mark.usefixtures("live_session")
def test_live_upright_l_bracket_is_recommended_flat() -> None:
    # Same upright L as the cantilever test: 300 mm² of arm overhangs as modelled.
    _padded("XZ", [(0, 0), (10, 0), (10, 30), (40, 30), (40, 40), (0, 40)], 10)
    report = _check()

    assert report["printable"] is True and report["current_orientation_printable"] is False
    best = report["recommended_orientation"]
    # Lying on its side: the whole 700 mm² L profile on the bed, no supports.
    assert best["up_axis"] in ("+Y", "-Y") and best["requires_supports"] is False
    assert best["bed_contact_mm2"] == pytest.approx(700.0, rel=1e-3)
    assert best["height_mm"] == pytest.approx(10.0, abs=1e-3)
    assert "no supports needed" in report["remediation_hint"]


@pytest.mark.e2e
@_live
@pytest.mark.usefixtures("live_session")
def test_live_jack_needs_supports_every_way() -> None:
    for name, size, corner in [
        ("BarX", (60, 10, 10), (-30, -5, -5)),
        ("BarY", (10, 60, 10), (-5, -30, -5)),
        ("BarZ", (10, 10, 60), (-5, -5, -30)),
    ]:
        assert json.loads(engine.create_box(*size, name=name, placement=corner))["ok"]
    assert json.loads(engine.apply_boolean("union", objects=["BarX", "BarY", "BarZ"], name="Jack"))["ok"]
    report = _check(object_name="Jack")

    assert report["printable"] is True and report["current_orientation_printable"] is False
    assert all(pose["requires_supports"] for pose in report["orientations"])
    assert report["remediation_hint"].startswith("Needs supports in every principal orientation")
