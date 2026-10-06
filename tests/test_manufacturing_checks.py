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
from dana.plugins.freecad.printability import DEFAULT_PRINTER, Facet, PrinterProfile, build_report
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


# -- live: real FreeCAD -------------------------------------------------------------------

_live = pytest.mark.skipif(
    engine.detect_freecadcmd() is None,
    reason="real FreeCADCmd not found (PATH or DANA_FREECADCMD_PATH) — this test drives the real engine",
)


@pytest.fixture
def live_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(engine, "_OUTPUT_DIR", tmp_path / "freecad_output")
    monkeypatch.setattr(engine, "_EXPORT_DIR", tmp_path / "exports")
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
    assert Path(report["stl_path"]).read_bytes()[:5] != b"solid"  # binary STL, not ASCII


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
