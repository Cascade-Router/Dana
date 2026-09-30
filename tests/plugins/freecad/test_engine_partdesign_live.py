"""Live tests for the real-FreeCAD Sketcher/PartDesign tools in
dana.plugins.freecad.engine (create_sketch ... create_loft).

These drive the actual FreeCADCmd binary, because the point is the gap the
mock hid: until these existed, RealFreeCADEngine called functions this module
didn't have. Geometry is checked by reopening the saved session document and
measuring volumes/bounding boxes, not by trusting the tool's own payload.
Skipped where FreeCAD isn't installed (e.g. CI).
"""

from __future__ import annotations

import json
import math
import re
import uuid
from pathlib import Path
from typing import Any

import pytest

from dana.plugins.freecad import engine
from dana.session_context import DEFAULT_SESSION_ID, set_session_id

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        engine.detect_freecadcmd() is None,
        reason="real FreeCADCmd not found (PATH or DANA_FREECADCMD_PATH) — this test drives the real engine",
    ),
]


@pytest.fixture(autouse=True)
def _isolated_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(engine, "_OUTPUT_DIR", tmp_path / "freecad_output")
    monkeypatch.setattr(engine, "_EXPORT_DIR", tmp_path / "exports")
    monkeypatch.setenv("DANA_HEADLESS", "true")
    monkeypatch.delenv("DANA_OS_DRY_RUN", raising=False)
    set_session_id(f"partdesign-{uuid.uuid4().hex[:8]}")
    yield
    set_session_id(DEFAULT_SESSION_ID)


def _ok(raw: str) -> dict[str, Any]:
    result = json.loads(raw)
    assert result.get("ok") is True, result
    return result


def _err(raw: str) -> str:
    result = json.loads(raw)
    assert result.get("ok") is False, result
    return result["error"]


def _measure(*names: str) -> dict[str, dict[str, Any]]:
    """Reopen the saved session document in FreeCADCmd and report each named
    object's volume, bounding box, validity and TypeId."""
    script = (
        "import FreeCAD as App, json\n"
        f"doc = App.openDocument({str(engine._session_document_path())!r})\n"
        "out = {}\n"
        f"for n in {list(names)!r}:\n"
        "    o = doc.getObject(n)\n"
        "    if o is None:\n"
        "        out[n] = None\n"
        "        continue\n"
        "    s = o.Shape\n"
        "    b = s.BoundBox\n"
        "    out[n] = {'type': o.TypeId, 'valid': o.isValid(), 'volume': s.Volume if s.Solids else 0.0,\n"
        "              'bbox': [b.XMin, b.YMin, b.ZMin, b.XMax, b.YMax, b.ZMax],\n"
        "              'constraints': len(o.Constraints) if hasattr(o, 'Constraints') else None,\n"
        "              'tip': o.Tip.Name if getattr(o, 'Tip', None) is not None else None}\n"
        f"print({engine._OK_MARKER!r} + '_MEASURE ' + json.dumps(out))\n"
    )
    result = engine._run_freecad_script(script)
    assert result["ok"], result
    match = re.search(re.escape(engine._OK_MARKER + "_MEASURE ") + r"(\{.*\})", result["stdout"])
    assert match, result["stdout"][-2000:]
    return json.loads(match.group(1))


def _rect(x0: float, y0: float, x1: float, y1: float) -> list[dict[str, Any]]:
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return [
        {"type": "line", "start": list(corners[i]), "end": list(corners[(i + 1) % 4])} for i in range(4)
    ]


def _bbox_close(actual: list[float], expected: list[float], tol: float = 1e-4) -> bool:
    return all(abs(a - e) <= tol for a, e in zip(actual, expected))


# -- create_sketch --------------------------------------------------------------


def test_create_sketch_builds_a_real_sketch_and_reports_dof() -> None:
    result = _ok(engine.create_sketch("Profile", "XY", _rect(0, 0, 20, 10)))
    assert result["name"] == "Profile"
    assert result["type"] == "Sketcher::SketchObject"
    assert result["sketch_fully_constrained"] is False
    assert result["sketch_dof"] and result["sketch_dof"] > 0
    assert "warning" in result
    measured = _measure("Profile")["Profile"]
    assert measured["type"] == "Sketcher::SketchObject"
    assert _bbox_close(measured["bbox"], [0, 0, 0, 20, 10, 0])


def test_create_sketch_maps_xz_and_yz_planes() -> None:
    _ok(engine.create_sketch("OnXZ", "XZ", [{"type": "line", "start": [0, 0], "end": [10, 5]}]))
    _ok(engine.create_sketch("OnYZ", "YZ", [{"type": "line", "start": [0, 0], "end": [10, 5]}]))
    measured = _measure("OnXZ", "OnYZ")
    assert _bbox_close(measured["OnXZ"]["bbox"], [0, 0, 0, 10, 0, 5])
    assert _bbox_close(measured["OnYZ"]["bbox"], [0, 0, 0, 0, 10, 5])


def test_create_sketch_rejects_malformed_geometry_before_running_freecad() -> None:
    assert "unknown type" in _err(engine.create_sketch("Bad", "XY", [{"type": "rectangle"}]))
    assert "missing or invalid field" in _err(engine.create_sketch("Bad", "XY", [{"type": "circle", "radius": 3}]))
    assert "unknown plane" in _err(engine.create_sketch("Bad", "AB", _rect(0, 0, 1, 1)))
    assert not engine._session_document_path().exists()


# -- apply_sketch_constraint ------------------------------------------------------


def test_constraints_reduce_dof_and_conflicts_fail_without_saving() -> None:
    before = _ok(engine.create_sketch("Profile", "XY", _rect(0, 0, 20, 10)))["sketch_dof"]
    after = _ok(engine.apply_sketch_constraint("Profile", "Horizontal", [0]))
    assert after["sketch_dof"] < before

    # Vertical on the line just made horizontal cannot solve: the error comes
    # back and the saved document keeps only the one good constraint.
    error = _err(engine.apply_sketch_constraint("Profile", "Vertical", [0]))
    assert "does not solve" in error or "conflict" in error.lower()
    assert _measure("Profile")["Profile"]["constraints"] == 1


def test_constraint_errors_bubble_up() -> None:
    _ok(engine.create_sketch("Profile", "XY", _rect(0, 0, 20, 10)))
    assert "out of range" in _err(engine.apply_sketch_constraint("Profile", "Distance", [7], value=5))
    assert "requires a numeric value" in _err(engine.apply_sketch_constraint("Profile", "Distance", [0]))
    _ok(engine.create_sketch("Hole", "XY", [{"type": "circle", "center": [5, 5], "radius": 2}]))
    radius = _ok(engine.apply_sketch_constraint("Hole", "Radius", [0], value=3))
    assert radius["sketch_dof"] == 2  # only the center is still free


# -- create_pad / create_pocket ---------------------------------------------------


def test_pad_extrudes_a_real_solid() -> None:
    _ok(engine.create_sketch("Profile", "XY", _rect(0, 0, 20, 10)))
    pad = _ok(engine.create_pad("Profile", 5))
    assert pad["type"] == "PartDesign::Pad"
    assert pad["body"] == "Body"
    # Regression: the sketch's DoF read 0 (falsely "fully constrained") once
    # it had been moved into the Body.
    assert pad["sketch_dof"] and pad["sketch_dof"] > 0
    assert pad["sketch_fully_constrained"] is False
    measured = _measure(pad["name"])[pad["name"]]
    assert math.isclose(measured["volume"], 20 * 10 * 5, rel_tol=1e-6)
    assert _bbox_close(measured["bbox"], [0, 0, 0, 20, 10, 5])


def test_symmetric_pad_straddles_the_sketch_plane() -> None:
    _ok(engine.create_sketch("Profile", "XY", _rect(0, 0, 10, 10)))
    pad = _ok(engine.create_pad("Profile", 6, symmetric_to_plane=True))
    assert _bbox_close(_measure(pad["name"])[pad["name"]]["bbox"], [0, 0, -3, 10, 10, 3])


def test_pad_of_an_open_profile_reports_freecads_reason() -> None:
    _ok(engine.create_sketch("OpenLine", "XY", [{"type": "line", "start": [0, 0], "end": [10, 0]}]))
    error = _err(engine.create_pad("OpenLine", 5))
    assert "Wire is not closed" in error
    assert _measure("Pad")["Pad"] is None  # nothing half-built was saved


def test_pocket_cuts_the_padded_solid() -> None:
    _ok(engine.create_sketch("Plate", "XY", _rect(-20, -20, 20, 20)))
    _ok(engine.create_pad("Plate", 4))
    _ok(engine.create_sketch("Hole", "XY", [{"type": "circle", "center": [0, 0], "radius": 5}]))
    pocket = _ok(engine.create_pocket("Hole", 1, through_all=True))
    assert pocket["type"] == "PartDesign::Pocket"
    # The hole sketch sits on the pad's base plane, so the default direction
    # cuts into empty space; the engine must notice and flip it.
    assert pocket["auto_reversed"] is True
    volume = _measure(pocket["name"])[pocket["name"]]["volume"]
    assert math.isclose(volume, 40 * 40 * 4 - math.pi * 25 * 4, rel_tol=1e-6)


def test_pocket_that_misses_the_solid_fails_instead_of_silently_succeeding() -> None:
    _ok(engine.create_sketch("Plate", "XY", _rect(-20, -20, 20, 20)))
    _ok(engine.create_pad("Plate", 4))
    _ok(engine.create_sketch("FarHole", "XY", [{"type": "circle", "center": [100, 100], "radius": 5}]))
    assert "removed no material" in _err(engine.create_pocket("FarHole", 1, through_all=True))
    assert _measure("Pocket")["Pocket"] is None


def test_pocket_without_a_padded_solid_fails_clearly() -> None:
    _ok(engine.create_sketch("Hole", "XY", [{"type": "circle", "center": [0, 0], "radius": 5}]))
    assert "no active body solid" in _err(engine.create_pocket("Hole", 3))


# -- patterns ----------------------------------------------------------------------


def _plate_with_hole() -> str:
    _ok(engine.create_sketch("Plate", "XY", _rect(-30, -30, 30, 30)))
    _ok(engine.create_pad("Plate", 4))
    _ok(engine.create_sketch("Hole", "XY", [{"type": "circle", "center": [15, 0], "radius": 3}]))
    return _ok(engine.create_pocket("Hole", 1, through_all=True))["name"]


def test_polar_pattern_repeats_a_pocket_around_z() -> None:
    pocket = _plate_with_hole()
    pattern = _ok(engine.create_polar_pattern(pocket, 4))
    assert pattern["type"] == "PartDesign::PolarPattern"
    measured = _measure(pattern["name"], "Body")
    expected = 60 * 60 * 4 - 4 * math.pi * 9 * 4
    assert math.isclose(measured[pattern["name"]]["volume"], expected, rel_tol=1e-6)
    # Regression: the pattern must become the Body's Tip, so the part's final
    # shape actually includes it.
    assert measured["Body"]["tip"] == pattern["name"]
    assert math.isclose(measured["Body"]["volume"], expected, rel_tol=1e-6)


def test_features_after_a_pattern_build_on_the_patterned_solid() -> None:
    pocket = _plate_with_hole()
    pattern = _ok(engine.create_polar_pattern(pocket, 4))["name"]
    _ok(engine.create_sketch("Boss", "XY", [{"type": "circle", "center": [0, 0], "radius": 4}]))
    boss = _ok(engine.create_pad("Boss", 10))["name"]
    measured = _measure(pattern, boss, "Body")
    assert measured["Body"]["tip"] == boss
    # 4 mm of the boss overlaps the plate; only the 6 mm above it adds volume.
    expected = measured[pattern]["volume"] + math.pi * 16 * 6
    assert math.isclose(measured["Body"]["volume"], expected, rel_tol=1e-6)


def test_linear_pattern_repeats_a_pocket_along_y() -> None:
    pocket = _plate_with_hole()
    pattern = _ok(engine.create_linear_pattern(pocket, 3, 20, direction="Y"))
    assert pattern["type"] == "PartDesign::LinearPattern"
    volume = _measure(pattern["name"])[pattern["name"]]["volume"]
    assert math.isclose(volume, 60 * 60 * 4 - 3 * math.pi * 9 * 4, rel_tol=1e-6)


def test_pattern_of_a_sketch_is_rejected() -> None:
    _plate_with_hole()
    assert "not a PartDesign feature" in _err(engine.create_polar_pattern("Hole", 4))


# -- sweep / loft ------------------------------------------------------------------


def test_sweep_carries_a_profile_along_a_path() -> None:
    _ok(engine.create_sketch("Section", "XY", [{"type": "circle", "center": [0, 0], "radius": 2}]))
    _ok(engine.create_sketch("Spine", "XZ", [{"type": "line", "start": [0, 0], "end": [0, 20]}]))
    sweep = _ok(engine.create_sweep("Section", "Spine"))
    assert sweep["type"] == "PartDesign::AdditivePipe"
    volume = _measure(sweep["name"])[sweep["name"]]["volume"]
    assert math.isclose(volume, math.pi * 4 * 20, rel_tol=1e-4)


def test_loft_blends_offset_sections_and_rejects_flat_ones() -> None:
    _ok(engine.create_sketch("Base", "XY", _rect(-10, -10, 10, 10)))
    _ok(engine.create_sketch("Top", "XY", [{"type": "circle", "center": [0, 0], "radius": 5}]))
    flat = _err(engine.create_loft(["Base", "Top"]))
    assert "AdditiveLoft" in flat

    _ok(engine.modify_parameter("Top", "Placement", [0, 0, 15]))
    loft = _ok(engine.create_loft(["Base", "Top"]))
    measured = _measure(loft["name"])[loft["name"]]
    assert measured["valid"] is True
    assert _bbox_close(measured["bbox"], [-10, -10, 0, 10, 10, 15], tol=1e-3)


def test_loft_argument_checks() -> None:
    assert "not a single string" in _err(engine.create_loft("Base"))
    assert "at least 2" in _err(engine.create_loft(["Base"]))
    assert "distinct" in _err(engine.create_loft(["Base", "Base"]))


# -- macro export parity ------------------------------------------------------------
#
# Build a session through the real engine while recording a CadCallLog the way
# dispatch_tool_call does (ReAct tool_id, resolved arguments, the engine's own
# result payload), export the macro, run it in a FRESH document, and require
# the replayed part to match the live one.

from dana.plugins.freecad import py_export  # noqa: E402
from dana.plugins.freecad.call_log import CadCallLog  # noqa: E402


def _run_logged(log: CadCallLog, tool_id: str, arguments: dict[str, Any], raw: str) -> dict[str, Any]:
    result = _ok(raw)
    log.record(tool_id, arguments, ok=True, result=result)
    return result


def _replay_measure(log: CadCallLog, *names: str) -> dict[str, dict[str, Any]]:
    script = py_export.render_macro_script(log) + (
        "\nimport json as _json\n"
        "_out = {}\n"
        f"for _n in {list(names)!r}:\n"
        "    _o = doc.getObject(_n)\n"
        "    if _o is None:\n"
        "        _out[_n] = None\n"
        "        continue\n"
        "    _out[_n] = {'volume': _o.Shape.Volume if _o.Shape.Solids else 0.0, 'valid': _o.isValid(),\n"
        "                'tip': _o.Tip.Name if getattr(_o, 'Tip', None) is not None else None}\n"
        f"print({engine._OK_MARKER!r} + '_REPLAY ' + _json.dumps(_out))\n"
    )
    result = engine._run_freecad_script(script)
    assert result["ok"], result["error"]
    match = re.search(re.escape(engine._OK_MARKER + "_REPLAY ") + r"(\{.*\})", result["stdout"])
    assert match, result["stdout"][-2000:]
    return json.loads(match.group(1))


def test_macro_replays_a_pad_pocket_pattern_session_identically() -> None:
    log = CadCallLog()
    _run_logged(log, "create_freecad_sketch", {"name": "Plate", "plane": "XY", "geometry": _rect(-30, -30, 30, 30)},
                engine.create_sketch("Plate", "XY", _rect(-30, -30, 30, 30)))
    _run_logged(log, "apply_sketch_constraint",
                {"sketch_name": "Plate", "constraint_type": "Horizontal", "geometry_indices": [0]},
                engine.apply_sketch_constraint("Plate", "Horizontal", [0]))
    _run_logged(log, "create_freecad_pad", {"sketch_name": "Plate", "length": 4}, engine.create_pad("Plate", 4))
    hole = [{"type": "circle", "center": [15, 0], "radius": 3}]
    _run_logged(log, "create_freecad_sketch", {"name": "Hole", "plane": "XY", "geometry": hole},
                engine.create_sketch("Hole", "XY", hole))
    pocket = _run_logged(log, "create_freecad_pocket", {"sketch_name": "Hole", "depth": 1, "through_all": True},
                         engine.create_pocket("Hole", 1, through_all=True))
    assert pocket["auto_reversed"] is True
    polar = _run_logged(log, "create_freecad_polar_pattern", {"feature_name": pocket["name"], "occurrences": 4},
                        engine.create_polar_pattern(pocket["name"], 4))
    boss = [{"type": "circle", "center": [0, 0], "radius": 4}]
    _run_logged(log, "create_freecad_sketch", {"name": "Boss", "plane": "XY", "geometry": boss},
                engine.create_sketch("Boss", "XY", boss))
    last = _run_logged(log, "create_freecad_pad", {"sketch_name": "Boss", "length": 10}, engine.create_pad("Boss", 10))

    live = _measure("Body", polar["name"], last["name"])
    replay = _replay_measure(log, "Body", polar["name"], last["name"])
    assert replay["Body"]["tip"] == live["Body"]["tip"] == last["name"]
    for name in ("Body", polar["name"], last["name"]):
        assert replay[name]["valid"] is True
        assert math.isclose(replay[name]["volume"], live[name]["volume"], rel_tol=1e-9), name


def test_macro_replays_sweep_and_loft_sessions_identically() -> None:
    log = CadCallLog()
    section = [{"type": "circle", "center": [0, 0], "radius": 2}]
    spine = [{"type": "line", "start": [0, 0], "end": [0, 20]}]
    _run_logged(log, "create_freecad_sketch", {"name": "Section", "plane": "XY", "geometry": section},
                engine.create_sketch("Section", "XY", section))
    _run_logged(log, "create_freecad_sketch", {"name": "Spine", "plane": "XZ", "geometry": spine},
                engine.create_sketch("Spine", "XZ", spine))
    sweep = _run_logged(log, "create_freecad_sweep", {"profile_sketch": "Section", "path_sketch": "Spine"},
                        engine.create_sweep("Section", "Spine"))
    _run_logged(log, "create_freecad_sketch", {"name": "Base", "plane": "XY", "geometry": _rect(-10, -10, 10, 10)},
                engine.create_sketch("Base", "XY", _rect(-10, -10, 10, 10)))
    top = [{"type": "circle", "center": [0, 0], "radius": 5}]
    _run_logged(log, "create_freecad_sketch", {"name": "Top", "plane": "XY", "geometry": top},
                engine.create_sketch("Top", "XY", top))
    _run_logged(log, "modify_freecad_parameter",
                {"target_object": "Top", "parameter_name": "Placement", "new_value": [0, 0, 30]},
                engine.modify_parameter("Top", "Placement", [0, 0, 30]))
    loft = _run_logged(log, "create_freecad_loft", {"cross_section_sketches": ["Base", "Top"]},
                       engine.create_loft(["Base", "Top"]))

    live = _measure("Body", sweep["name"], loft["name"])
    replay = _replay_measure(log, "Body", sweep["name"], loft["name"])
    assert replay["Body"]["tip"] == live["Body"]["tip"] == loft["name"]
    for name in ("Body", sweep["name"], loft["name"]):
        assert replay[name]["valid"] is True
        assert math.isclose(replay[name]["volume"], live[name]["volume"], rel_tol=1e-9), name
