"""Parametric rebuild against the real FreeCADCmd: a plate with a face hole
and a chamfer is rebuilt with new dimensions, and an impossible dimension
fails without touching the live document."""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path

import pytest

from dana.core import react_dispatch as rd
from dana.platform import get_control_plane
from dana.plugins.freecad import engine
from dana.plugins.freecad import rebuild as rb
from dana.plugins.freecad.call_log import CadCallLog
from dana.session_context import DEFAULT_SESSION_ID, set_session_id
from dana.tools.schema import ToolCall

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        engine.detect_freecadcmd() is None,
        reason="real FreeCADCmd not found (PATH or DANA_FREECADCMD_PATH) — this test drives the real engine",
    ),
]

GEOMETRY = frozenset({"create_freecad_box"})


@pytest.fixture(autouse=True)
def _isolated_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(engine, "_OUTPUT_DIR", tmp_path / "freecad_output")
    monkeypatch.setenv("DANA_HEADLESS", "true")
    monkeypatch.delenv("DANA_OS_DRY_RUN", raising=False)
    monkeypatch.setattr(rd, "_get_has_plan", lambda *_a, **_k: True)
    set_session_id(f"rebuild-{uuid.uuid4().hex[:8]}")
    yield
    set_session_id(DEFAULT_SESSION_ID)


def test_rebuild_with_new_dimensions_and_reject_an_impossible_one() -> None:
    from dana.platform.win32 import RealFreeCADEngine

    log = CadCallLog()
    cad, control = RealFreeCADEngine(), get_control_plane()
    for tool, args in (
        ("create_freecad_box", {"name": "Plate", "length": 60, "width": 40, "height": 5}),
        ("create_freecad_feature_on_face", {"object_name": "Plate", "face": "top", "shape": "circle", "u": 0,
                                            "v": 0, "extent": 5, "operation": "cut", "radius": 4, "name": "Holed"}),
        ("perform_freecad_edge_operation", {"target_object": "Holed", "operation": "chamfer", "value": 0.5}),
    ):
        assert rd.dispatch_tool_call(ToolCall(tool, args), cad, control, call_log=log).ok, tool

    live = engine._session_document_path()
    params = {p["label"]: p["id"] for p in rb.parameter_table(log.records, GEOMETRY)["params"]}
    kwargs = dict(session_path=live, geometry_tool_ids=GEOMETRY, run_script=engine._run_freecad_script,
                  ok_marker=engine._OK_MARKER, backup=engine._backup_session_document, timeout=180)

    result = rb.rebuild(log, {params["Plate.length"]: 80, params["Plate top feature.radius"]: 6}, **kwargs)
    assert result["ok"], result
    assert result["bounding_box"][3] == pytest.approx(80.0)
    assert {"Plate", "Holed", "Chamfer"} <= (engine.list_document_objects(live) or set())

    before = hashlib.sha256(live.read_bytes()).hexdigest()
    bad = rb.rebuild(log, {params["Plate top feature.radius"]: 100}, **kwargs)  # wider than the plate
    assert bad["ok"] is False and "model left unchanged" in bad["error"]
    assert hashlib.sha256(live.read_bytes()).hexdigest() == before
    lengths = {p["label"]: p["value"] for p in rb.parameter_table(log.records, GEOMETRY)["params"]}
    assert lengths["Plate.length"] == 80.0 and lengths["Plate top feature.radius"] == 6.0


def test_pyramid_pipe_and_helix_build_in_the_session_document_and_rebuild() -> None:
    from dana.platform.win32 import RealFreeCADEngine

    log = CadCallLog()
    cad, control = RealFreeCADEngine(), get_control_plane()
    for tool, args in (
        ("create_freecad_pyramid", {"name": "Spire", "length": 20, "width": 20, "height": 30}),
        ("create_freecad_pipe", {"name": "Elbow", "pipe_radius": 4, "path_type": "arc", "length_or_angle": 90,
                                 "placement_x": 60}),
        ("create_freecad_helix", {"name": "Coil", "coil_radius": 10, "pitch": 5, "height": 20, "pipe_radius": 1.5,
                                  "angle_offset": 45, "placement_y": 60}),
    ):
        assert rd.dispatch_tool_call(ToolCall(tool, args), cad, control, call_log=log).ok, tool

    live = engine._session_document_path()
    assert {"Spire", "Elbow", "Coil"} <= (engine.list_document_objects(live) or set())
    table = rb.parameter_table(log.records, GEOMETRY)
    assert table["unsupported"] == []
    ids = {p["label"]: p["id"] for p in table["params"]}
    result = rb.rebuild(
        log, {ids["Spire.height"]: 45, ids["Elbow.pipe_radius"]: 6, ids["Coil.pitch"]: 8},
        session_path=live, geometry_tool_ids=GEOMETRY, run_script=engine._run_freecad_script,
        ok_marker=engine._OK_MARKER, backup=engine._backup_session_document, timeout=180,
    )
    assert result["ok"], result
    spire = json.loads(engine.get_bounding_box(str(live), "Spire"))
    elbow = json.loads(engine.get_bounding_box(str(live), "Elbow"))
    assert spire["z_max"] == pytest.approx(45.0)
    assert elbow["z_max"] == pytest.approx(6.0)  # the tube is now 6 mm in radius
