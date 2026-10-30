"""Zero-token parametric rebuild (dana.plugins.freecad.rebuild): parameters
come from the call log, overrides re-render the model in a scratch document,
and only a successful run replaces the session document and the log."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from dana.api import server as server_module
from dana.plugins.freecad import rebuild as rb
from dana.plugins.freecad.call_log import CadCallLog

GEOMETRY = frozenset({"create_freecad_box", "execute_freecad_script"})


def _log(*, centroid: bool = True) -> CadCallLog:
    log = CadCallLog()
    log.record("create_plan", {"tasks": ["plate"]}, ok=True)
    log.record(
        "create_freecad_box",
        {"name": "Plate", "length": 60, "width": 40, "height": 5},
        ok=True,
        result={"name": "Plate", "dimensions": {"length": 60.0, "width": 40.0, "height": 5.0}, "placement": [0, 0, 0]},
    )
    log.record(
        "create_freecad_feature_on_face",
        {"object_name": "Plate", "face": "top", "shape": "circle", "u": 0, "v": 0, "extent": 5,
         "operation": "cut", "radius": 3, "name": "Holed"},
        ok=True,
        result={"name": "Holed", "dimensions": {"u": 0.0, "v": 0.0, "extent": 5.0}},
    )
    log.record("generate_2d_blueprint", {"object_name": "Holed"}, ok=True, result={"path": "x.pdf"})
    log.record(
        "perform_freecad_edge_operation",
        {"target_object": "Holed", "operation": "fillet", "value": 1.0,
         **({"face_centroid": [30.0, 20.0, 5.0]} if centroid else {})},
        ok=True,
        result={"name": "Fillet", "type": "Part::Fillet"},
    )
    log.record("create_freecad_box", {"name": "Broken"}, ok=False, error="boom")
    return log


def _params(log: CadCallLog) -> dict[str, dict[str, Any]]:
    return {p["id"]: p for p in rb.parameter_table(log.records, GEOMETRY)["params"]}


def test_parameter_table_lists_dimensions_with_units_and_labels() -> None:
    params = _params(_log())
    assert params["1.length"] == {
        "id": "1.length", "label": "Plate.length", "value": 60.0, "baseline": 60.0, "modified": False,
        "unit": "mm", "min": 0, "warning": rb.DRESSUP_WARNING,
    }
    assert params["2.radius"]["label"] == "Plate top feature.radius"  # composite unrolled
    assert params["2.u"]["min"] is None  # positions may be negative
    assert params["4.value"]["label"] == "Fillet.radius"
    assert "warning" not in params["4.value"]  # nothing downstream of the fillet itself
    assert not any(pid.startswith("3.") or pid.startswith("5.") for pid in params)  # blueprint, failed call


def test_no_shift_warning_when_the_dressup_has_no_centroid_selection() -> None:
    assert "warning" not in _params(_log(centroid=False))["1.length"]


def test_non_replayable_geometry_is_reported_not_dropped() -> None:
    log = _log()
    log.record("execute_freecad_script", {"python_script_str": "..."}, ok=True, result={})
    table = rb.parameter_table(log.records, GEOMETRY)
    assert table["unsupported"] == ["execute_freecad_script"]
    assert "execute_freecad_script" in table["reason"]


class _Runner:
    """Stands in for engine._run_freecad_script: records the script and
    writes the scratch document it was told to save."""

    def __init__(self, ok: bool = True) -> None:
        self.ok = ok
        self.scripts: list[str] = []

    def __call__(self, script: str, *, timeout: float) -> dict[str, Any]:
        self.scripts.append(script)
        scratch = next(line.split("=", 1)[1].strip().strip("'\"") for line in script.splitlines()
                       if line.startswith("_session_path ="))
        Path(scratch).write_bytes(b"rebuilt")
        if not self.ok:
            return {"ok": False, "error": "Topology collapse"}
        return {"ok": True, "resolved_name": "Fillet", "bounding_box": [0, 0, 0, 80, 40, 5]}


def _rebuild(log: CadCallLog, overrides: dict[str, Any], tmp_path: Path, runner: _Runner) -> tuple[dict, Path, list]:
    live = tmp_path / "Session_Active.FCStd"
    live.write_bytes(b"original")
    backups: list[Path] = []
    result = rb.rebuild(
        log, overrides, session_path=live, geometry_tool_ids=GEOMETRY, run_script=runner,
        ok_marker="OK", backup=backups.append, timeout=1.0,
    )
    return result, live, backups


def test_rebuild_applies_overrides_swaps_the_document_and_updates_the_log(tmp_path: Path) -> None:
    log = _log()
    runner = _Runner()
    result, live, backups = _rebuild(log, {"1.length": 80, "2.radius": "4.5", "4.value": 2}, tmp_path, runner)

    assert result["ok"] is True and result["applied"] == {"1.length": 80.0, "2.radius": 4.5, "4.value": 2.0}
    script = runner.scripts[0]
    assert "80.0" in script and "4.5" in script
    assert live.read_bytes() == b"rebuilt" and backups == [live]
    assert not (tmp_path / "Session_Rebuild.FCStd").exists()
    params = _params(log)  # re-derived from the rewritten records
    assert (params["1.length"]["value"], params["2.radius"]["value"], params["4.value"]["value"]) == (80.0, 4.5, 2.0)
    assert params["1.width"]["value"] == 40.0


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"1.length": -5}, "greater than 0"),
        ({"1.length": "abc"}, "not a number"),
        ({"1.length": float("nan")}, "finite"),
        ({"9.length": 5}, "unknown parameter"),
        ({"1.placement": 5}, "unknown parameter"),
    ],
)
def test_invalid_values_are_rejected_before_freecad_runs(
    tmp_path: Path, overrides: dict[str, Any], message: str
) -> None:
    log = _log()
    before = log.records
    runner = _Runner()
    result, live, _ = _rebuild(log, overrides, tmp_path, runner)
    assert result["ok"] is False and message in result["error"]
    assert runner.scripts == [] and live.read_bytes() == b"original" and log.records == before


def test_failed_rebuild_leaves_the_model_and_log_untouched(tmp_path: Path) -> None:
    log = _log()
    before = log.records
    result, live, backups = _rebuild(log, {"2.radius": 100}, tmp_path, _Runner(ok=False))
    assert result["ok"] is False and "model left unchanged" in result["error"]
    assert live.read_bytes() == b"original" and backups == [] and log.records == before
    assert not (tmp_path / "Session_Rebuild.FCStd").exists()


def test_unsupported_log_is_refused(tmp_path: Path) -> None:
    log = _log()
    log.record("execute_freecad_script", {"python_script_str": "..."}, ok=True, result={})
    runner = _Runner()
    result, live, _ = _rebuild(log, {"1.length": 80}, tmp_path, runner)
    assert result["ok"] is False and result["unsupported"] == ["execute_freecad_script"]
    assert runner.scripts == [] and live.read_bytes() == b"original"


def test_write_back_round_trips_modify_parameter_and_pad() -> None:
    log = CadCallLog()
    log.record("modify_freecad_parameter", {"target_object": "Plate", "parameter_name": "Length", "new_value": 70},
               ok=True, result={"parameter_name": "Length", "new_value": 70.0})
    log.record("create_freecad_pad", {"sketch_name": "Sketch", "length": 10, "name": "Pad"}, ok=True,
               result={"name": "Pad", "dimensions": {"length": 10.0}})
    plan = rb.plan_rebuild(log.records, GEOMETRY)
    rb._apply_overrides(plan, {"0.new_value": 75, "1.length": 12})
    records = [rb._write_back(log.records[0], "new_value", 75.0), rb._write_back(log.records[1], "length", 12.0)]
    log.replace(records)
    params = _params(log)
    assert params["0.new_value"]["value"] == 75.0 and params["0.new_value"]["unit"] == "mm"
    assert params["1.length"]["value"] == 12.0


# --- WebSocket -------------------------------------------------------------------


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from dana.platform.mock import MockFreeCADEngine

    monkeypatch.setattr(server_module, "get_cad_engine", lambda: MockFreeCADEngine())
    return TestClient(server_module.app)


def _session_for(ready: dict[str, Any]) -> dict[str, Any]:
    return next(s for s in server_module._active_sessions.values() if s["session_id"] == ready["session_id"])


def test_list_parameters_over_websocket(client: TestClient) -> None:
    with client.websocket_connect("/ws/chat") as ws:
        _session_for(ws.receive_json())["call_log"] = _log()
        ws.send_json({"type": "list_parameters"})
        event = ws.receive_json()
    assert event["type"] == "parameters"
    assert {"1.length", "2.radius", "4.value"} <= {p["id"] for p in event["params"]}
    assert event["reason"] is None


def _mock_log(*, plate_length: float = 60.0) -> tuple[CadCallLog, Any]:
    """A real call log built by dispatching through the mock driver."""
    from dana.core import react_dispatch as rd
    from dana.platform.mock import MockFreeCADEngine
    from dana.tools.schema import ToolCall

    engine = MockFreeCADEngine()
    log = CadCallLog()
    for tool, args in (
        ("create_freecad_box", {"name": "Plate", "length": plate_length, "width": 40, "height": 5}),
        ("create_freecad_pyramid", {"name": "Spire", "length": 10, "width": 10, "height": 20}),
        ("create_freecad_pipe", {"name": "Rod", "pipe_radius": 2, "path_type": "straight", "length_or_angle": 30}),
    ):
        assert rd.dispatch_tool_call(ToolCall(tool, args), engine, None, call_log=log).ok, tool
    return log, engine


@pytest.fixture
def mock_session(monkeypatch: pytest.MonkeyPatch):
    from dana.core import react_dispatch as rd
    from dana.platform import mock as mock_driver
    from dana.session_context import DEFAULT_SESSION_ID, set_session_id

    sid = "rebuild-mock"
    monkeypatch.setattr(rd, "_get_has_plan", lambda *_a, **_k: True)
    set_session_id(sid)
    yield sid
    set_session_id(DEFAULT_SESSION_ID)
    mock_driver.replace_mock_session(sid, {})
    for registry in rd._PERSISTED_SESSION_REGISTRIES.values():
        registry.pop(sid, None)


def test_mock_engine_rebuild_replays_the_patched_log(mock_session: str) -> None:
    from dana.core import react_dispatch as rd
    from dana.platform import mock as mock_driver

    log, engine = _mock_log()
    result = rb.rebuild_mock(
        log, {"0.length": 80, "2.pipe_radius": 3}, session_id=mock_session, engine=engine,
        handlers=rd.TOOL_HANDLERS, geometry_tool_ids=GEOMETRY,
    )
    assert result["ok"] is True, result
    assert result["name"] == "Rod" and Path(result["path"]).is_file()
    plate_mesh = mock_driver._MOCK_OBJECT_REGISTRY[mock_session]["Plate"]
    import trimesh

    assert trimesh.load(plate_mesh).extents[0] == pytest.approx(80.0)
    params = _params(log)
    assert params["0.length"]["value"] == 80.0 and params["2.pipe_radius"]["value"] == 3.0


def test_mock_engine_rebuild_failure_restores_the_previous_state(mock_session: str) -> None:
    from dana.core import react_dispatch as rd
    from dana.platform import mock as mock_driver

    log, engine = _mock_log()
    before_state = mock_driver.snapshot_mock_session(mock_session)
    before_log = log.records

    def _broken_pyramid(*_a: Any, **_k: Any) -> dict[str, Any]:
        return {"ok": False, "error": "mesh boolean failed"}

    handlers = {**rd.TOOL_HANDLERS, "create_freecad_pyramid": _broken_pyramid}
    result = rb.rebuild_mock(
        log, {"0.length": 80}, session_id=mock_session, engine=engine, handlers=handlers, geometry_tool_ids=GEOMETRY
    )
    assert result["ok"] is False and "model left unchanged" in result["error"]
    assert mock_driver.snapshot_mock_session(mock_session) == before_state
    assert log.records == before_log


def test_rebuild_over_websocket_on_the_mock_engine(client: TestClient, mock_session: str) -> None:
    log, _engine = _mock_log()
    with client.websocket_connect("/ws/chat") as ws:
        session = _session_for(ws.receive_json())
        session["session_id"] = mock_session  # the session the log was built in
        session["call_log"] = log
        ws.send_json({"type": "rebuild", "overrides": {"0.length": 75}})
        result = ws.receive_json()
        refreshed = ws.receive_json()
    assert result["type"] == "rebuild_result" and result["ok"] is True, result
    assert result["mesh_url"] and result["applied"] == {"0.length": 75.0}
    assert result["zero_token_rebuilds"] == 1
    assert refreshed["type"] == "parameters"
    assert next(p for p in refreshed["params"] if p["id"] == "0.length")["value"] == 75.0


def test_pipe_bend_radius_follows_an_overridden_pipe_radius() -> None:
    log = CadCallLog()
    log.record("create_freecad_pipe", {"pipe_radius": 4, "path_type": "arc", "length_or_angle": 90}, ok=True,
               result={"name": "Elbow", "dimensions": {"pipe_radius": 4.0, "path_type": "arc", "length_or_angle": 90.0}})
    plan = rb.plan_rebuild(log.records, GEOMETRY)
    assert plan.steps[0]["arc_radius"] == 20.0  # 4 * 3 is below the 20 mm minimum
    rb._apply_overrides(plan, {"0.pipe_radius": 10})
    assert plan.steps[0]["arc_radius"] == 30.0
    assert _params(log)["0.length_or_angle"]["unit"] == "deg"


def test_rebuild_without_overrides_is_rejected(client: TestClient) -> None:
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()
        ws.send_json({"type": "rebuild", "overrides": {}})
        assert ws.receive_json()["ok"] is False


def test_rebuild_remembers_the_agents_value_as_the_baseline() -> None:
    """The Parameters tab's Reset: the first rebuild that changes a field
    records the agent's original value; later edits keep it; rebuilding back
    to it clears "modified"."""
    log = _log()
    records = log.records
    applied = {(1, "length"): 80.0}
    patched = rb._patched_records(records, applied, {(1, "length"): 60.0})
    assert patched[1].baseline == {"length": 60.0}
    log.replace(patched)
    row = _params(log)["1.length"]
    assert (row["value"], row["baseline"], row["modified"]) == (80.0, 60.0, True)

    again = rb._patched_records(log.records, {(1, "length"): 95.0}, {(1, "length"): 80.0})
    assert again[1].baseline == {"length": 60.0}  # still the agent's value, not the previous edit
    log.replace(rb._patched_records(again, {(1, "length"): 60.0}, {(1, "length"): 95.0}))
    row = _params(log)["1.length"]
    assert (row["value"], row["modified"]) == (60.0, False)

    restored = type(log).from_payload(log.to_payload())
    assert restored.records[1].baseline == {"length": 60.0}  # survives a restart
