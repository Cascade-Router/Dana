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
        "id": "1.length", "label": "Plate.length", "value": 60.0, "unit": "mm", "min": 0,
        "warning": rb.DRESSUP_WARNING,
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
    assert "real FreeCAD" in event["reason"]  # the mock engine can list but not rebuild


def test_rebuild_on_the_mock_engine_reports_instead_of_pretending(client: TestClient) -> None:
    with client.websocket_connect("/ws/chat") as ws:
        _session_for(ws.receive_json())["call_log"] = _log()
        ws.send_json({"type": "rebuild", "overrides": {"1.length": 80}})
        event = ws.receive_json()
    assert event == {"type": "rebuild_result", "ok": False, "error": "Rebuild needs real FreeCAD (not available here)."}


def test_rebuild_without_overrides_is_rejected(client: TestClient) -> None:
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()
        ws.send_json({"type": "rebuild", "overrides": {}})
        assert ws.receive_json()["ok"] is False
