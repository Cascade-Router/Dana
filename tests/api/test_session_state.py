"""Session state persistence across server restarts (dana.api.session_state):
registries, plan, call log and the paired FreeCAD document survive a fresh
process; approvals and secrets never do."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from dana.api import server as server_module
from dana.api import session_state, sessions
from dana.core import react_dispatch as rd
from dana.platform import mock as mock_module
from dana.platform.mock import MockFreeCADEngine
from dana.plugins.freecad import engine as freecad_engine
from dana.plugins.freecad.call_log import CadCallLog
from dana.plugins.planning import task_board

SID = "sess-restore-1"


class _RealEngine:
    """Anything that is not MockFreeCADEngine takes the real-document path."""


@pytest.fixture(autouse=True)
def _clean_registries(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(freecad_engine, "_OUTPUT_DIR", tmp_path / "freecad_output")
    yield
    for registry in (*rd._PERSISTED_SESSION_REGISTRIES.values(), *mock_module._PERSISTED_MOCK_REGISTRIES.values()):
        registry.pop(SID, None)
    task_board._PLANS_BY_SESSION.pop(SID, None)


def _forget_process_state() -> None:
    """What a server restart does to the in-memory registries."""
    for registry in (*rd._PERSISTED_SESSION_REGISTRIES.values(), *mock_module._PERSISTED_MOCK_REGISTRIES.values()):
        registry.pop(SID, None)
    task_board._PLANS_BY_SESSION.pop(SID, None)


def _session(**extra: Any) -> dict[str, Any]:
    return {
        "session_id": SID,
        "call_log": CadCallLog(),
        "capability_unlocked_at_turn": {},
        "hidden_tool_ids": set(),
        "turn_counter": 0,
        **extra,
    }


def _seed_engine_state() -> CadCallLog:
    rd._OBJECT_PATH_REGISTRY[SID] = {"Plate": "doc.FCStd", "Hole": "doc.FCStd"}
    rd._TOPOLOGY_DAG_REGISTRY[SID] = {
        "nodes": {n: {"id": n, "label": n, "type": "solid"} for n in ("Plate", "Hole")},
        "edges": [{"source": "Plate", "target": "Hole"}],
    }
    rd._PLAN_STATE_REGISTRY[SID] = {
        "has_plan": True,
        "plan_text": "1. plate",
        "tasks": [{"id": 1, "status": "active", "executed_tools": {"create_freecad_box"}}],
    }
    rd._COLLISIONS_VALIDATED_BY_SESSION[SID] = {"Asm"}
    rd._PLAN_GATE_REGISTRY[SID] = (1, 3)
    task_board._PLANS_BY_SESSION[SID] = {**task_board._new_plan(), "objective": "make a plate"}
    log = CadCallLog()
    log.record("create_freecad_box", {"length": 10}, ok=True, result={"name": "Plate"})
    return log


def test_codec_round_trips_sets_tuples_and_non_string_keys() -> None:
    value = {"a": {1, 2}, "b": (1, "x"), "c": frozenset({"k"}), "d": {(1, 2): "pair"}, "e": [{"__x": None}]}
    assert session_state.decode(json.loads(json.dumps(session_state.encode(value)))) == value


def test_save_is_atomic_and_load_keeps_state() -> None:
    sessions.save_session(SID, title="t", created_at=None, messages=[], state={"version": 1, "k": [1]})
    assert sessions.load_session(SID)["state"] == {"version": 1, "k": [1]}
    assert [p.name for p in sessions.SESSIONS_DIR.iterdir()] == [f"{SID}.json"]  # no temp file left


def test_legacy_record_without_state_is_a_clean_no_op() -> None:
    sessions.SESSIONS_DIR.mkdir(parents=True)
    (sessions.SESSIONS_DIR / f"{SID}.json").write_text(json.dumps({"messages": []}), encoding="utf-8")
    stored = sessions.load_session(SID)
    assert stored["state"] is None
    session = _session()
    assert session_state.restore_session_state(session, stored["state"], MockFreeCADEngine()) is None
    assert rd.session_state_is_empty(SID)


def test_failed_write_keeps_the_previous_file(monkeypatch: pytest.MonkeyPatch) -> None:
    sessions.save_session(SID, title="first", created_at=None, messages=[])

    def _boom(*_a: Any, **_k: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(sessions.os, "replace", _boom)
    with pytest.raises(OSError):
        sessions.save_session(SID, title="second", created_at=None, messages=[])
    assert sessions.load_session(SID)["title"] == "first"
    assert [p.name for p in sessions.SESSIONS_DIR.iterdir()] == [f"{SID}.json"]


def test_restart_restores_registries_plan_and_call_log() -> None:
    log = _seed_engine_state()
    session = _session(call_log=log, turn_counter=7, hidden_tool_ids={"create_freecad_cone"})
    payload = json.loads(json.dumps(session_state.capture_session_state(session, _RealEngine())))
    _forget_process_state()

    fresh = _session()
    assert session_state.restore_session_state(fresh, payload, _RealEngine()) is None
    assert rd._OBJECT_PATH_REGISTRY[SID] == {"Plate": "doc.FCStd", "Hole": "doc.FCStd"}
    assert rd._PLAN_STATE_REGISTRY[SID]["tasks"][0]["executed_tools"] == {"create_freecad_box"}
    assert rd._COLLISIONS_VALIDATED_BY_SESSION[SID] == {"Asm"}
    assert rd._PLAN_GATE_REGISTRY[SID] == (1, 3)
    assert rd.get_topology_dag(SID)["edges"] == [{"source": "Plate", "target": "Hole"}]
    assert task_board._PLANS_BY_SESSION[SID]["objective"] == "make a plate"
    assert [r.tool_id for r in fresh["call_log"].records] == ["create_freecad_box"]
    assert fresh["turn_counter"] == 7 and fresh["hidden_tool_ids"] == {"create_freecad_cone"}


def test_same_process_reconnect_keeps_live_memory() -> None:
    log = _seed_engine_state()
    payload = session_state.capture_session_state(_session(call_log=log), _RealEngine())
    rd._OBJECT_PATH_REGISTRY[SID]["Newer"] = "doc.FCStd"  # built after the save
    fresh = _session()
    session_state.restore_session_state(fresh, payload, _RealEngine())
    assert "Newer" in rd._OBJECT_PATH_REGISTRY[SID]
    assert len(fresh["call_log"]) == 1  # the connection's own log still comes from disk


def test_secrets_approvals_and_suspended_turns_never_reach_disk() -> None:
    session = _session(
        session_title="t",
        session_created_at=None,
        chat_history=[{"role": "user", "content": "hi"}],
        hitl_approved_tools={"execute_freecad_script"},
        auto_approve=True,
        api_keys={"openrouter": "sk-or-SECRET"},
        react_state={"pending_call": "SUSPENDED-HITL-TURN"},
        visual_state={"pending": "SUSPENDED-VISUAL-TURN"},
    )
    server_module._save_session_record(session)
    text = (sessions.SESSIONS_DIR / f"{SID}.json").read_text(encoding="utf-8")
    for leaked in ("hitl_approved_tools", "auto_approve", "api_keys", "sk-or-SECRET", "SUSPENDED", "react_state"):
        assert leaked not in text

    # A forged state file cannot smuggle them back in either.
    forged = json.loads(text)["state"]
    forged["server"].update({"auto_approve": True, "hitl_approved_tools": ["x"], "api_keys": {"a": "b"}})
    fresh = _session(auto_approve=False, hitl_approved_tools=set(), api_keys={})
    session_state.restore_session_state(fresh, forged, _RealEngine())
    assert fresh["auto_approve"] is False and fresh["hitl_approved_tools"] == set() and fresh["api_keys"] == {}


def _live_doc() -> Path:
    path = freecad_engine.session_document_path_for(SID)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def test_session_without_a_document_records_none_and_restores_cleanly() -> None:
    payload = session_state.capture_session_state(_session(), _RealEngine())
    assert payload["document"] is None
    assert not freecad_engine.session_document_path_for(SID).parent.exists()  # probing created nothing
    assert session_state.restore_session_state(_session(), payload, _RealEngine()) is None


def test_save_snapshots_the_document_and_restore_copies_it_back() -> None:
    _seed_engine_state()
    live = _live_doc()
    live.write_bytes(b"geometry-v1")
    payload = session_state.capture_session_state(_session(), _RealEngine())
    assert session_state.document_snapshot_path(SID).read_bytes() == b"geometry-v1"

    live.write_bytes(b"another-session-overwrote-this")
    _forget_process_state()
    assert session_state.restore_session_state(_session(), payload, _RealEngine()) is None
    assert live.read_bytes() == b"geometry-v1"


def test_external_edit_resyncs_objects_from_the_document(monkeypatch: pytest.MonkeyPatch) -> None:
    _seed_engine_state()
    live = _live_doc()
    live.write_bytes(b"geometry-v1")
    payload = session_state.capture_session_state(_session(), _RealEngine())
    live.write_bytes(b"edited-in-freecad")
    session_state.document_snapshot_path(SID).unlink()
    _forget_process_state()
    calls: list[Path] = []
    monkeypatch.setattr(
        freecad_engine, "list_document_objects", lambda path: calls.append(path) or {"Plate", "Fillet"}
    )

    note = session_state.restore_session_state(_session(), payload, _RealEngine())
    assert calls == [live]
    assert "Hole" in note
    assert set(rd._OBJECT_PATH_REGISTRY[SID]) == {"Plate", "Fillet"}
    assert set(rd.get_topology_dag(SID)["nodes"]) == {"Plate"}
    assert rd.get_topology_dag(SID)["edges"] == []


def test_mock_engine_prunes_objects_whose_mesh_is_gone_without_shelling_out(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _seed_engine_state()
    kept = tmp_path / "plate.stl"
    kept.write_bytes(b"solid")
    mock_module._MOCK_OBJECT_REGISTRY[SID] = {"Plate": str(kept), "Hole": str(tmp_path / "gone.stl")}
    payload = session_state.capture_session_state(_session(), MockFreeCADEngine())
    assert "document" not in payload
    _forget_process_state()

    def _no_freecad(path: Path) -> None:
        raise AssertionError("must not shell out under the mock engine")

    monkeypatch.setattr(freecad_engine, "list_document_objects", _no_freecad)
    note = session_state.restore_session_state(_session(), payload, MockFreeCADEngine())
    assert "Hole" in note
    assert set(rd._OBJECT_PATH_REGISTRY[SID]) == {"Plate"}
    assert set(mock_module._MOCK_OBJECT_REGISTRY[SID]) == {"Plate"}


def test_ws_resume_seeds_ready_with_the_restored_state() -> None:
    from fastapi.testclient import TestClient

    _seed_engine_state()
    state = session_state.capture_session_state(_session(), server_module.get_cad_engine())
    sessions.save_session(SID, title="t", created_at=None, messages=[], state=state)
    _forget_process_state()

    with TestClient(server_module.app).websocket_connect(f"/ws/chat?session_id={SID}") as ws:
        ready = ws.receive_json()
    assert ready["type"] == "ready"
    assert set(ready["topology_dag"]["nodes"]) == {"Plate", "Hole"}
    assert ready["active_plan"]["objective"] == "make a plate"
